from datetime import datetime
import os
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from .architectures import HACQIModel
from .losses.dice import DICELoss
from .losses.focal import FocalLoss


def build_hacqi_model(backbone_name="efficientnet_b0", fpn_channels=128, n_layers=None, **kwargs):
    """构建 HA-CQI 网络主体。"""
    return HACQIModel(backbone=backbone_name, fpn_channels=fpn_channels, n_layers=n_layers, **kwargs)


def resolve_unique_run_name(checkpoint_dir: str, base_name: str) -> str:
    """为训练实验名追加日期后缀，并在重名时递增序号。"""
    date_suffix = datetime.now().strftime("%Y%m%d")
    candidate_name = f"{base_name}-{date_suffix}"
    candidate_dir = os.path.join(checkpoint_dir, candidate_name)
    if not os.path.exists(candidate_dir):
        return candidate_name

    index = 1
    while True:
        candidate_name = f"{base_name}-{date_suffix}-{index}"
        candidate_dir = os.path.join(checkpoint_dir, candidate_name)
        if not os.path.exists(candidate_dir):
            return candidate_name
        index += 1


class HACQIEngine(nn.Module):
    """封装 HA-CQI 的优化器、损失、checkpoint 与推理接口。"""

    AUX_BASE_WEIGHTS = (1.0, 1.0, 0.5, 0.25, 0.25)

    def __init__(self, opt):
        super().__init__()
        use_cuda = torch.cuda.is_available() and len(getattr(opt, "gpu_ids", [])) > 0
        self.device = torch.device("cuda:%s" % opt.gpu_ids[0] if use_cuda else "cpu")
        self.opt = opt
        self.base_lr = opt.lr

        resolved_name = opt.name
        if getattr(opt, "phase", "train") == "train":
            resolved_name = resolve_unique_run_name(opt.checkpoint_dir, opt.name)
        self.opt.name = resolved_name
        self.save_dir = os.path.join(opt.checkpoint_dir, resolved_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"save_dir resolved to: {self.save_dir}")

        self.model = build_hacqi_model(
            backbone_name=opt.backbone,
            backbone_weight=opt.backbone_weight,
            fpn_channels=opt.fpn_channels,
            deform_groups=opt.deform_groups,
            gamma_mode=opt.gamma_mode,
            beta_mode=opt.beta_mode,
            n_layers=getattr(opt, "n_layers", None),
            disable_soft_alignment=bool(getattr(opt, "disable_soft_alignment", False)),
            align_window=opt.align_window,
            align_points=opt.align_points,
            align_heads=opt.align_heads,
            align_on_levels=opt.align_on_levels,
            align_qkv_bias=opt.align_qkv_bias,
            align_offset_groups=opt.align_offset_groups,
            num_change_queries=int(getattr(opt, "num_change_queries", 16)),
            cqi_heads=int(getattr(opt, "cqi_heads", 4)),
            mask_dim=int(getattr(opt, "mask_dim", 128)),
            mask_queries=int(getattr(opt, "mask_queries", 32)),
            mask_decoder_layers=int(getattr(opt, "mask_decoder_layers", 3)),
            mask_heads=int(getattr(opt, "mask_heads", 4)),
            dino_arch=opt.dino_arch,
            extract_ids=opt.extract_ids,
            dino_weight=opt.dino_weight,
            device=self.device,
        )
        self.aux_base_weights = [
            weight / sum(self.AUX_BASE_WEIGHTS) for weight in self.AUX_BASE_WEIGHTS
        ]
        self.focal = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        self.dice = DICELoss()
        self.last_loss_stats = {}
        self.optimizer = self._build_optimizer(opt)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, opt.num_epochs, eta_min=1e-7
        )
        self.schedular = self.scheduler
        if opt.load_pretrain:
            self.load_ckpt(self.model, self.optimizer, opt.name, opt.backbone)
        self.model.to(self.device)
        print("---------- HA-CQI network initialized -------------")

    def _amp_autocast_context(self):
        amp_enabled = bool(getattr(self.opt, "amp", False)) and self.device.type == "cuda"
        if not amp_enabled:
            return nullcontext()
        amp_dtype = torch.bfloat16 if str(getattr(self.opt, "amp_dtype", "fp16")).lower() == "bf16" else torch.float16
        return torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=True)

    def _loss_autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.amp.autocast(device_type=self.device.type, enabled=False)

    def _build_optimizer(self, opt):
        """为预训练编码器和随机初始化任务头设置不同学习率。"""
        head_prefixes = (
            "ha.",
            "cqi.",
            "mask_head.",
            "aux_heads.",
            "encoder.dino_adapter.",
            "encoder.semantic_fusion.",
            "encoder.p1_from_p2.",
        )
        base_params = []
        head_params = []
        frozen_params = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                frozen_params += param.numel()
                continue
            if name.startswith(head_prefixes):
                head_params.append(param)
            else:
                base_params.append(param)

        param_groups = []
        if base_params:
            param_groups.append(
                {
                    "params": base_params,
                    "lr": opt.lr,
                    "weight_decay": opt.weight_decay,
                    "name": "base",
                }
            )
        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": opt.lr * float(getattr(opt, "head_lr_mult", 3.0)),
                    "weight_decay": opt.weight_decay,
                    "name": "head",
                }
            )
        print(
            "optimizer param groups | "
            f"base={sum(p.numel() for p in base_params) / 1e6:.3f}M@{opt.lr:.2e} "
            f"| head={sum(p.numel() for p in head_params) / 1e6:.3f}M@"
            f"{opt.lr * float(getattr(opt, 'head_lr_mult', 3.0)):.2e} "
            f"| frozen={frozen_params / 1e6:.3f}M"
        )
        return optim.AdamW(param_groups, lr=opt.lr, weight_decay=opt.weight_decay)

    def _query_gate_tensor(self):
        query_gate = getattr(self.model.mask_head, "query_gate", None)
        if callable(query_gate):
            return query_gate().detach()
        return torch.tensor(float("nan"), device=self.device)

    @staticmethod
    def _linear_progress(epoch: int | None, total_epochs: int, start_epoch: int = 1) -> float:
        if epoch is None:
            return 0.0
        if total_epochs <= 1:
            return 1.0
        return max(0.0, min(1.0, (float(epoch) - float(start_epoch)) / float(total_epochs - 1)))

    def _tversky_params(self, epoch: int | None) -> tuple[float, float]:
        progress = self._linear_progress(
            epoch,
            int(getattr(self.opt, "loss_anneal_epochs", 20)),
            start_epoch=1,
        )
        beta_start = float(getattr(self.opt, "tversky_beta_start", 0.70))
        beta_end = float(getattr(self.opt, "tversky_beta_end", 0.55))
        beta = beta_start + (beta_end - beta_start) * progress
        alpha = 1.0 - beta
        return alpha, beta

    def _aux_loss_scale(self, epoch: int | None) -> float:
        base_weight = float(getattr(self.opt, "aux_loss_weight", 1.0))
        end_weight = float(getattr(self.opt, "aux_loss_weight_end", 0.5))
        start_epoch = int(getattr(self.opt, "aux_decay_start_epoch", 5))
        end_epoch = max(start_epoch, int(getattr(self.opt, "loss_anneal_epochs", 20)))
        if epoch is None or epoch <= start_epoch:
            progress = 0.0
        elif end_epoch == start_epoch:
            progress = 1.0
        else:
            progress = max(0.0, min(1.0, (float(epoch) - start_epoch) / (end_epoch - start_epoch)))
        return base_weight + (end_weight - base_weight) * progress

    def _consistency_scale(self, epoch: int | None, target_weight: float) -> float:
        warmup = int(getattr(self.opt, "consistency_warmup_epochs", 5))
        ramp_epochs = int(getattr(self.opt, "consistency_ramp_epochs", 10))
        if epoch is None or epoch <= warmup or target_weight <= 0.0:
            return 0.0
        progress = self._linear_progress(epoch, ramp_epochs, start_epoch=warmup + 1)
        return float(target_weight) * progress

    @staticmethod
    def _foreground_prob(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float(), dim=1)[:, 1:2]

    def _high_resolution_support(self, aux_preds: tuple[torch.Tensor, ...]) -> torch.Tensor | None:
        if len(aux_preds) < 2:
            return None
        p1_fg = self._foreground_prob(aux_preds[0]).detach()
        p2_fg = self._foreground_prob(aux_preds[1]).detach()
        return torch.maximum(p1_fg, p2_fg)

    def _support_preserve_loss(
        self,
        final_pred: torch.Tensor,
        aux_preds: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        support = self._high_resolution_support(aux_preds)
        if support is None:
            return final_pred.sum() * 0.0
        support_mask = (support > 0.60).float()
        if torch.count_nonzero(support_mask).item() == 0:
            return final_pred.sum() * 0.0
        final_fg = self._foreground_prob(final_pred)
        miss = F.relu(support.detach() - final_fg)
        return (miss * support_mask).sum() / support_mask.sum().clamp_min(1.0)

    def _coarse_suppression_loss(
        self,
        final_pred: torch.Tensor,
        aux_preds: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        support = self._high_resolution_support(aux_preds)
        if support is None:
            return final_pred.sum() * 0.0
        weak_support_mask = (support < 0.35).float()
        if torch.count_nonzero(weak_support_mask).item() == 0:
            return final_pred.sum() * 0.0
        final_fg = self._foreground_prob(final_pred)
        loss = F.relu(final_fg - support.detach() - 0.15)
        loss = (loss * weak_support_mask).sum() / weak_support_mask.sum().clamp_min(1.0)
        coarse_preds = aux_preds[3:5] if len(aux_preds) >= 5 else aux_preds[2:]
        for pred in coarse_preds:
            coarse_fg = self._foreground_prob(pred)
            excess = F.relu(coarse_fg - support.detach() - 0.15)
            loss = loss + (excess * weak_support_mask).sum() / weak_support_mask.sum().clamp_min(1.0)
        return loss / (1.0 + float(len(coarse_preds)))

    def forward(self, x1, x2, label, epoch: int | None = None):
        final_pred, aux_preds = self.model(x1, x2, gt_mask=label)
        label = label.long()
        tversky_alpha, tversky_beta = self._tversky_params(epoch)
        aux_loss_scale = self._aux_loss_scale(epoch)
        aux_head_weights = [aux_loss_scale * weight for weight in self.aux_base_weights]
        support_weight = self._consistency_scale(
            epoch, float(getattr(self.opt, "support_consistency_weight", 0.03))
        )
        coarse_weight = self._consistency_scale(
            epoch, float(getattr(self.opt, "coarse_consistency_weight", 0.02))
        )
        with self._loss_autocast_context():
            main_focal = 0.5 * self.focal(final_pred.float(), label)
            main_tversky = self.dice(
                final_pred.float(),
                label,
                alpha=tversky_alpha,
                beta=tversky_beta,
            )
            aux_focal = main_focal.new_zeros(())
            aux_tversky = main_tversky.new_zeros(())
            for weight, pred in zip(aux_head_weights, aux_preds):
                if weight <= 0.0:
                    continue
                aux_focal = aux_focal + 0.5 * weight * self.focal(pred.float(), label)
                aux_tversky = aux_tversky + weight * self.dice(
                    pred.float(),
                    label,
                    alpha=tversky_alpha,
                    beta=tversky_beta,
                )

            focal = main_focal + aux_focal
            tversky = main_tversky + aux_tversky
            support_loss = self._support_preserve_loss(final_pred.float(), aux_preds)
            coarse_loss = self._coarse_suppression_loss(final_pred.float(), aux_preds)
            consistency = support_weight * support_loss + coarse_weight * coarse_loss

        self.last_loss_stats = {
            "main_focal": main_focal.detach(),
            "main_tversky": main_tversky.detach(),
            "aux_focal": aux_focal.detach(),
            "aux_tversky": aux_tversky.detach(),
            "support_loss": support_loss.detach(),
            "coarse_loss": coarse_loss.detach(),
            "support_weight": torch.tensor(support_weight, device=self.device),
            "coarse_weight": torch.tensor(coarse_weight, device=self.device),
            "aux_loss_scale": torch.tensor(aux_loss_scale, device=self.device),
            "tversky_beta": torch.tensor(tversky_beta, device=self.device),
            "query_gate": self._query_gate_tensor(),
        }
        return final_pred, focal, tversky + consistency

    @torch.inference_mode()
    def inference(self, x1, x2):
        with self._amp_autocast_context():
            return self.model.predict_logits(x1, x2)

    def load_ckpt(self, network, optimizer, name, backbone):
        save_filename = "%s_%s_best.pth" % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            raise FileNotFoundError(f"{save_filename} must exist")

        checkpoint = torch.load(save_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint["network"]
        current_state = network.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in current_state:
                continue
            if current_state[key].shape != value.shape:
                skipped.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                continue
            filtered_state[key] = value
        network.load_state_dict(filtered_state, strict=False)
        if skipped:
            print(
                "skip incompatible pretrain keys:",
                [f"{key}:{src}->{dst}" for key, src, dst in skipped[:5]],
            )
        print("load HA-CQI checkpoint")

    def _build_checkpoint_meta(self):
        """保存推理重建 HA-CQI 所需的最小结构配置。"""
        return {
            "model_config": {
                "architecture": "HA-CQI",
                "backbone": self.opt.backbone,
                "backbone_weight": self.opt.backbone_weight,
                "fpn_channels": int(self.opt.fpn_channels),
                "deform_groups": int(self.opt.deform_groups),
                "gamma_mode": self.opt.gamma_mode,
                "beta_mode": self.opt.beta_mode,
                "disable_soft_alignment": bool(getattr(self.opt, "disable_soft_alignment", False)),
                "align_window": int(self.opt.align_window),
                "align_points": int(self.opt.align_points),
                "align_heads": int(self.opt.align_heads),
                "align_on_levels": [int(v) for v in self.opt.align_on_levels],
                "align_qkv_bias": bool(self.opt.align_qkv_bias),
                "align_offset_groups": int(self.opt.align_offset_groups),
                "num_change_queries": int(getattr(self.opt, "num_change_queries", 16)),
                "cqi_heads": int(getattr(self.opt, "cqi_heads", 4)),
                "mask_dim": int(getattr(self.opt, "mask_dim", 128)),
                "mask_queries": int(getattr(self.opt, "mask_queries", 32)),
                "mask_decoder_layers": int(getattr(self.opt, "mask_decoder_layers", 3)),
                "mask_heads": int(getattr(self.opt, "mask_heads", 4)),
                "head_lr_mult": float(getattr(self.opt, "head_lr_mult", 3.0)),
                "aux_loss_weight": float(getattr(self.opt, "aux_loss_weight", 1.0)),
                "aux_loss_weight_end": float(getattr(self.opt, "aux_loss_weight_end", 0.5)),
                "aux_decay_start_epoch": int(getattr(self.opt, "aux_decay_start_epoch", 5)),
                "tversky_beta_start": float(getattr(self.opt, "tversky_beta_start", 0.70)),
                "tversky_beta_end": float(getattr(self.opt, "tversky_beta_end", 0.55)),
                "loss_anneal_epochs": int(getattr(self.opt, "loss_anneal_epochs", 20)),
                "support_consistency_weight": float(
                    getattr(self.opt, "support_consistency_weight", 0.03)
                ),
                "coarse_consistency_weight": float(
                    getattr(self.opt, "coarse_consistency_weight", 0.02)
                ),
                "consistency_warmup_epochs": int(
                    getattr(self.opt, "consistency_warmup_epochs", 5)
                ),
                "consistency_ramp_epochs": int(getattr(self.opt, "consistency_ramp_epochs", 10)),
                "focal_gamma": float(getattr(self.opt, "gamma", 2.0)),
                "amp": bool(getattr(self.opt, "amp", False)),
                "amp_dtype": str(getattr(self.opt, "amp_dtype", "fp16")),
                "dino_arch": self.opt.dino_arch,
                "dino_weight": self.opt.dino_weight,
                "extract_ids": [int(v) for v in self.opt.extract_ids],
                "best_metric": getattr(self.opt, "best_metric", "iou_1"),
                "eval_fg_threshold": float(getattr(self.opt, "eval_fg_threshold", 0.5)),
                "eval_thresholds": [float(v) for v in getattr(self.opt, "eval_thresholds", [])],
            }
        }

    def save_ckpt(self, network, optimizer, model_name, backbone, tag: str = "best"):
        if tag == "best":
            save_filename = "%s_%s_best.pth" % (model_name, backbone)
        else:
            save_filename = f"{model_name}_{backbone}_{tag}.pth"
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save(
            {
                "network": network.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "meta": self._build_checkpoint_meta(),
            },
            save_path,
        )
        network.to(self.device)

    def save(self, model_name, backbone, tag: str = "best"):
        self.save_ckpt(self.model, self.optimizer, model_name, backbone, tag=tag)

    def save_epoch_ckpt(self, network, optimizer, model_name, backbone, epoch):
        save_filename = f"{model_name}_{backbone}_epoch{epoch}.pth"
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(
            {
                "network": network.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "meta": self._build_checkpoint_meta(),
                "epoch": epoch,
            },
            save_path,
        )
        network.to(self.device)

    def save_periodic(self, model_name, backbone, epoch):
        self.save_epoch_ckpt(self.model, self.optimizer, model_name, backbone, epoch)

    def name(self):
        return self.opt.name


def build_hacqi_engine(opt):
    engine = HACQIEngine(opt)
    print("HA-CQI engine [%s] was created" % engine.name())
    return engine.to(engine.device)
