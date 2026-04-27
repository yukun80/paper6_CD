from datetime import datetime
import os
from contextlib import nullcontext

import torch
from torch import nn
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
        self.aux_head_weights = [1.0, 1.0, 0.5, 0.25, 0.25]
        self.focal = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        self.dice = DICELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
        )
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

    def forward(self, x1, x2, label, epoch: int | None = None):
        del epoch
        final_pred, aux_preds = self.model(x1, x2, gt_mask=label)
        label = label.long()
        focal = 0.5 * self.focal(final_pred, label)
        dice = self.dice(final_pred, label)
        for weight, pred in zip(self.aux_head_weights, aux_preds):
            focal += 0.5 * weight * self.focal(pred, label)
            dice += weight * self.dice(pred, label)
        return final_pred, focal, dice

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
                "amp": bool(getattr(self.opt, "amp", False)),
                "amp_dtype": str(getattr(self.opt, "amp_dtype", "fp16")),
                "dino_arch": self.opt.dino_arch,
                "dino_weight": self.opt.dino_weight,
                "extract_ids": [int(v) for v in self.opt.extract_ids],
                "best_metric": getattr(self.opt, "best_metric", "iou_1"),
                "eval_fg_threshold": float(getattr(self.opt, "eval_fg_threshold", 0.5)),
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
