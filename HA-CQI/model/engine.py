from datetime import datetime
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from .architectures import HACQIModel
from .backbones import DEFAULT_BACKBONE_NAME
from .checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_torch_save,
    checkpoint_model_config,
    cpu_state_dict,
    load_checkpoint_payload,
    load_network_state,
)
from .losses.dice import DICELoss
from .losses.focal import FocalLoss
from utils.provenance import portable_repo_path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent


def build_hacqi_model(fpn_channels=128, **kwargs):
    """构建 HA-CQI 网络主体。"""
    return HACQIModel(fpn_channels=fpn_channels, **kwargs)


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

        resolved_name = opt.name
        resume_path = str(getattr(opt, "resume", "") or "")
        if getattr(opt, "phase", "train") == "train" and resume_path:
            self.save_dir = str(Path(resume_path).resolve().parent)
            resolved_name = Path(self.save_dir).name
        elif getattr(opt, "phase", "train") == "train":
            resolved_name = resolve_unique_run_name(opt.checkpoint_dir, opt.name)
            self.save_dir = os.path.join(opt.checkpoint_dir, resolved_name)
        else:
            self.save_dir = os.path.join(opt.checkpoint_dir, resolved_name)
        self.opt.name = resolved_name
        if getattr(opt, "phase", "train") == "train":
            os.makedirs(self.save_dir, exist_ok=True)
        print(f"save_dir resolved to: {self.save_dir}")

        self.model = build_hacqi_model(
            backbone_weight=opt.backbone_weight,
            fpn_channels=opt.fpn_channels,
            deform_groups=opt.deform_groups,
            gamma_mode=opt.gamma_mode,
            beta_mode=opt.beta_mode,
            disable_soft_alignment=bool(getattr(opt, "disable_soft_alignment", False)),
            align_window=opt.align_window,
            align_points=opt.align_points,
            align_heads=opt.align_heads,
            align_on_levels=opt.align_on_levels,
            align_qkv_bias=opt.align_qkv_bias,
            align_offset_groups=opt.align_offset_groups,
            num_change_queries=int(getattr(opt, "num_change_queries", 16)),
            cqi_heads=int(getattr(opt, "cqi_heads", 4)),
            dino_arch=opt.dino_arch,
            extract_ids=opt.extract_ids,
            dino_weight=opt.dino_weight,
            input_mean=[float(value) for value in opt.mean],
            input_std=[float(value) for value in opt.std],
            device=self.device,
        )
        self.aux_base_weights = [
            weight / sum(self.AUX_BASE_WEIGHTS) for weight in self.AUX_BASE_WEIGHTS
        ]
        self.focal = FocalLoss(
            class_weights=opt.focal_class_weights,
            gamma=opt.gamma,
        )
        self.dice = DICELoss()
        self.last_loss_stats = {}
        self.optimizer = self._build_optimizer(opt)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, opt.num_epochs, eta_min=1e-7
        )
        if getattr(opt, "init_checkpoint", ""):
            load_network_state(self.model, opt.init_checkpoint, strict=True, map_location="cpu")
            print(f"initialized network from: {opt.init_checkpoint}")
        self.model.to(self.device)
        print("---------- HA-CQI network initialized -------------")

    def _amp_autocast_context(self):
        amp_enabled = bool(getattr(self.opt, "amp", False)) and self.device.type == "cuda"
        if not amp_enabled:
            return nullcontext()
        amp_dtype = (
            torch.bfloat16
            if str(getattr(self.opt, "amp_dtype", "bf16")).lower() == "bf16"
            else torch.float16
        )
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
            "aux_heads.",
            "encoder.dino_adapter.",
            "encoder.semantic_fusion.",
            "decoder.",
        )
        grouped_params: dict[tuple[str, bool], list[nn.Parameter]] = {
            ("base", True): [],
            ("base", False): [],
            ("head", True): [],
            ("head", False): [],
        }
        frozen_params = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                frozen_params += param.numel()
                continue
            scope = "head" if name.startswith(head_prefixes) else "base"
            use_weight_decay = not bool(getattr(param, "_no_weight_decay", False))
            grouped_params[(scope, use_weight_decay)].append(param)

        param_groups = []
        head_lr = opt.lr * float(getattr(opt, "head_lr_mult", 2.0))
        for scope in ("base", "head"):
            for use_weight_decay in (True, False):
                params = grouped_params[(scope, use_weight_decay)]
                if not params:
                    continue
                param_groups.append(
                    {
                        "params": params,
                        "lr": opt.lr if scope == "base" else head_lr,
                        "weight_decay": opt.weight_decay if use_weight_decay else 0.0,
                        "name": scope if use_weight_decay else f"{scope}_no_decay",
                    }
                )
        base_params = grouped_params[("base", True)] + grouped_params[("base", False)]
        head_params = grouped_params[("head", True)] + grouped_params[("head", False)]
        no_decay_params = grouped_params[("base", False)] + grouped_params[("head", False)]
        print(
            "optimizer param groups | "
            f"base={sum(p.numel() for p in base_params) / 1e6:.3f}M@{opt.lr:.2e} "
            f"| head={sum(p.numel() for p in head_params) / 1e6:.3f}M@"
            f"{head_lr:.2e} "
            f"| no_decay={sum(p.numel() for p in no_decay_params) / 1e6:.3f}M "
            f"| frozen={frozen_params / 1e6:.3f}M"
        )
        return optim.AdamW(param_groups, lr=opt.lr, weight_decay=opt.weight_decay)

    def _decoder_diagnostics(self) -> dict[str, torch.Tensor]:
        """导出 decoder 上下文/细节幅值诊断，不参与反向传播。"""
        diagnostics = self.model.decoder.diagnostics()
        return {name: value.detach() for name, value in diagnostics.items()}

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
        final_pred, aux_preds = self.model(x1, x2)
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
            **self._decoder_diagnostics(),
        }
        return final_pred, focal, tversky + consistency

    @torch.inference_mode()
    def inference(self, x1, x2):
        with self._amp_autocast_context():
            return self.model.predict_logits(x1, x2)

    def _build_checkpoint_meta(
        self,
        *,
        epoch: int,
        global_step: int,
        checkpoint_role: str,
        selection: dict[str, Any] | None,
        data_provenance: dict[str, Any],
    ) -> dict[str, Any]:
        """保存结构、训练、数据和阈值选择契约。"""
        portable_data_provenance = dict(data_provenance)
        for path_key in ("dataset_dir", "split_report"):
            if portable_data_provenance.get(path_key):
                portable_data_provenance[path_key] = portable_repo_path(
                    portable_data_provenance[path_key], REPO_ROOT
                )
        model_config = {
            "architecture": "HA-CQI",
            "backbone": DEFAULT_BACKBONE_NAME,
            "backbone_weight": portable_repo_path(self.opt.backbone_weight, REPO_ROOT),
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
            "decoder": "oscd_v1",
            "decoder_channels": 128,
            "ssm_state_dim": 1,
            "ssm_directions": 4,
            "context_levels": [3, 4, 5],
            "detail_levels": [2, 1],
            "dino_arch": self.opt.dino_arch,
            "dino_weight": portable_repo_path(self.opt.dino_weight, REPO_ROOT),
            "extract_ids": [int(v) for v in self.opt.extract_ids],
            "dino_input_norm": "imagenet",
            "input_mean": [float(v) for v in self.opt.mean],
            "input_std": [float(v) for v in self.opt.std],
        }
        loss_config = {
            "focal_class_weights": [float(v) for v in self.opt.focal_class_weights],
            "focal_gamma": float(getattr(self.opt, "gamma", 2.0)),
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
        }
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "run_name": self.opt.name,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "checkpoint_role": checkpoint_role,
            "model_config": model_config,
            "loss_config": loss_config,
            "training_config": {
                "optimizer": "AdamW",
                "lr": float(self.opt.lr),
                "weight_decay": float(self.opt.weight_decay),
                "head_lr_mult": float(getattr(self.opt, "head_lr_mult", 2.0)),
                "batch_size": int(self.opt.batch_size),
                "num_epochs": int(self.opt.num_epochs),
                "num_workers": int(self.opt.num_workers),
                "input_size": int(self.opt.input_size),
                "dataset_mode": str(self.opt.dataset_mode),
                "max_train_steps": int(getattr(self.opt, "max_train_steps", -1)),
                "max_val_steps": int(getattr(self.opt, "max_val_steps", -1)),
                "seed": int(getattr(self.opt, "seed", 1)),
                "deterministic": bool(getattr(self.opt, "deterministic", True)),
                "amp": bool(getattr(self.opt, "amp", False)),
                "amp_dtype": str(getattr(self.opt, "amp_dtype", "bf16")),
                "grad_scaler_init_scale": float(
                    getattr(self.opt, "grad_scaler_init_scale", 4096.0)
                ),
                "eval_fg_threshold": float(self.opt.eval_fg_threshold),
                "threshold_min": float(getattr(self.opt, "threshold_min", 0.05)),
                "threshold_max": float(getattr(self.opt, "threshold_max", 0.95)),
                "threshold_step": float(getattr(self.opt, "threshold_step", 0.01)),
            },
            "data_config": {
                **portable_data_provenance,
                "dataroot": portable_repo_path(self.opt.dataroot, REPO_ROOT),
                "stats_file": portable_repo_path(self.opt.stats_file, REPO_ROOT),
            },
            "selection": dict(selection or {}),
        }

    def save_training_checkpoint(
        self,
        *,
        tag: str,
        epoch: int,
        global_step: int,
        selection: dict[str, Any] | None,
        data_provenance: dict[str, Any],
        scaler_state: dict[str, Any] | None,
        data_loader_generator_state: torch.Tensor | None,
    ) -> Path:
        if tag == "periodic":
            filename = f"{self.opt.name}_{self.opt.backbone}_epoch{epoch}.pth"
        elif tag in {"best_primary", "last"}:
            filename = f"{self.opt.name}_{self.opt.backbone}_{tag}.pth"
        else:
            raise ValueError(f"Unsupported checkpoint tag: {tag}")
        meta = self._build_checkpoint_meta(
            epoch=epoch,
            global_step=global_step,
            checkpoint_role=tag,
            selection=selection,
            data_provenance=data_provenance,
        )
        payload: dict[str, Any] = {
            "network": cpu_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": scaler_state,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "data_loader_generator_state": data_loader_generator_state,
            "meta": meta,
        }
        return atomic_torch_save(payload, Path(self.save_dir) / filename)

    def restore_training_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        expected_dataset_fingerprint: str | None,
        scaler,
        data_loader_generator=None,
    ) -> tuple[int, int, dict[str, Any]]:
        payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
        checkpoint_model_config(payload)
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("--resume requires checkpoint v2 metadata")
        data_config = meta.get("data_config", {})
        actual_fingerprint = data_config.get("dataset_fingerprint") if isinstance(data_config, dict) else None
        if expected_dataset_fingerprint and actual_fingerprint != expected_dataset_fingerprint:
            raise ValueError(
                "Resume dataset fingerprint mismatch: "
                f"checkpoint={actual_fingerprint}, current={expected_dataset_fingerprint}"
            )
        expected_meta = self._build_checkpoint_meta(
            epoch=0,
            global_step=0,
            checkpoint_role="resume_validation",
            selection=None,
            data_provenance={"dataset_fingerprint": expected_dataset_fingerprint},
        )
        mismatches: list[str] = []
        for section_name in ("model_config", "loss_config", "training_config"):
            checkpoint_section = meta.get(section_name)
            expected_section = expected_meta[section_name]
            if not isinstance(checkpoint_section, dict):
                mismatches.append(f"{section_name}=missing")
                continue
            for key, expected_value in expected_section.items():
                actual_value = checkpoint_section.get(key)
                if actual_value != expected_value:
                    mismatches.append(
                        f"{section_name}.{key}: checkpoint={actual_value!r}, current={expected_value!r}"
                    )
        if mismatches:
            details = "\n  - ".join(mismatches)
            raise ValueError(f"Resume configuration mismatch:\n  - {details}")
        self.model.load_state_dict(payload["network"], strict=True)
        if "optimizer" not in payload or "scheduler" not in payload:
            raise ValueError("Resume checkpoint lacks optimizer or scheduler state")
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        if data_loader_generator is not None and payload.get("data_loader_generator_state") is not None:
            data_loader_generator.set_state(payload["data_loader_generator_state"])
        epoch = int(payload.get("epoch", meta.get("epoch", 0)))
        global_step = int(payload.get("global_step", meta.get("global_step", 0)))
        return epoch + 1, global_step, payload

    def name(self):
        return self.opt.name


def build_hacqi_engine(opt):
    engine = HACQIEngine(opt)
    print("HA-CQI engine [%s] was created" % engine.name())
    return engine.to(engine.device)
