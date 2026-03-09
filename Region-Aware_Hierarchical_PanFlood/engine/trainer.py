import math
import time
from contextlib import nullcontext
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from engine.metrics import summarize_metrics, update_confusion_matrix


class Trainer:
    """封装训练与评估流程，支持 AMP、梯度裁剪与阶段化开关。"""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        num_classes: int,
        ignore_index: int,
        class_names,
        amp_enabled: bool = True,
        amp_dtype: str = "fp16",
        grad_clip_norm: float = 0.0,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = list(class_names)
        self.grad_clip_norm = float(grad_clip_norm)

        self.amp_enabled = bool(amp_enabled and device.type == "cuda")
        self.amp_dtype = torch.float16 if amp_dtype.lower() == "fp16" else torch.bfloat16
        self.use_scaler = self.amp_enabled and self.amp_dtype == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

    def _move_xdict(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in x_dict.items()}

    def train_one_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        logger,
        log_interval: int = 50,
        max_steps: int = -1,
    ) -> Dict[str, float]:
        self.model.train()

        running = {
            "loss_total": 0.0,
            "loss_floodness": 0.0,
            "loss_router": 0.0,
            "loss_open": 0.0,
            "loss_urban": 0.0,
            "loss_final": 0.0,
            "loss_boundary": 0.0,
            "loss_consistency": 0.0,
        }
        n = 0
        t0 = time.time()

        for step, (x_dict, target, _) in enumerate(loader, start=1):
            x_dict = self._move_xdict(x_dict)
            target = target.to(self.device, non_blocking=True)

            amp_ctx = torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled) if self.amp_enabled else nullcontext()
            with amp_ctx:
                out = self.model(x_dict)
                loss_dict = self.criterion(out, target)
                loss = loss_dict["loss_total"]

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            for k in running.keys():
                running[k] += float(loss_dict[k].item())
            n += 1

            if step % max(1, log_interval) == 0:
                logger.info(
                    f"[train][epoch={epoch} step={step}] "
                    f"loss={loss_dict['loss_total'].item():.4f} "
                    f"flood={loss_dict['loss_floodness'].item():.4f} "
                    f"router={loss_dict['loss_router'].item():.4f} "
                    f"open={loss_dict['loss_open'].item():.4f} "
                    f"urban={loss_dict['loss_urban'].item():.4f}"
                )

            if max_steps > 0 and step >= max_steps:
                break

        if self.scheduler is not None:
            self.scheduler.step()

        elapsed = time.time() - t0
        ret = {f"train_{k}": v / max(n, 1) for k, v in running.items()}
        ret["train_time_sec"] = elapsed
        ret["train_lr"] = float(self.optimizer.param_groups[0]["lr"])
        return ret

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val", max_steps: int = -1) -> Dict[str, float]:
        self.model.eval()

        conf = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=self.device)
        running = {"loss_total": 0.0}
        n = 0

        for step, (x_dict, target, _) in enumerate(loader, start=1):
            x_dict = self._move_xdict(x_dict)
            target = target.to(self.device, non_blocking=True)

            amp_ctx = torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled) if self.amp_enabled else nullcontext()
            with amp_ctx:
                out = self.model(x_dict)
                loss_dict = self.criterion(out, target)

            pred = out["final_logits"].argmax(dim=1)
            conf = update_confusion_matrix(conf, pred, target, self.num_classes, self.ignore_index)
            running["loss_total"] += float(loss_dict["loss_total"].item())
            n += 1

            if max_steps > 0 and step >= max_steps:
                break

        metrics = summarize_metrics(conf, self.class_names)
        metrics[f"{split}_loss_total"] = running["loss_total"] / max(n, 1)
        metrics = {f"{split}_{k}": v for k, v in metrics.items()}
        return metrics


def is_better(candidate: float, current_best: Optional[float], mode: str = "max") -> bool:
    if current_best is None:
        return True
    if mode == "max":
        return candidate > current_best
    if mode == "min":
        return candidate < current_best
    raise ValueError(f"Unsupported mode: {mode}")


def resolve_amp_dtype(name: str) -> str:
    name = str(name).lower()
    if name not in {"fp16", "bf16"}:
        return "fp16"
    return name
