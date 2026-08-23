"""HA-CQI 可复现训练、联合阈值验证与 checkpoint v2 主入口。"""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

# cuBLAS 的可复现工作区必须在首次 CUDA 运算前配置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from data.cd_dataset import DataLoader
from model.checkpointing import atomic_json_save
from model.engine import build_hacqi_engine
from option import Options
from utils.flood_evaluation import FloodEvaluationAccumulator, build_threshold_grid
from utils.provenance import load_dataset_provenance
from utils.util import de_norm, make_numpy_grid


def setup_seed(seed: int, deterministic: bool = True) -> None:
    """在模型与 DataLoader 创建前固定全部可控随机源。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    print(
        f"[INFO] seed={seed} | deterministic={deterministic} | "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
    )


def _json_safe_options(opt) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(opt).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, torch.dtype):
            result[key] = str(value)
        else:
            result[key] = value
    return result


class HACQITrainer:
    """保持网络结构不变，只重建训练、评估与保存契约。"""

    def __init__(self, opt) -> None:
        self.opt = opt
        self.data_provenance = load_dataset_provenance(opt.dataroot, opt.dataset)
        if str(opt.dataset) == "S1GFloods_CD_DINO_BG_75_25" and not self.data_provenance.get(
            "dataset_fingerprint"
        ):
            raise ValueError(
                "Corrected dataset requires split_report.json with dataset_fingerprint: "
                f"{self.data_provenance['split_report']}"
            )

        opt.phase = "train"
        self.train_loader = DataLoader(opt)
        self.train_data = self.train_loader.load_data()
        train_size = len(self.train_loader)
        print(f"#training images = {train_size}")

        opt.phase = "val"
        self.val_loader = DataLoader(opt)
        self.val_data = self.val_loader.load_data()
        val_size = len(self.val_loader)
        print(f"#validation images = {val_size}")
        opt.phase = "train"

        self.model = build_hacqi_engine(opt)
        self.optimizer = self.model.optimizer
        self.scheduler = self.model.scheduler
        self.amp_enabled = bool(opt.amp) and self.model.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if str(opt.amp_dtype).lower() == "bf16" else torch.float16
        )
        scaler_enabled = self.amp_enabled and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=scaler_enabled,
            init_scale=float(opt.grad_scaler_init_scale),
        )
        print(
            f"[INFO] AMP enabled={self.amp_enabled}, dtype={opt.amp_dtype}, "
            f"GradScaler enabled={scaler_enabled}"
        )

        self.global_step = 0
        self.start_epoch = 1
        self.best_selection: dict[str, Any] | None = None
        self.thresholds = build_threshold_grid(
            opt.threshold_min,
            opt.threshold_max,
            opt.threshold_step,
        )

        self.save_dir = Path(self.model.save_dir)
        self.vis_path = self.save_dir / opt.vis_path
        self.vis_path.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.save_dir / "metrics.jsonl"
        snapshot_name = "resume_options.json" if opt.resume else "options.json"
        atomic_json_save(_json_safe_options(opt), self.save_dir / snapshot_name)

        if opt.resume:
            self.start_epoch, self.global_step, _ = self.model.restore_training_checkpoint(
                opt.resume,
                expected_dataset_fingerprint=self.data_provenance.get("dataset_fingerprint"),
                scaler=self.scaler,
                data_loader_generator=self.train_loader.generator,
            )
            self._restore_best_selection()
            print(
                f"[INFO] resumed from {opt.resume}: start_epoch={self.start_epoch}, "
                f"global_step={self.global_step}"
            )

    def _restore_best_selection(self) -> None:
        path = self.save_dir / "selection.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("iou_1") is not None:
                self.best_selection = payload

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.amp.autocast(
            device_type=self.model.device.type,
            dtype=self.amp_dtype,
            enabled=True,
        )

    @staticmethod
    def _mean_stats(sums: dict[str, float], count: int) -> dict[str, float]:
        denominator = max(count, 1)
        return {key: value / denominator for key, value in sums.items()}

    def _plot_cd_result(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        stage: str,
        threshold: float,
    ) -> None:
        if prediction.ndim == 4:
            probability = torch.softmax(prediction.detach().float(), dim=1)[:, 1]
            prediction = (probability >= float(threshold)).long()
        vis_input = make_numpy_grid(de_norm(x1[0:8].clone(), self.opt.mean, self.opt.std))
        vis_input2 = make_numpy_grid(de_norm(x2[0:8].clone(), self.opt.mean, self.opt.std))
        vis_pred = make_numpy_grid(prediction[0:8].unsqueeze(1).repeat(1, 3, 1, 1))
        vis_gt = make_numpy_grid(target[0:8].unsqueeze(1).repeat(1, 3, 1, 1))
        visual = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        plt.imsave(self.vis_path / f"{stage}_{epoch}.jpg", np.clip(visual, 0.0, 1.0))

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.opt.phase = "train"
        self.model.model.train()
        sums = {
            "loss": 0.0,
            "focal": 0.0,
            "tversky": 0.0,
            "main_focal": 0.0,
            "main_tversky": 0.0,
            "aux_focal": 0.0,
            "aux_tversky": 0.0,
            "support_loss": 0.0,
            "coarse_loss": 0.0,
            "support_weight": 0.0,
            "coarse_weight": 0.0,
            "aux_loss_scale": 0.0,
            "tversky_beta": 0.0,
            "query_gate": 0.0,
        }
        step_count = 0
        optimizer_steps = 0
        last_batch = None
        last_logits = None
        progress = tqdm(self.train_data, ncols=100, desc=f"train e{epoch}")
        for step_index, batch in enumerate(progress):
            img1 = batch["img1"].to(self.model.device, non_blocking=True)
            img2 = batch["img2"].to(self.model.device, non_blocking=True)
            label = batch["cd_label"].to(self.model.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                logits, focal, tversky = self.model(img1, img2, label, epoch=epoch)
                loss = focal + tversky
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, step={step_index}: {loss.item()}"
                )
            if self.scaler.is_enabled():
                scale_before = float(self.scaler.get_scale())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_steps += int(float(self.scaler.get_scale()) >= scale_before)
            else:
                loss.backward()
                self.optimizer.step()
                optimizer_steps += 1

            sums["loss"] += float(loss.detach())
            sums["focal"] += float(focal.detach())
            sums["tversky"] += float(tversky.detach())
            for key in sums:
                if key in {"loss", "focal", "tversky"}:
                    continue
                value = self.model.last_loss_stats.get(key)
                if value is not None:
                    sums[key] += float(value.detach())
            self.global_step += 1
            step_count += 1
            last_batch = batch
            last_logits = logits.detach()
            progress.set_postfix(
                loss=f"{sums['loss'] / step_count:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )
            if self.opt.max_train_steps > 0 and step_count >= self.opt.max_train_steps:
                break

        if step_count == 0:
            raise RuntimeError("No training steps were executed")
        if optimizer_steps > 0:
            self.scheduler.step()
        else:
            print("[WARN] GradScaler skipped every optimizer step; scheduler was not advanced.")
        averaged = self._mean_stats(sums, step_count)
        averaged["lr"] = float(self.optimizer.param_groups[0]["lr"])
        averaged["steps"] = int(step_count)
        averaged["optimizer_steps"] = int(optimizer_steps)
        if last_batch is not None and last_logits is not None:
            self._plot_cd_result(
                last_batch["img1"],
                last_batch["img2"],
                last_logits,
                last_batch["cd_label"],
                epoch,
                "train",
                self.opt.eval_fg_threshold,
            )
        return averaged

    @torch.inference_mode()
    def validate_epoch(self, epoch: int) -> dict[str, Any]:
        self.opt.phase = "val"
        self.model.eval()
        evaluator = FloodEvaluationAccumulator(
            self.thresholds,
            reference_threshold=float(self.opt.eval_fg_threshold),
            tiny_area_thresh=int(self.opt.tiny_area_thresh),
            small_area_thresh=int(self.opt.small_area_thresh),
        )
        step_count = 0
        last_batch = None
        last_logits = None
        progress = tqdm(self.val_data, ncols=100, desc=f"val e{epoch}")
        for batch in progress:
            img1 = batch["img1"].to(self.model.device, non_blocking=True)
            img2 = batch["img2"].to(self.model.device, non_blocking=True)
            with self._autocast_context():
                logits = self.model.inference(img1, img2)
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            target = batch["cd_label"].cpu().numpy()
            evaluator.update(probabilities, target)
            step_count += 1
            last_batch = batch
            last_logits = logits.detach()
            if self.opt.max_val_steps > 0 and step_count >= self.opt.max_val_steps:
                break

        if step_count == 0:
            raise RuntimeError("No validation steps were executed")
        result = evaluator.scores()
        result["steps"] = int(step_count)
        selected = result["selected"]
        print(
            "validation primary | "
            f"threshold={selected['threshold']:.2f} "
            f"IoU={selected['iou_1'] * 100:.3f} "
            f"F1={selected['F1_1'] * 100:.3f} "
            f"P={selected['precision_1'] * 100:.3f} "
            f"R={selected['recall_1'] * 100:.3f}"
        )
        reference = result["reference"]
        print(
            "validation diagnostic | "
            f"threshold={reference['threshold']:.2f} "
            f"background_tile_fp_rate={reference['background_tile_fp_rate'] * 100:.3f} "
            f"largest_fp_component={reference['largest_fp_component']} "
            f"tiny_cov10={reference['tiny_recall_cov10'] * 100:.3f} "
            f"tiny_cov25={reference['tiny_recall_cov25'] * 100:.3f}"
        )
        if last_batch is not None and last_logits is not None:
            self._plot_cd_result(
                last_batch["img1"],
                last_batch["img2"],
                last_logits,
                last_batch["cd_label"],
                epoch,
                "val",
                float(selected["threshold"]),
            )
        return result

    @staticmethod
    def _selection_key(selection: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(selection["iou_1"]),
            float(selection["precision_1"]),
            float(selection["threshold"]),
        )

    def _selection_payload(self, epoch: int, validation: dict[str, Any]) -> dict[str, Any]:
        selected = validation["selected"]
        return {
            "epoch": int(epoch),
            "threshold": float(selected["threshold"]),
            "iou_1": float(selected["iou_1"]),
            "F1_1": float(selected["F1_1"]),
            "precision_1": float(selected["precision_1"]),
            "recall_1": float(selected["recall_1"]),
            "threshold_grid": [float(value) for value in self.thresholds],
            "dataset_fingerprint": self.data_provenance.get("dataset_fingerprint"),
        }

    def _save_epoch_checkpoints(self, epoch: int, validation: dict[str, Any]) -> None:
        selection = self._selection_payload(epoch, validation)
        is_best = self.best_selection is None or self._selection_key(selection) > self._selection_key(
            self.best_selection
        )
        if is_best:
            checkpoint_path = self.model.save_training_checkpoint(
                tag="best_primary",
                epoch=epoch,
                global_step=self.global_step,
                selection=selection,
                data_provenance=self.data_provenance,
                scaler_state=self.scaler.state_dict(),
                data_loader_generator_state=self.train_loader.get_generator_state(),
            )
            selection["checkpoint_file"] = checkpoint_path.name
            self.best_selection = selection
            atomic_json_save(selection, self.save_dir / "selection.json")
            print(f"[INFO] new best primary checkpoint: {checkpoint_path}")

        last_path = self.model.save_training_checkpoint(
            tag="last",
            epoch=epoch,
            global_step=self.global_step,
            selection=selection,
            data_provenance=self.data_provenance,
            scaler_state=self.scaler.state_dict(),
            data_loader_generator_state=self.train_loader.get_generator_state(),
        )
        print(f"[INFO] last checkpoint: {last_path}")
        if epoch % 10 == 0:
            periodic_path = self.model.save_training_checkpoint(
                tag="periodic",
                epoch=epoch,
                global_step=self.global_step,
                selection=selection,
                data_provenance=self.data_provenance,
                scaler_state=self.scaler.state_dict(),
                data_loader_generator_state=self.train_loader.get_generator_state(),
            )
            print(f"[INFO] periodic checkpoint: {periodic_path}")

    def _append_logs(
        self,
        epoch: int,
        train_metrics: dict[str, Any],
        validation: dict[str, Any],
    ) -> None:
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "train": train_metrics,
            "validation": validation,
            "dataset_fingerprint": self.data_provenance.get("dataset_fingerprint"),
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def run(self) -> None:
        if self.start_epoch > self.opt.num_epochs:
            raise ValueError(
                f"Resume epoch {self.start_epoch - 1} already reaches num_epochs={self.opt.num_epochs}"
            )
        total_batches = math.ceil(len(self.train_loader) / self.opt.batch_size)
        print(
            f"[INFO] epochs={self.start_epoch}..{self.opt.num_epochs} | "
            f"nominal_train_batches={total_batches}"
        )
        for epoch in range(self.start_epoch, self.opt.num_epochs + 1):
            previous = (
                float(self.best_selection["iou_1"]) * 100.0 if self.best_selection else float("nan")
            )
            print(f"\n==> Name {self.opt.name}, Epoch {epoch}, previous best IoU={previous:.3f}")
            train_metrics = self.train_epoch(epoch)
            validation = self.validate_epoch(epoch)
            self._append_logs(epoch, train_metrics, validation)
            self._save_epoch_checkpoints(epoch, validation)
        print("Done!")


if __name__ == "__main__":
    options = Options().parse()
    setup_seed(options.seed, options.deterministic)
    HACQITrainer(options).run()
