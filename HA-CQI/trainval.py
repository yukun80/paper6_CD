import torch
from option import Options
from data.cd_dataset import DataLoader
from model.engine import build_hacqi_engine
from tqdm import tqdm
import math
from utils.metric_tool import (
    ConfuseMatrixMeter,
    component_recall_scores,
    init_component_recall_stats,
    init_prediction_blob_stats,
    prediction_blob_scores,
    update_component_recall_stats,
    update_prediction_blob_stats,
)
import os
import json
import numpy as np
import random
from datetime import datetime
from contextlib import nullcontext
from utils.util import make_numpy_grid, de_norm
import matplotlib.pyplot as plt


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    disable_cudnn = os.environ.get("HA_CQI_DISABLE_CUDNN", "0") == "1"
    cudnn_benchmark = os.environ.get("HA_CQI_CUDNN_BENCHMARK", "0") == "1"
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = cudnn_benchmark and not disable_cudnn
    torch.backends.cudnn.enabled = not disable_cudnn
    if disable_cudnn:
        print("[INFO] HA_CQI_DISABLE_CUDNN=1: cuDNN is disabled globally; CUDA tensors still run on GPU.")
    elif cudnn_benchmark:
        print("[INFO] HA_CQI_CUDNN_BENCHMARK=1: cuDNN benchmark is enabled.")
    else:
        print("[INFO] cuDNN benchmark is disabled for stable HA-CQI training.")


class HACQITrainer(object):
    def __init__(self, opt):
        self.opt = opt

        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()
        train_size = len(train_loader)
        print("#training images = %d" % train_size)
        opt.phase = "val"
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()
        val_size = len(val_loader)
        print("#validation images = %d" % val_size)
        opt.phase = "train"

        self.model = build_hacqi_engine(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular
        self.amp_enabled = bool(getattr(opt, "amp", False)) and self.model.device.type == "cuda"
        self.amp_dtype = self._resolve_amp_dtype(getattr(opt, "amp_dtype", "fp16"))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        if self.amp_enabled:
            print(f"[INFO] AMP is enabled for HA-CQI training/validation: dtype={opt.amp_dtype}")
        elif getattr(opt, "amp", False):
            print("[INFO] AMP was requested but CUDA is unavailable; fallback to FP32.")
        else:
            print("[INFO] AMP is disabled; HA-CQI uses FP32 training.")

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.alpha = 0.5
        self.num_epochs = opt.num_epochs
        self.eval_fg_threshold = float(getattr(opt, "eval_fg_threshold", 0.5))
        self.eval_thresholds = [float(v) for v in getattr(opt, "eval_thresholds", [])]
        self.last_threshold_scan = {}
        self.best_metric = str(getattr(opt, "best_metric", "iou_1"))
        self.best_scores = {
            "default": float("-inf"),
            "iou_1": float("-inf"),
            "tiny_recall": float("-inf"),
            "tiny_combo": float("-inf"),
            "tiny_safe_combo": float("-inf"),
        }
        self.best_epochs = {
            "default": 0,
            "iou_1": 0,
            "tiny_recall": 0,
            "tiny_combo": 0,
            "tiny_safe_combo": 0,
        }
        self.best_summary_path = os.path.join(self.model.save_dir, "best_metrics.json")

        self.log_path = os.path.join(self.model.save_dir, "record.txt")
        self.vis_path = os.path.join(self.model.save_dir, opt.vis_path)
        os.makedirs(self.vis_path, exist_ok=True)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("# Record of training/validation metrics\n")
                f.write("# name: %s | backbone: %s\n" % (opt.name, getattr(opt, "backbone", "NA")))
                f.write(
                    "# amp: %s | amp_dtype: %s\n"
                    % (str(self.amp_enabled), getattr(opt, "amp_dtype", "fp16"))
                )
                f.write(
                    "# time,epoch,train_loss,train_focal,train_tversky,"
                    "main_focal,main_tversky,aux_focal,aux_tversky,"
                    "support_loss,coarse_loss,support_weight,coarse_weight,"
                    "aux_loss_scale,tversky_beta,query_gate,lr,"
                )
                f.write("val_metrics(json)\n")

    @staticmethod
    def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
        if str(dtype_name).lower() == "bf16":
            return torch.bfloat16
        return torch.float16

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.amp.autocast(
            device_type=self.model.device.type,
            dtype=self.amp_dtype,
            enabled=True,
        )

    @staticmethod
    def _threshold_prediction(logits: torch.Tensor, threshold: float) -> torch.Tensor:
        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        return (probs >= float(threshold)).long()

    @staticmethod
    def _compute_tiny_combo(val_scores: dict) -> float:
        return (
            0.45 * float(val_scores.get("iou_1", 0.0))
            + 0.45 * float(val_scores.get("tiny_recall", 0.0))
            + 0.10 * float(val_scores.get("small_recall", 0.0))
        )

    @staticmethod
    def _compute_tiny_safe_combo(val_scores: dict) -> float:
        return (
            0.40 * float(val_scores.get("iou_1", 0.0))
            + 0.25 * float(val_scores.get("tiny_recall", 0.0))
            + 0.10 * float(val_scores.get("small_recall", 0.0))
            + 0.15 * float(val_scores.get("precision_1", 0.0))
            + 0.10 * float(val_scores.get("blob_precision", 0.0))
        )

    def _metric_value(self, metric_name: str, val_scores: dict) -> float:
        if metric_name == "tiny_combo":
            return float(val_scores.get("tiny_combo", self._compute_tiny_combo(val_scores)))
        if metric_name == "tiny_safe_combo":
            return float(
                val_scores.get(
                    "tiny_safe_combo",
                    self._compute_tiny_safe_combo(val_scores),
                )
            )
        return float(val_scores.get(metric_name, 0.0))

    def _write_best_summary(self):
        payload = {
            "best_metric": self.best_metric,
            "eval_fg_threshold": self.eval_fg_threshold,
            "best_scores": self.best_scores,
            "best_epochs": self.best_epochs,
        }
        with open(self.best_summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _update_best_checkpoints(self, epoch: int, val_scores: dict):
        track_to_tag = {
            "iou_1": "best_iou",
            "tiny_recall": "best_tiny_recall",
            "tiny_combo": "best_tiny_combo",
            "tiny_safe_combo": "best_tiny_safe",
        }

        selected_metric_value = self._metric_value(self.best_metric, val_scores)
        if selected_metric_value >= self.best_scores["default"]:
            self.model.save(self.opt.name, self.opt.backbone)
            self.best_scores["default"] = selected_metric_value
            self.best_epochs["default"] = epoch
            self.previous_best = selected_metric_value

        for metric_name, tag in track_to_tag.items():
            metric_value = self._metric_value(metric_name, val_scores)
            if metric_value >= self.best_scores[metric_name]:
                self.model.save(self.opt.name, self.opt.backbone, tag=tag)
                self.best_scores[metric_name] = metric_value
                self.best_epochs[metric_name] = epoch

        self._write_best_summary()

    def _append_log_line(self, epoch: int, train_stats: dict, val_scores: dict):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        val_payload = dict(val_scores)
        if self.last_threshold_scan:
            val_payload["threshold_scan"] = self.last_threshold_scan

        line = (
            f"{ts},{epoch},"
            f"{train_stats.get('loss', float('nan')):.6f},"
            f"{train_stats.get('focal', float('nan')):.6f},"
            f"{train_stats.get('tversky', float('nan')):.6f},"
            f"{train_stats.get('main_focal', float('nan')):.6f},"
            f"{train_stats.get('main_tversky', float('nan')):.6f},"
            f"{train_stats.get('aux_focal', float('nan')):.6f},"
            f"{train_stats.get('aux_tversky', float('nan')):.6f},"
            f"{train_stats.get('support_loss', float('nan')):.6f},"
            f"{train_stats.get('coarse_loss', float('nan')):.6f},"
            f"{train_stats.get('support_weight', float('nan')):.6f},"
            f"{train_stats.get('coarse_weight', float('nan')):.6f},"
            f"{train_stats.get('aux_loss_scale', float('nan')):.6f},"
            f"{train_stats.get('tversky_beta', float('nan')):.6f},"
            f"{train_stats.get('query_gate', float('nan')):.6f},"
            f"{train_stats.get('lr', float('nan')):.8f}," + json.dumps(val_payload, ensure_ascii=False) + "\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _new_eval_state(self):
        return {
            "metric": ConfuseMatrixMeter(n_class=2),
            "component": init_component_recall_stats(),
            "blob": init_prediction_blob_stats(),
        }

    def _update_eval_state(self, state: dict, pred_np: np.ndarray, gt_np: np.ndarray) -> None:
        _ = state["metric"].update_cm(pr=pred_np, gt=gt_np)
        for pred_item, gt_item in zip(pred_np, gt_np):
            update_component_recall_stats(
                state["component"],
                gt_item,
                pred_item,
                tiny_area_thresh=int(getattr(self.opt, "tiny_area_thresh", 100)),
                small_area_thresh=int(getattr(self.opt, "small_area_thresh", 400)),
            )
            update_prediction_blob_stats(state["blob"], gt_item, pred_item)

    def _scores_from_eval_state(self, state: dict) -> dict:
        scores = state["metric"].get_scores()
        scores.update(component_recall_scores(state["component"]))
        scores.update(prediction_blob_scores(state["blob"]))
        scores["tiny_combo"] = self._compute_tiny_combo(scores)
        scores["tiny_safe_combo"] = self._compute_tiny_safe_combo(scores)
        return scores

    def _plot_cd_result(self, x1, x2, pred, target, epoch, stage):
        if len(pred.shape) == 4:
            pred = self._threshold_prediction(pred, threshold=self.eval_fg_threshold)
        vis_input = make_numpy_grid(de_norm(x1[0:8], self.opt.mean, self.opt.std))
        vis_input2 = make_numpy_grid(de_norm(x2[0:8], self.opt.mean, self.opt.std))
        vis_pred = make_numpy_grid(pred[0:8].unsqueeze(1).repeat(1, 3, 1, 1))
        vis_gt = make_numpy_grid(target[0:8].unsqueeze(1).repeat(1, 3, 1, 1))
        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(self.vis_path, f"{stage}_" + str(epoch) + ".jpg")
        plt.imsave(file_name, vis)

    def train(self, epoch):
        tbar = tqdm(self.train_data, ncols=80)
        opt.phase = "train"
        _loss = 0.0
        _focal_loss = 0.0
        _tversky_loss = 0.0
        loss_stat_sums = {
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
        last_lr = self.optimizer.param_groups[0]["lr"]

        for i, data in enumerate(tbar):
            self.model.model.train()
            img1 = data["img1"].to(self.model.device)
            img2 = data["img2"].to(self.model.device)
            label = data["cd_label"].to(self.model.device)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                pred, focal, tversky = self.model(img1, img2, label, epoch=epoch)
                loss = focal + tversky

            if self.amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            _loss += loss.item()
            _focal_loss += focal.item()
            _tversky_loss += tversky.item()
            for key in loss_stat_sums:
                value = self.model.last_loss_stats.get(key)
                if value is not None:
                    loss_stat_sums[key] += float(value.detach().item())
            last_lr = self.optimizer.param_groups[0]["lr"]
            del loss

            tbar.set_description(
                "L:%.3f F:%.3f T:%.3f B:%.2f Q:%.3f LR:%.6f"
                % (
                    _loss / (i + 1),
                    _focal_loss / (i + 1),
                    _tversky_loss / (i + 1),
                    loss_stat_sums["tversky_beta"] / (i + 1),
                    loss_stat_sums["query_gate"] / (i + 1),
                    last_lr,
                )
            )

            if i == len(tbar) - 1:
                self._plot_cd_result(data["img1"], data["img2"], pred, data["cd_label"], epoch, "train")
        self.schedular.step()

        n = max(1, i + 1)
        return {
            "loss": _loss / n,
            "focal": _focal_loss / n,
            "tversky": _tversky_loss / n,
            "main_focal": loss_stat_sums["main_focal"] / n,
            "main_tversky": loss_stat_sums["main_tversky"] / n,
            "aux_focal": loss_stat_sums["aux_focal"] / n,
            "aux_tversky": loss_stat_sums["aux_tversky"] / n,
            "support_loss": loss_stat_sums["support_loss"] / n,
            "coarse_loss": loss_stat_sums["coarse_loss"] / n,
            "support_weight": loss_stat_sums["support_weight"] / n,
            "coarse_weight": loss_stat_sums["coarse_weight"] / n,
            "aux_loss_scale": loss_stat_sums["aux_loss_scale"] / n,
            "tversky_beta": loss_stat_sums["tversky_beta"] / n,
            "query_gate": loss_stat_sums["query_gate"] / n,
            "lr": last_lr,
        }

    def val(self, epoch):
        tbar = tqdm(self.val_data, ncols=80)
        self.running_metric.clear()
        component_stats = init_component_recall_stats()
        blob_stats = init_prediction_blob_stats()
        opt.phase = "val"
        self.model.eval()
        threshold_states = {threshold: self._new_eval_state() for threshold in self.eval_thresholds}
        self.last_threshold_scan = {}

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                with self._autocast_context():
                    val_logits = self.model.inference(
                        _data["img1"].to(self.model.device), _data["img2"].to(self.model.device)
                    )
                val_target = _data["cd_label"].detach()
                val_pred = self._threshold_prediction(val_logits, threshold=self.eval_fg_threshold)
                _ = self.running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())
                pred_np = val_pred.cpu().numpy()
                gt_np = val_target.cpu().numpy()
                for threshold, state in threshold_states.items():
                    scan_pred = self._threshold_prediction(val_logits, threshold=threshold)
                    self._update_eval_state(state, scan_pred.cpu().numpy(), gt_np)
                for pred_item, gt_item in zip(pred_np, gt_np):
                    update_component_recall_stats(
                        component_stats,
                        gt_item,
                        pred_item,
                        tiny_area_thresh=int(getattr(self.opt, "tiny_area_thresh", 100)),
                        small_area_thresh=int(getattr(self.opt, "small_area_thresh", 400)),
                    )
                    update_prediction_blob_stats(blob_stats, gt_item, pred_item)
                if i == len(tbar) - 1:
                    self._plot_cd_result(
                        _data["img1"],
                        _data["img2"],
                        val_logits,
                        _data["cd_label"],
                        epoch,
                        "val",
                    )
            val_scores = self.running_metric.get_scores()
            val_scores.update(component_recall_scores(component_stats))
            val_scores.update(prediction_blob_scores(blob_stats))
            val_scores["tiny_combo"] = self._compute_tiny_combo(val_scores)
            val_scores["tiny_safe_combo"] = self._compute_tiny_safe_combo(val_scores)
            self.last_threshold_scan = {
                f"{threshold:.2f}": {
                    key: float(scores.get(key, 0.0))
                    for key in [
                        "iou_1",
                        "precision_1",
                        "recall_1",
                        "tiny_recall",
                        "small_recall",
                        "tiny_safe_combo",
                    ]
                }
                for threshold, scores in (
                    (threshold, self._scores_from_eval_state(state))
                    for threshold, state in threshold_states.items()
                )
            }
            message = "(phase: %s) " % (self.opt.phase)
            for k, v in val_scores.items():
                if k.endswith("_components") or k.endswith("_count") or k.endswith("_area"):
                    message += "%s: %d " % (k, int(v))
                else:
                    message += "%s: %.3f " % (k, v * 100)
            print(message)
            if self.last_threshold_scan:
                scan_message = "threshold-scan "
                for threshold, scores in self.last_threshold_scan.items():
                    scan_message += (
                        f"| th={threshold} "
                        f"iou_1={scores['iou_1'] * 100:.2f} "
                        f"tiny={scores['tiny_recall'] * 100:.2f} "
                        f"safe={scores['tiny_safe_combo'] * 100:.2f} "
                    )
                print(scan_message)
        print(
            "best-metric summary | "
            f"selected={self.best_metric}:{self._metric_value(self.best_metric, val_scores) * 100:.3f} "
            f"| iou_1={float(val_scores.get('iou_1', 0.0)) * 100:.3f} "
            f"| tiny_recall={float(val_scores.get('tiny_recall', 0.0)) * 100:.3f} "
            f"| small_recall={float(val_scores.get('small_recall', 0.0)) * 100:.3f} "
            f"| precision_1={float(val_scores.get('precision_1', 0.0)) * 100:.3f} "
            f"| blob_precision={float(val_scores.get('blob_precision', 0.0)) * 100:.3f} "
            f"| tiny_combo={float(val_scores.get('tiny_combo', 0.0)) * 100:.3f} "
            f"| tiny_safe_combo={float(val_scores.get('tiny_safe_combo', 0.0)) * 100:.3f}"
        )
        self._update_best_checkpoints(epoch, val_scores)

        return val_scores


if __name__ == "__main__":
    opt = Options().parse()
    trainval = HACQITrainer(opt)
    setup_seed(seed=1)

    for epoch in range(1, opt.num_epochs + 1):
        print("\n==> Name %s, Epoch %i, previous best = %.3f" % (opt.name, epoch, trainval.previous_best * 100))
        train_stats = trainval.train(epoch)
        val_scores = trainval.val(epoch)

        trainval._append_log_line(epoch, train_stats, val_scores)

        if epoch % 10 == 0:
            trainval.model.save_periodic(opt.name, opt.backbone, epoch)
            print(f"[INFO] Periodic checkpoint saved at epoch {epoch}")

    print("Done!")
