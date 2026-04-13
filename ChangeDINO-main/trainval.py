import torch
from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model
from tqdm import tqdm
import math
from util.metric_tool import (
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
from util.util import make_numpy_grid, de_norm
import matplotlib.pyplot as plt


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.enabled = True  


class Trainval(object):
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

        self.model = create_model(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.alpha = 0.5
        self.topo_loss_weight = self.model.topo_loss_weight
        self.topo_warmup_epochs = int(getattr(opt, "topo_warmup_epochs", 10))
        self.num_epochs = opt.num_epochs
        self.eval_fg_threshold = float(getattr(opt, "eval_fg_threshold", 0.5))
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
                f.write(
                    "# name: %s | backbone: %s\n"
                    % (opt.name, getattr(opt, "backbone", "NA"))
                )
                f.write(
                    "# time,epoch,train_loss,train_focal,train_dice,train_topo,train_consistency,train_coarse_fp,lr,"
                )
                f.write("val_metrics(json)\n")

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

        line = (
            f"{ts},{epoch},"
            f"{train_stats.get('loss', float('nan')):.6f},"
            f"{train_stats.get('focal', float('nan')):.6f},"
            f"{train_stats.get('dice', float('nan')):.6f},"
            f"{train_stats.get('topo', float('nan')):.6f},"
            f"{train_stats.get('consistency', float('nan')):.6f},"
            f"{train_stats.get('coarse_fp', float('nan')):.6f},"
            f"{train_stats.get('lr', float('nan')):.8f},"
            + json.dumps(val_scores, ensure_ascii=False)
            + "\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

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
        _dice_loss = 0.0
        _topo_loss = 0.0
        _consistency_loss = 0.0
        _coarse_fp_loss = 0.0
        last_lr = self.optimizer.param_groups[0]["lr"]

        topo_w = self.topo_loss_weight if epoch > self.topo_warmup_epochs else 0.0
        consistency_w = float(getattr(self.model, "branch_consistency_weight", 0.0))
        coarse_fp_w = float(getattr(self.model, "coarse_fp_consistency_weight", 0.0))

        for i, data in enumerate(tbar):
            self.model.model.train()
            pred, focal, dice, topo_loss, consistency_loss, coarse_fp_loss = self.model(
                data["img1"].cuda(),
                data["img2"].cuda(),
                data["cd_label"].cuda(),
                epoch=epoch,
            )

            loss = focal + dice
            if topo_loss is not None and topo_w > 0:
                loss = loss + topo_w * topo_loss
            if consistency_loss is not None and consistency_w > 0:
                loss = loss + consistency_w * consistency_loss
            if coarse_fp_loss is not None and coarse_fp_w > 0:
                loss = loss + coarse_fp_w * coarse_fp_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _loss += loss.item()
            _focal_loss += focal.item()
            _dice_loss += dice.item()
            if topo_loss is not None:
                _topo_loss += topo_loss.item()
            if consistency_loss is not None:
                _consistency_loss += consistency_loss.item()
            if coarse_fp_loss is not None:
                _coarse_fp_loss += coarse_fp_loss.item()
            last_lr = self.optimizer.param_groups[0]["lr"]
            del loss

            tbar.set_description(
                "L:%.3f F:%.3f D:%.3f T:%.3f C:%.3f CF:%.3f LR:%.6f"
                % (
                    _loss / (i + 1),
                    _focal_loss / (i + 1),
                    _dice_loss / (i + 1),
                    _topo_loss / (i + 1),
                    _consistency_loss / (i + 1),
                    _coarse_fp_loss / (i + 1),
                    last_lr,
                )
            )

            if i == len(tbar) - 1:
                self._plot_cd_result(
                    data["img1"], data["img2"], pred, data["cd_label"], epoch, "train"
                )
        self.schedular.step()

        n = max(1, i + 1)
        return {
            "loss": _loss / n,
            "focal": _focal_loss / n,
            "dice": _dice_loss / n,
            "topo": _topo_loss / n,
            "consistency": _consistency_loss / n,
            "coarse_fp": _coarse_fp_loss / n,
            "lr": last_lr,
        }

    def val(self, epoch):
        tbar = tqdm(self.val_data, ncols=80)
        self.running_metric.clear()
        component_stats = init_component_recall_stats()
        blob_stats = init_prediction_blob_stats()
        opt.phase = "val"
        self.model.eval()

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                val_logits = self.model.inference(
                    _data["img1"].cuda(), _data["img2"].cuda()
                )
                val_target = _data["cd_label"].detach()
                val_pred = self._threshold_prediction(
                    val_logits, threshold=self.eval_fg_threshold
                )
                _ = self.running_metric.update_cm(
                    pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy()
                )
                pred_np = val_pred.cpu().numpy()
                gt_np = val_target.cpu().numpy()
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
            message = "(phase: %s) " % (self.opt.phase)
            for k, v in val_scores.items():
                if k.endswith("_components") or k.endswith("_count") or k.endswith("_area"):
                    message += "%s: %d " % (k, int(v))
                else:
                    message += "%s: %.3f " % (k, v * 100)
            print(message)
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
    trainval = Trainval(opt)
    setup_seed(seed=1)

    for epoch in range(1, opt.num_epochs + 1):
        print(
            "\n==> Name %s, Epoch %i, previous best = %.3f"
            % (opt.name, epoch, trainval.previous_best * 100)
        )
        train_stats = trainval.train(epoch)
        val_scores = trainval.val(epoch)

        trainval._append_log_line(epoch, train_stats, val_scores)

        if epoch % 10 == 0:
            trainval.model.save_periodic(opt.name, opt.backbone, epoch)
            print(f"[INFO] Periodic checkpoint saved at epoch {epoch}")

    print("Done!")
