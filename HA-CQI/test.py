"""HA-CQI patch 数据集显式 checkpoint 评估入口。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data.cd_dataset import DataLoader
from model.checkpointing import (
    apply_checkpoint_model_config,
    atomic_json_save,
    checkpoint_data_config,
    checkpoint_model_config,
    extract_network_state,
    load_checkpoint_payload,
    resolve_inference_threshold,
)
from model.engine import build_hacqi_engine
from option import Options, _resolve_existing_project_path
from utils.flood_evaluation import FloodEvaluationAccumulator, build_threshold_grid


def parse_and_prepare():
    builder = Options()
    builder.init()
    builder.parser.set_defaults(phase="val")
    builder.parser.add_argument(
        "--report_json",
        type=str,
        default="",
        help="评估 JSON；默认写入 checkpoint 同目录。",
    )
    builder.parser.add_argument(
        "--prediction_dir",
        type=str,
        default="",
        help="--save_test 时的预测目录；默认写入 checkpoint 同目录/pred。",
    )
    raw_opt = builder.parser.parse_args()
    if not raw_opt.checkpoint:
        builder.parser.error("--checkpoint is required for test.py")

    explicit = {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    checkpoint_path = Path(_resolve_existing_project_path(raw_opt.checkpoint))
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    model_config = checkpoint_model_config(payload)
    raw_opt = apply_checkpoint_model_config(raw_opt, model_config, explicit)
    data_config = checkpoint_data_config(payload) or {}
    for field in ("dataset", "dataroot", "stats_file"):
        if field not in explicit and data_config.get(field):
            setattr(raw_opt, field, data_config[field])
    raw_opt.checkpoint = str(checkpoint_path)
    raw_opt.threshold, raw_opt.threshold_source = resolve_inference_threshold(
        payload,
        explicit_threshold=raw_opt.threshold,
    )
    opt = builder.prepare(raw_opt)
    if opt.phase != "val":
        raise ValueError("test.py evaluates the locked validation split; --phase must be val")
    opt.report_json = str(
        Path(opt.report_json).expanduser().resolve()
        if opt.report_json
        else checkpoint_path.parent / f"{checkpoint_path.stem}_test_metrics.json"
    )
    opt.prediction_dir = str(
        Path(opt.prediction_dir).expanduser().resolve()
        if opt.prediction_dir
        else checkpoint_path.parent / "pred"
    )
    return opt, payload


@torch.inference_mode()
def main() -> None:
    opt, payload = parse_and_prepare()
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    print(f"#testing images = {len(test_loader)}")

    model = build_hacqi_engine(opt)
    model.model.load_state_dict(extract_network_state(payload), strict=True)
    model.eval()
    thresholds = build_threshold_grid(opt.threshold_min, opt.threshold_max, opt.threshold_step)
    evaluator = FloodEvaluationAccumulator(
        thresholds,
        reference_threshold=float(opt.threshold),
        tiny_area_thresh=int(opt.tiny_area_thresh),
        small_area_thresh=int(opt.small_area_thresh),
    )
    prediction_dir = Path(opt.prediction_dir)
    if opt.save_test:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    steps = 0
    for batch in tqdm(test_data, ncols=100):
        logits = model.inference(
            batch["img1"].to(model.device, non_blocking=True),
            batch["img2"].to(model.device, non_blocking=True),
        )
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        target = batch["cd_label"].cpu().numpy()
        evaluator.update(probabilities, target)
        steps += 1
        if opt.save_test:
            predictions = probabilities >= float(opt.threshold)
            for index, filename in enumerate(batch["fname"]):
                output_name = f"{Path(filename).stem}.png"
                Image.fromarray((predictions[index].astype(np.uint8) * 255)).save(
                    prediction_dir / output_name
                )
        if opt.max_val_steps > 0 and steps >= opt.max_val_steps:
            break

    if steps == 0:
        raise RuntimeError("No test steps were executed")
    result = evaluator.scores()
    report = {
        "checkpoint": str(opt.checkpoint),
        "dataset": str(opt.dataset),
        "phase": str(opt.phase),
        "stats_file": str(opt.stats_file),
        "threshold": float(opt.threshold),
        "threshold_source": str(opt.threshold_source),
        "steps": int(steps),
        "evaluation": result,
    }
    atomic_json_save(report, opt.report_json)
    reference = result["reference"]
    print(
        f"threshold={opt.threshold:.2f} ({opt.threshold_source}) | "
        f"IoU={reference['iou_1'] * 100:.4f} "
        f"F1={reference['F1_1'] * 100:.4f} "
        f"P={reference['precision_1'] * 100:.4f} "
        f"R={reference['recall_1'] * 100:.4f}"
    )
    print(f"report={opt.report_json}")


if __name__ == "__main__":
    main()
