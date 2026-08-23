"""A 路线数据、loss、阈值与 checkpoint 的快速契约测试。"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tif_io import (  # noqa: E402
    build_valid_mask,
    read_binary_label_tif_with_valid_mask,
    stretch_sar_array,
)
from data.cd_dataset import Load_Dataset  # noqa: E402
from model.checkpointing import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    atomic_torch_save,
    checkpoint_model_config,
    load_checkpoint_payload,
    load_network_state,
    resolve_inference_threshold,
)
from model.losses.focal import FocalLoss  # noqa: E402
from model.losses.dice import DICELoss  # noqa: E402
from scripts.prepare_fused_sar_cd_dataset import (  # noqa: E402
    assign_splits,
    build_s1gfloods_records,
    build_varfloods_records,
    compute_cross_split_overlap,
)
from scripts.evaluate_sar_scene import resolve_inputs  # noqa: E402
from utils.flood_evaluation import (  # noqa: E402
    FloodEvaluationAccumulator,
    select_primary_threshold,
)
from option import validate_stats_provenance  # noqa: E402


def valid_model_config() -> dict[str, object]:
    return {
        "architecture": "HA-CQI",
        "backbone": "efficientnet_b0",
        "backbone_weight": "pretrained/efficientnet_b0.pth",
        "fpn_channels": 128,
        "deform_groups": 4,
        "gamma_mode": "SE",
        "beta_mode": "contextgatedconv",
        "disable_soft_alignment": False,
        "align_window": 5,
        "align_points": 9,
        "align_heads": 4,
        "align_on_levels": [1, 2, 3],
        "align_qkv_bias": False,
        "align_offset_groups": 4,
        "num_change_queries": 16,
        "cqi_heads": 4,
        "mask_dim": 128,
        "mask_queries": 32,
        "mask_decoder_layers": 3,
        "mask_heads": 4,
        "dino_arch": "dinov3_vits16",
        "dino_weight": "dinov3/weights/vits16.pth",
        "extract_ids": [2, 5, 8, 11],
        "dino_input_norm": "shared",
        "input_mean": [0.5, 0.5, 0.5],
        "input_std": [0.5, 0.5, 0.5],
    }


class PipelineContractTests(unittest.TestCase):
    def test_train_and_val_dataset_phase_is_frozen(self) -> None:
        opt = SimpleNamespace(
            dataroot=str(REPO_ROOT / "datasets"),
            dataset="S1GFloods_CD_DINO_BG_75_25",
            phase="train",
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            input_size=256,
            dataset_mode="sar",
        )
        train_dataset = Load_Dataset(opt)
        opt.phase = "val"
        val_dataset = Load_Dataset(opt)
        opt.phase = "train"
        self.assertEqual(train_dataset.phase, "train")
        self.assertEqual(val_dataset.phase, "val")

    def test_corrected_split_is_reproducible_and_keeps_background(self) -> None:
        class Args:
            varfloods_root = REPO_ROOT / "datasets" / "VarFloods"
            tile_size = 256
            stride = 128
            min_valid_ratio = 0.0
            strict = False

        records = build_s1gfloods_records(
            REPO_ROOT / "datasets" / "S1GFloods", strict=False
        )
        var_records, _ = build_varfloods_records(Args())
        all_records = records + var_records
        first = assign_splits(all_records, 0.75, 42)
        second = assign_splits(all_records, 0.75, 42)
        self.assertEqual(
            [item.sample_id for item in first["train"]],
            [item.sample_id for item in second["train"]],
        )
        self.assertEqual((len(first["train"]), len(first["val"])), (5064, 1688))
        self.assertEqual(
            (
                sum(item.is_background for item in first["train"]),
                sum(item.is_background for item in first["val"]),
            ),
            (437, 144),
        )
        self.assertEqual(compute_cross_split_overlap(first)["overlap_pairs"], 2044)

    def test_focal_contract_and_background_penalty(self) -> None:
        generator = torch.Generator().manual_seed(123)
        logits = torch.randn(2, 2, 8, 8, generator=generator)
        target = torch.randint(0, 2, (2, 8, 8), generator=generator)
        criterion = FocalLoss(class_weights=[0.25, 0.75], gamma=2.0)
        actual = criterion(logits, target)
        flat_logits = logits.float().permute(0, 2, 3, 1).reshape(-1, 2)
        flat_target = target.long().reshape(-1)
        logpt = torch.log_softmax(flat_logits, dim=1)
        logpt = logpt.gather(1, flat_target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        sample_weights = torch.tensor([0.25, 0.75]).gather(0, flat_target)
        expected = (-sample_weights * (1.0 - pt).pow(2.0) * logpt).mean()
        self.assertTrue(torch.equal(actual, expected))
        with self.assertRaisesRegex(ValueError, "class_weights"):
            FocalLoss(class_weights=0.25, gamma=2.0)

        background = torch.zeros((1, 8, 8), dtype=torch.long)
        correct_logits = torch.stack(
            [torch.full((1, 8, 8), 4.0), torch.full((1, 8, 8), -4.0)], dim=1
        )
        false_positive_logits = -correct_logits
        correct_loss = criterion(correct_logits, background)
        false_positive_loss = criterion(false_positive_logits, background)
        self.assertTrue(torch.isfinite(false_positive_loss))
        self.assertGreater(float(false_positive_loss), float(correct_loss))
        tversky = DICELoss()
        correct_composite = correct_loss + tversky(
            correct_logits, background, alpha=0.30, beta=0.70
        )
        false_positive_composite = false_positive_loss + tversky(
            false_positive_logits, background, alpha=0.30, beta=0.70
        )
        self.assertTrue(torch.isfinite(false_positive_composite))
        self.assertGreater(float(false_positive_composite), float(correct_composite))

    def test_threshold_tie_break_is_iou_precision_then_higher_threshold(self) -> None:
        selected = select_primary_threshold(
            [
                {"threshold": 0.40, "iou_1": 0.7, "precision_1": 0.8},
                {"threshold": 0.50, "iou_1": 0.7, "precision_1": 0.9},
                {"threshold": 0.60, "iou_1": 0.7, "precision_1": 0.9},
            ]
        )
        self.assertEqual(selected["threshold"], 0.60)

    def test_checkpoint_threshold_priority(self) -> None:
        payload = {
            "network": {"weight": torch.ones(1)},
            "meta": {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "selection": {"threshold": 0.63},
                "model_config": valid_model_config(),
            }
        }
        self.assertEqual(
            resolve_inference_threshold(payload),
            (0.63, "checkpoint_selection"),
        )
        self.assertEqual(
            resolve_inference_threshold(payload, explicit_threshold=0.71),
            (0.71, "explicit_cli"),
        )
        payload["meta"]["selection"] = {}
        with self.assertRaisesRegex(ValueError, "requires --threshold"):
            resolve_inference_threshold(payload)

    def test_threshold_accumulator_uses_probability_ge_threshold(self) -> None:
        with self.assertRaises(TypeError):
            FloodEvaluationAccumulator([0.4])
        evaluator = FloodEvaluationAccumulator([0.4], reference_threshold=0.4)
        evaluator.update(
            np.asarray([[0.4, 0.399]], dtype=np.float32),
            np.asarray([[1, 0]], dtype=np.uint8),
        )
        metrics = evaluator.scores()["selected"]
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 0))

    def test_scene_evaluator_requires_threshold_source(self) -> None:
        args = SimpleNamespace(
            prediction_dir=None,
            probability=Path("probability.tif"),
            ground_truth=Path("ground_truth.tif"),
            filtered_binary=None,
            valid_mask=None,
            threshold=None,
            report=None,
        )
        with self.assertRaisesRegex(ValueError, "requires --threshold"):
            resolve_inputs(args)

    def test_legacy_nodata_three_is_never_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.tif"
            array = np.asarray([[0, 1], [3, 3]], dtype=np.uint8)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="uint8",
                nodata=3,
                transform=rasterio.transform.from_origin(0, 2, 1, 1),
            ) as dataset:
                dataset.write(array, 1)
            label, valid = read_binary_label_tif_with_valid_mask(path)
            self.assertEqual(label.tolist(), [[0, 1], [0, 0]])
            self.assertEqual(valid.tolist(), [[True, True], [False, False]])

    def test_checkpoint_loading_is_v2_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = torch.nn.Conv2d(2, 2, 1)
            invalid_payloads = {
                "raw.pth": source.state_dict(),
                "v1.pth": {
                    "network": source.state_dict(),
                    "meta": {"format_version": 1},
                },
                "missing_meta.pth": {"network": source.state_dict()},
            }
            for filename, invalid_payload in invalid_payloads.items():
                path = root / filename
                torch.save(invalid_payload, path)
                with self.subTest(filename=filename), self.assertRaisesRegex(
                    ValueError, "format v2"
                ):
                    load_checkpoint_payload(path)

            optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            v2_path = atomic_torch_save(
                {
                    "network": source.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": {},
                    "epoch": 1,
                    "global_step": 2,
                    "torch_rng_state": torch.get_rng_state(),
                    "data_loader_generator_state": torch.Generator().manual_seed(1).get_state(),
                    "meta": {
                        "format_version": CHECKPOINT_FORMAT_VERSION,
                        "model_config": valid_model_config(),
                        "data_config": {"dataset_fingerprint": "test"},
                        "selection": {"threshold": 0.55},
                    },
                },
                root / "v2.pth",
            )
            payload = load_checkpoint_payload(v2_path)
            self.assertEqual(payload["meta"]["format_version"], 2)
            self.assertIn("optimizer", payload)
            v2_target = torch.nn.Conv2d(2, 2, 1)
            load_network_state(v2_target, v2_path, strict=True)
            for expected, actual in zip(source.parameters(), v2_target.parameters()):
                self.assertTrue(torch.equal(expected, actual))

            payload["meta"].pop("model_config")
            with self.assertRaisesRegex(ValueError, "meta.model_config"):
                checkpoint_model_config(payload)

            payload["meta"]["model_config"] = {"architecture": "HA-CQI"}
            with self.assertRaisesRegex(ValueError, "lacks required inference fields"):
                checkpoint_model_config(payload)

    def test_shared_sar_helpers_preserve_validity_and_stretch(self) -> None:
        array = np.asarray([[0.0, 1.0], [2.0, -9999.0]], dtype=np.float32)
        valid = build_valid_mask(array, -9999.0)
        self.assertEqual(valid.tolist(), [[True, True], [True, False]])
        stretched = stretch_sar_array(array, valid, low=0.0, high=100.0)
        self.assertTrue(np.allclose(stretched[valid], [0.0, 0.5, 1.0]))
        self.assertEqual(float(stretched[1, 1]), 0.0)

    def test_shared_stretch_matches_materialized_varflood_tile(self) -> None:
        dataset_root = REPO_ROOT / "datasets" / "S1GFloods_CD_DINO_BG_75_25"
        with (dataset_root / "manifest_train.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            row = next(item for item in csv.DictReader(handle) if item["source"] == "varfloods_pro")
        source_root = REPO_ROOT / "datasets" / "VarFloods" / row["region"] / "PRO"
        a_path = next((source_root / "A").glob("*.tif"))
        b_path = next((source_root / "B").glob("*.tif"))
        window = Window(
            col_off=int(row["col_off"]),
            row_off=int(row["row_off"]),
            width=int(row["width"]),
            height=int(row["height"]),
        )
        with rasterio.open(a_path) as dataset_a, rasterio.open(b_path) as dataset_b:
            array_a = dataset_a.read(1, window=window).astype(np.float32, copy=False)
            array_b = dataset_b.read(1, window=window).astype(np.float32, copy=False)
            valid = build_valid_mask(array_a, dataset_a.nodata)
            valid &= build_valid_mask(array_b, dataset_b.nodata)
        expected = np.rint(stretch_sar_array(array_a, valid) * 255.0).astype(np.uint8)
        materialized = np.asarray(Image.open(dataset_root / row["a_png"]).convert("RGB"))[:, :, 0]
        self.assertTrue(np.array_equal(expected, materialized))

    def test_stats_and_split_fingerprints_match(self) -> None:
        root = REPO_ROOT / "datasets" / "S1GFloods_CD_DINO_BG_75_25"
        if not root.is_dir():
            self.skipTest("corrected dataset has not been materialized yet")
        report = json.loads((root / "split_report.json").read_text(encoding="utf-8"))
        stats = json.loads(
            (root / "channel_stats_s1gfloods_train.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["dataset_fingerprint"], stats["dataset_fingerprint"])

    def test_all_materialized_background_tiles_exist(self) -> None:
        root = REPO_ROOT / "datasets" / "S1GFloods_CD_DINO_BG_75_25"
        background_rows: list[dict[str, str]] = []
        split_counts: dict[str, int] = {}
        for split in ("train", "val"):
            with (root / f"manifest_{split}.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = [row for row in csv.DictReader(handle) if row["is_background"] == "1"]
            split_counts[split] = len(rows)
            background_rows.extend(rows)
        self.assertEqual(split_counts, {"train": 437, "val": 144})
        self.assertEqual(len(background_rows), 581)
        for row in background_rows:
            self.assertTrue((root / row["a_png"]).is_file())
            self.assertTrue((root / row["b_png"]).is_file())
            self.assertTrue((root / row["label_png"]).is_file())

    def test_stats_manifest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "split_report.json").write_text(
                json.dumps({"dataset_fingerprint": "fingerprint"}), encoding="utf-8"
            )
            (dataset_dir / "manifest_train.csv").write_text("sample_id\na\n", encoding="utf-8")
            stats_path = dataset_dir / "stats.json"
            stats_path.write_text(
                json.dumps(
                    {
                        "split": "train",
                        "dataset_fingerprint": "fingerprint",
                        "manifest_sha256": "wrong",
                    }
                ),
                encoding="utf-8",
            )
            opt = SimpleNamespace(
                dataroot=str(root), dataset="dataset", stats_file=str(stats_path)
            )
            with self.assertRaisesRegex(ValueError, "Stats/split manifest mismatch"):
                validate_stats_provenance(opt)


if __name__ == "__main__":
    unittest.main()
