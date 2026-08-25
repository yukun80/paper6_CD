"""HA-CQI 数据、B2/DINO、loss、阈值与 checkpoint 快速契约测试。"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from unittest import mock
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
from model.backbones import build_feature_backbone  # noqa: E402
from model.decode_heads import OmniScaleStateSpaceChangeDecoder  # noqa: E402
from model.decode_heads.state_space_scan import (  # noqa: E402
    FourDirectionSelectiveScan2D,
    selective_scan_reference,
)
from model.losses.focal import FocalLoss  # noqa: E402
from model.losses.dice import DICELoss  # noqa: E402
from model.modules.dino_adapter import DinoV3FeatureExtractor  # noqa: E402
from model.modules.semantic_encoder import HierarchicalCnnDinoEncoder  # noqa: E402
from scripts.prepare_fused_sar_cd_dataset import (  # noqa: E402
    assign_splits,
    build_s1gfloods_records,
    build_varfloods_records,
    compute_cross_split_overlap,
)
from scripts.compute_s1gfloods_cd_stats import (  # noqa: E402
    compute_stats_payload,
    write_stats_payload,
)
from scripts.evaluate_sar_scene import resolve_inputs  # noqa: E402
from utils.flood_evaluation import (  # noqa: E402
    FloodEvaluationAccumulator,
    select_primary_threshold,
)
from option import Options, validate_stats_provenance  # noqa: E402


def valid_model_config() -> dict[str, object]:
    return {
        "architecture": "HA-CQI",
        "backbone": "efficientnet_b2",
        "backbone_weight": "pretrained/efficientnet_b2_ra-bcdf34b7.pth",
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
        "decoder": "oscd_v1",
        "decoder_channels": 128,
        "ssm_state_dim": 1,
        "ssm_directions": 4,
        "context_levels": [3, 4, 5],
        "detail_levels": [2, 1],
        "dino_arch": "dinov3_vits16",
        "dino_weight": "dinov3/weights/vits16.pth",
        "extract_ids": [2, 5, 8, 11],
        "dino_input_norm": "imagenet",
        "input_mean": [0.5, 0.5, 0.5],
        "input_std": [0.5, 0.5, 0.5],
    }


class PipelineContractTests(unittest.TestCase):
    def test_b2_backbone_contract_and_pyramid_shapes(self) -> None:
        weight = PROJECT_ROOT / "pretrained" / "efficientnet_b2_ra-bcdf34b7.pth"
        if not weight.is_file():
            self.skipTest(f"missing local EfficientNet-B2 weights: {weight}")
        backbone = build_feature_backbone(str(weight)).eval()
        self.assertEqual(backbone.channels, [16, 24, 48, 120, 352])
        self.assertEqual(backbone.reductions, [2, 4, 8, 16, 32])
        with torch.inference_mode():
            features = backbone(torch.randn(1, 3, 256, 256))
        self.assertEqual(
            [tuple(feature.shape) for feature in features],
            [
                (1, 16, 128, 128),
                (1, 24, 64, 64),
                (1, 48, 32, 32),
                (1, 120, 16, 16),
                (1, 352, 8, 8),
            ],
        )

    def test_b2_backbone_rejects_b0_weights(self) -> None:
        weight = PROJECT_ROOT / "pretrained" / "efficientnet_b0_ra-3dd342df.pth"
        if not weight.is_file():
            self.skipTest(f"missing archived EfficientNet-B0 weights: {weight}")
        with self.assertRaisesRegex(RuntimeError, "B0 and other backbone weights"):
            build_feature_backbone(str(weight))

    def test_checkpoint_model_contract_rejects_b0_and_shared_norm(self) -> None:
        for field, invalid_value in (
            ("backbone", "efficientnet_b0"),
            ("dino_input_norm", "shared"),
        ):
            model_config = valid_model_config()
            model_config[field] = invalid_value
            payload = {
                "network": {"weight": torch.ones(1)},
                "meta": {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "model_config": model_config,
                },
            }
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "model contract mismatch"
            ):
                checkpoint_model_config(payload)

    def test_checkpoint_contract_rejects_non_oscd_decoder(self) -> None:
        model_config = valid_model_config()
        model_config["decoder"] = "query_dense_v0"
        payload = {
            "network": {"weight": torch.ones(1)},
            "meta": {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "model_config": model_config,
            },
        }
        with self.assertRaisesRegex(ValueError, "OSCD-only decoder contract mismatch"):
            checkpoint_model_config(payload)

    def test_dino_input_contract_reverses_dataset_norm_then_uses_imagenet(self) -> None:
        encoder = object.__new__(HierarchicalCnnDinoEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.register_buffer(
            "input_mean",
            torch.tensor([0.2, 0.4, 0.6]).view(1, 3, 1, 1),
            persistent=False,
        )
        encoder.register_buffer(
            "input_std",
            torch.tensor([0.1, 0.2, 0.25]).view(1, 3, 1, 1),
            persistent=False,
        )
        encoder.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        encoder.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        raw = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1)
        dataset_normalized = (raw - encoder.input_mean) / encoder.input_std
        actual = encoder.prepare_dino_input(dataset_normalized)
        expected = (raw - encoder.dino_mean) / encoder.dino_std
        torch.testing.assert_close(actual, expected)

    def test_dino_selected_layers_match_legacy_all_layer_selection(self) -> None:
        weight = (
            PROJECT_ROOT
            / "dinov3"
            / "weights"
            / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        )
        if not weight.is_file():
            self.skipTest(f"missing local DINOv3 weights: {weight}")
        extractor = DinoV3FeatureExtractor(
            dino_arch="dinov3_vits16",
            weights_path=str(weight),
            extract_ids=[2, 5, 8, 11],
            device="cpu",
        )
        sample = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            legacy_all = extractor.model.get_intermediate_layers(
                sample, n=12, reshape=True, norm=True
            )
            selected = extractor.model.get_intermediate_layers(
                sample, n=[2, 5, 8, 11], reshape=True, norm=True
            )
        for expected_index, actual in zip([2, 5, 8, 11], selected):
            torch.testing.assert_close(actual, legacy_all[expected_index])

    def test_dino_forward_inherits_outer_autocast_context(self) -> None:
        class RecordingDino(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.autocast_enabled = False
                self.autocast_dtype = None

            def get_intermediate_layers(self, x, *, n, reshape, norm):
                self.autocast_enabled = torch.is_autocast_enabled("cpu")
                self.autocast_dtype = torch.get_autocast_dtype("cpu")
                self.requested_layers = list(n)
                self.reshape = reshape
                self.norm = norm
                return [x[:, :1] for _ in n]

        extractor = object.__new__(DinoV3FeatureExtractor)
        torch.nn.Module.__init__(extractor)
        extractor.device = torch.device("cpu")
        extractor.extract_ids = [2, 5, 8, 11]
        extractor.model = RecordingDino()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = extractor(torch.randn(1, 3, 32, 32))
        self.assertEqual(len(output), 4)
        self.assertTrue(extractor.model.autocast_enabled)
        self.assertEqual(extractor.model.autocast_dtype, torch.bfloat16)
        self.assertEqual(extractor.model.requested_layers, [2, 5, 8, 11])

    def test_oscd_decoder_shape_diagnostics_and_backward(self) -> None:
        torch.manual_seed(123)
        decoder = OmniScaleStateSpaceChangeDecoder(channels=8).eval()
        features = [
            torch.randn(1, 8, size, size, requires_grad=True)
            for size in (128, 64, 32, 16, 8)
        ]
        logits = decoder(features, output_size=(256, 256))
        logits.square().mean().backward()
        diagnostics = decoder.diagnostics()
        shapes = decoder.last_feature_shapes()
        self.assertEqual(tuple(logits.shape), (1, 2, 256, 256))
        self.assertEqual(shapes["fused_p3"], (1, 8, 32, 32))
        self.assertEqual(shapes["region_full"], (1, 8, 32, 32))
        self.assertEqual(shapes["region_half"], (1, 16, 16, 16))
        self.assertEqual(shapes["region_quarter"], (1, 32, 8, 8))
        self.assertEqual(shapes["scan_input"], (1, 24, 8, 8))
        self.assertEqual(shapes["scan_output"], (1, 24, 8, 8))
        self.assertEqual(
            set(diagnostics),
            {
                "decoder_scan_rms",
                "decoder_context_rms",
                "decoder_p2_detail_rms",
                "decoder_p1_detail_rms",
                "decoder_context_detail_ratio",
            },
        )
        self.assertTrue(all(torch.isfinite(value) for value in diagnostics.values()))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in decoder.parameters()
            )
        )
        self.assertLessEqual(sum(parameter.numel() for parameter in decoder.parameters()), 4_500_000)

    def test_oscd_pixel_unshuffle_is_lossless_and_padding_is_cropped(self) -> None:
        source = torch.randn(1, 3, 8, 12)
        restored = torch.nn.functional.pixel_shuffle(
            torch.nn.functional.pixel_unshuffle(source, 4), 4
        )
        torch.testing.assert_close(restored, source)

        decoder = OmniScaleStateSpaceChangeDecoder(channels=4).eval()
        features = [
            torch.randn(1, 4, height, width)
            for height, width in ((30, 38), (15, 19), (7, 9), (4, 5), (2, 3))
        ]
        with torch.inference_mode():
            logits = decoder(features, output_size=(61, 77))
        self.assertEqual(tuple(logits.shape), (1, 2, 61, 77))
        shapes = decoder.last_feature_shapes()
        self.assertEqual(shapes["fused_p3"][-2:], (7, 9))
        self.assertEqual(shapes["region_full"][-2:], (8, 12))
        self.assertEqual(shapes["scan_input"][-2:], (2, 3))

    def test_selective_scan_cpu_reference_is_finite_and_differentiable(self) -> None:
        generator = torch.Generator().manual_seed(17)
        u = torch.randn(1, 8, 5, generator=generator, requires_grad=True)
        delta = torch.randn(1, 8, 5, generator=generator, requires_grad=True)
        A = (-torch.rand(8, 1, generator=generator)).requires_grad_()
        B = torch.randn(1, 4, 1, 5, generator=generator, requires_grad=True)
        C = torch.randn(1, 4, 1, 5, generator=generator, requires_grad=True)
        D = torch.randn(8, generator=generator, requires_grad=True)
        delta_bias = torch.randn(8, generator=generator, requires_grad=True)
        output = selective_scan_reference(
            u,
            delta,
            A,
            B,
            C,
            D,
            delta_bias=delta_bias,
        )
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        for tensor in (u, delta, A, B, C, D, delta_bias):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_scan_matches_cpu_reference_output_and_gradients(self) -> None:
        self.assertTrue(FourDirectionSelectiveScan2D.cuda_backend_available())
        torch.manual_seed(29)
        cpu_module = FourDirectionSelectiveScan2D(channels=4).eval()
        cuda_module = FourDirectionSelectiveScan2D(channels=4).cuda().eval()
        cuda_module.load_state_dict(cpu_module.state_dict())
        cpu_input = torch.randn(1, 4, 2, 3, requires_grad=True)
        cuda_input = cpu_input.detach().cuda().requires_grad_(True)
        cpu_output = cpu_module(cpu_input)
        cuda_output = cuda_module(cuda_input)
        torch.testing.assert_close(cuda_output.cpu(), cpu_output, rtol=3e-4, atol=3e-4)
        cpu_output.square().mean().backward()
        cuda_output.square().mean().backward()
        torch.testing.assert_close(
            cuda_input.grad.cpu(), cpu_input.grad, rtol=8e-4, atol=8e-4
        )
        for (_, cpu_parameter), (_, cuda_parameter) in zip(
            cpu_module.named_parameters(), cuda_module.named_parameters()
        ):
            torch.testing.assert_close(
                cuda_parameter.grad.cpu(), cpu_parameter.grad, rtol=2e-3, atol=2e-3
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_scan_bf16_keeps_state_parameters_fp32(self) -> None:
        module = FourDirectionSelectiveScan2D(channels=8).cuda().train()
        sample = torch.randn(2, 8, 3, 4, device="cuda", requires_grad=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = module(sample)
            loss = output.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(module.A_logs.dtype, torch.float32)
        self.assertEqual(module.Ds.dtype, torch.float32)
        self.assertTrue(torch.isfinite(sample.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_scan_missing_kernel_fails_without_fallback(self) -> None:
        import model.decode_heads.state_space_scan as scan_module

        module = FourDirectionSelectiveScan2D(channels=4).cuda().eval()
        sample = torch.randn(1, 4, 2, 3, device="cuda")
        with mock.patch.object(scan_module, "selective_scan_cuda", None), mock.patch.object(
            scan_module,
            "_SELECTIVE_SCAN_IMPORT_ERROR",
            ImportError("undefined symbol: selective_scan_cuda"),
        ):
            with self.assertRaisesRegex(RuntimeError, "no PyTorch/CPU fallback is allowed"):
                module(sample)

    def test_oscd_contract_has_no_decoder_query_interface(self) -> None:
        decoder = OmniScaleStateSpaceChangeDecoder(channels=8)
        state_keys = tuple(decoder.state_dict())
        self.assertFalse(any("query" in key for key in state_keys))
        parser_builder = Options()
        parser_builder.init()
        option_names = {
            option
            for action in parser_builder.parser._actions
            for option in action.option_strings
        }
        self.assertFalse(any(option.startswith("--mask_") for option in option_names))

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

    def test_stats_payload_write_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_a = root / "a.png"
            image_b = root / "b.png"
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(image_a)
            Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(image_b)
            payload = compute_stats_payload(
                data_root=root,
                split="train",
                paths=[image_a, image_b],
                dataset_fingerprint="fingerprint",
                manifest_sha256="manifest",
            )
            self.assertEqual(payload["num_images"], 2)
            self.assertEqual(payload["pixel_count"], 8)
            self.assertEqual(payload["recommended_config_fields"]["mean"], [0.5] * 3)
            output = root / "stats.json"
            write_stats_payload(payload, output)
            self.assertEqual(json.loads(output.read_text())["dataset_fingerprint"], "fingerprint")

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

            payload["meta"]["model_config"] = {
                "architecture": "HA-CQI",
                "decoder": "oscd_v1",
            }
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
