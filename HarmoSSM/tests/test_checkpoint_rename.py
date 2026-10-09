"""验证更名兼容仅影响名称与路径，不放宽模型契约。"""

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from model.checkpointing import (
    FIXED_MODEL_CONFIG,
    INFERENCE_MODEL_CONFIG_FIELDS,
    apply_checkpoint_model_config,
    checkpoint_data_config,
    checkpoint_model_config,
    relocate_checkpoint_path,
)


class RenameCompatibilityTests(unittest.TestCase):
    def payload(self, architecture="HA-CQI"):
        config = dict.fromkeys(INFERENCE_MODEL_CONFIG_FIELDS)
        config.update(FIXED_MODEL_CONFIG)
        config.update(
            architecture=architecture,
            dino_arch="dinov3_vits16",
            dino_fusion_layers=[5, 8, 11],
            dino_weight="HA-CQI/dinov3/weights/example.pth",
            backbone_weight="HA-CQI/pretrained/example.pth",
        )
        return {"network": {"weight": torch.zeros(1)}, "meta": {
            "format_version": 2, "model_config": config,
            "data_config": {"stats_file": "HA-CQI/checkpoints/cache.json"},
        }}

    def test_old_and_new_architectures_preserve_original_metadata(self):
        for name in ("HA-CQI", "HarmoSSM"):
            payload = self.payload(name)
            before = copy.deepcopy(payload["meta"])
            config = checkpoint_model_config(payload)
            self.assertEqual(config["architecture"], name)
            self.assertIn("/HarmoSSM/dinov3/", config["dino_weight"])
            self.assertIn("/HarmoSSM/checkpoints/", checkpoint_data_config(payload)["stats_file"])
            self.assertEqual(payload["meta"], before)

    def test_explicit_weight_paths_win(self):
        config = checkpoint_model_config(self.payload())
        opt = SimpleNamespace(dino_weight="/custom/dino.pth", backbone_weight="/custom/b2.pth")
        apply_checkpoint_model_config(opt, config, {"dino_weight", "backbone_weight"})
        self.assertEqual(opt.dino_weight, "/custom/dino.pth")
        self.assertEqual(opt.backbone_weight, "/custom/b2.pth")

    def test_relocation_is_limited_to_this_project(self):
        root = Path(__file__).resolve().parents[2]
        old = str(root / "HA-CQI/checkpoints/HA-CQI-old/model.pth")
        expected = str(root / "HarmoSSM/checkpoints/HarmoSSM-old/model.pth")
        self.assertEqual(relocate_checkpoint_path(old), expected)
        for path in ("/external/HA-CQI/model.pth", "other/HA-CQI/model.pth", "datasets/stats.json", "/envs/hacqi/bin/python"):
            self.assertEqual(relocate_checkpoint_path(path), path)

    def test_unknown_architecture_and_old_format_still_fail(self):
        with self.assertRaises(ValueError):
            checkpoint_model_config(self.payload("unknown"))
        payload = self.payload()
        payload["meta"]["format_version"] = 1
        with self.assertRaises(ValueError):
            checkpoint_model_config(payload)


if __name__ == "__main__":
    unittest.main()
