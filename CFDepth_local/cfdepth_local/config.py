"""Frozen baseline with explicit, configurable convergence tolerances."""
from pathlib import Path
from types import SimpleNamespace
import json
import math

DEFAULTS = json.loads(Path(__file__).with_name("defaults.json").read_text())
ALGORITHM = {k: v for k, v in DEFAULTS.items() if k not in {"tileScale", "maxPixels", "noData", "diagnosticDetails"}}
CONVERGENCE_OVERRIDES = frozenset({"residualTolerance", "objectiveTolerance"})
RUNTIME = dict(mask="", dem="", output_dir="", solver_backend="python_reference",
               spatial_profile="geographic_v321", slope_method="central4_geographic_v1",
               vertical_unit="m", alignment_tolerance_pixels=1e-5, memory_gib=None,
               threads=4, checkpoint_seconds=300, component_ids=None, tile_size=512)

def parameters(overrides=None, *, test_only=False):
    overrides = overrides or {}
    if set(overrides) - ALGORITHM.keys():
        raise ValueError("Unknown algorithm fields")
    for k, v in overrides.items():
        if k in CONVERGENCE_OVERRIDES and isinstance(v, list):
            raise ValueError(f"Invalid scalar algorithm parameter: {k}")
        values = v if isinstance(v, list) else [v]
        if not values or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f"Invalid algorithm parameter: {k}")
        if not test_only and k not in CONVERGENCE_OVERRIDES and v != ALGORITHM[k]:
            raise ValueError(f"Frozen production parameter: {k}")
    return SimpleNamespace(**(ALGORITHM | overrides))

def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if set(cfg) - {"algorithm", "runtime"}:
        raise ValueError("Configuration must contain algorithm/runtime only")
    algorithm = vars(parameters(cfg.get("algorithm")))
    user = cfg.get("runtime", {})
    if set(user) - RUNTIME.keys():
        raise ValueError("Unknown runtime fields")
    r = RUNTIME | user
    if r["spatial_profile"] != "geographic_v321" or r["slope_method"] != "central4_geographic_v1" or r["vertical_unit"] != "m":
        raise ValueError("Unsupported spatial/vertical profile")
    if r["alignment_tolerance_pixels"] != 1e-5:
        raise ValueError("Alignment tolerance is fixed at 1e-5 pixels")
    if r["solver_backend"] not in {"python_reference", "numba_serial", "numba_parallel"}:
        raise ValueError("Unknown solver backend")
    for key in ("mask", "dem", "output_dir"):
        if not r[key]:
            raise ValueError(f"Missing {key}")
        r[key] = str((path.parent / r[key]).resolve())
    for key in ("threads", "tile_size"):
        if type(r[key]) is not int or r[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not isinstance(r["checkpoint_seconds"], (int, float)) or r["checkpoint_seconds"] <= 0:
        raise ValueError("Invalid checkpoint interval")
    if r["memory_gib"] is not None and (not math.isfinite(r["memory_gib"]) or r["memory_gib"] <= 0):
        raise ValueError("Invalid memory budget")
    if r["component_ids"] is not None and (not r["component_ids"] or any(type(i) is not int or i <= 0 for i in r["component_ids"])):
        raise ValueError("Invalid component IDs")
    return dict(algorithm=algorithm, runtime=r)
