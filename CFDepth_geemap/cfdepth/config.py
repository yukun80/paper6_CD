"""配置严格校验，所有相对路径以仓库根目录解析。"""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import re
from . import VERSION

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = json.loads(Path(__file__).with_name("defaults.json").read_text())


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Config:
    def __init__(self, value: dict) -> None:
        self.data = dict(value)
        allowed = {
            "project",
            "run_id",
            "asset_root",
            "flood_asset",
            "fixture",
            "run_root",
            "output_root",
            "poll_seconds",
            "cleanup_previous_states",
            "download_threads",
            "parameters",
        }
        if set(value) - allowed:
            raise ValueError(
                "Unknown configuration fields: " + str(set(value) - allowed)
            )
        for k in allowed - {"parameters"}:
            if k not in value:
                raise ValueError("Missing configuration field: " + k)
        if not re.fullmatch(r"cfdepth_py321_[A-Za-z0-9_-]+", value["run_id"]):
            raise ValueError("Use independent run_id beginning cfdepth_py321_")
        if not re.fullmatch(
            r"projects/[^/\s]+/assets(?:/[A-Za-z0-9_-]+)*", value["asset_root"]
        ):
            raise ValueError("Invalid asset_root")
        if value["fixture"] not in ("", "small", "large"):
            raise ValueError("Invalid fixture")
        if not value["fixture"] and not isinstance(value["flood_asset"], str):
            raise ValueError("Missing flood_asset")
        if type(value["cleanup_previous_states"]) is not bool:
            raise ValueError("cleanup_previous_states must be bool")
        if type(value["poll_seconds"]) is not int or value["poll_seconds"] < 10:
            raise ValueError("poll_seconds >=10")
        if (
            type(value["download_threads"]) is not int
            or not 1 <= value["download_threads"] <= 8
        ):
            raise ValueError("download_threads 1..8")
        params = dict(DEFAULTS)
        changes = value.get("parameters", {})
        if set(changes) - params.keys():
            raise ValueError("Unknown numerical parameters")
        params.update(changes)
        canonical(params)
        for key, v in params.items():
            if key in ("muRatios", "diagnosticDetails"):
                continue
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError("Invalid parameter: " + key)
            if key != "noData" and v < 0:
                raise ValueError("Negative parameter: " + key)
        for k in (
            "lambdaC",
            "lambdaB",
            "lambdaT",
            "sigmaFloor",
            "slopeScaleDeg",
            "dispersionScaleMeters",
            "residualTolerance",
            "objectiveTolerance",
            "objectiveFloor",
            "budgetRelative",
            "budgetAbsolute",
        ):
            if params[k] <= 0:
                raise ValueError("Expected positive " + k)
        for k in (
            "pairRadius",
            "peerRadius",
            "minSamplesPerSide",
            "minAnchors",
            "minSpanPixels",
            "sweepsPerStage",
            "maxSweeps",
            "tileScale",
        ):
            if type(params[k]) is not int or params[k] < 1:
                raise ValueError("Expected positive integer " + k)
        if (
            params["maxSweeps"] < 2 * params["sweepsPerStage"]
            or params["maxSweeps"] % params["sweepsPerStage"]
        ):
            raise ValueError("Invalid sweep schedule")
        if not 0 < params["highWeight"] <= 1 or params["noData"] >= 0:
            raise ValueError("Invalid weight/NoData")
        ratios = params["muRatios"]
        if (
            not isinstance(ratios, list)
            or not ratios
            or any(type(r) not in (int, float) or not 0 < r < 1 for r in ratios)
            or any(a <= b for a, b in zip(ratios, ratios[1:]))
        ):
            raise ValueError("muRatios must decrease in (0,1)")
        self.params = params
        self.p = SimpleNamespace(**params)
        self.run_id = value["run_id"]
        self.project = value["project"]
        self.fixture = value["fixture"]
        self.flood_asset = value["flood_asset"]
        self.prefix = value["asset_root"] + "/" + self.run_id
        self.run_dir = (ROOT / value["run_root"] / self.run_id).resolve()
        self.output_dir = (ROOT / value["output_root"] / self.run_id).resolve()
        self.identity = digest(
            dict(
                version=VERSION,
                project=self.project,
                asset_prefix=self.prefix,
                fixture=self.fixture,
                flood=self.flood_asset,
                params=params,
                output=str(self.output_dir),
            )
        )
        self.max_stages = (
            (1 + len(ratios)) * params["maxSweeps"] // params["sweepsPerStage"]
        )

    def asset(self, kind: str, step: int = 0) -> str:
        if kind not in ("components", "prepared", "state"):
            raise ValueError("Invalid kind")
        if kind == "state" and (
            type(step) is not int or not 1 <= step <= self.max_stages
        ):
            raise ValueError("Invalid step")
        return self.prefix + ("_state_%05d" % step if kind == "state" else "_" + kind)

    @classmethod
    def read(cls, path):
        return cls(json.loads((ROOT / path).read_text()))
