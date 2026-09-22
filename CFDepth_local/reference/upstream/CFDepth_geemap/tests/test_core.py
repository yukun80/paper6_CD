import copy
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize_scalar
import ee
from ee.apitestcase import ApiTestCase
from cfdepth.config import Config, ROOT
from cfdepth.numerics import (
    coordinate_minimum,
    numeric_ops,
    boundary_terms,
    advance_component,
    directional_difference,
    STATE_FIELDS,
)
from cfdepth.grid import grouped_layout, validate_grid, pack_stage
from cfdepth.components import inputs, parse_histogram
from cfdepth.solver import Solver
from cfdepth.backend import validate_contract
from cfdepth import VERSION


def config():
    return Config.read("CFDepth_geemap/configs/small.json")


class NumericTests(unittest.TestCase):
    def test_js_and_scipy(self):
        rng = random.Random(42)
        cases = []
        p = config().p
        for _ in range(300):
            lower = rng.uniform(-20, 20)
            upper = lower + rng.uniform(0, 8)
            cases.append(
                dict(
                    degree=rng.uniform(0.01, 8),
                    neighborSum=rng.uniform(-100, 100),
                    boundary=rng.uniform(0, 10),
                    lower=lower,
                    upper=upper,
                    terrain=rng.uniform(-10, 25),
                    hard=bool(rng.randrange(2)),
                    soft=rng.random(),
                    midWeight=rng.random() / 100,
                    mid=(lower + upper) / 2,
                )
            )
        boundaries = [
            dict(
                wet={
                    "median": rng.uniform(0, 20),
                    "mad": rng.random(),
                    "count": rng.randrange(8),
                },
                dry={
                    "median": rng.uniform(0, 20),
                    "mad": rng.random(),
                    "count": rng.randrange(8),
                },
                slope=rng.uniform(0, 40),
                peer=None if i % 2 else {"median": 10, "mad": 0.2, "count": i % 6},
            )
            for i in range(50)
        ]
        states = []
        for i in range(200):
            s = {k: 0 for k in STATE_FIELDS}
            s.update(
                status=i % 6,
                attempt=i % 4,
                sweeps=1990 if i % 3 == 0 else 10,
                stable=i % 3,
                prev_primary=1,
                prev_total=1,
                base_primary=1,
                base_mid=1,
            )
            r = dict(
                primary=1 + (i % 4) * 1e-5,
                total=1,
                mid=1,
                residual=0 if i % 2 else 0.1,
                hard=0,
                bad=1 if i % 17 == 0 else 0,
            )
            states.append([s, r])
        js = "const k=require(process.argv[1]);let x=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify({coord:x.c.map(v=>k.coordinateMinimum(v,k.numericOps())),boundary:x.b.map(v=>k.boundaryTerms(v.wet,v.dry,v.slope,v.peer,k.CONFIG,k.numericOps())),state:x.s.map(v=>k.advanceComponent(v[0],v[1],k.CONFIG,k.numericOps()))}));"
        ref = json.loads(
            subprocess.check_output(
                ["node", "-e", js, str(ROOT / "CFDepth/CFDepth_0919.txt")],
                input=json.dumps({"c": cases, "b": boundaries, "s": states}).encode(),
            )
        )
        for v, want in zip(cases, ref["coord"]):
            got = coordinate_minimum(v)
            self.assertAlmostEqual(got, want, places=11)

            def f(s):
                return (
                    v["degree"] * (s - v["neighborSum"] / v["degree"]) ** 2
                    + v["boundary"] * max(0, v["lower"] - s, s - v["upper"]) ** 2
                    + v["soft"] * max(v["terrain"] - s, 0) ** 2
                    + v["midWeight"] * (s - v["mid"]) ** 2
                )

            lower = v["terrain"] if v["hard"] else -20000
            optimum = minimize_scalar(
                f, bounds=(lower, 20000), method="bounded", options={"xatol": 1e-10}
            )
            self.assertLessEqual(f(got), optimum.fun + 1e-7)
            if v["hard"]:
                self.assertGreaterEqual(got, v["terrain"])
        for v, want in zip(boundaries, ref["boundary"]):
            got = boundary_terms(**v, cfg=p, o=numeric_ops())
            for k in want:
                self.assertAlmostEqual(got[k], want[k], places=12)
        for (s, r), want in zip(states, ref["state"]):
            self.assertEqual(advance_component(s, r, p, numeric_ops()), want)

    def test_gradient_missing(self):
        for hp, hm in [(True, True), (True, False), (False, True), (False, False)]:
            r = directional_difference(10, 12, 8, 2, 2, hp, hm, numeric_ops())
            self.assertEqual(r["valid"], hp or hm)
            if r["valid"]:
                self.assertEqual(r["value"], 1)

    def test_histogram(self):
        hist, n = parse_histogram({"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "null": 7})
        self.assertEqual(len(hist), 5)
        self.assertEqual(n, 7)
        self.assertEqual(parse_histogram({"null": 12}), ({}, 12))
        for value in ({"x": 2}, {"0": 1}, {"1": 0.4}, {"null": -1}, {"1": "1"}):
            with self.assertRaises(ValueError):
                parse_histogram(value)

    def test_layout_grid(self):
        self.assertEqual(
            grouped_layout(["max", "a", "min", "b"], ["max", "sum", "min", "sum"]),
            [("a", "sum"), ("b", "sum"), ("max", "max"), ("min", "min")],
        )
        for t in ("WKT", [1, 0, 0, 0, 0, 0], [float("nan"), 0, 0, 0, -1, 0]):
            with self.assertRaises(ValueError):
                validate_grid({"crs": "EPSG:4326", "transform": t})


class GraphTests(ApiTestCase):
    def test_real_official_expression_all_stages(self):
        cfg = config()
        raw, dem, grid, region = inputs(cfg)
        D = (
            ee.Image.constant([99, 1, 1, 0, 1])
            .rename(["dem", "cid", "support", "dry", "hard"])
            .setDefaultProjection(dem.projection())
        )
        prepare = Solver(D, grid, region, cfg).prepare()
        self.assertIn("Reducer.group", prepare.serialize())
        # 阶段边界以资产替换，检查真实Python API签名及序列化，不伪称服务器求值。
        s = Solver(D, grid, region, cfg, ee.Image("projects/test/assets/prepared"))
        state = s.iterate(ee.Image("projects/test/assets/state"))
        encoded = state.serialize()
        self.assertIn("Image.where", encoded)
        self.assertIn("Reducer.combine", encoded)
        self.assertEqual(len(json.loads(encoded)["values"]) > 100, True)
        self.assertIn(
            "Image.sqrt", s.gradients(ee.Image("S"), D.select("support")).serialize()
        )
        with patch.object(
            ee.Dictionary, "getInfo", return_value={"running": 0, "audit_failed": 0}
        ):
            images, _ = s.finalize(ee.Image("state"))
            self.assertIn("CFDepth_WSE_Gradient", images["gradient"].serialize())
        task = ee.batch.Export.image.toAsset(
            image=pack_stage(state, "state"),
            assetId="projects/test/assets/state",
            crs=grid["crs"],
            crsTransform=grid["transform"],
            region=region,
            pyramidingPolicy={".default": "sample"},
        )
        self.assertNotIn("driveDestination", str(task.config))


class ContractTests(unittest.TestCase):
    def test_saved_contract(self):
        from cfdepth.grid import band_names

        cfg = config()
        grid = {"crs": "EPSG:4326", "transform": [1 / 3600, 0, 110, 0, -1 / 3600, 30]}
        props = dict(
            implementation=VERSION,
            identity=cfg.identity,
            run_id=cfg.run_id,
            kind="state",
            step=2,
            source="s",
            token="t",
            grid_crs=grid["crs"],
            grid_transform_json=json.dumps(grid["transform"]),
            components_token="c",
            prepared_token="p",
        )
        info = {
            "properties": props,
            "bands": [
                {"id": k, "crs": grid["crs"], "crs_transform": grid["transform"]}
                for k in band_names("state")
            ],
        }
        validate_contract(
            info,
            cfg,
            "state",
            2,
            "s",
            grid,
            {"components_token": "c", "prepared_token": "p"},
            "t",
        )
        for field, bad in [
            ("token", "bad"),
            ("grid_transform_json", "bad"),
            ("source", "bad"),
            ("components_token", "bad"),
        ]:
            wrong = copy.deepcopy(info)
            wrong["properties"][field] = bad
            with self.assertRaises(ValueError):
                validate_contract(
                    wrong, cfg, "state", 2, "s", grid, {"components_token": "c"}, "t"
                )
        wrong = copy.deepcopy(info)
        wrong["bands"].pop()
        with self.assertRaises(ValueError):
            validate_contract(wrong, cfg, "state", 2, "s")


if __name__ == "__main__":
    unittest.main()
