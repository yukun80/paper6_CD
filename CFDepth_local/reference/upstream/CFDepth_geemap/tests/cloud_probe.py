"""可选真实GEE小图求值验收；不创建任务或资产。"""

import json
import math
from pathlib import Path
import ee
from cfdepth.config import Config
from cfdepth.grid import Grid, grouped, R
from cfdepth.solver import Solver
from cfdepth.storage import atomic_json


def main():
    cfg = Config.read("CFDepth_geemap/configs/small.json")
    ee.Initialize(project=cfg.project)
    grid = {"crs": "EPSG:4326", "transform": [1 / 3600, 0, 110, 0, -1 / 3600, 30]}
    region = ee.Geometry.Rectangle(
        [110, 30 - 10 / 3600, 110 + 10 / 3600, 30], None, False
    )
    g = Grid(grid, region, cfg.p)
    x = g.xy.select("x")
    y = g.xy.select("y")
    a = x.gte(1).And(x.lte(3)).And(y.gte(1)).And(y.lte(3))
    b = x.gte(6).And(x.lte(8)).And(y.gte(6)).And(y.lte(8))
    support = a.Or(b)
    ids = g.C(0).where(a, 1).where(b, 2).rename("cid")
    dem = g.C(99).where(b, 109).rename("dem")
    hard = x.eq(2).And(y.eq(2)).Or(x.eq(7).And(y.eq(7)))
    D = g.pack(
        [
            dem,
            ids,
            support.rename("support"),
            support.Not().rename("dry"),
            hard.rename("hard"),
        ]
    )
    prepared = g.pack(
        [
            dem.add(0.4).rename("lower"),
            dem.add(0.6).rename("upper"),
            dem.add(0.5).rename("mid"),
            g.C(1).rename("weight"),
        ]
    ).updateMask(support)
    solver = Solver(D, grid, region, cfg, prepared)
    initial = dem.add(1.5).where(hard, dem.add(3)).updateMask(support).rename("S")
    after = solver.sweep(initial, g.C(0), support)
    other = solver.sweep(initial.where(b, 130), g.C(0), support)
    peak = x.multiply(10).add(7).rename("peak").updateMask(g.C(0.4))
    v = x.add(1).rename("a").updateMask(g.C(0.4))
    groups = grouped(
        g.pack([peak, v]), ["peak", "a"], ["max", "sum"], ids.updateMask(support), g
    )
    reference = v.updateMask(v.mask().multiply(a)).reduceRegion(
        **g.args(ee.Reducer.sum())
    )
    plane = g.lat.multiply(math.pi / 180 * R * 0.01).updateMask(support)
    gradient = solver.gradients(plane, support)
    result = ee.Dictionary(
        {
            "change": g.total(
                after.subtract(initial).abs().rename("v"), ee.Reducer.max()
            ),
            "before": g.total(
                solver.energy(initial, g.C(0)).select("primary"), ee.Reducer.sum()
            ),
            "after": g.total(
                solver.energy(after, g.C(0)).select("primary"), ee.Reducer.sum()
            ),
            "hard_violation": g.total(
                dem.add(0.01).subtract(after).max(0).updateMask(hard).rename("v"),
                ee.Reducer.max(),
            ),
            "cross_component": g.total(
                after.subtract(other).abs().updateMask(a).rename("v"), ee.Reducer.max()
            ),
            "gradient_error": g.total(
                gradient.subtract(0.01).abs().rename("v"), ee.Reducer.max()
            ),
            "groups": groups.toList(10),
            "reference": reference,
        }
    ).getInfo()
    assert result["change"]["v"] > 0, result
    assert result["after"]["primary"] < result["before"]["primary"], result
    assert result["hard_violation"]["v"] <= 1e-9, result
    assert result["cross_component"]["v"] == 0, result
    assert result["gradient_error"]["v"] <= 1e-10, result
    rows = {
        int(row["properties"]["cid"]): row["properties"] for row in result["groups"]
    }
    assert abs(rows[1]["a"] - result["reference"]["a"]) < 1e-10, result
    assert rows[1]["peak"] == 37, result
    atomic_json(
        Path("CFDepth_geemap/environment/cloud-probe.json"),
        {"passed": True, "result": result},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
