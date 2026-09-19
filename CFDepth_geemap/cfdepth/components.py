"""原网格输入与一次八连通矢量化；严格核对回写成员和编号。"""

import math
import ee
from .grid import Grid, validate_grid, pack_stage

FABDEM = "projects/sat-io/open-datasets/FABDEM"


def inputs(cfg):
    if cfg.fixture:
        size = 36 if cfg.fixture == "small" else 360
        grid = {"crs": "EPSG:4326", "transform": [1 / 3600, 0, 110, 0, -1 / 3600, 30]}
        region = ee.Geometry.Rectangle(
            [110, 30 - size / 3600, 110 + size / 3600, 30], None, False
        )
        g = Grid(grid, region, cfg.p)
        x = g.xy.select("x")
        y = g.xy.select("y")
        shape = x.gte(4).And(x.lt(size - 12)).And(y.gte(4)).And(y.lt(size - 12))
        if cfg.fixture == "small":
            shape = shape.Or(x.gte(25).And(x.lte(28)).And(y.gte(25)).And(y.lte(28)))
            shape = (
                shape.Or(x.eq(27).And(y.eq(32)))
                .Or(x.eq(28).And(y.eq(33)))
                .Or(x.eq(34).And(y.eq(34)))
            )
        raw = shape.toByte().clip(region).rename("raw")
        if cfg.fixture == "small":
            raw = raw.updateMask(x.eq(14).And(y.eq(14)).Not())
        dem = g.C(100).reproject(g.P).where(shape, 99).rename("dem")
        return raw, dem, grid, region
    original = ee.Image(cfg.flood_asset)
    if original.bandNames().size().getInfo() != 1:
        raise ValueError("Flood input must contain exactly one band")
    raw = original.select([0]).rename("raw")
    region = raw.geometry().bounds(1).buffer(1000).bounds(1)
    collection = ee.ImageCollection(FABDEM).filterBounds(region)
    if collection.size().getInfo() < 1:
        raise ValueError("FABDEM is empty over input")
    grid = validate_grid(
        ee.Image(collection.first()).select([0]).projection().getInfo()
    )
    dem = (
        collection.mosaic()
        .select([0])
        .setDefaultProjection(ee.Projection(grid["crs"], grid["transform"]))
        .rename("dem")
        .clip(region)
    )
    return raw, dem, grid, region


def check_input(raw, cfg):
    native = validate_grid(raw.projection().getInfo(), dem=False)
    bad = raw.neq(0).And(raw.neq(1)).rename("bad")
    out = bad.reduceRegion(
        reducer=ee.Reducer.max().combine(ee.Reducer.count(), sharedInputs=True),
        geometry=raw.geometry(),
        crs=native["crs"],
        crsTransform=native["transform"],
        maxPixels=cfg.p.maxPixels,
        tileScale=cfg.p.tileScale,
    ).getInfo()
    if not out.get("bad_count") or out.get("bad_max") != 0:
        raise ValueError("Input must have valid binary 0/1 pixels: " + str(out))
    return {"grid": native, "valid_pixels": out["bad_count"]}


def parse_histogram(hist: dict) -> tuple[dict[int, int], int]:
    parsed = {}
    null_count = 0
    for key, count in hist.items():
        if (
            not isinstance(count, (float, int))
            or not math.isfinite(count)
            or count < 0
            or count % 1
        ):
            raise ValueError("Invalid histogram count")
        if key == "null":
            null_count = count
            continue
        try:
            value = float(key)
        except ValueError as exc:
            raise ValueError("Invalid component ID " + key) from exc
        if not math.isfinite(value) or not 0 < value < 2**53 or value % 1:
            raise ValueError("Invalid component ID " + key)
        if int(value) in parsed:
            raise ValueError("Duplicate numerical component ID")
        parsed[int(value)] = int(count)
    return parsed, null_count


def label_components(support, g):
    expected = g.indicator(support)
    xy = g.xy.updateMask(expected)
    info = ee.Dictionary(
        {
            "extent": g.total(xy, ee.Reducer.minMax().unweighted()),
            "count": g.total(expected.rename("n"), ee.Reducer.sum().unweighted()),
        }
    ).getInfo()
    extent = info["extent"]
    count = info["count"]["n"]
    if not count:
        raise ValueError("Empty flood support")
    width = extent["x_max"] - extent["x_min"] + 1
    height = extent["y_max"] - extent["y_min"] + 1
    if width * height >= 2**53:
        raise ValueError("Component key exceeds exact integer range")
    key = (
        xy.select("y")
        .subtract(extent["y_min"])
        .multiply(width)
        .add(xy.select("x").subtract(extent["x_min"]))
        .add(1)
        .toInt64()
        .rename("pixel_key")
    )
    args = g.args(ee.Reducer.min().unweighted().setOutputs(["component_id"]))
    args.update(
        geometryType="polygon",
        eightConnected=True,
        labelProperty="support_value",
        geometryInNativeProjection=True,
    )
    vectors = support.selfMask().toInt().addBands(key).reduceToVectors(**args)
    vector_ids = vectors.aggregate_array("component_id")
    raw = (
        g.C(0)
        .toInt64()
        .reproject(g.P)
        .paint(vectors, "component_id")
        .reproject(g.P)
        .rename("cid")
    )
    ids = raw.updateMask(support)
    restored = g.indicator(ids.gt(0))
    checks = g.pack(
        [
            expected.rename("expected"),
            restored.rename("restored"),
            expected.And(restored.Not()).rename("missing"),
            restored.And(expected.Not()).rename("added"),
            g.indicator(
                ids.lte(0).Or(ids.gte(2**53)).Or(ids.neq(ids.floor()))
            ).rename("invalid"),
        ]
    )
    result = ee.Dictionary(
        {
            "counts": g.total(checks, ee.Reducer.sum().unweighted()),
            "histogram": g.total(ids, ee.Reducer.frequencyHistogram().unweighted()),
            "distinct": g.total(ids, ee.Reducer.countDistinctNonNull().unweighted()),
            "vector_ids": vector_ids,
        }
    ).getInfo()
    counts = result["counts"]
    hist, null_count = parse_histogram(result["histogram"].get("cid") or {})
    vs = result["vector_ids"]
    if any(
        type(v) not in (int, float)
        or not math.isfinite(v)
        or not 0 < v < 2**53
        or v % 1
        for v in vs
    ):
        raise ValueError("Invalid vector component ID")
    if (
        counts["expected"] != counts["restored"]
        or counts["missing"]
        or counts["added"]
        or counts["invalid"]
        or sum(hist.values()) != counts["restored"]
        or len(set(vs)) != len(vs)
        or set(hist) != set(vs)
        or len(hist) != result["distinct"].get("cid")
    ):
        raise ValueError("Component membership/ID audit failed: " + str(result))
    return ids, {
        "pixels": count,
        "components": len(vs),
        "histogram_null_count": null_count,
    }


def build_components(raw, dem, grid, region, cfg):
    g = Grid(grid, region, cfg.p)
    observed = (
        raw.mask()
        .unmask(0, False)
        .clip(region)
        .reduceResolution(ee.Reducer.min(), False, 4096)
        .reproject(g.P)
        .eq(1)
    )
    flood = (
        raw.eq(1)
        .reduceResolution(ee.Reducer.max(), False, 4096)
        .reproject(g.P)
        .unmask(0)
        .gt(0)
        .And(dem.mask())
        .And(g.finite(dem))
    )
    support = flood.rename("support")
    dry = observed.And(flood.Not()).And(dem.mask()).And(g.finite(dem)).rename("dry")
    ids, audit = label_components(support, g)
    hard = (
        support.And(observed)
        .unmask(0)
        .focalMin(1, "square", "pixels")
        .And(support)
        .rename("hard")
    )
    if cfg.fixture:
        expected = (418, 4) if cfg.fixture == "small" else (118336, 1)
        if (audit["pixels"], audit["components"]) != expected:
            raise ValueError("Fixture component mismatch: " + str(audit))
    return (
        pack_stage(g.pack([dem, ids.unmask(0), support, dry, hard]), "components"),
        audit,
    )
