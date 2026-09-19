"""固定原DEM网格与GEE运算类型；无任务提交。"""

import math
import ee
from .numerics import ee_ops, STATE_FIELDS

R = 6378137
DIRS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def validate_grid(grid: dict, dem: bool = True) -> dict:
    crs, t = grid.get("crs"), grid.get("transform")
    if (
        not isinstance(crs, str)
        or not crs
        or not isinstance(t, list)
        or len(t) != 6
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in t)
    ):
        raise ValueError("Grid requires CRS and six finite affine numbers")
    if t[0] * t[4] - t[1] * t[3] == 0:
        raise ValueError("Singular grid")
    if dem and (crs != "EPSG:4326" or t[1] != 0 or t[3] != 0 or t[0] <= 0 or t[4] >= 0):
        raise ValueError("DEM requires north-up EPSG:4326 grid")
    return grid


def band_names(kind: str) -> list[str]:
    base = ["dem", "cid", "support", "dry", "hard"]
    state = ["S", "baseS"] + STATE_FIELDS
    return {
        "components": base,
        "prepared": base + ["lower", "upper", "mid", "weight", "eligible"] + state,
        "state": state,
    }[kind]


def pack_stage(image: ee.Image, kind: str) -> ee.Image:
    return image.select(band_names(kind)).toDouble()


class Grid:
    def __init__(self, grid, region, cfg):
        validate_grid(grid)
        self.grid = grid
        self.t = grid["transform"]
        self.region = region
        self.cfg = cfg
        self.P = ee.Projection(grid["crs"], self.t)
        self.constants = {}
        self.shift = {}
        self.distances = {}
        self.xy = ee.Image.pixelCoordinates(self.P).floor().toInt64().reproject(self.P)
        x, y = self.xy.select("x"), self.xy.select("y")
        self.colors = (
            x.mod(2)
            .add(2)
            .mod(2)
            .add(y.mod(2).add(2).mod(2).multiply(2))
            .rename("color")
        )
        self.lat = ee.Image.pixelLonLat().select("latitude").reproject(self.P)
        self.ops = ee_ops(self.C)
        self.nops = ee_ops(ee.Number, True)

    def C(self, x):
        if type(x) not in (int, float):
            return (
                ee.Image.constant(x)
                .toDouble()
                .setDefaultProjection(self.P)
                .clip(self.region)
            )
        if x not in self.constants:
            self.constants[x] = (
                ee.Image.constant(x)
                .toDouble()
                .setDefaultProjection(self.P)
                .clip(self.region)
            )
        return self.constants[x]

    def args(self, reducer):
        return dict(
            reducer=reducer,
            geometry=self.region,
            crs=self.grid["crs"],
            crsTransform=self.t,
            maxPixels=self.cfg.maxPixels,
            tileScale=self.cfg.tileScale,
            bestEffort=False,
        )

    def total(self, image, reducer):
        return image.reduceRegion(**self.args(reducer))

    def at(self, image, dx, dy):
        return image.translate(-dx, -dy, "pixels", self.P).reproject(self.P)

    def static_at(self, image, dx, dy):
        key = (id(image), dx, dy)
        if key not in self.shift:
            self.shift[key] = (image, self.at(image, dx, dy))
        return self.shift[key][1]

    def finite(self, image):
        return image.eq(image).And(image.abs().lt(1e11))

    def pack(self, images):
        return ee.Image.cat(images).setDefaultProjection(self.P)

    def indicator(self, image):
        return (
            ee.Image.constant(0)
            .toByte()
            .reproject(self.P)
            .where(image.unmask(0, False).gt(0), 1)
        )

    def distance(self, dx, dy):
        if (dx, dy) not in self.distances:
            mid = self.lat.add(self.static_at(self.lat, dx, dy)).multiply(math.pi / 360)
            east = mid.cos().multiply(R * abs(self.t[0]) * math.pi / 180 * dx)
            self.distances[dx, dy] = (
                east.pow(2).add((R * abs(self.t[4]) * math.pi / 180 * dy) ** 2).sqrt()
            )
        return self.distances[dx, dy]


def grouped_layout(names: list[str], methods: list[str]) -> list[tuple[str, str]]:
    if (
        not names
        or len(names) != len(methods)
        or len(set(names)) != len(names)
        or "cid" in names
        or any(m not in ("sum", "min", "max") for m in methods)
    ):
        raise ValueError("Invalid grouped reducer fields")
    fields = list(zip(names, methods))
    return [p for p in fields if p[1] == "sum"] + [p for p in fields if p[1] != "sum"]


def grouped(image, names, methods, ids, g):
    fields = grouped_layout(names, methods)
    reducer = None
    for name, method in fields:
        part = getattr(ee.Reducer, method)().setOutputs([name])
        reducer = (
            part
            if reducer is None
            else reducer.combine(reducer2=part, sharedInputs=False)
        )
    reducer = reducer.group(groupField=len(fields), groupName="cid")
    valid = ids.gt(0).And(ids.lt(2**53)).And(ids.eq(ids.floor()))
    admitted = g.indicator(valid)
    selected = image.select([n for n, _ in fields])
    data = selected.updateMask(selected.mask().multiply(admitted))
    labels = ids.unmask(0, False).updateMask(admitted).rename("cid")
    rows = ee.List(
        data.addBands(labels).reduceRegion(**g.args(reducer)).get("groups", ee.List([]))
    )
    return ee.FeatureCollection(
        rows.map(lambda row: ee.Feature(None, ee.Dictionary(row)))
    )
