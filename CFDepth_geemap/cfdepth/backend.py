"""公开EE任务接口与严格阶段合同；不调用Drive或浏览器调度接口。"""

import json
import logging
import math
import time
import uuid
import ee
from . import VERSION
from .config import canonical, digest
from .components import FABDEM, inputs, check_input, build_components
from .grid import Grid, validate_grid, band_names, pack_stage
from .solver import Solver

LOG = logging.getLogger("cfdepth")


def is_missing(exc: Exception) -> bool:
    # 权限错误即使写有not found也不视为可安全创建/删除的证据。
    status = getattr(getattr(exc, "resp", None), "status", None)
    text = str(exc).lower()
    return status == 404 or (
        "not found" in text and "permission" not in text and "access" not in text
    )


def retry_read(fn, sleep=time.sleep):
    for attempt in range(4):
        try:
            return fn()
        except Exception as exc:
            text = str(exc).lower()
            transient = any(
                s in text
                for s in (
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "timed out",
                    "timeout",
                    "connection reset",
                    "connection aborted",
                    "temporarily unavailable",
                )
            )
            if not transient or attempt == 3:
                raise
            sleep((30, 60, 120)[attempt])


def grid_equal(a: dict, b: dict) -> bool:
    if a["crs"] != b["crs"]:
        return False
    scale = max(abs(a["transform"][0]), abs(a["transform"][4]))
    return all(
        abs(x - y) <= scale * 1e-6 for x, y in zip(a["transform"], b["transform"])
    )


def validate_contract(
    info, cfg, kind, step, source, grid=None, snapshots=None, token=None
):
    props = info.get("properties", {})
    expected = dict(
        implementation=VERSION,
        identity=cfg.identity,
        run_id=cfg.run_id,
        kind=kind,
        step=step,
        source=source,
    )
    for key, value in expected.items():
        if props.get(key) != value:
            raise ValueError("Stage metadata mismatch " + key)
    if token is not None and props.get("token") != token:
        raise ValueError("Stage token changed")
    if not isinstance(props.get("token"), str) or not props["token"]:
        raise ValueError("Missing stage token")
    try:
        stored = validate_grid(
            {
                "crs": props["grid_crs"],
                "transform": json.loads(props["grid_transform_json"]),
            }
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Missing/invalid grid metadata") from exc
    if grid and not grid_equal(stored, grid):
        raise ValueError("Cross-stage grid mismatch")
    bands = info.get("bands", [])
    if [b["id"] for b in bands] != band_names(kind):
        raise ValueError("Stage band contract mismatch")
    for b in bands:
        actual = validate_grid(
            {"crs": b.get("crs"), "transform": b.get("crs_transform")}
        )
        if not grid_equal(actual, stored):
            raise ValueError("Actual band grid differs: " + b["id"])
    for key, value in (snapshots or {}).items():
        if props.get(key) != value:
            raise ValueError("Snapshot token changed: " + key)
    return props, stored


def parse_status_histogram(hist: dict) -> dict[int, int]:
    result = {}
    for key, count in hist.items():
        if key == "null":
            continue
        try:
            value = float(key)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid solver status key: " + str(key)) from exc
        if (
            not math.isfinite(value)
            or value % 1
            or not 0 <= value <= 5
            or int(value) in result
        ):
            raise ValueError("Invalid or duplicate solver status: " + str(hist))
        if (
            type(count) not in (int, float)
            or not math.isfinite(count)
            or count < 0
            or count % 1
        ):
            raise ValueError("Invalid solver status count")
        result[int(value)] = int(count)
    if not result:
        raise ValueError("Empty solver status histogram")
    return result


class Backend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cache = {}
        self.grid = None
        self.region = None
        self.source = None
        self.raw = None
        self.dem = None

    def initialize(self):
        ee.Initialize(project=self.cfg.project)
        # 两种输入均验证计算项目与FABDEM权限；不打印认证对象。
        root = self.cfg.data["asset_root"]
        retry_read(lambda: ee.data.listAssets({"parent": root, "pageSize": 1}))
        meta = (
            None
            if self.cfg.fixture
            else retry_read(lambda: ee.data.getAsset(self.cfg.flood_asset))
        )
        if meta and str(meta.get("type", "")).upper() != "IMAGE":
            raise ValueError("Input asset is not Image")
        demmeta = retry_read(lambda: ee.data.getAsset(FABDEM))
        self.source = digest(
            {
                "fixture": self.cfg.fixture,
                "input": meta,
                "dem": demmeta,
                "version": VERSION,
            }
        )
        return {
            "project": self.cfg.project,
            "source": self.source,
            "asset_root_readable": True,
            "write_permission_verified": False,
        }

    def metadata(self, path):
        try:
            return retry_read(lambda: ee.data.getAsset(path))
        except Exception as exc:
            if is_missing(exc) or "does not exist" in str(exc).lower():
                # EE把缺失与无权限合并为同一消息。只有父目录完整列表也证明缺失才返回None。
                parent = path.rsplit("/", 1)[0]
                page_token = None
                while True:
                    args = {"parent": parent, "pageSize": 1000}
                    if page_token:
                        args["pageToken"] = page_token
                    page = retry_read(lambda: ee.data.listAssets(args))
                    if any(
                        a.get("name", a.get("id")) == path
                        for a in page.get("assets", [])
                    ):
                        raise exc
                    page_token = page.get("nextPageToken")
                    if not page_token:
                        return None
            raise

    def inventory(self):
        assets = []
        token = None
        while True:
            params = {"parent": self.cfg.data["asset_root"], "pageSize": 1000}
            if token:
                params["pageToken"] = token
            page = retry_read(lambda: ee.data.listAssets(params))
            assets.extend(
                a
                for a in page.get("assets", [])
                if a.get("name", a.get("id", "")).startswith(self.cfg.prefix + "_")
            )
            token = page.get("nextPageToken")
            if not token:
                break
        tasks = retry_read(ee.data.getTaskList)
        return assets, [
            t
            for t in tasks
            if t.get("description", "").startswith(self.cfg.run_id + "_")
        ]

    def check(self):
        raw, dem, grid, region = inputs(self.cfg)
        report = check_input(raw, self.cfg)
        report["dem_grid"] = grid
        self.raw, self.dem, self.grid, self.region = raw, dem, grid, region
        return report

    def restore(self, record, journal):
        path = record["path"]
        kind = record["kind"]
        step = record["step"]
        # 不缓存跨阶段资产：新state必须重新读取。
        info = retry_read(lambda: ee.Image(path).getInfo())
        snapshots = {}
        if kind != "components":
            snapshots["components_token"] = journal["components"]["token"]
        if kind == "state":
            snapshots["prepared_token"] = journal["prepared"]["token"]
        props, grid = validate_contract(
            info,
            self.cfg,
            kind,
            step,
            self.source,
            self.grid,
            snapshots,
            record["token"],
        )
        region = ee.Geometry(json.loads(props["region_json"]))
        if self.region is not None and canonical(
            json.loads(props["region_json"])
        ) != canonical(self.region.getInfo()):
            raise ValueError("Stage region mismatch")
        self.grid, self.region = grid, region
        self.cache[path] = ee.Image(path)
        return self.cache[path]

    def build(self, kind, step, journal, token):
        if kind == "components":
            self.check()
            image, audit = build_components(
                self.raw, self.dem, self.grid, self.region, self.cfg
            )
            LOG.info("Component audit %s", audit)
        else:
            D = self.cache[journal["components"]["path"]]
            prepared = (
                None if kind == "prepared" else self.cache[journal["prepared"]["path"]]
            )
            solver = Solver(D, self.grid, self.region, self.cfg, prepared)
            if kind == "prepared":
                image = solver.prepare()
            else:
                current = (
                    pack_stage(prepared, "state")
                    if step == 1
                    else self.cache[journal["latest"]["path"]]
                )
                image = solver.iterate(current)
        props = dict(
            implementation=VERSION,
            identity=self.cfg.identity,
            run_id=self.cfg.run_id,
            kind=kind,
            step=step,
            token=token,
            source=self.source,
            grid_crs=self.grid["crs"],
            grid_transform_json=canonical(self.grid["transform"]),
            region_json=canonical(self.region.getInfo()),
        )
        if kind != "components":
            props["components_token"] = journal["components"]["token"]
        if kind == "state":
            props["prepared_token"] = journal["prepared"]["token"]
        return image.set(props)

    def reserve(self):
        return retry_read(lambda: ee.data.newTaskId(1))[0]

    def submit(self, record, journal):
        image = self.build(record["kind"], record["step"], journal, record["token"])
        task = ee.batch.Export.image.toAsset(
            image=image,
            description=record["description"],
            assetId=record["path"],
            region=self.region,
            crs=self.grid["crs"],
            crsTransform=self.grid["transform"],
            maxPixels=self.cfg.p.maxPixels,
            pyramidingPolicy={".default": "sample"},
        )
        # request ID仅用于幂等提交，不一定等于服务端operation ID。
        operation = ee.data.exportImage(
            record.get("request_id", record["task_id"]), task.config
        )
        return {
            "task_id": operation["name"].rsplit("/", 1)[1],
            "operation": operation["name"],
        }

    def task_status(self, record):
        rows = retry_read(lambda: ee.data.getTaskStatus(record["task_id"]))
        if len(rows) != 1:
            raise RuntimeError("Ambiguous task status")
        row = rows[0]
        if row.get("state") == "UNKNOWN":
            matches = [
                t
                for t in retry_read(ee.data.getTaskList)
                if t.get("description") == record["description"]
            ]
            if len(matches) > 1:
                raise RuntimeError(
                    "Ambiguous submission receipt; multiple tasks with same description"
                )
            if matches:
                row = matches[0]
        if row.get("description") not in (None, record["description"]):
            raise RuntimeError("Task description changed")
        return row

    def status(self, journal):
        state = self.cache[journal["latest"]["path"]]
        D = self.cache[journal["components"]["path"]]
        g = Grid(self.grid, self.region, self.cfg.p)
        result = ee.Dictionary(
            {
                "histogram": g.total(
                    state.select("status").updateMask(D.select("support")),
                    ee.Reducer.frequencyHistogram().unweighted(),
                ),
                "support": g.total(
                    g.indicator(D.select("support")).rename("n"),
                    ee.Reducer.sum().unweighted(),
                ),
            }
        ).getInfo()
        hist = parse_status_histogram(result["histogram"].get("status", {}))
        if sum(hist.values()) != result["support"]["n"]:
            raise ValueError("State status coverage differs from support")
        return {
            "running_pixels": hist.get(1, 0) + hist.get(2, 0),
            "failed_pixels": hist.get(5, 0),
            "histogram": hist,
        }

    def remove(self, record, journal):
        if self.metadata(record["path"]) is None:
            return
        self.restore(record, journal)
        ee.data.deleteAsset(record["path"])
        if self.metadata(record["path"]) is not None:
            raise RuntimeError("Deletion not yet confirmed")

    def products(self, journal):
        s = Solver(
            self.cache[journal["components"]["path"]],
            self.grid,
            self.region,
            self.cfg,
            self.cache[journal["prepared"]["path"]],
        )
        return s.finalize(self.cache[journal["latest"]["path"]])
