"""本地单进程控制器：先落盘授权，再清理；未知提交结果不重复提交。"""

import json
import logging
import time
import uuid
from .storage import atomic_json, run_lock
from .backend import retry_read
from .download import download_products

LOG = logging.getLogger("cfdepth")


class Scheduler:
    def __init__(self, cfg, backend, sleep=time.sleep, downloader=download_products):
        self.cfg = cfg
        self.b = backend
        self.sleep = sleep
        self.downloader = downloader
        self.j = None
        self.path = cfg.run_dir / "journal.json"

    def save(self) -> None:
        atomic_json(self.path, self.j)

    def open(self, resume: bool) -> None:
        if self.path.exists():
            if not resume:
                raise ValueError("Run already exists; use --resume")
            self.j = json.loads(self.path.read_text())
            if (
                self.j["identity"] != self.cfg.identity
                or self.j["source"] != self.b.source
            ):
                raise ValueError("Run parameters/source changed")
        else:
            assets, tasks = self.b.inventory()
            if assets or tasks:
                raise ValueError(
                    "Missing local journal with existing cloud history; restore journal or use a new run_id"
                )
            self.j = {
                "format": 1,
                "identity": self.cfg.identity,
                "source": self.b.source,
                "history": [],
                "deleted": [],
                "cleanup": [],
                "pending": None,
                "complete": False,
            }
            atomic_json(self.cfg.run_dir / "config.json", self.cfg.data)
            self.save()
        # 未记录资产/未经授权缺口均停止。缺失判定来自完整目录列表，不猜测权限错误。
        assets, _ = self.b.inventory()
        present = {a.get("name", a.get("id")) for a in assets}
        known = {r["path"] for r in self.j["history"]}
        if self.j["pending"]:
            known.add(self.j["pending"]["path"])
        if present - known:
            raise ValueError("Unrecorded cloud assets: " + str(present - known))
        permitted = set(self.j["deleted"]) | {r["path"] for r in self.j["cleanup"]}
        required = {r["path"] for r in self.j["history"]} - permitted
        if required - present:
            raise ValueError(
                "Unauthorized missing stage assets: " + str(required - present)
            )
        if present & set(self.j["deleted"]):
            raise ValueError("Deleted asset path reappeared")
        for key in ("components", "prepared", "latest"):
            if key in self.j:
                self.b.restore(self.j[key], self.j)

    def clean(self) -> None:
        while self.j["cleanup"]:
            rec = self.j["cleanup"][0]
            latest = self.j.get("latest")
            if (
                not latest
                or rec["kind"] != "state"
                or rec["step"] >= latest["step"]
                or rec["path"] != self.cfg.asset("state", rec["step"])
                or rec not in self.j["history"]
            ):
                raise ValueError("Invalid cleanup authorization")
            self.b.restore(latest, self.j)

            def remove():
                try:
                    self.b.remove(rec, self.j)
                except Exception:
                    if self.b.metadata(rec["path"]) is not None:
                        raise

            retry_read(remove, sleep=self.sleep)
            self.j["deleted"].append(rec["path"])
            self.j["cleanup"].pop(0)
            self.save()
            LOG.info("Deleted verified old state: %s", rec["path"])

    def wait_pending(self) -> None:
        rec = self.j["pending"]
        started = time.monotonic()
        unknown_queries = 0
        while True:
            status = self.b.task_status(rec)
            state = status.get("state")
            if status.get("id") and status["id"] != rec["task_id"]:
                rec["task_id"] = status["id"]
                self.save()
            LOG.info(
                "%s task=%s state=%s elapsed=%.0fs",
                rec["description"],
                rec["task_id"],
                state,
                time.monotonic() - started,
            )
            if state == "COMPLETED":
                rec["receipt"] = {
                    key: status[key]
                    for key in (
                        "creation_timestamp_ms",
                        "start_timestamp_ms",
                        "update_timestamp_ms",
                        "batch_eecu_usage_seconds",
                        "attempt",
                    )
                    if key in status
                }
                self.save()
                break
            if state == "UNKNOWN" and unknown_queries < 3:
                self.sleep((30, 60, 120)[unknown_queries])
                unknown_queries += 1
                continue
            if state not in ("READY", "RUNNING"):
                raise RuntimeError(
                    "Task stopped/ambiguous; no automatic resubmit: " + str(status)
                )
            self.sleep(self.cfg.data["poll_seconds"])
        # 任务完成后资产可见性有延迟，有限等待；不忽略合同失败。
        for attempt in range(4):
            if self.b.metadata(rec["path"]) is not None:
                break
            if attempt == 3:
                raise RuntimeError("Completed task asset not visible; resume later")
            self.sleep((30, 60, 120)[attempt])
        self.b.restore(rec, self.j)
        self.j["history"].append(rec)
        self.j["latest" if rec["kind"] == "state" else rec["kind"]] = rec
        self.j["pending"] = None
        if self.cfg.data["cleanup_previous_states"] and rec["kind"] == "state":
            self.j["cleanup"] = [
                r
                for r in self.j["history"]
                if r["kind"] == "state"
                and r["step"] < rec["step"]
                and r["path"] not in self.j["deleted"]
            ]
        self.save()
        self.clean()

    def submit(self, kind: str, step: int) -> None:
        path = self.cfg.asset(kind, step)
        if self.b.metadata(path) is not None:
            raise ValueError("Refusing to overwrite " + path)
        rec = {
            "kind": kind,
            "step": step,
            "path": path,
            "token": uuid.uuid4().hex,
            "task_id": self.b.reserve(),
            "description": self.cfg.run_id
            + "_"
            + kind
            + ("_%05d" % step if kind == "state" else ""),
        }
        rec["request_id"] = rec["task_id"]
        rec["description"] += "_" + rec["token"][:12]
        self.j["pending"] = rec
        self.save()
        try:
            receipt = self.b.submit(rec, self.j)
            if receipt:
                rec.update(receipt)
                self.save()
        except Exception:
            status = self.b.task_status(rec)
            if status.get("state") not in ("READY", "RUNNING", "COMPLETED"):
                raise
            LOG.warning(
                "Submission response lost; existing task found, not resubmitted"
            )
        self.wait_pending()

    def run(self, resume: bool = False, stop_after_step: int | None = None) -> None:
        with run_lock(self.cfg.run_dir):
            self.open(resume)
            if self.j["complete"]:
                self.downloader(self.b, self.j, self.save)  # 校验收据，不重复求解。
                return
            self.clean()
            if self.j["pending"]:
                self.wait_pending()
            for kind in ("components", "prepared"):
                if kind not in self.j:
                    self.submit(kind, 0)
            while True:
                if "latest" in self.j:
                    summary = self.b.status(self.j)
                    LOG.info("Saved solver status: %s", summary)
                    if not summary["running_pixels"]:
                        break
                    step = self.j["latest"]["step"] + 1
                else:
                    step = 1
                if stop_after_step is not None and step > stop_after_step:
                    LOG.info(
                        "Requested pause after saved step %s; resume with same configuration",
                        stop_after_step,
                    )
                    return
                if step > self.cfg.max_stages:
                    raise RuntimeError("Stage limit reached with active components")
                self.submit("state", step)
            self.j["products"] = self.downloader(self.b, self.j, self.save)
            self.j["complete"] = True
            self.save()
            LOG.info("Complete: both local products verified")
