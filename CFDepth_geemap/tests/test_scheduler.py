import copy
import json
import tempfile
from pathlib import Path
import unittest
from cfdepth.config import Config
from cfdepth.scheduler import Scheduler
from cfdepth.storage import run_lock, atomic_json


class FakeBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.source = "source"
        self.assets = {}
        self.tasks = {}
        self.next_id = 0
        self.submits = []
        self.removed = []
        self.lose_submit = False
        self.lose_delete = False
        self.fail_task = False
        self.fail_delete = False
        self.bad_restore = None

    def inventory(self):
        return [{"name": p} for p in self.assets], list(self.tasks.values())

    def metadata(self, path):
        return self.assets.get(path)

    def restore(self, rec, j):
        if (
            self.bad_restore == rec["path"]
            or self.assets[rec["path"]]["token"] != rec["token"]
        ):
            raise ValueError("Token mismatch")

    def reserve(self):
        self.next_id += 1
        return str(self.next_id)

    def submit(self, rec, j):
        if rec["kind"] == "state":
            previous = j.get("latest")
            expected = 1 if previous is None else previous["step"] + 1
            assert rec["step"] == expected
        self.submits.append(copy.deepcopy(rec))
        self.tasks[rec["task_id"]] = {
            "state": "FAILED" if self.fail_task else "COMPLETED",
            "description": rec["description"],
        }
        if not self.fail_task:
            self.assets[rec["path"]] = copy.deepcopy(rec)
        if self.lose_submit:
            self.lose_submit = False
            raise ConnectionError("response lost")

    def task_status(self, rec):
        return self.tasks.get(rec["task_id"], {"state": "UNKNOWN"})

    def status(self, j):
        return {"running_pixels": int(j["latest"]["step"] < 4)}

    def remove(self, rec, j):
        self.restore(rec, j)
        if self.fail_delete:
            raise PermissionError("permission denied")
        # 授权已写入磁盘，且最新状态已核验。
        disk = json.loads((self.cfg.run_dir / "journal.json").read_text())
        assert rec in disk["cleanup"]
        assert disk["latest"]["step"] > rec["step"]
        del self.assets[rec["path"]]
        self.removed.append(rec["path"])
        if self.lose_delete:
            self.lose_delete = False
            raise ConnectionError("response lost")


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        data = json.loads(Path("CFDepth_geemap/configs/small.json").read_text())
        data.update(
            run_root=self.tmp.name + "/runs", output_root=self.tmp.name + "/out"
        )
        self.cfg = Config(data)
        self.b = FakeBackend(self.cfg)
        self.downloads = []

    def downloader(self, b, j, save):
        self.assertEqual(j["latest"]["step"], 4)
        self.downloads.append(1)
        return {"depth": {}, "gradient": {}}

    def engine(self):
        return Scheduler(
            self.cfg, self.b, sleep=lambda _: None, downloader=self.downloader
        )

    def test_clean_resume(self):
        self.engine().run(stop_after_step=2)
        self.assertEqual(len(self.b.assets), 3)
        self.assertFalse(self.downloads)
        self.assertIn(self.cfg.asset("state", 2), self.b.assets)
        self.engine().run(resume=True)
        self.assertEqual(len(self.b.assets), 3)
        self.assertEqual(len(self.downloads), 1)
        self.assertEqual(
            [r["step"] for r in self.b.submits if r["kind"] == "state"], [1, 2, 3, 4]
        )
        self.assertEqual(
            json.loads((self.cfg.run_dir / "journal.json").read_text())["latest"][
                "step"
            ],
            4,
        )

    def test_keep_all_same_stages(self):
        self.cfg.data["cleanup_previous_states"] = False
        self.engine().run()
        self.assertEqual(len(self.b.assets), 6)
        self.assertFalse(self.b.removed)

    def test_responses_lost(self):
        self.b.lose_submit = True
        self.b.lose_delete = True
        self.engine().run()
        self.assertEqual(len(self.b.submits), 6)

    def test_failed_task_no_next(self):
        self.b.fail_task = True
        with self.assertRaises(RuntimeError):
            self.engine().run()
        self.assertEqual(len(self.b.submits), 1)
        self.assertFalse(self.b.removed)

    def test_delete_failure_preserves_latest_and_resume(self):
        self.b.fail_delete = True
        with self.assertRaises(PermissionError):
            self.engine().run()
        self.assertIn(self.cfg.asset("state", 1), self.b.assets)
        self.assertIn(self.cfg.asset("state", 2), self.b.assets)
        self.b.fail_delete = False
        self.engine().run(resume=True)
        self.assertEqual(len(self.b.assets), 3)

    def test_missing_latest_and_missing_journal(self):
        self.engine().run(stop_after_step=2)
        saved = self.b.assets.pop(self.cfg.asset("state", 2))
        with self.assertRaises(ValueError):
            self.engine().run(resume=True)
        self.b.assets[self.cfg.asset("state", 2)] = saved
        (self.cfg.run_dir / "journal.json").unlink()
        with self.assertRaises(ValueError):
            self.engine().run(resume=True)

    def test_new_asset_bad_token_keeps_old(self):
        self.b.bad_restore = self.cfg.asset("state", 2)
        with self.assertRaises(ValueError):
            self.engine().run()
        self.assertIn(self.cfg.asset("state", 1), self.b.assets)
        self.assertFalse(self.downloads)

    def test_lock_duplicate_start(self):
        with run_lock(self.cfg.run_dir):
            with self.assertRaises(RuntimeError):
                self.engine().run()

    def test_download_failure_resumes_without_iterations(self):
        def fail(*args):
            raise OSError("download interrupted")

        with self.assertRaises(OSError):
            Scheduler(self.cfg, self.b, lambda _: None, fail).run()
        count = len(self.b.submits)
        self.engine().run(resume=True)
        self.assertEqual(len(self.b.submits), count)

    def test_unrelated_asset_never_deleted(self):
        self.engine().run(stop_after_step=2)
        self.b.assets[self.cfg.prefix + "_state_99999"] = {"token": "other"}
        with self.assertRaises(ValueError):
            self.engine().run(resume=True)
        self.assertIn(self.cfg.prefix + "_state_99999", self.b.assets)


if __name__ == "__main__":
    unittest.main()
