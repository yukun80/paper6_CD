import unittest
from unittest.mock import patch
from cfdepth.backend import Backend, retry_read
from cfdepth.config import Config
import ee


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.b = Backend(Config.read("CFDepth_geemap/configs/small.json"))

    def test_ambiguous_missing_requires_readable_parent(self):
        error = ee.EEException("Asset does not exist or doesn't allow this operation")
        with patch.object(ee.data, "getAsset", side_effect=error), patch.object(
            ee.data, "listAssets", return_value={"assets": []}
        ):
            self.assertIsNone(self.b.metadata(self.b.cfg.asset("components")))
        with patch.object(ee.data, "getAsset", side_effect=error), patch.object(
            ee.data, "listAssets", side_effect=PermissionError("denied")
        ):
            with self.assertRaises(PermissionError):
                self.b.metadata(self.b.cfg.asset("components"))
        with patch.object(ee.data, "getAsset", side_effect=error), patch.object(
            ee.data,
            "listAssets",
            return_value={"assets": [{"name": self.b.cfg.asset("components")}]},
        ):
            with self.assertRaises(ee.EEException):
                self.b.metadata(self.b.cfg.asset("components"))

    def test_request_id_differs_operation_id(self):
        record = {"task_id": "request", "description": "unique"}
        with patch.object(
            ee.data, "getTaskStatus", return_value=[{"state": "UNKNOWN"}]
        ), patch.object(
            ee.data,
            "getTaskList",
            return_value=[
                {"state": "COMPLETED", "id": "operation", "description": "unique"}
            ],
        ):
            self.assertEqual(self.b.task_status(record)["id"], "operation")
        with patch.object(
            ee.data, "getTaskStatus", return_value=[{"state": "UNKNOWN"}]
        ), patch.object(
            ee.data,
            "getTaskList",
            return_value=[{"description": "unique"}, {"description": "unique"}],
        ):
            with self.assertRaises(RuntimeError):
                self.b.task_status(record)

    def test_finite_retry(self):
        waits = []
        calls = []

        def read():
            calls.append(1)
            raise RuntimeError("503 temporarily unavailable")

        with self.assertRaises(RuntimeError):
            retry_read(read, waits.append)
        self.assertEqual(waits, [30, 60, 120])
        self.assertEqual(len(calls), 4)
        waits.clear()
        with self.assertRaises(PermissionError):
            retry_read(
                lambda: (_ for _ in ()).throw(PermissionError("denied")), waits.append
            )
        self.assertFalse(waits)


if __name__ == "__main__":
    unittest.main()


class StatusTests(unittest.TestCase):
    def test_double_histogram_keys(self):
        from cfdepth.backend import parse_status_histogram

        self.assertEqual(
            parse_status_histogram({"0.0": 3, "1.0": 415, "null": 878}), {0: 3, 1: 415}
        )
        for invalid in (
            {"null": 4},
            {"1": 1, "1.0": 1},
            {"NaN": 1},
            {"1": 0.5},
            {"6": 1},
        ):
            with self.assertRaises(ValueError):
                parse_status_histogram(invalid)
