"""持久化后才授权删除；进程锁只覆盖本次运行目录。"""

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from .config import canonical


def atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if canonical(json.loads(path.read_text())) != content:
            raise RuntimeError("Journal readback mismatch")
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "process.lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this run") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
