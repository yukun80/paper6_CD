import hashlib
import json
import os
from pathlib import Path

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def source_hash():
    root = Path(__file__).parent
    return object_hash({p.name: file_hash(p) for p in sorted(root.glob("*.py"))} | {"defaults.json": file_hash(root / "defaults.json")})

def atomic_json(path, value, durable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        if durable:os.fsync(f.fileno())
    os.replace(temp, path)
