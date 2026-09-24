"""Bounded, transactional working state; never a published product."""
from pathlib import Path
import json
import shutil
import sqlite3
import os
import numpy as np


class WorkState:
    def __init__(self, root, create=False):
        self.path = Path(root) / '.work' / 'state.sqlite'
        if not create and not self.path.is_file():
            raise ValueError('No compatible working state; use a new run directory')
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA synchronous=FULL')
        if create:
            self.db.executescript('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);'
                                 'CREATE TABLE components (id INTEGER PRIMARY KEY, value TEXT NOT NULL);')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def get(self, key):
        row = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        if row is None:
            raise ValueError(f'Incomplete working state: {key}')
        return json.loads(row[0])

    def put(self, key, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                            (key, json.dumps(value, allow_nan=False)))

    def completed(self):
        return {cid: json.loads(value) for cid, value in self.db.execute('SELECT id,value FROM components')}

    def commit_components(self, records):
        with self.db:
            self.db.executemany('INSERT OR REPLACE INTO components VALUES (?,?)',
                                [(r['id'], json.dumps(r, allow_nan=False)) for r in records])


def close_maps(*arrays):
    seen = set()
    for array in arrays:
        if isinstance(array, np.memmap) and id(array._mmap) not in seen:
            seen.add(id(array._mmap))
            array._mmap.close()


def remove_work_path(root, path):
    """Remove only a verified descendant of this run's .work, without links."""
    root = Path(root).absolute()
    work = root / '.work'
    path = Path(path).absolute()
    if not path.exists():
        return
    if path != work and work not in path.parents:
        raise ValueError(f'Not a work path: {path}')
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink() or (hasattr(ancestor, 'is_junction') and ancestor.is_junction()):
            raise ValueError(f'Refusing linked work path: {ancestor}')
        if ancestor == root:
            break
    if path.resolve() != path or root.resolve() != root:
        raise ValueError(f'Unresolved work path: {path}')
    for base, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            item = Path(base) / name
            # Reparse points include Windows junctions on Python 3.11.
            if item.is_symlink() or getattr(item.lstat(), 'st_file_attributes', 0) & 1024:
                raise ValueError(f'Refusing work reparse point: {item}')
    shutil.rmtree(path)
