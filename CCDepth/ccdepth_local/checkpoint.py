"""Versioned checkpoint generations; a partial generation is never selected."""
from pathlib import Path
import json
import uuid
import numpy as np
from .provenance import atomic_json,file_hash

def save_checkpoint(root,payload,identity):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex
    path=root/(token+'.npz')
    with path.open('wb') as f:
        np.savez(f,S=payload['S'],baseS=payload['baseS']);f.flush()
    meta={k:v for k,v in payload.items() if k not in ('S','baseS')}
    meta.update(identity=identity,array_file=path.name,array_sha256=file_hash(path))
    atomic_json(root/(token+'.json'),meta)
    atomic_json(root/'latest.json',dict(metadata=token+'.json',sha256=file_hash(root/(token+'.json'))))
    # Keep latest and preceding generation for crash recovery without unbounded growth.
    generations=sorted(root.glob('*.npz'),key=lambda p:p.stat().st_mtime,reverse=True)
    for old in generations[2:]:
        old.unlink();old.with_suffix('.json').unlink(missing_ok=True)

def load_checkpoint(root,identity):
    root=Path(root)
    if not (root/'latest.json').exists():return None
    pointer=json.loads((root/'latest.json').read_text());path=root/pointer['metadata']
    if file_hash(path)!=pointer['sha256']:raise ValueError('Checkpoint metadata checksum mismatch')
    meta=json.loads(path.read_text())
    if meta['identity']!=identity:raise ValueError('Checkpoint identity mismatch')
    arrays=root/meta['array_file']
    if file_hash(arrays)!=meta['array_sha256']:raise ValueError('Checkpoint array checksum mismatch')
    with np.load(arrays,allow_pickle=False) as d:
        return {k:v for k,v in meta.items() if k not in ('identity','array_file','array_sha256')}|dict(S=d['S'].copy(),baseS=d['baseS'].copy())
