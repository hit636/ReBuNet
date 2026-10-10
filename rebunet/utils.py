from __future__ import annotations
import hashlib
import json
import os
import platform
import random
from pathlib import Path
import numpy as np
import torch
import yaml


def read_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if 'extends' in cfg:
        base = read_config(path.parent / cfg.pop('extends'))
        cfg = merge(base, cfg)
    return cfg


def merge(base, changes):
    import copy
    result = copy.deepcopy(base)
    for key, value in changes.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        if isinstance(x, Path): return str(x)
        return x
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def seed_all(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
