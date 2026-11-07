"""Episode dataset for the world model (recorded by record.py).

Episodes are split into train / test by a hash of their path (10% test), so the split is the same
on every run and for every model. `load_episodes` reads them into memory as numpy arrays.
"""
import glob
import hashlib
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "runs", "wm", "data")
TEST_FRACTION = 0.10


def _bucket(p, salt=""):
    return int(hashlib.md5((salt + os.path.relpath(p, DATA)).encode()).hexdigest(), 16) % 1000


def episode_paths(split, sources="all"):
    """split: "train" | "test" | "fit" (train minus val) | "val" (5% of train, for early stopping).
    sources: "all", "stage1" (the Stage 1 recording only) or "b" (the Stage 1b recording only).
    The test split is the same per episode whatever else was recorded, so the Stage 1 test set stays fixed."""
    paths = sorted(glob.glob(os.path.join(DATA, "*", "ep_*.npz")))
    if sources == "stage1":
        paths = [p for p in paths if not os.path.basename(os.path.dirname(p)).endswith("_b")]
    elif sources == "b":
        paths = [p for p in paths if os.path.basename(os.path.dirname(p)).endswith("_b")]
    test = [p for p in paths if _bucket(p) < TEST_FRACTION * 1000]
    if split == "test":
        return test
    train = [p for p in paths if _bucket(p) >= TEST_FRACTION * 1000]
    if split == "train":
        return train
    val = [p for p in train if _bucket(p, "val") < 50]
    return val if split == "val" else [p for p in train if _bucket(p, "val") >= 50]


def load_episodes(split, keys=("frames", "actions", "inventory", "unlocked", "daylight", "sem", "died"),
                  limit=None, extra_dir=None, sources="all"):
    """List of dicts of arrays. `extra_dir`: also load <extra_dir>/<source>/ep_k.npz (e.g. cached tokens)."""
    out = []
    for p in episode_paths(split, sources)[:limit]:
        d = np.load(p)
        ep = {k: d[k] for k in keys if k in d.files}
        ep["path"] = p
        if extra_dir:
            q = os.path.join(extra_dir, os.path.relpath(p, DATA))
            if os.path.exists(q):
                e = np.load(q)
                ep.update({k: e[k] for k in e.files})
        out.append(ep)
    return out
