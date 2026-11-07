"""Stage 1d map memory, precomputed for training: what the agent's top-down map shows where it will be next.

For every step t of an episode, a map is built from all real frames 0..t: each world tile code (tile rows 0-6
of the 9x9 grid) is written at its world position (agent position + offset in view; the agent stands at row 3,
column 4). Then the 7x9 window around the *next* position p_{t+1} is read out:
  mtok (T, 7, 9) uint16   last code seen at each tile, or N_CODES (= 1024) if never seen
  mage (T, 7, 9) uint8    steps since it was seen (0 = seen in frame t), 255 if never
Positions come from derive_positions.py (default, runs/wm/pos) or pixel_odometry.py (--pos pos_pix).
Output: runs/wm/<out>/<source>/ep_k.npz

  .venv/bin/python methods/08_world_model/build_maps.py [--pos pos_pix --out maps_tile_pix]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import DATA, ROOT, episode_paths  # noqa: E402

TOK = os.path.join(ROOT, "runs", "wm", "tokens_tile")
R = 72  # map half-size in tiles (Crafter's world is 64 x 64)


class TileMap:
    """Top-down memory of tile codes, indexed by world position relative to the episode start."""

    def __init__(self, n_codes):
        self.n = n_codes
        self.tok = np.full((2 * R + 1, 2 * R + 1), n_codes, np.int32)  # [y, x]
        self.last = np.full((2 * R + 1, 2 * R + 1), -1, np.int64)

    def write(self, grid, pos, t):  # grid (9, 9) codes of one frame; pos (x, y)
        x0, y0 = pos[0] - 4 + R, pos[1] - 3 + R
        self.tok[y0:y0 + 7, x0:x0 + 9] = grid[:7]
        self.last[y0:y0 + 7, x0:x0 + 9] = t

    def read(self, pos, t):  # window that frame at position `pos` will show; ages relative to step t
        x0, y0 = pos[0] - 4 + R, pos[1] - 3 + R
        tok = self.tok[y0:y0 + 7, x0:x0 + 9].copy()
        last = self.last[y0:y0 + 7, x0:x0 + 9]
        age = np.where(last < 0, 255, np.minimum(t - last, 254)).astype(np.uint8)
        return tok.astype(np.uint16), age


def episode_maps(tokens, pos, n_codes):
    T = len(tokens) - 1
    m = TileMap(n_codes)
    mtok = np.zeros((T, 7, 9), np.uint16)
    mage = np.zeros((T, 7, 9), np.uint8)
    for t in range(T):
        m.write(tokens[t], pos[t], t)
        mtok[t], mage[t] = m.read(pos[t + 1], t)
    return mtok, mage


def main(args):
    POS, OUT = os.path.join(ROOT, "runs", "wm", args.pos), os.path.join(ROOT, "runs", "wm", args.out)
    n_codes = json.load(open(os.path.join(ROOT, "runs", "wm", "tokenizer_tile", "info.json")))["codes"]
    paths = episode_paths("train") + episode_paths("test")
    for k, p in enumerate(paths):
        rel = os.path.relpath(p, DATA)
        tokens = np.load(os.path.join(TOK, rel))["tokens"].astype(np.int32)
        pos = np.load(os.path.join(POS, rel))["pos"].astype(np.int64)
        mtok, mage = episode_maps(tokens, pos, n_codes)
        q = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(q), exist_ok=True)
        np.savez_compressed(q, mtok=mtok, mage=mage)
        if k % 1000 == 0:
            print(f"{k}/{len(paths)}", flush=True)
    print("wrote maps for", len(paths), "episodes to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="pos")
    ap.add_argument("--out", default="maps_tile")
    main(ap.parse_args())
