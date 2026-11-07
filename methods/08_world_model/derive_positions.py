"""Stage 1c: recover the agent's path from the recorded 9x7 tile maps (no re-recording needed).

A move action (1-4: left, right, up, down) either moves the player one tile or is blocked. If it moved, the
next tile map is the previous one shifted by one tile. We compare both hypotheses on material tiles only
(creatures move on their own). 1.3% of move actions look identical either way (e.g. open grass); those are
counted as moved. Output runs/wm/pos/<source>/ep_k.npz:
  pos  (T+1, 2) int16   position relative to the episode start
  move (T,)    int8     0 stay, 1 left, 2 right, 3 up, 4 down  (the step taken from t to t+1)

  .venv/bin/python methods/08_world_model/derive_positions.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import DATA, ROOT, episode_paths  # noqa: E402

OUT = os.path.join(ROOT, "runs", "wm", "pos")
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def mismatch(a, b, d):
    dx, dy = d  # if the player moved by d, b[i, j] == a[i + dx, j + dy] on the overlap
    A = a[max(0, dx):9 + min(0, dx), max(0, dy):7 + min(0, dy)]
    B = b[max(0, -dx):9 + min(0, -dx), max(0, -dy):7 + min(0, -dy)]
    m = (A < 13) & (B < 13)  # materials only
    return np.mean(A[m] != B[m]) if m.any() else 1.0


def path(sem, actions):
    T = len(actions)
    move = np.zeros(T, np.int8)
    for t in range(T):
        a = int(actions[t])
        if a in MOVES and mismatch(sem[t], sem[t + 1], MOVES[a]) <= mismatch(sem[t], sem[t + 1], (0, 0)):
            move[t] = a
    d = np.array([(0, 0)] + [MOVES[int(m)] if m else (0, 0) for m in move], np.int16)
    return np.cumsum(d, 0).astype(np.int16), move


def main():
    paths = episode_paths("train") + episode_paths("test")
    amb = 0
    for k, p in enumerate(paths):
        d = np.load(p)
        pos, move = path(d["sem"], d["actions"])
        q = os.path.join(OUT, os.path.relpath(p, DATA))
        os.makedirs(os.path.dirname(q), exist_ok=True)
        np.savez_compressed(q, pos=pos, move=move)
        if k % 1000 == 0:
            print(f"{k}/{len(paths)}", flush=True)
    print("wrote positions for", len(paths), "episodes to", OUT)


if __name__ == "__main__":
    main()
