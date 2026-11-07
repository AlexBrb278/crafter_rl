"""Agent path from pixels only (no game internals), as an honest replacement for derive_positions.py.

A move action (1-4) either moves the player one tile or is blocked. If it moved, the world part of frame t+1
is frame t shifted by one 7x7-px tile. Both hypotheses are scored by a trimmed mean of per-tile pixel error
(the 15% worst tiles are dropped: creatures, the player sprite). On uniform ground (open grass, forest,
water) both scores tie. Then a self-learned table decides: for each tile code, how often a move onto a tile
with that code succeeded on the clear (non-tied) moves of the training episodes. A sleeping player (grey view)
never moves, except on the step it wakes up, which is compared in luminance.

Output runs/wm/pos_pix/<source>/ep_k.npz with the same fields as derive_positions.py (pos, move).

  .venv/bin/python methods/08_world_model/pixel_odometry.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import DATA, ROOT, episode_paths  # noqa: E402

TOK = os.path.join(ROOT, "runs", "wm", "tokens_tile")
OUT = os.path.join(ROOT, "runs", "wm", "pos_pix")
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
TIE = 0.5  # score gap (mean abs pixel error, 0-255 scale) below which the pictures cannot tell


def tiles(frame):  # (64, 64, C) -> (9 x, 7 y, 49 C) world tiles
    C = frame.shape[-1]
    w = frame[:49, :63].astype(np.float32).reshape(7, 7, 9, 7, C).transpose(2, 0, 1, 3, 4)
    return w.reshape(9, 7, -1)


def luminance(frame):  # (64, 64, 1); a sleeping view is grey L tinted to 0.5 L (+ blue), so red * 2 undoes it
    f = frame.astype(np.float32)
    if asleep(frame):
        return 2 * f[..., :1]
    return (f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114)[..., None]


def shift_err(a, b, d):
    dx, dy = d  # if the player moved by d, b[i, j] == a[i + dx, j + dy] on the overlap
    A = a[max(0, dx):9 + min(0, dx), max(0, dy):7 + min(0, dy)]
    B = b[max(0, -dx):9 + min(0, -dx), max(0, -dy):7 + min(0, -dy)]
    e = np.sort(np.abs(A - B).mean(-1).ravel())
    return e[:int(len(e) * 0.85)].mean()


def asleep(frame):
    """A sleeping player sees the world in grey (engine.py _sleep: colour removed, then a dark blue tint),
    and the game ignores its actions until energy is full."""
    w = frame[:49, :63].astype(np.float32)
    return np.abs(w[..., 0] - w[..., 1]).mean() < 1.0


def step_scores(frames, actions):
    """Per step: (kind, score if moved, score if stayed). kind: "awake" or "wake" (grey frame, then colour:
    the player wakes up and its action runs; compared in luminance) for move actions, else None."""
    T = [tiles(f) for f in frames]
    sl = [asleep(f) for f in frames]
    out = []
    for t, a in enumerate(actions):
        a = int(a)
        if a not in MOVES or (sl[t] and sl[t + 1]):
            out.append((None, 0.0, 0.0))
        elif sl[t]:
            A, B = tiles(luminance(frames[t])), tiles(luminance(frames[t + 1]))
            out.append(("wake", shift_err(A, B, MOVES[a]), shift_err(A, B, (0, 0))))
        else:
            out.append(("awake", shift_err(T[t], T[t + 1], MOVES[a]), shift_err(T[t], T[t + 1], (0, 0))))
    return out


def clear(e_m, e_s):
    return abs(e_m - e_s) >= TIE


def target_code(tokens_t, a):  # code of the tile the player tries to step onto; tokens (9 rows y, 9 cols x)
    dx, dy = MOVES[a]
    return int(tokens_t[3 + dy, 4 + dx])


def path(frames, actions, tokens, walk):
    """walk: (n_codes,) probability that a move onto a tile with this code succeeds (or None: tie -> moved)."""
    move = np.zeros(len(actions), np.int8)
    for t, (kind, e_m, e_s) in enumerate(step_scores(frames, actions)):
        if kind is None:
            continue
        a = int(actions[t])
        if clear(e_m, e_s):
            move[t] = a if e_m < e_s else 0
        elif kind == "wake" or walk is None or walk[target_code(tokens[t], a)] >= 0.5:
            move[t] = a
    d = np.array([(0, 0)] + [MOVES[int(m)] if m else (0, 0) for m in move], np.int16)
    return np.cumsum(d, 0).astype(np.int16), move


def learn_walk(paths, n_codes):
    ok, n = np.zeros(n_codes), np.zeros(n_codes)
    for p in paths:
        d = np.load(p)
        tok = np.load(os.path.join(TOK, os.path.relpath(p, DATA)))["tokens"]
        for t, (kind, e_m, e_s) in enumerate(step_scores(d["frames"], d["actions"])):
            if kind == "awake" and clear(e_m, e_s):
                c = target_code(tok[t], int(d["actions"][t]))
                n[c] += 1
                ok[c] += e_m < e_s
    walk = np.where(n > 0, (ok + 1) / (n + 2), 1.0)  # unseen codes: assume walkable (most ground is)
    return walk, int((n > 0).sum())


def main():
    n_codes = json.load(open(os.path.join(ROOT, "runs", "wm", "tokenizer_tile", "info.json")))["codes"]
    fit = episode_paths("fit")
    walk, seen = learn_walk(fit[::8], n_codes)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "walk.npy"), walk)
    print(f"walkability table from {len(fit[::8])} episodes, {seen} codes seen", flush=True)
    paths = episode_paths("train") + episode_paths("test")
    agree = n = 0
    for k, p in enumerate(paths):
        d = np.load(p)
        tok = np.load(os.path.join(TOK, os.path.relpath(p, DATA)))["tokens"]
        pos, move = path(d["frames"], d["actions"], tok, walk)
        q = os.path.join(OUT, os.path.relpath(p, DATA))
        os.makedirs(os.path.dirname(q), exist_ok=True)
        np.savez_compressed(q, pos=pos, move=move)
        ref = np.load(os.path.join(ROOT, "runs", "wm", "pos", os.path.relpath(p, DATA)))["move"]
        agree += (ref == move).sum()
        n += len(move)
        if k % 1000 == 0:
            print(f"{k}/{len(paths)}  agreement with the tile-map path so far {agree / n:.4f}", flush=True)
    print(f"wrote pixel positions for {len(paths)} episodes to {OUT}; per-step agreement with tile maps {agree / n:.4f}")


if __name__ == "__main__":
    main()
