"""Is the dream an easier world? Creatures per frame and damage in dreams vs reality, by day and night.

A Stage 2 agent that practises in dreams learns from what the dream shows. If dreamed zombies fade away, the
dream is safer than Crafter and the agent learns to ignore the night. This measures that directly:
  - creatures per frame (zombie, cow, skeleton) and zombies within reach of the player (Manhattan distance 1,
    where Crafter's zombies hit), at every dream step 1..H;
  - "walk-ins": share of dreams with no zombie in view at the start in which one appears later;
  - health: how much the player has lost by step h, from the dream's inventory output vs the real inventory;
  - terrain accuracy (semantic_eval.scores), to see what sampling costs.
Each dreamed frame is classified by semantic_eval's grader (tile code -> most common class). Reality is
counted both from the game's true labels and through the same grader on the real frames (the fair
comparison: the grader misses some real creatures). Day / night is the real daylight at that step (< 0.3 =
night). Starts are stratified: half at night, half by day, the same for every model and decoding.
Decoding: greedy (temperature 0, as in all evaluations so far) and sampled (`--temps 0 1`).

  .venv/bin/python methods/08_world_model/creature_eval.py v4:model_refit.pt v3_nomap:model_refit.pt --temps 0 1
"""
import argparse
import json
import os
import sys

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402
from cache_tokens import load_tokenizer  # noqa: E402
from data import DATA, ROOT, load_episodes  # noqa: E402
from evaluate import dream, load  # noqa: E402
from semantic_eval import classify, code_classes, scores  # noqa: E402

COW, ZOMBIE, SKELETON = 14, 15, 16
NIGHT = 0.3
_yy, _xx = np.mgrid[0:7, 0:9]
REACH = (np.abs(_yy - 3) + np.abs(_xx - 4)) == 1  # the player is drawn at tile (x 4, y 3)
KINDS = ("zombie", "zombie_reach", "cow", "skeleton")


def counts(cls):  # (N, 7, 9) classes -> {kind: (N,) count per frame}
    z = cls == ZOMBIE
    return {"zombie": z.sum((1, 2)), "zombie_reach": (z & REACH).sum((1, 2)),
            "cow": (cls == COW).sum((1, 2)), "skeleton": (cls == SKELETON).sum((1, 2))}


def pick_starts(raw, n, H, rng):
    """n (episode, start) pairs, half with the start at night; the same for every model."""
    long_eps = [i for i, e in enumerate(raw) if len(e["actions"]) > 64 + H]
    day, night = [], []
    while len(day) < n // 2 or len(night) < n - n // 2:
        i = int(rng.choice(long_eps))
        s = int(rng.integers(32, len(raw[i]["actions"]) - H))
        (night if raw[i]["daylight"][s] < NIGHT else day).append((i, s))
    return day[:n // 2] + night[:n - n // 2]


def split(x, is_night):  # (N, H) -> {"day": [H], "night": [H]} means
    x = x.astype(np.float64)
    return {k: [float(x[m[:, h], h].mean()) if m[:, h].any() else None for h in range(x.shape[1])]
            for k, m in (("day", ~is_night), ("night", is_night))}


def walk_in(c, c0):  # c (N, H) zombie counts, c0 (N,) at the start -> share of zombie-free starts that get one
    free = c0 == 0
    return float((c[free] > 0).any(1).mean()) if free.any() else None


def main(args):
    dev = th.device("cuda")
    table = code_classes()
    tile_tok = load_tokenizer("tokenizer_tile", dev)
    raw = load_episodes("test", keys=("frames", "actions", "inventory", "unlocked", "daylight", "sem", "died"),
                        sources="stage1")
    H = args.H
    starts = pick_starts(raw, args.n, H, np.random.default_rng(0))
    N = len(starts)
    is_night = np.stack([raw[i]["daylight"][s + 1:s + 1 + H] < NIGHT for i, s in starts])  # (N, H)
    health0 = np.array([int(raw[i]["inventory"][s][0]) for i, s in starts])
    real_health = np.stack([raw[i]["inventory"][s + 1:s + 1 + H, 0].astype(np.int64) for i, s in starts])
    sem0 = np.stack([raw[i]["sem"][s].T for i, s in starts]).astype(np.int64)

    real_true = {k: np.zeros((N, H), np.int64) for k in KINDS}
    real_graded = {k: np.zeros((N, H), np.int64) for k in KINDS}
    real_cls = []
    for h in range(1, H + 1):
        rs = np.stack([raw[i]["sem"][s + h].T for i, s in starts]).astype(np.int64)
        rc = classify(np.stack([raw[i]["frames"][s + h] for i, s in starts]), tile_tok, table, dev)
        real_cls.append(rs)
        for k, v in counts(rs).items():
            real_true[k][:, h - 1] = v
        for k, v in counts(rc).items():
            real_graded[k][:, h - 1] = v
    out = {"n": N, "H": H, "night_share": float(is_night.mean()),
           "real_true": {k: split(v, is_night) for k, v in real_true.items()},
           "real_graded": {k: split(v, is_night) for k, v in real_graded.items()},
           "walk_in": {"real_true": walk_in(real_true["zombie"], counts(sem0)["zombie"])},
           "health_lost": {"real": split(health0[:, None] - real_health, is_night)},
           "runs": {}}

    for run in args.runs:
        wm, tok, tokens_dir = load(run, dev)
        wm.pos_dir = args.pos_dir or wm.pos_dir
        eps = []
        for i, _ in starts:
            e = raw[i]
            rel = os.path.relpath(e["path"], DATA)
            qp = os.path.join(ROOT, "runs", "wm", wm.pos_dir, rel)
            pos = np.load(qp)["pos"] if os.path.exists(qp) else np.zeros((len(e["actions"]) + 1, 2), np.int16)
            eps.append({**e, "tokens": np.load(os.path.join(tokens_dir, rel))["tokens"], "pos": pos})
        for T in args.temps:
            evaluate.SAMPLE_T = T
            th.manual_seed(0)
            pt, pi = dream(wm, eps, [s for _, s in starts], H, dev)
            dc = {k: np.zeros((N, H), np.int64) for k in KINDS}
            terrain = {}
            with th.no_grad():
                for h in range(1, H + 1):
                    df = tok.decode(pt[:, h - 1].to(dev)).cpu().numpy()
                    cls = classify(df, tile_tok, table, dev)
                    for k, v in counts(cls).items():
                        dc[k][:, h - 1] = v
                    if h in (1, 5, 15, 50):
                        terrain[h] = scores(cls, real_cls[h - 1])["terrain_acc"]
            dream_health = pi[:, :, 0].cpu().numpy().astype(np.int64)
            key = f"{run}|T={T:g}"
            out["runs"][key] = {"counts": {k: split(v, is_night) for k, v in dc.items()},
                                "walk_in": walk_in(dc["zombie"], counts(sem0)["zombie"]),
                                "health_lost": split(health0[:, None] - dream_health, is_night),
                                "terrain_acc": terrain}
            r = out["runs"][key]
            hs = [h for h in (1, 15, 50) if h <= H]
            print(key, f"zombies/frame night h{hs}:",
                  [round(r["counts"]["zombie"]["night"][h - 1], 3) for h in hs],
                  "real", [round(out["real_graded"]["zombie"]["night"][h - 1], 3) for h in hs],
                  "| walk-in", round(r["walk_in"], 3), "real", round(out["walk_in"]["real_true"], 3),
                  "| health lost @50 night", round(r["health_lost"]["night"][-1], 2),
                  "real", round(out["health_lost"]["real"]["night"][-1], 2),
                  "| terrain", {h: round(v, 3) for h, v in terrain.items()}, flush=True)
    path = os.path.join(ROOT, "results", "world_model", args.out)
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--pos_dir", default=None)
    ap.add_argument("--out", default="creature_eval.json")
    sys.path.insert(0, ROOT)
    main(ap.parse_args())
