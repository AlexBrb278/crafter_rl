"""What is on each tile of a dream: terrain accuracy and ghost creatures (P14), for any world model.

Pixel PSNR mixes everything together. Here every dreamed frame is decoded to pixels, re-encoded with the tile
tokenizer, and each world tile is classified (water, grass, stone, ..., cow, zombie, ...) by a lookup table
code -> most common semantic class, learned from real training frames (semantic labels are used only to
build this grader, never by the world models). Then, against the real future frames:
  terrain_acc       share of material tiles whose class matches (creatures and the player excluded)
  creature_recall   share of real creature tiles where the dream also draws a creature
  creature_precision share of dreamed creature tiles where reality has one; 1 - precision = ghosts
  ghost_rate        dreamed creatures per frame on tiles with no real creature
The grader's own accuracy on real frames (re-encoded the same way) is reported as the ceiling.

  .venv/bin/python methods/08_world_model/semantic_eval.py v2_long:model_refit.pt v3:model_refit.pt --out stage1d_semantic.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cache_tokens import load_tokenizer  # noqa: E402
from data import DATA, ROOT, episode_paths, load_episodes  # noqa: E402
from evaluate import dream, load  # noqa: E402

CREATURE = (14, 15, 16, 17)  # cow, zombie, skeleton, arrow (plants do not move; the player is excluded)
PLAYER = 13
HORIZONS = (1, 5, 15, 50)


def code_classes(n_codes=1024, n_eps=800):
    """code -> most common semantic class of the tile, from real training frames."""
    counts = np.zeros((n_codes, 256), np.int64)
    for p in episode_paths("fit")[::max(1, len(episode_paths("fit")) // n_eps)]:
        tok = np.load(os.path.join(ROOT, "runs", "wm", "tokens_tile", os.path.relpath(p, DATA)))["tokens"][:, :7]
        sem = np.load(p)["sem"].transpose(0, 2, 1)  # (T+1, 7 y, 9 x)
        np.add.at(counts, (tok.ravel(), sem.ravel()), 1)
    return counts.argmax(1)


def classify(frames, tile_tok, table, dev):  # (N, 64, 64, 3) uint8 -> (N, 7, 9) classes
    out = []
    with th.no_grad():
        for i in range(0, len(frames), 512):
            out.append(tile_tok.encode(th.from_numpy(frames[i:i + 512]).to(dev)).cpu().numpy()[:, :7])
    return table[np.concatenate(out)]


def scores(pred, real):
    mat = (real < PLAYER) & (pred < PLAYER)  # both terrain
    rc, pc = np.isin(real, CREATURE), np.isin(pred, CREATURE)
    return {"terrain_acc": float((pred == real)[mat].mean()),
            "creature_recall": float((pc & rc).sum() / max(rc.sum(), 1)),
            "creature_precision": float((pc & rc).sum() / max(pc.sum(), 1)),
            "ghost_rate": float((pc & ~rc).sum() / len(pred))}


def main(args):
    dev = th.device("cuda")
    table = code_classes()
    tile_tok = load_tokenizer("tokenizer_tile", dev)
    raw = load_episodes("test", keys=("frames", "actions", "inventory", "unlocked", "daylight", "sem", "died"), sources="stage1")
    H = max(HORIZONS)
    results = {}
    for run in args.runs:
        wm, tok, tokens_dir = load(run, dev)
        wm.pos_dir = args.pos_dir or wm.pos_dir
        rng = np.random.default_rng(0)  # the same episodes and starts as evaluate.py
        eps = []
        for e in raw:
            rel = os.path.relpath(e["path"], DATA)
            qp = os.path.join(ROOT, "runs", "wm", wm.pos_dir, rel)
            pos = np.load(qp)["pos"] if os.path.exists(qp) else np.zeros((len(e["actions"]) + 1, 2), np.int16)
            eps.append({**e, "tokens": np.load(os.path.join(tokens_dir, rel))["tokens"], "pos": pos})
        long_eps = [e for e in eps if len(e["actions"]) > 64 + H]
        starts = [(e, int(rng.integers(32, len(e["actions"]) - H))) for e in
                  [long_eps[i] for i in rng.choice(len(long_eps), min(args.n, len(long_eps)), replace=False)]]
        pt, _ = dream(wm, [e for e, _ in starts], [s for _, s in starts], H, dev)
        R = {"run": run}
        with th.no_grad():
            for h in HORIZONS:
                df = tok.decode(pt[:, h - 1].to(dev)).cpu().numpy()
                rf = np.stack([e["frames"][s + h] for e, s in starts])
                rs = np.stack([e["sem"][s + h].T for e, s in starts]).astype(np.int64)
                last = np.stack([e["frames"][s] for e, s in starts])
                R[h] = {"dream": scores(classify(df, tile_tok, table, dev), rs),
                        "copy": scores(classify(last, tile_tok, table, dev), rs),
                        "grader_on_real": scores(classify(rf, tile_tok, table, dev), rs)}
                print(run, h, json.dumps(R[h]["dream"]), flush=True)
        results[run] = R
    out = os.path.join(ROOT, "results", "world_model", args.out)
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pos_dir", default=None)
    ap.add_argument("--out", default="semantic_eval.json")
    sys.path.insert(0, ROOT)
    main(ap.parse_args())
