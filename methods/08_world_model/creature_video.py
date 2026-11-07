"""Videos for creature_eval: the same night moment three times, REALITY | GREEDY DREAM | SAMPLED DREAM (T = 1),
with each one's health under it. Moments are taken from creature_eval's start set (same seed): night starts
in which the real player loses at least `--min_loss` health within `--H` steps.

  .venv/bin/python methods/08_world_model/creature_video.py v4:model_refit.pt --videos 3
"""
import argparse
import os
import sys

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402
from creature_eval import NIGHT, pick_starts  # noqa: E402
from data import DATA, ROOT, load_episodes  # noqa: E402
from evaluate import dream, load  # noqa: E402


def main(args):
    sys.path.insert(0, ROOT)
    from common.video import write_mp4
    from PIL import Image, ImageDraw, ImageFont
    dev = th.device("cuda")
    raw = load_episodes("test", keys=("frames", "actions", "inventory", "unlocked", "daylight", "sem", "died"),
                        sources="stage1")
    H = args.H
    starts = pick_starts(raw, args.n, 50, np.random.default_rng(0))
    chosen = [(i, s) for i, s in starts if raw[i]["daylight"][s] < NIGHT
              and raw[i]["inventory"][s][0] - raw[i]["inventory"][s + 1:s + 1 + H, 0].min() >= args.min_loss]
    chosen = chosen[:args.videos]
    wm, tok, tokens_dir = load(args.run, dev)
    eps = []
    for i, _ in chosen:
        e = raw[i]
        rel = os.path.relpath(e["path"], DATA)
        pos = np.load(os.path.join(ROOT, "runs", "wm", wm.pos_dir, rel))["pos"]
        eps.append({**e, "tokens": np.load(os.path.join(tokens_dir, rel))["tokens"], "pos": pos})
    dreams = {}
    for T in (0.0, 1.0):
        evaluate.SAMPLE_T = T
        th.manual_seed(0)
        pt, pi = dream(wm, eps, [s for _, s in chosen], H, dev)
        with th.no_grad():
            f = np.stack([tok.decode(pt[:, k].to(dev)).cpu().numpy() for k in range(H)], 1)
        dreams[T] = (f, pi[:, :, 0].cpu().numpy())
    out_dir = os.path.join(ROOT, "runs", "wm", "eval", "creatures")
    os.makedirs(out_dir, exist_ok=True)
    font = ImageFont.load_default(size=13)
    for v, (i, s) in enumerate(chosen):
        frames = []
        for k in range(H):
            real = raw[i]["frames"][s + 1 + k]
            hp = [int(raw[i]["inventory"][s + 1 + k][0]), int(dreams[0.0][1][v, k]), int(dreams[1.0][1][v, k])]
            gap = np.full((64, 2, 3), 255, np.uint8)
            row = np.kron(np.concatenate([real, gap, dreams[0.0][0][v, k], gap, dreams[1.0][0][v, k]], 1),
                          np.ones((4, 4, 1), np.uint8))
            img = Image.new("RGB", (row.shape[1], row.shape[0] + 48), (20, 20, 24))
            img.paste(Image.fromarray(row), (0, 26))
            dr = ImageDraw.Draw(img)
            for j, label in enumerate(("REALITY", "DREAM, greedy", "DREAM, sampled T = 1")):
                dr.text((j * 264 + 6, 5), label, fill=(235, 235, 235), font=font)
                col = (170, 200, 255) if j == 0 else (150, 230, 150) if hp[j] == hp[0] else (255, 150, 120)
                dr.text((j * 264 + 6, row.shape[0] + 30), f"health {hp[j]}", fill=col, font=font)
            dr.text((row.shape[1] - 120, 5), f"{k + 1} steps ahead", fill=(255, 220, 90), font=font)
            frames.append(np.asarray(img))
        path = os.path.join(out_dir, f"night_{v:02d}.mp4")
        write_mp4(path, frames, fps=6)
        print(path, "episode", os.path.relpath(raw[i]["path"], DATA), "start", s,
              "health real / greedy / sampled at end:", hp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--min_loss", type=int, default=3)
    ap.add_argument("--videos", type=int, default=3)
    main(ap.parse_args())
