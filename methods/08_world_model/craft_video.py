"""Video for craft_check: REALITY | DREAM (T = 1) around a held-out iron-pickaxe craft (runs/wm/probe), the
dream started `--k` steps before it and driven by the real actions, with both inventories under the frames.

  .venv/bin/python methods/08_world_model/craft_video.py v4:model_refit.pt --videos 2
  .venv/bin/python methods/08_world_model/craft_video.py v4:model_refit.pt --craft --item iron_sword   # craft layer
"""
import argparse
import os
import sys

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from craft_check import events, probe_episodes, track_of  # noqa: E402
from data import ROOT  # noqa: E402
from dream_env import CONTEXT, INVENTORY, Dreamer  # noqa: E402
from evaluate import load  # noqa: E402

SHOW = [("wood", "wood"), ("stone", "stone"), ("coal", "coal"), ("iron", "iron")]


def main(args):
    sys.path.insert(0, ROOT)
    from common.video import write_mp4
    from PIL import Image, ImageDraw, ImageFont
    dev = th.device("cuda")
    th.manual_seed(0)
    wm, tok, _ = load(args.run, dev)
    craft = None
    if args.craft:
        from craft_layer import load_layer
        craft = load_layer(args.run, dev)
    item, show = args.item, SHOW + [(args.item, args.item.replace("_", " "))]
    eps = probe_episodes(tok, dev)
    evs = events(eps, (item,), CONTEXT - 1 + args.k, np.random.default_rng(0), 0)[:args.videos]
    H = args.k + 1 + args.after
    font = ImageFont.load_default(size=13)
    out_dir = os.path.join(ROOT, "runs", "wm", "eval", "crafts")
    os.makedirs(out_dir, exist_ok=True)
    for v, (_, j, t) in enumerate(evs):
        e, tr = eps[j], track_of(eps[j])
        frames_real = np.load(e["path"])["frames"]
        dv = Dreamer(wm, tok, 1, dev, temperature=1.0, craft=craft)
        s0 = t - args.k
        dv.load([0], [(tr, s0)])
        out = []
        for s in range(H):
            f, inv, _, _ = dv.step(th.tensor([tr.act[s0 + s]], device=dev))
            real, dream = frames_real[s0 + s + 1], f[0].cpu().numpy()
            ri, di = e["inventory"][s0 + s + 1], inv[0].cpu().numpy()
            gap = np.full((64, 2, 3), 255, np.uint8)
            row = np.kron(np.concatenate([real, gap, dream], 1), np.ones((4, 4, 1), np.uint8))
            img = Image.new("RGB", (row.shape[1], row.shape[0] + 92), (20, 20, 24))
            img.paste(Image.fromarray(row), (0, 46))
            dr = ImageDraw.Draw(img)
            dr.text((6, 5), "REALITY", fill=(235, 235, 235), font=font)
            dr.text((264, 5), "DREAM, sampled T = 1" + (", craft layer" if craft is not None else ""), fill=(235, 235, 235), font=font)
            when = f"{s - args.k:+d} steps from the craft" if s != args.k else "CRAFT STEP"
            dr.text((6, 25), when, fill=(255, 220, 90), font=font)
            for col, inv_, x in ((0, ri, 6), (1, di, 264)):
                parts = [f"{n} {int(inv_[INVENTORY.index(k)])}" for k, n in show]
                good = int(inv_[INVENTORY.index(item)]) == int(ri[INVENTORY.index(item)])
                fill = (170, 200, 255) if col == 0 else (150, 230, 150) if good else (255, 150, 120)
                dr.text((x, row.shape[0] + 52), "  ".join(parts[:3]), fill=fill, font=font)
                dr.text((x, row.shape[0] + 70), "  ".join(parts[3:]), fill=fill, font=font)
            out.append(np.asarray(img))
        path = os.path.join(out_dir, f"{item}{'_layer' if craft is not None else ''}_{v:02d}.mp4")
        write_mp4(path, out, fps=3)
        print(path, os.path.relpath(e["path"], ROOT), "craft at", t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--after", type=int, default=6)
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--item", default="iron_pickaxe")
    ap.add_argument("--craft", action="store_true")
    main(ap.parse_args())
