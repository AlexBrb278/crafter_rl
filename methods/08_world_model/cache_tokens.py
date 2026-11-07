"""Encode every recorded frame with a trained tokenizer once: <out>/<source>/ep_k.npz -> tokens (T+1, 8, 8).

  .venv/bin/python methods/08_world_model/cache_tokens.py                                   # Stage 1
  .venv/bin/python methods/08_world_model/cache_tokens.py --tokenizer tokenizer_v2 --out tokens_v2
"""
import argparse
import os
import sys

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import DATA, ROOT, episode_paths  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402


def load_tokenizer(name, dev):
    """runs/wm/<name>/tokenizer.pt; an info.json with type "tile" means the Stage 1d tile tokenizer."""
    d = os.path.join(ROOT, "runs", "wm", name)
    info = os.path.join(d, "info.json")
    if os.path.exists(info) and __import__("json").load(open(info)).get("type") == "tile":
        from tile_tokenizer import TileTokenizer
        tok = TileTokenizer(n_codes=__import__("json").load(open(info))["codes"])
    else:
        tok = Tokenizer()
    tok = tok.to(dev).eval()
    tok.load_state_dict(th.load(os.path.join(d, "tokenizer.pt"), map_location=dev))
    return tok


def main(args):
    OUT = os.path.join(ROOT, "runs", "wm", args.out)
    dev = th.device("cuda")
    tok = load_tokenizer(args.tokenizer, dev)
    paths = episode_paths("train") + episode_paths("test")
    for k, p in enumerate(paths):
        q = os.path.join(OUT, os.path.relpath(p, DATA))
        os.makedirs(os.path.dirname(q), exist_ok=True)
        frames = th.from_numpy(np.load(p)["frames"]).to(dev)
        with th.no_grad():
            codes = th.cat([tok.encode(frames[i:i + 1024]) for i in range(0, len(frames), 1024)])
        np.savez_compressed(q, tokens=codes.cpu().numpy().astype(np.uint16))
        if k % 500 == 0:
            print(f"{k}/{len(paths)}", flush=True)
    print("cached", len(paths), "episodes to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer", help="folder under runs/wm/")
    ap.add_argument("--out", default="tokens", help="folder under runs/wm/")
    main(ap.parse_args())
