"""Stage 1d tile tokenizer: one code per Crafter tile.

Crafter draws the 64x64 observation as 9 x 9 tiles of 7x7 px (7 rows of world view, 2 rows of inventory) in
the top-left 63x63 pixels; the last row and column are always black. Each tile is encoded on its own by a
small MLP into one of 1,024 codes (EMA vector quantiser from tokenizer.py) and decoded back. Because a code is
exactly one tile, a one-tile move of the player is an exact one-code shift of the grid, which is what the map
memory needs. Loss: L1 with the "hard pixel" weighting of tokenizer v2.

  .venv/bin/python methods/08_world_model/tile_tokenizer.py --steps 20000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import ROOT, load_episodes  # noqa: E402
from tokenizer import CREATURES, RESOURCES, VQ, tile_mask  # noqa: E402

GRID = 9  # 9 x 9 tiles; rows 0-6 world, rows 7-8 inventory
T = 7     # tile size in pixels


def to_tiles(frames):  # (B, 64, 64, 3) uint8 -> (B, 9, 9, 7, 7, 3); [row = y, col = x]
    f = frames[:, :63, :63]
    return f.reshape(len(f), GRID, T, GRID, T, 3).permute(0, 1, 3, 2, 4, 5)


def from_tiles(tiles):  # (B, 9, 9, 7, 7, 3) -> (B, 64, 64, 3)
    B = len(tiles)
    f = tiles.permute(0, 1, 3, 2, 4, 5).reshape(B, 63, 63, 3)
    out = th.zeros(B, 64, 64, 3, dtype=f.dtype, device=f.device)
    out[:, :63, :63] = f
    return out


class TileTokenizer(nn.Module):
    grid = (GRID, GRID)

    def __init__(self, dim=32, n_codes=1024, hid=512):
        super().__init__()
        n = T * T * 3
        self.n_codes = n_codes
        self.enc = nn.Sequential(nn.Linear(n, hid), nn.SiLU(), nn.Linear(hid, hid // 2), nn.SiLU(), nn.Linear(hid // 2, dim))
        self.vq = VQ(n_codes, dim)
        self.dec = nn.Sequential(nn.Linear(dim, hid // 2), nn.SiLU(), nn.Linear(hid // 2, hid), nn.SiLU(), nn.Linear(hid, n))

    @staticmethod
    def prep(tiles_u8):  # (..., 7, 7, 3) -> (N, 147) in [-1, 1]
        return tiles_u8.reshape(-1, T * T * 3).float() / 127.5 - 1.0

    def encode(self, frames_u8):  # -> (B, 9, 9) codes
        z = self.enc(self.prep(to_tiles(frames_u8))).float()
        return self.vq.codes(z).reshape(len(frames_u8), GRID, GRID)

    def decode(self, codes):  # (B, 9, 9) -> uint8 frames (B, 64, 64, 3)
        x = self.dec(self.vq.codebook[codes.reshape(-1)])
        tiles = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().reshape(len(codes), GRID, GRID, T, T, 3)
        return from_tiles(tiles)

    def forward(self, tiles_u8):  # tiles (N, 7, 7, 3) -> loss
        x = self.prep(tiles_u8)
        q, idx, commit = self.vq(self.enc(x)[:, :, None, None])
        rec = self.dec(q[:, :, 0, 0])
        err = (rec.float() - x).abs()
        thr = err.detach().quantile(0.9, dim=1, keepdim=True)
        w = 1.0 + 4.0 * (err.detach() >= thr)  # hard-pixel weighting, as tokenizer v2
        return (w * err).sum() / w.sum() + commit


def main(args):
    dev = th.device("cuda")
    th.manual_seed(0)
    out_dir = os.path.join(ROOT, "runs", "wm", args.out)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    tr = load_episodes("train", keys=("frames",))
    train = np.concatenate([e["frames"] for e in tr])
    del tr
    if args.max_frames and len(train) > args.max_frames:
        train = train[np.sort(np.random.default_rng(0).choice(len(train), args.max_frames, replace=False))]
    te = load_episodes("test", keys=("frames", "sem"), limit=300)  # same 4,096 frames as tokenizer.py
    test, test_sem = np.concatenate([e["frames"] for e in te]), np.concatenate([e["sem"] for e in te])
    sel = np.random.default_rng(0).choice(len(test), 4096, replace=False)
    test_t = th.from_numpy(test[sel]).to(dev)
    creat_m, res_m = tile_mask(test_sem[sel], CREATURES), tile_mask(test_sem[sel], RESOURCES)
    train_t = th.from_numpy(train)
    print(f"frames: train {len(train):,} (loaded in {time.time() - t0:.0f} s)", flush=True)
    model = TileTokenizer(n_codes=args.codes).to(dev)
    opt = th.optim.AdamW(model.parameters(), lr=args.lr)
    sched = th.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    log = open(os.path.join(out_dir, "train_log.jsonl"), "w")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        fr = train_t[th.randint(len(train_t), (args.frames_per_batch,))].to(dev, non_blocking=True)
        tiles = to_tiles(fr).reshape(-1, T, T, 3)
        tiles = tiles[th.randint(len(tiles), (args.batch,), device=dev)]
        model.train()
        loss = model(tiles)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == args.steps:
            model.eval()
            with th.no_grad():
                codes = th.cat([model.encode(test_t[i:i + 512]) for i in range(0, len(test_t), 512)])
                rec = th.cat([model.decode(codes[i:i + 512]) for i in range(0, len(codes), 512)])
            se = ((rec.float() - test_t.float()) ** 2).mean(-1).cpu().numpy()
            reg = lambda m: round(float(10 * np.log10(255 ** 2 / max(se[m].mean(), 1e-9))), 2)
            used = th.bincount(codes.flatten(), minlength=model.n_codes)
            row = {"step": step, "loss": round(loss.item(), 5),
                   "test_psnr": round(float(10 * np.log10(255 ** 2 / se.mean())), 2),
                   "creature_psnr": reg(creat_m), "resource_psnr": reg(res_m),
                   "codes_used": int((used > 0).sum()), "min": round((time.time() - t0) / 60, 1)}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(row, flush=True)
    th.save(model.state_dict(), os.path.join(out_dir, "tokenizer.pt"))
    json.dump({"type": "tile", "codes": args.codes}, open(os.path.join(out_dir, "info.json"), "w"))
    import imageio.v2 as iio
    sel12 = test_t[:12]
    with th.no_grad():
        rec = model.decode(model.encode(sel12))
    sheet = th.cat([th.cat(list(sel12), 1), th.cat(list(rec), 1)], 0).cpu().numpy()
    iio.imwrite(os.path.join(out_dir, "reconstructions.png"), np.kron(sheet, np.ones((3, 3, 1), np.uint8)))
    print("saved", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8192, help="tiles per step")
    ap.add_argument("--frames_per_batch", type=int, default=256)
    ap.add_argument("--codes", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_frames", type=int, default=1200000)
    ap.add_argument("--out", default="tokenizer_tile")
    main(ap.parse_args())
