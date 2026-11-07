"""Frame tokenizer: a VQ autoencoder that turns a 64x64 Crafter frame into an 8x8 grid of codes.

Encoder: 3 stride-2 conv stages (64 -> 32 -> 16 -> 8) with residual blocks, then a 1x1 conv to the
code dimension. Quantiser: 512 codes, EMA codebook updates, commitment loss 0.25, and dead codes
are restarted from random encoder outputs. Decoder mirrors the encoder. Loss: pixel L1
(MSE blurred small sprites such as coal and cows away in a first test).

  .venv/bin/python methods/08_world_model/tokenizer.py --steps 30000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import ROOT, load_episodes  # noqa: E402

OUT = os.path.join(ROOT, "runs", "wm", "tokenizer")  # Stage 1 tokenizer; Stage 1b variants use --out
CREATURES = (14, 15, 16, 17, 18)   # Cow, Zombie, Skeleton, Arrow, Plant (semantic ids, runs/wm/data/meta.json)
RESOURCES = (8, 9, 10)             # coal, iron, diamond


def tile_mask(sem, ids):
    """sem (N, 9, 7) semantic ids of the world tiles -> (N, 64, 64) bool pixel mask of tiles in `ids`.
    Each tile is 7x7 px; the world view fills the top 49 rows (Crafter renders unit = 64 // 9 = 7, no border)."""
    hit = np.isin(sem, ids).transpose(0, 2, 1)  # (N, 7 rows, 9 cols)
    m = np.zeros((len(sem), 64, 64), bool)
    m[:, :49, :63] = hit.repeat(7, 1).repeat(7, 2)
    return m


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
                                 nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class VQ(nn.Module):
    def __init__(self, n_codes=512, dim=64, decay=0.99, beta=0.25):
        super().__init__()
        self.n, self.dim, self.decay, self.beta = n_codes, dim, decay, beta
        self.register_buffer("codebook", th.randn(n_codes, dim) * 0.1)
        self.register_buffer("ema_count", th.ones(n_codes))
        self.register_buffer("ema_sum", self.codebook.clone())

    def codes(self, z):  # z: (N, dim) -> indices
        d = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.codebook.t() + self.codebook.pow(2).sum(1))
        return d.argmin(1)

    def forward(self, z):  # z: (B, dim, H, W)
        z = z.float()  # nearest-code search and EMA updates in fp32, also under autocast
        B, D, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        idx = self.codes(flat)
        q = self.codebook[idx]
        if self.training:
            with th.no_grad():
                onehot = F.one_hot(idx, self.n).type(flat.dtype)
                self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
                n = self.ema_count.sum()
                count = (self.ema_count + 1e-5) / (n + self.n * 1e-5) * n
                self.codebook.copy_(self.ema_sum / count[:, None])
                dead = self.ema_count < 1e-2  # restart unused codes from random encoder outputs
                if dead.any():
                    pick = flat[th.randint(len(flat), (int(dead.sum()),), device=flat.device)]
                    self.codebook[dead] = pick
                    self.ema_sum[dead] = pick
                    self.ema_count[dead] = 1.0
        commit = F.mse_loss(flat, q.detach())
        q = flat + (q - flat).detach()  # straight-through
        return q.reshape(B, H, W, D).permute(0, 3, 1, 2), idx.reshape(B, H, W), self.beta * commit


class Tokenizer(nn.Module):
    def __init__(self, ch=(64, 128, 256), dim=64, n_codes=512):
        super().__init__()
        c0, c1, c2 = ch
        self.enc = nn.Sequential(
            nn.Conv2d(3, c0, 3, padding=1), ResBlock(c0),
            nn.Conv2d(c0, c1, 4, 2, 1), ResBlock(c1),         # 32
            nn.Conv2d(c1, c2, 4, 2, 1), ResBlock(c2),         # 16
            nn.Conv2d(c2, c2, 4, 2, 1), ResBlock(c2), ResBlock(c2),  # 8
            nn.GroupNorm(8, c2), nn.SiLU(), nn.Conv2d(c2, dim, 1))
        self.vq = VQ(n_codes, dim)
        self.dec = nn.Sequential(
            nn.Conv2d(dim, c2, 3, padding=1), ResBlock(c2), ResBlock(c2),
            nn.ConvTranspose2d(c2, c2, 4, 2, 1), ResBlock(c2),  # 16
            nn.ConvTranspose2d(c2, c1, 4, 2, 1), ResBlock(c1),  # 32
            nn.ConvTranspose2d(c1, c0, 4, 2, 1), ResBlock(c0),  # 64
            nn.GroupNorm(8, c0), nn.SiLU(), nn.Conv2d(c0, 3, 3, padding=1))

    @staticmethod
    def prep(frames):  # uint8 (B, 64, 64, 3) -> float (B, 3, 64, 64) in [-1, 1]
        return frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0

    def encode(self, frames_u8):
        z = self.enc(self.prep(frames_u8)).float()
        return self.vq.codes(z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])).reshape(len(z), 8, 8)

    def decode(self, idx):  # (B, 8, 8) codes -> uint8 frames
        q = self.vq.codebook[idx].permute(0, 3, 1, 2)
        x = self.dec(q)
        return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1)

    def forward(self, frames_u8, loss="l1", weight=None):
        """loss: "l1" (Stage 1), "hard" (the 10% worst pixels of each frame count 5x, Stage 1b),
        "sem" (pixels in `weight`, e.g. creature tiles, count 10x; diagnostic only)."""
        x = self.prep(frames_u8)
        q, idx, commit = self.vq(self.enc(x))
        rec = self.dec(q)
        err = (rec.float() - x).abs().mean(1)  # (B, 64, 64)
        if loss == "hard":
            thr = err.detach().flatten(1).quantile(0.9, dim=1)[:, None, None]
            w = 1.0 + 4.0 * (err.detach() >= thr)
        elif loss == "sem":
            w = 1.0 + 9.0 * weight.float()
        else:
            w = th.ones_like(err)
        return rec, idx, (w * err).sum() / w.sum() + commit


def psnr(a_u8, b_u8):
    mse = ((a_u8.float() - b_u8.float()) ** 2).mean().item()
    return 10 * np.log10(255 ** 2 / max(mse, 1e-9))


def main(args):
    dev = th.device("cuda")
    th.manual_seed(0)
    t0 = time.time()
    out_dir = os.path.join(ROOT, "runs", "wm", args.out)
    os.makedirs(out_dir, exist_ok=True)
    tr = load_episodes("train", keys=("frames", "sem"), limit=args.episodes)
    te = load_episodes("test", keys=("frames", "sem"), limit=300)
    train, train_sem = np.concatenate([e["frames"] for e in tr]), np.concatenate([e["sem"] for e in tr])
    del tr
    if args.max_frames and len(train) > args.max_frames:  # cap memory: a random subset of frames
        keep = np.sort(np.random.default_rng(0).choice(len(train), args.max_frames, replace=False))
        train, train_sem = train[keep], train_sem[keep]
    test, test_sem = np.concatenate([e["frames"] for e in te]), np.concatenate([e["sem"] for e in te])
    print(f"frames: train {len(train):,}  test {len(test):,}  (loaded in {time.time() - t0:.0f} s)", flush=True)
    train_t = th.from_numpy(train)  # stays on CPU, batches go to the GPU
    train_w = th.from_numpy(tile_mask(train_sem, CREATURES)) if args.loss == "sem" else None
    sel = np.random.default_rng(0).choice(len(test), 4096, replace=False)
    test_t = th.from_numpy(test[sel]).to(dev)
    creat_m, res_m = tile_mask(test_sem[sel], CREATURES), tile_mask(test_sem[sel], RESOURCES)
    model = Tokenizer().to(dev)
    opt = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = th.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    log = open(os.path.join(out_dir, "train_log.jsonl"), "w")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = th.randint(len(train_t), (args.batch,))
        x = train_t[idx].to(dev, non_blocking=True)
        w = train_w[idx].to(dev, non_blocking=True) if train_w is not None else None
        model.train()
        with th.autocast("cuda", dtype=th.bfloat16):
            rec, codes, loss = model(x, args.loss, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            model.eval()
            with th.no_grad():
                codes = th.cat([model.encode(test_t[i:i + 512]) for i in range(0, len(test_t), 512)])
                rec = th.cat([model.decode(codes[i:i + 512]) for i in range(0, len(codes), 512)])
            used = th.bincount(codes.flatten(), minlength=512).float()
            p = used / used.sum()
            perplexity = float(th.exp(-(p[p > 0] * p[p > 0].log()).sum()))
            se = ((rec.float() - test_t.float()) ** 2).mean(-1).cpu().numpy()  # (N, 64, 64)
            region = lambda m: round(float(10 * np.log10(255 ** 2 / max(se[m].mean(), 1e-9))), 2)
            row = {"step": step, "loss": round(loss.item(), 5), "test_psnr": round(psnr(rec, test_t), 2),
                   "creature_psnr": region(creat_m), "resource_psnr": region(res_m),
                   "codes_used": int((used > 0).sum()), "perplexity": round(perplexity, 1), "min": round((time.time() - t0) / 60, 1)}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(row, flush=True)
    th.save(model.state_dict(), os.path.join(out_dir, "tokenizer.pt"))
    # reconstruction sheet: top row real, bottom row decoded
    import imageio.v2 as iio
    sel = test_t[:12]
    with th.no_grad():
        rec = model.decode(model.encode(sel))
    sheet = th.cat([th.cat(list(sel), 1), th.cat(list(rec), 1)], 0).cpu().numpy()
    iio.imwrite(os.path.join(out_dir, "reconstructions.png"), np.kron(sheet, np.ones((3, 3, 1), np.uint8)))
    print("saved", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--episodes", type=int, default=None, help="limit training episodes (debug)")
    ap.add_argument("--loss", choices=("l1", "hard", "sem"), default="l1")
    ap.add_argument("--out", default="tokenizer", help="output folder under runs/wm/")
    ap.add_argument("--max_frames", type=int, default=None, help="train on a random subset of frames (memory)")
    main(ap.parse_args())
