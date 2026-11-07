"""World-model dynamics: predict the next frame tokens, inventory, unlocks and death.

Per step t the input is  frame(o_t) + inventory(o_t) + action(a_t) [+ clock(t)]:
  frame:     the 8x8 token grid -> token embeddings + 2D positions -> 2 spatial Transformer layers
             -> flattened and projected to one vector (keeps where things are)
  inventory: 16 counts (0-9), one embedding table per slot, summed
  action:    embedding of the action taken at step t
  clock:     (Stage 1b, --clock) the episode step modulo Crafter's 300-step day, so night is predictable
A causal Transformer over the last <= 64 steps gives h_t, from which we predict step t+1:
  tokens:    a small spatial decoder, MaskGIT-style: it sees h_t and the next grid with some tokens
             masked, and predicts the masked ones. At imagination time it starts fully masked and keeps
             the most confident predictions over S passes.
  inventory: 16 x 10-way classification (its own output, not read back from pixels)
  unlocks:   22 sigmoid outputs, "achievement i is unlocked on this step"
  death:     1 sigmoid output
Stage 1b additions (all off by default, so Stage 1 models load unchanged):
  --skip     skip-ahead output: from h_t and the actions a_{t+1..t+k-1}, predict the grid k in {4,8,16,32}
             steps ahead. Predicting a lake that comes back 16 steps later needs memory of the lake.
  --unroll   for a share of training windows, the last 8 inputs are the model's own predictions (made
             without gradients), so it practises on its own dreams and learns to recover from drift.
  --pos      (Stage 1c, diagnostic) the agent's position relative to the window start as an input, plus an
             odometry output (stay / one tile left, right, up, down). Dreams update the position from the
             model's own odometry. Positions come from derive_positions.py (recorded tile maps).
  --refit    freeze a trained model and re-fit the inventory / unlock / death outputs from scratch with
             early stopping on a validation split (they overfit in Stage 1 while the frames kept improving).

  .venv/bin/python methods/08_world_model/dynamics.py --run uniform --steps 30000                  # Stage 1
  .venv/bin/python methods/08_world_model/dynamics.py --run v2 --tokens tokens_v2 --clock --skip --unroll 0.25
  .venv/bin/python methods/08_world_model/dynamics.py --run v2 --refit
"""
import argparse
import json
import math
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

TOKENS = os.path.join(ROOT, "runs", "wm", "tokens")
POS = os.path.join(ROOT, "runs", "wm", "pos")
MOVE_DELTA = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))  # odometry classes
POS_CLIP = 32
N_CODES, MASK = 512, 512  # token id 512 = [MASK]
N_ACT, N_INV, N_ACH = 17, 16, 22
DAY = 300
SKIPS = (4, 8, 16, 32)


def block(d, heads, ff_mult=4, dropout=0.0):
    return nn.TransformerEncoderLayer(d, heads, ff_mult * d, dropout=dropout, activation="gelu",
                                      batch_first=True, norm_first=True)


class WorldModel(nn.Module):
    def __init__(self, d_s=256, d_t=512, n_spatial=2, n_temporal=6, n_dec=3, context=64, dropout=0.1,
                 clock=False, skip=False, pos=False, grid=(8, 8), n_codes=N_CODES, map_codebook=None):
        super().__init__()
        self.context, self.use_clock, self.use_skip, self.use_pos = context, clock, skip, pos
        self.grid, self.n_tok, self.n_codes, self.mask_id = tuple(grid), grid[0] * grid[1], n_codes, n_codes
        self.tok_emb = nn.Embedding(n_codes + 1, d_s)
        self.pos_s = nn.Parameter(th.randn(self.n_tok, d_s) * 0.02)
        self.spatial = nn.TransformerEncoder(block(d_s, 4), n_spatial)
        self.frame_proj = nn.Sequential(nn.Linear(d_s, 64), nn.Flatten(-2), nn.Linear(self.n_tok * 64, d_t))
        self.use_map = map_codebook is not None
        if self.use_map:  # Stage 1d: the top-down map window, as codebook vectors (robust to lighting) + age
            self.register_buffer("map_cb", map_codebook.clone().float())
            self.map_proj = nn.Linear(map_codebook.shape[1], d_s)
            self.map_unseen = nn.Parameter(th.zeros(d_s))
            self.map_age = nn.Embedding(11, d_s)
        self.inv_emb = nn.Embedding(N_INV * 10, d_t)
        self.act_emb = nn.Embedding(N_ACT, d_t)
        self.pos_t = nn.Parameter(th.randn(context, d_t) * 0.02)
        self.temporal = nn.TransformerEncoder(block(d_t, 8, dropout=dropout), n_temporal)
        self.norm_t = nn.LayerNorm(d_t)
        self.cond = nn.Linear(d_t, d_s)
        self.decoder = nn.TransformerEncoder(block(d_s, 4), n_dec)
        self.tok_head = nn.Linear(d_s, n_codes)
        self.inv_head = nn.Linear(d_t, N_INV * 10)
        self.ach_head = nn.Linear(d_t, N_ACH)
        self.done_head = nn.Linear(d_t, 1)
        self.register_buffer("inv_offset", th.arange(N_INV) * 10)
        if clock:
            self.clock_emb = nn.Embedding(DAY, d_t)
        if pos:
            self.pos_x = nn.Embedding(2 * POS_CLIP + 1, d_t)
            self.pos_y = nn.Embedding(2 * POS_CLIP + 1, d_t)
            self.odo_head = nn.Linear(d_t, len(MOVE_DELTA))
            self.register_buffer("move_delta", th.tensor(MOVE_DELTA))
        if skip:
            self.skip_act = nn.Embedding(N_ACT, d_t)
            self.skip_k = nn.Embedding(max(SKIPS) + 1, d_t)
            self.skip_cond = nn.Sequential(nn.Linear(2 * d_t, d_t), nn.GELU(), nn.Linear(d_t, d_s))

    def embed_frames(self, tok):  # (..., H, W) -> (..., d_t)
        lead = tok.shape[:-2]
        x = self.tok_emb(tok.reshape(-1, self.n_tok)) + self.pos_s
        x = self.spatial(x)
        return self.frame_proj(x).reshape(*lead, -1)

    def history(self, tok, inv, act, time=None, pos=None):
        """tok (B, L, 8, 8), inv (B, L, 16), act (B, L), time (B, L) episode steps, pos (B, L, 2) -> h (B, L, d_t)."""
        L = tok.shape[1]
        x = self.embed_frames(tok) + self.inv_emb(inv.clamp(0, 9) + self.inv_offset).sum(-2) + self.act_emb(act)
        if self.use_clock:
            x = x + self.clock_emb(time % DAY)
        if self.use_pos:  # relative to the first step of the window, so it does not depend on where in the world
            rel = (pos - pos[:, :1]).clamp(-POS_CLIP, POS_CLIP) + POS_CLIP
            x = x + self.pos_x(rel[..., 0]) + self.pos_y(rel[..., 1])
        x = x + self.pos_t[:L]
        causal = th.triu(th.full((L, L), float("-inf"), device=x.device), 1)
        return self.norm_t(self.temporal(x, mask=causal, is_causal=True))

    @staticmethod
    def age_bucket(age):  # 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, 129-254, never seen (255)
        a = age.long()
        b = th.where(a <= 2, a, (th.log2(a.clamp(min=1).float()).ceil().long() + 1).clamp(max=9))
        return th.where(a >= 255, th.full_like(b, 10), b)

    def map_prior(self, mtok, mage):
        """Map window (N, 7, 9) codes + ages -> (N, 63, d_s) additions for the world rows of the next frame."""
        seen = (mtok < self.n_codes)[..., None]
        v = self.map_proj(self.map_cb[mtok.clamp(max=self.n_codes - 1).long()])
        v = th.where(seen, v, self.map_unseen.to(v.dtype))
        return (v + self.map_age(self.age_bucket(mage))).reshape(len(mtok), -1, v.shape[-1])

    def decode_logits(self, h, partial, cond=None, prior=None):
        """h (N, d_t), partial (N, H, W) with MASK where unknown -> token logits (N, H*W, n_codes).
        `cond` (N, d_s) replaces the next-step conditioning (used by the skip-ahead output).
        `prior` (N, 63, d_s): Stage 1d map window, added to the world rows (the first 63 tiles)."""
        c = self.cond(h) if cond is None else cond
        x = self.tok_emb(partial.reshape(-1, self.n_tok)) + self.pos_s + c[:, None]
        if prior is not None:
            x = th.cat([x[:, :prior.shape[1]] + prior.to(x.dtype), x[:, prior.shape[1]:]], 1)
        return self.tok_head(self.decoder(x))

    def heads(self, h):
        return (self.inv_head(h).reshape(*h.shape[:-1], N_INV, 10), self.ach_head(h), self.done_head(h).squeeze(-1))

    def _mask(self, n, device):
        k = self.n_tok
        r = th.cos(th.rand(n, device=device) * math.pi / 2)
        n_mask = (r * k).ceil().clamp(1, k)
        return th.rand(n, k, device=device).argsort(1).argsort(1) < n_mask[:, None]

    def _masked_ce(self, logits, tgt, masked):
        ce = F.cross_entropy(logits.reshape(-1, self.n_codes), tgt.reshape(-1), reduction="none").reshape(len(tgt), -1)
        return (ce * masked).sum(1) / masked.sum(1)

    @th.no_grad()
    def self_feed(self, batch, rows, U=8):
        """Replace the last U inputs of the given full-length windows with the model's own predictions,
        made one step at a time from the (partly predicted) history, without gradients.
        With the map, each prediction reads the precomputed (real-frame) map window, an approximation: in a
        dream the window would also hold the model's own last few frames."""
        tok, inv, act, tm, ps = (batch[k][rows].clone() for k in ("tok", "inv", "act", "time", "pos"))
        L = tok.shape[1]
        for pos in range(L - U, L):
            h = self.history(tok[:, :pos], inv[:, :pos], act[:, :pos], tm[:, :pos], ps[:, :pos])[:, -1].float()
            # input `pos` is the frame that followed step pos-1, whose map window is mtok[:, pos - 1]
            prior = self.map_prior(batch["mtok"][rows, pos - 1], batch["mage"][rows, pos - 1]) if self.use_map else None
            nt, ni, _, _ = self.predict_next(h, passes=1, prior=prior)
            tok[:, pos], inv[:, pos] = nt, ni
            if self.use_pos:
                ps[:, pos] = ps[:, pos - 1] + self.move_delta[self.odo_head(h).argmax(-1)]
        batch["tok"][rows], batch["inv"][rows], batch["pos"][rows] = tok, inv, ps

    def loss(self, batch, unroll=0.0, skip_w=0.5):
        if unroll > 0:
            full = batch["valid"].all(1).nonzero().squeeze(1)
            rows = full[th.rand(len(full), device=full.device) < unroll]
            if len(rows):
                self.self_feed(batch, rows)
        tok, inv, act, tm = batch["tok"], batch["inv"], batch["act"], batch["time"]
        ntok, ninv, nach, ndone, valid = batch["ntok"], batch["ninv"], batch["nach"], batch["ndone"], batch["valid"]
        h = self.history(tok, inv, act, tm, batch["pos"])
        B, L = act.shape
        hf, tgt = h.reshape(B * L, -1), ntok.reshape(B * L, self.n_tok)
        masked = self._mask(B * L, h.device)
        prior = self.map_prior(batch["mtok"].reshape(B * L, 7, 9), batch["mage"].reshape(B * L, 7, 9)) if self.use_map else None
        logits = self.decode_logits(hf, th.where(masked, self.mask_id, tgt).reshape(-1, *self.grid), prior=prior)
        v = valid.reshape(B * L)
        tok_loss = self._masked_ce(logits, tgt, masked)[v].mean()
        inv_logits, ach_logits, done_logits = self.heads(h)
        inv_loss = F.cross_entropy(inv_logits[valid].reshape(-1, 10), ninv.clamp(0, 9)[valid].reshape(-1))
        ach_loss = F.binary_cross_entropy_with_logits(ach_logits[valid], nach[valid].float(),
                                                      pos_weight=th.full((N_ACH,), 10.0, device=h.device))
        done_loss = F.binary_cross_entropy_with_logits(done_logits[valid], ndone[valid].float(),
                                                       pos_weight=th.tensor(10.0, device=h.device))
        total = tok_loss + inv_loss + ach_loss + done_loss
        parts = {"tok": tok_loss.item(), "inv": inv_loss.item(), "ach": ach_loss.item(), "done": done_loss.item()}
        if self.use_pos:
            odo_loss = F.cross_entropy(self.odo_head(h)[valid], batch["move"][valid])
            total = total + odo_loss
            parts["odo"] = odo_loss.item()
        if self.use_skip:
            # one k per window; position t predicts the grid at t+k from h_t and the actions a_{t+1..t+k-1}
            k = th.tensor(SKIPS, device=h.device)[th.randint(len(SKIPS), (B,), device=h.device)]
            emb = self.skip_act(act)                                     # (B, L, d)
            csum = F.pad(emb.cumsum(1), (0, 0, 1, 0))                    # csum[:, j] = sum of emb[:, :j]
            t = th.arange(L, device=h.device)[None]                      # (1, L)
            end = (t + k[:, None]).clamp(max=L)                          # actions t+1 .. t+k-1 -> csum[t+k] - csum[t+1]
            summary = (csum.gather(1, end[..., None].expand(-1, -1, emb.shape[-1]))
                       - csum[:, 1:].gather(1, t.expand(B, -1)[..., None].expand(-1, -1, emb.shape[-1])))
            summary = summary / (k[:, None, None] - 1).clamp(min=1)
            tgt_idx = (t + k[:, None] - 1).clamp(max=L - 1)              # ntok[t+k-1] = tokens at step t+k
            ok = valid & (t + k[:, None] - 1 < L) & valid.gather(1, tgt_idx)
            if ok.any():
                c = self.skip_cond(th.cat([h, summary + self.skip_k(k)[:, None]], -1))[ok]
                st = ntok.reshape(B, L, self.n_tok).gather(1, tgt_idx[..., None].expand(-1, -1, self.n_tok))[ok]
                m = self._mask(len(st), h.device)
                sl = self.decode_logits(None, th.where(m, self.mask_id, st).reshape(-1, *self.grid), cond=c)
                skip_loss = self._masked_ce(sl, st, m).mean()
                total = total + skip_w * skip_loss
                parts["skip"] = skip_loss.item()
        return total, parts

    @th.no_grad()
    def predict_next(self, h, passes=3, cond=None, prior=None, temperature=0.0, inv_sampled=None):
        """h (N, d_t) -> next tokens (N, H, W), inventory (N, 16), unlock probs (N, 22), death prob (N,).
        `prior`: Stage 1d map window for the next frame (see map_prior).
        `temperature` 0: greedy (each token / count is its most likely value); > 0: sampled from the
        softmax at that temperature, so unlikely events (a zombie walking in, a hit) happen at their rate.
        `inv_sampled`: inventory slots that are sampled (default all); the others take the most likely count."""
        N = len(h) if h is not None else len(cond)
        dev = (h if h is not None else cond).device
        K = self.n_tok
        cur = th.full((N, K), self.mask_id, dtype=th.long, device=dev)
        known = th.zeros(N, K, dtype=th.bool, device=dev)
        for s in range(passes):
            logits = self.decode_logits(h, cur.reshape(N, *self.grid), cond=cond, prior=prior).float()
            if temperature > 0:
                p = (logits / temperature).softmax(-1)
                pred = th.multinomial(p.reshape(-1, p.shape[-1]), 1).reshape(N, K)
                prob = p.gather(-1, pred[..., None])[..., 0]
            else:
                prob, pred = logits.softmax(-1).max(-1)
            prob = th.where(known, th.full_like(prob, 2.0), prob)  # keep what is already fixed
            n_keep = K if s == passes - 1 else int(round(K * (1 - math.cos(math.pi / 2 * (s + 1) / passes))))
            keep = prob.argsort(1, descending=True).argsort(1) < n_keep
            cur = th.where(keep & ~known, pred, cur)
            known = known | keep
        if h is None:
            return cur.reshape(N, *self.grid)
        inv_logits, ach_logits, done_logits = self.heads(h)
        inv = inv_logits.argmax(-1)
        if temperature > 0:
            p = (inv_logits.float() / temperature).softmax(-1)
            drawn = th.multinomial(p.reshape(-1, p.shape[-1]), 1).reshape(p.shape[:-1])
            if inv_sampled is None:
                inv = drawn
            else:
                inv[:, list(inv_sampled)] = drawn[:, list(inv_sampled)]
        return cur.reshape(N, *self.grid), inv, ach_logits.sigmoid(), done_logits.sigmoid()

    @th.no_grad()
    def predict_skip(self, h, future_actions, k, passes=3):
        """Grid k steps after the state h, given the k-1 actions in between (N, k-1)."""
        emb = self.skip_act(future_actions).mean(1) if future_actions.shape[1] else th.zeros_like(h)
        kk = th.full((len(h),), k, device=h.device, dtype=th.long)
        return self.predict_next(None, passes, cond=self.skip_cond(th.cat([h, emb + self.skip_k(kk)], -1)))


# ---------------------------------------------------------------- data

def load_token_episodes(split, tokens_dir=TOKENS, sources="all", maps_dir=None, pos_dir=POS):
    from data import DATA
    eps = load_episodes(split, keys=("actions", "inventory", "unlocked", "died"), extra_dir=tokens_dir, sources=sources)
    eps = [e for e in eps if "tokens" in e]
    for e in eps:
        e["events"] = e["unlocked"][1:] & ~e["unlocked"][:-1]  # (T, 22): unlocked on step t -> t+1
        q = os.path.join(pos_dir, os.path.relpath(e["path"], DATA))
        if os.path.exists(q):
            d = np.load(q)
            e["pos"], e["move"] = d["pos"], d["move"]
        else:
            e["pos"], e["move"] = np.zeros((len(e["actions"]) + 1, 2), np.int16), np.zeros(len(e["actions"]), np.int8)
        if maps_dir:
            m = np.load(os.path.join(maps_dir, os.path.relpath(e["path"], DATA)))
            e["mtok"], e["mage"] = m["mtok"], m["mage"]
    return eps


def rare_achievements(eps, threshold=0.10):
    rate = np.mean([e["unlocked"][-1] for e in eps], axis=0)
    return np.where(rate < threshold)[0], rate


class Sampler:
    """Random 64-step windows; windows whose targets contain a rare unlock get `rare_weight`."""

    def __init__(self, eps, L, rare=(), rare_weight=1.0, seed=0):
        self.eps, self.L = eps, L
        self.rng = np.random.default_rng(seed)
        idx, w = [], []
        for i, e in enumerate(eps):
            T = len(e["actions"])
            starts = np.arange(max(1, T - L + 1))
            ww = np.ones(len(starts))
            if len(rare) and rare_weight != 1.0:
                ev = e["events"][:, rare].any(1).astype(np.int64)  # (T,)
                c = np.concatenate([[0], np.cumsum(ev)])
                ends = np.minimum(starts + L, T)
                ww[(c[ends] - c[starts]) > 0] = rare_weight
            idx.append(np.stack([np.full(len(starts), i), starts], 1))
            w.append(ww)
        self.idx = np.concatenate(idx)
        w = np.concatenate(w)
        self.p = w / w.sum()
        self.rare_share = float(w[w > 1].sum() / w.sum()) if (w > 1).any() else 0.0

    def batch(self, B, device):
        L = self.L
        pick = self.idx[self.rng.choice(len(self.idx), B, p=self.p)]
        g = self.eps[0]["tokens"].shape[1:]
        spec = [("tok", (B, L, *g), np.int64), ("inv", (B, L, 16), np.int64), ("act", (B, L), np.int64),
                ("time", (B, L), np.int64), ("pos", (B, L, 2), np.int64), ("move", (B, L), np.int64),
                ("ntok", (B, L, *g), np.int64), ("ninv", (B, L, 16), np.int64), ("nach", (B, L, 22), bool),
                ("ndone", (B, L), bool), ("valid", (B, L), bool)]
        has_map = "mtok" in self.eps[0]
        if has_map:
            spec += [("mtok", (B, L, 7, 9), np.int64), ("mage", (B, L, 7, 9), np.int64)]
        out = {k: np.zeros(s, dt) for k, s, dt in spec}
        if has_map:
            out["mage"][:] = 255
        for b, (i, s) in enumerate(pick):
            e = self.eps[i]
            T = len(e["actions"])
            n = min(L, T - s)
            out["tok"][b, :n] = e["tokens"][s:s + n]
            out["inv"][b, :n] = e["inventory"][s:s + n]
            out["act"][b, :n] = e["actions"][s:s + n]
            out["time"][b, :n] = np.arange(s, s + n)
            out["pos"][b, :n] = e["pos"][s:s + n]
            out["move"][b, :n] = e["move"][s:s + n]
            if has_map:
                out["mtok"][b, :n] = e["mtok"][s:s + n]
                out["mage"][b, :n] = e["mage"][s:s + n]
            out["ntok"][b, :n] = e["tokens"][s + 1:s + n + 1]
            out["ninv"][b, :n] = e["inventory"][s + 1:s + n + 1]
            out["nach"][b, :n] = e["events"][s:s + n]
            out["ndone"][b, n - 1] = bool(e["died"]) and s + n == T
            out["valid"][b, :n] = True
        return {k: th.from_numpy(v).to(device, non_blocking=True) for k, v in out.items()}


def tokenizer_codebook(tokens):
    """(grid, n_codes, codebook) of the tokenizer behind a token cache ("tokens_X" <- "tokenizer_X")."""
    d = os.path.join(ROOT, "runs", "wm", "tokenizer" + tokens[len("tokens"):])
    sd = th.load(os.path.join(d, "tokenizer.pt"), map_location="cpu")
    cb = sd["vq.codebook"]
    info = os.path.join(d, "info.json")
    tile = os.path.exists(info) and json.load(open(info)).get("type") == "tile"
    return ((9, 9) if tile else (8, 8)), len(cb), cb


def build(info, dev):
    a = info["args"]
    grid, n_codes, cb = tokenizer_codebook(a.get("tokens", "tokens"))
    return WorldModel(context=a.get("context", 64), clock=a.get("clock", False), skip=a.get("skip", False),
                      pos=a.get("pos", False), grid=grid, n_codes=n_codes,
                      map_codebook=cb if a.get("map") else None).to(dev)


def refit(args):
    """Freeze the trained model; re-fit inventory / unlock / death outputs with early stopping on 'val'."""
    dev = th.device("cuda")
    out_dir = os.path.join(ROOT, "runs", "wm", "dynamics", args.run)
    info = json.load(open(os.path.join(out_dir, "info.json")))
    tokens_dir = os.path.join(ROOT, "runs", "wm", info["args"].get("tokens", "tokens"))
    model = build(info, dev)
    model.load_state_dict(th.load(os.path.join(out_dir, "model.pt"), map_location=dev))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    heads = [model.inv_head, model.ach_head, model.done_head]
    for m in heads:
        m.reset_parameters()
        for p in m.parameters():
            p.requires_grad_(True)
    maps_dir = os.path.join(ROOT, "runs", "wm", info["args"]["maps"]) if info["args"].get("map") else None
    pos_dir = os.path.join(ROOT, "runs", "wm", info["args"].get("pos_dir", "pos"))
    fit = load_token_episodes("fit", tokens_dir, maps_dir=maps_dir, pos_dir=pos_dir)
    val = load_token_episodes("val", tokens_dir, maps_dir=maps_dir, pos_dir=pos_dir)
    s_fit, s_val = Sampler(fit, model.context, seed=1), Sampler(val, model.context, seed=2)
    val_batches = [s_val.batch(64, dev) for _ in range(10)]
    opt = th.optim.AdamW([p for m in heads for p in m.parameters()], lr=1e-3, weight_decay=0.05)

    def head_loss(b):
        with th.no_grad(), th.autocast("cuda", dtype=th.bfloat16):
            h = model.history(b["tok"], b["inv"], b["act"], b["time"], b["pos"]).float()
        il, al, dl = model.heads(h)
        v = b["valid"]
        return (F.cross_entropy(il[v].reshape(-1, 10), b["ninv"].clamp(0, 9)[v].reshape(-1)),
                F.binary_cross_entropy_with_logits(al[v], b["nach"][v].float(), pos_weight=th.full((N_ACH,), 10.0, device=dev)),
                F.binary_cross_entropy_with_logits(dl[v], b["ndone"][v].float(), pos_weight=th.tensor(10.0, device=dev)))

    best, best_state, since, log = float("inf"), None, 0, []
    for step in range(1, args.refit_steps + 1):
        losses = head_loss(s_fit.batch(64, dev))
        opt.zero_grad(set_to_none=True)
        sum(losses).backward()
        opt.step()
        if step % 200 == 0:
            with th.no_grad():
                vl = np.mean([[x.item() for x in head_loss(b)] for b in val_batches], axis=0)
            log.append({"step": step, "val_inv": round(vl[0], 4), "val_ach": round(vl[1], 4), "val_done": round(vl[2], 4)})
            print(log[-1], flush=True)
            if vl.sum() < best:
                best, since = vl.sum(), 0
                best_state = {f"{n}.{k}": v.clone() for n, m in zip(("inv_head", "ach_head", "done_head"), heads)
                              for k, v in m.state_dict().items()}
            else:
                since += 200
                if since >= 1000:
                    break
    sd = model.state_dict()
    sd.update(best_state)
    th.save(sd, os.path.join(out_dir, "model_refit.pt"))
    json.dump(log, open(os.path.join(out_dir, "refit_log.json"), "w"), indent=1)
    print("saved model_refit.pt; best at step", log[int(np.argmin([r["val_inv"] + r["val_ach"] + r["val_done"] for r in log]))]["step"])


def main(args):
    if args.refit:
        return refit(args)
    dev = th.device("cuda")
    th.manual_seed(args.seed)
    out_dir = os.path.join(ROOT, "runs", "wm", "dynamics", args.run)
    os.makedirs(out_dir, exist_ok=True)
    tokens_dir = os.path.join(ROOT, "runs", "wm", args.tokens)
    maps_dir = os.path.join(ROOT, "runs", "wm", args.maps) if args.map else None
    pos_dir = os.path.join(ROOT, "runs", "wm", args.pos_dir)
    train = load_token_episodes(args.train_split, tokens_dir, args.sources, maps_dir, pos_dir)
    test = load_token_episodes("test", tokens_dir, "stage1", maps_dir, pos_dir)
    rare, rate = rare_achievements(train, args.rare_threshold)
    from common.crafter_score import ACHIEVEMENTS
    sampler = Sampler(train, args.context, rare, args.rare_weight, args.seed)
    test_sampler = Sampler(test, args.context, seed=123)
    info = {"run": args.run, "args": vars(args), "train_episodes": len(train), "test_episodes": len(test),
            "rare": [ACHIEVEMENTS[i] for i in rare], "unlock_rate": dict(zip(ACHIEVEMENTS, map(float, rate))),
            "rare_window_share": sampler.rare_share}
    print(json.dumps({k: v for k, v in info.items() if k != "unlock_rate"}), flush=True)
    model = build({"args": vars(args)}, dev)
    if args.init:  # start from another run's weights; new parts (e.g. --pos) keep their fresh initialisation
        missing, unexpected = model.load_state_dict(
            th.load(os.path.join(ROOT, "runs", "wm", "dynamics", args.init, "model.pt"), map_location=dev), strict=False)
        print("initialised from", args.init, "| new parameters:", missing, "| unused:", unexpected, flush=True)
    info["params"] = sum(p.numel() for p in model.parameters())
    print(f"params {info['params'] / 1e6:.1f}M", flush=True)
    json.dump(info, open(os.path.join(out_dir, "info.json"), "w"), indent=1)
    opt = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    sched = th.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)
    log = open(os.path.join(out_dir, "train_log.jsonl"), "w")
    fixed_test = [test_sampler.batch(args.batch, dev) for _ in range(8)]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        batch = sampler.batch(args.batch, dev)
        with th.autocast("cuda", dtype=th.bfloat16):
            loss, parts = model.loss(batch, unroll=args.unroll, skip_w=args.skip_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with th.no_grad(), th.autocast("cuda", dtype=th.bfloat16):
                tp = [model.loss({k: v.clone() for k, v in b.items()})[1] for b in fixed_test]
            row = {"step": step, "min": round((time.time() - t0) / 60, 1),
                   **{f"train_{k}": round(v, 4) for k, v in parts.items()},
                   **{f"test_{k}": round(float(np.mean([p[k] for p in tp])), 4) for k in tp[0]}}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(row, flush=True)
        if step % 10000 == 0 or step == args.steps:
            th.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            if args.keep_checkpoints:
                th.save(model.state_dict(), os.path.join(out_dir, f"model_{step // 1000}k.pt"))
    print("saved", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rare_weight", type=float, default=1.0)
    ap.add_argument("--rare_threshold", type=float, default=0.30,
                    help="an achievement is rare if unlocked in fewer than this share of training episodes")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", default="tokens", help="token cache folder under runs/wm/")
    ap.add_argument("--sources", default="all", help="all | stage1 | b")
    ap.add_argument("--train_split", default="train", help="train | fit (keeps 'val' out for --refit)")
    ap.add_argument("--clock", action="store_true")
    ap.add_argument("--skip", action="store_true")
    ap.add_argument("--pos", action="store_true", help="Stage 1c: position input + odometry output (diagnostic)")
    ap.add_argument("--map", action="store_true", help="Stage 1d: top-down map memory as decoder input (needs --pos)")
    ap.add_argument("--maps", default="maps_tile", help="map cache folder under runs/wm/ (build_maps.py)")
    ap.add_argument("--pos_dir", default="pos", help="positions under runs/wm/: pos (from tile maps) | pos_pix (pixels only)")
    ap.add_argument("--unroll", type=float, default=0.0, help="share of full windows trained on own predictions")
    ap.add_argument("--keep_checkpoints", action="store_true")
    ap.add_argument("--init", default=None, help="start from runs/wm/dynamics/<init>/model.pt")
    ap.add_argument("--skip_w", type=float, default=0.5, help="weight of the skip-ahead (memory) loss")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--refit_steps", type=int, default=6000)
    sys.path.insert(0, ROOT)
    main(ap.parse_args())
