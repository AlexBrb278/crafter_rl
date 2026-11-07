"""Stage 1 evaluation: how well does the world model dream? (paper/designs/world_model_stage1.md)

On held-out episodes, starting from a real moment with up to 64 real past steps as context, the model
imagines the future using the *real* actions. We compare dream and reality:
  * accuracy vs horizon (1, 5, 15, 50, 100): token accuracy, PSNR of decoded frames, exact inventory,
    each against a "nothing changes" baseline (copy the last real frame / inventory);
  * one-step (teacher-forced) unlock and death prediction: precision / recall per achievement;
  * probes: night (dream brightness follows real daylight), off-screen water memory, crafting
    (inventory after unlocks), rare chain (collect iron, iron tools);
  * side-by-side videos: reality | dream | difference.

  .venv/bin/python methods/08_world_model/evaluate.py uniform rarity
"""
import argparse
import json
import os
import sys

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import ROOT, load_episodes  # noqa: E402
from dynamics import build  # noqa: E402
from tokenizer import CREATURES, Tokenizer, tile_mask  # noqa: E402

HORIZONS = (1, 5, 15, 50, 100)
INVENTORY = ["health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond",
             "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"]


TOKENIZER_OF = {"tokens": "tokenizer"}  # token cache -> tokenizer that made it (others: tokens_X -> tokenizer_X)


def load(run, dev):
    """run: dynamics run name, optionally "name:model_refit.pt". Uses the tokenizer and options it was trained with."""
    name, _, fname = run.partition(":")
    d = os.path.join(ROOT, "runs", "wm", "dynamics", name)
    info = json.load(open(os.path.join(d, "info.json")))
    wm = build(info, dev).eval()
    wm.load_state_dict(th.load(os.path.join(d, fname or "model.pt"), map_location=dev))
    tokens = info["args"].get("tokens", "tokens")
    from cache_tokens import load_tokenizer
    tok = load_tokenizer(TOKENIZER_OF.get(tokens, tokens.replace("tokens", "tokenizer")), dev)
    wm.pos_dir = info["args"].get("pos_dir", "pos")  # where the real-context positions come from
    return wm, tok, os.path.join(ROOT, "runs", "wm", tokens)


ANCHOR_EVERY = 0  # >0: "anchored dreaming", see dream()
SAMPLE_T = 0.0    # >0: dreams sample tokens and inventory at this temperature instead of taking the most likely


@th.no_grad()
def dream(wm, eps, starts, H, dev, context=64):
    """For each (episode, start s): real steps max(0, s-context+1)..s as context, then H imagined steps
    driven by the real actions a_{s+1..}. Returns predicted tokens (N, H, 8, 8), inventory (N, H, 16).
    With ANCHOR_EVERY = n and a skip-ahead model: every n-th imagined frame (up to 32 steps out) is replaced by
    the skip-ahead prediction made from the last *real* state, so long dreams re-anchor to what was really seen."""
    out_tok, out_inv = [], []
    for e, s in zip(eps, starts):
        c0 = max(0, s - context + 1)
        tok = th.from_numpy(e["tokens"][c0:s + 1].astype(np.int64)).to(dev)[None]
        inv = th.from_numpy(e["inventory"][c0:s + 1].astype(np.int64)).to(dev)[None]
        act = th.from_numpy(e["actions"][c0:s + 1].astype(np.int64)).to(dev)[None]
        tm = th.arange(c0, s + 1, device=dev)[None]
        ps = th.from_numpy(e["pos"][c0:s + 1].astype(np.int64)).to(dev)[None]  # true path for the real context
        tmap = None
        if getattr(wm, "use_map", False):  # Stage 1d: the map holds every real frame of the episode up to s
            from build_maps import TileMap
            tmap = TileMap(wm.n_codes)
            for t in range(s + 1):
                tmap.write(e["tokens"][t], e["pos"][t], t)
        pt, pi = [], []
        h_real = None
        for k in range(H):
            with th.autocast("cuda", dtype=th.bfloat16):
                h = wm.history(tok[:, -context:], inv[:, -context:], act[:, -context:], tm[:, -context:],
                               ps[:, -context:])[:, -1].float()
            if k == 0:
                h_real = h
            step = wm.move_delta[wm.odo_head(h).argmax(-1)] if getattr(wm, "use_pos", False) else th.zeros(1, 2, dtype=th.long, device=dev)
            prior = None
            if tmap is not None:  # read the map where the model thinks it will be next
                nxt = (ps[0, -1] + step[0]).cpu().numpy()
                mt, ma = tmap.read(nxt, s + k)
                prior = wm.map_prior(th.from_numpy(mt.astype(np.int64)).to(dev)[None], th.from_numpy(ma.astype(np.int64)).to(dev)[None])
            nt, ni, _, _ = wm.predict_next(h, prior=prior, temperature=SAMPLE_T)
            if tmap is not None:
                tmap.write(nt[0].cpu().numpy(), nxt, s + k + 1)  # the dream writes what it imagines
            if ANCHOR_EVERY and getattr(wm, "use_skip", False) and (k + 1) % ANCHOR_EVERY == 0 and 4 <= k + 1 <= 32:
                fut = th.from_numpy(e["actions"][s + 1:s + 1 + k].astype(np.int64)).to(dev)[None]
                nt = wm.predict_skip(h_real, fut, k + 1)
            pt.append(nt[0]); pi.append(ni[0])
            a_next = int(e["actions"][s + 1 + k]) if s + 1 + k < len(e["actions"]) else 0
            tok = th.cat([tok, nt[:, None]], 1)
            inv = th.cat([inv, ni[:, None]], 1)
            act = th.cat([act, th.tensor([[a_next]], device=dev)], 1)
            tm = th.cat([tm, tm[:, -1:] + 1], 1)
            ps = th.cat([ps, ps[:, -1:] + step[:, None]], 1)  # imagined steps: the model's own odometry
        out_tok.append(th.stack(pt)); out_inv.append(th.stack(pi))
    return th.stack(out_tok), th.stack(out_inv)


def psnr(a, b):
    """PSNR of the mean squared error over all frames. (A per-frame PSNR averaged afterwards lets frames
    that happen to be reconstructed perfectly count as ~138 dB and inflate the mean.)"""
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return np.float64(10 * np.log10(255 ** 2 / max(mse, 1e-9)))


def water_pixels(frames):
    """Share of the world area (top 7 of 9 rows) that looks like Crafter water (strong blue)."""
    f = frames[..., :50, :, :].astype(np.int16)  # 64 px * 7/9 ~= 50 rows of world view
    blue = (f[..., 2] > f[..., 0] + 40) & (f[..., 2] > f[..., 1] + 15)
    return blue.mean(axis=(-1, -2))


def brightness(frames):
    return frames[..., :50, :, :].astype(np.float32).mean(axis=(-1, -2, -3))


def main(args):
    global ANCHOR_EVERY
    ANCHOR_EVERY = args.anchor
    dev = th.device("cuda")
    meta = json.load(open(os.path.join(ROOT, "runs", "wm", "data", "meta.json")))
    water_id = meta["mat_ids"]["water"]
    from common.crafter_score import ACHIEVEMENTS
    H = max(HORIZONS)
    raw = load_episodes("test", keys=("frames", "actions", "inventory", "unlocked", "daylight", "sem", "died"),
                        sources=args.test_set)
    results = {}
    for run in args.runs:
        wm, tok, tokens_dir = load(run, dev)
        wm.pos_dir = args.pos_dir or wm.pos_dir
        # same episodes and the same starts for every model (rng reset per model); tokens from this model's tokenizer
        rng = np.random.default_rng(0)
        eps = []
        for e in raw:
            q = os.path.join(tokens_dir, os.path.relpath(e["path"], os.path.join(ROOT, "runs", "wm", "data")))
            qp = os.path.join(ROOT, "runs", "wm", wm.pos_dir, os.path.relpath(e["path"], os.path.join(ROOT, "runs", "wm", "data")))
            pos = np.load(qp)["pos"] if os.path.exists(qp) else np.zeros((len(e["actions"]) + 1, 2), np.int16)
            eps.append({**e, "tokens": np.load(q)["tokens"], "pos": pos})
        long_eps = [e for e in eps if len(e["actions"]) > 64 + H]
        starts = [(e, int(rng.integers(32, len(e["actions"]) - H))) for e in
                  [long_eps[i] for i in rng.choice(len(long_eps), min(args.n, len(long_eps)), replace=False)]]
        R = {"run": run, "test_set": args.test_set}
        # ---- accuracy vs horizon, against "nothing changes"
        pt, pi = dream(wm, [e for e, _ in starts], [s for _, s in starts], H, dev)
        pt, pi = pt.cpu().numpy(), pi.cpu().numpy()
        real_t = np.stack([e["tokens"][s + 1:s + 1 + H] for e, s in starts]).astype(np.int64)
        real_i = np.stack([e["inventory"][s + 1:s + 1 + H] for e, s in starts]).astype(np.int64)
        real_f = np.stack([e["frames"][s + 1:s + 1 + H] for e, s in starts])
        last_t = np.stack([e["tokens"][s] for e, s in starts]).astype(np.int64)
        last_i = np.stack([e["inventory"][s] for e, s in starts]).astype(np.int64)
        last_f = np.stack([e["frames"][s] for e, s in starts])
        with th.no_grad():
            dream_f = np.stack([tok.decode(th.from_numpy(x).to(dev)).cpu().numpy() for x in pt])
            # fair pixel baselines go through the same tokenizer: "nothing changes" = the last real frame as the
            # tokenizer draws it; "ceiling" = the real future frame as the tokenizer draws it (best a dream can do)
            last_dec = tok.decode(th.from_numpy(last_t).to(dev)).cpu().numpy()
            real_dec = np.stack([tok.decode(th.from_numpy(x).to(dev)).cpu().numpy() for x in real_t])
        hz = {}
        for h in HORIZONS:
            k = h - 1
            hz[h] = {
                "token_acc": float((pt[:, k] == real_t[:, k]).mean()),
                "token_acc_copy": float((last_t == real_t[:, k]).mean()),
                "psnr": float(psnr(dream_f[:, k], real_f[:, k]).mean()),
                "psnr_copy": float(psnr(last_dec, real_f[:, k]).mean()),
                "psnr_copy_raw": float(psnr(last_f, real_f[:, k]).mean()),
                "psnr_ceiling": float(psnr(real_dec[:, k], real_f[:, k]).mean()),
                "inv_exact": float((pi[:, k] == real_i[:, k]).all(-1).mean()),
                "inv_exact_copy": float((last_i == real_i[:, k]).all(-1).mean()),
                "inv_slot_acc": float((pi[:, k] == real_i[:, k]).mean()),
                # materials and tools only: food/drink/energy (and health) change on hidden timers
                "items_exact": float((pi[:, k, 4:] == real_i[:, k, 4:]).all(-1).mean()),
                "items_exact_copy": float((last_i[:, 4:] == real_i[:, k, 4:]).all(-1).mean()),
            }
        R["horizon"] = hz
        # creatures inside the dream: PSNR on tiles that really hold a creature, 1 and 5 steps ahead
        R["creature_psnr"] = {}
        for h in (1, 5):
            sem = np.stack([e["sem"][s + h] for e, s in starts])
            m = tile_mask(sem, CREATURES)
            if m.any():
                se = ((dream_f[:, h - 1].astype(np.float32) - real_f[:, h - 1].astype(np.float32)) ** 2).mean(-1)
                se_c = ((last_dec.astype(np.float32) - real_f[:, h - 1].astype(np.float32)) ** 2).mean(-1)
                R["creature_psnr"][h] = {"dream": float(10 * np.log10(255 ** 2 / se[m].mean())),
                                         "copy": float(10 * np.log10(255 ** 2 / se_c[m].mean()))}
        # ---- probe 1: night (dream brightness follows real brightness over the rollout)
        b_dream, b_real = brightness(dream_f), brightness(real_f)
        R["night_corr"] = float(np.mean([np.corrcoef(b_dream[i], b_real[i])[0, 1] for i in range(len(starts))
                                         if b_real[i].std() > 5]))
        # ---- one-step unlock / death prediction (teacher forced) on all test steps
        tp, fp, fn = np.zeros(22), np.zeros(22), np.zeros(22)
        inv_after_unlock = []
        with th.no_grad():
            for e in eps:
                T = len(e["actions"])
                for c0 in range(0, T, 64):
                    sl = slice(c0, min(c0 + 64, T))
                    t_ = th.from_numpy(e["tokens"][sl].astype(np.int64)).to(dev)[None]
                    i_ = th.from_numpy(e["inventory"][sl].astype(np.int64)).to(dev)[None]
                    a_ = th.from_numpy(e["actions"][sl].astype(np.int64)).to(dev)[None]
                    tm_ = th.arange(sl.start, sl.stop, device=dev)[None]
                    p_ = th.from_numpy(e["pos"][sl].astype(np.int64)).to(dev)[None]
                    with th.autocast("cuda", dtype=th.bfloat16):
                        hh = wm.history(t_, i_, a_, tm_, p_)[0].float()
                    inv_l, ach_l, _ = wm.heads(hh)
                    pred = (ach_l.sigmoid() > 0.5).cpu().numpy()
                    ev = e["unlocked"][sl.start + 1:sl.stop + 1] & ~e["unlocked"][sl.start:sl.stop]
                    tp += (pred & ev).sum(0); fp += (pred & ~ev).sum(0); fn += (~pred & ev).sum(0)
                    inv_p = inv_l.argmax(-1).cpu().numpy()
                    for j in np.where(ev.any(1))[0]:
                        inv_after_unlock.append(bool((inv_p[j] == e["inventory"][sl.start + j + 1]).all()))
        R["unlock"] = {a: {"precision": float(tp[i] / max(tp[i] + fp[i], 1)), "recall": float(tp[i] / max(tp[i] + fn[i], 1)),
                           "events": int(tp[i] + fn[i])} for i, a in enumerate(ACHIEVEMENTS)}
        R["unlock_recall_all"] = float(tp.sum() / max(tp.sum() + fn.sum(), 1))
        R["unlock_precision_all"] = float(tp.sum() / max(tp.sum() + fp.sum(), 1))
        R["inv_exact_after_unlock"] = float(np.mean(inv_after_unlock)) if inv_after_unlock else None
        # ---- probe 2: off-screen water memory (water in view at s, gone for g >= 5 steps, back at s + g + ...)
        mem = []
        for e in eps:
            w = (e["sem"] == water_id).any(axis=(1, 2))
            T = len(e["actions"])
            t = 0
            while t < T - 1 and len(mem) < args.n:
                if w[t] and not w[t + 1]:
                    back = next((u for u in range(t + 1, min(t + 60, T)) if w[u]), None)
                    if back is not None and back - t >= 5:
                        mem.append((e, t, back - t)); t = back; continue
                t += 1
        if mem:
            md_t, _ = dream(wm, [m[0] for m in mem], [m[1] for m in mem], max(m[2] for m in mem), dev)
            hit, det_ok = [], []
            for (e, s, g), seq in zip(mem, md_t):
                f = tok.decode(seq[g - 1:g]).cpu().numpy()
                hit.append(water_pixels(f)[0] > 0.01)
                det_ok.append(water_pixels(e["frames"][s + g:s + g + 1])[0] > 0.01)
            R["water_memory"] = {"cases": len(mem), "dream_shows_water": float(np.mean(hit)),
                                 "detector_on_real": float(np.mean(det_ok)),
                                 "gap_median": float(np.median([m[2] for m in mem]))}
            if wm.use_skip:  # the skip-ahead output, asked directly for the frame g steps later (g <= 32)
                sk = []
                for (e, s, g) in mem:
                    if g > 32:
                        continue
                    c0 = max(0, s - 63)
                    ten = lambda x: th.from_numpy(x.astype(np.int64)).to(dev)[None]
                    with th.no_grad(), th.autocast("cuda", dtype=th.bfloat16):
                        h = wm.history(ten(e["tokens"][c0:s + 1]), ten(e["inventory"][c0:s + 1]), ten(e["actions"][c0:s + 1]),
                                       th.arange(c0, s + 1, device=dev)[None], ten(e["pos"][c0:s + 1]))[:, -1].float()
                    fut = ten(e["actions"][s + 1:s + g])
                    grid = wm.predict_skip(h, fut, g)
                    sk.append(water_pixels(tok.decode(grid).cpu().numpy())[0] > 0.01)
                R["water_memory"]["skip_shows_water"] = float(np.mean(sk)) if sk else None
                R["water_memory"]["skip_cases"] = len(sk)
        # ---- probe 4: rare chain (unlocks of collect_iron / iron tools in the test set, one-step)
        R["rare_chain"] = {a: R["unlock"][a] for a in ("collect_iron", "make_iron_pickaxe", "make_iron_sword", "collect_diamond")}
        # ---- videos: reality | dream | difference
        if args.videos:
            from common.video import write_mp4
            from PIL import Image, ImageDraw, ImageFont
            vid_dir = os.path.join(ROOT, "runs", "wm", "eval", run.replace(":", "_").replace(".pt", ""))
            os.makedirs(vid_dir, exist_ok=True)
            font = ImageFont.load_default(size=13)
            short = [("health", "H"), ("food", "F"), ("drink", "D"), ("energy", "E"), ("wood", "wood"), ("stone", "stone"),
                     ("coal", "coal"), ("iron", "iron"), ("stone_pickaxe", "st.pick"), ("iron_pickaxe", "ir.pick")]
            fmt = lambda inv: "  ".join(f"{n} {inv[INVENTORY.index(k)]}" for k, n in short)
            for v in range(min(args.videos, len(starts))):
                frames = []
                for k in range(H):
                    r, d = real_f[v, k], dream_f[v, k]
                    diff = np.abs(r.astype(np.int16) - d.astype(np.int16)).clip(0, 255).astype(np.uint8)
                    gap = np.full((64, 2, 3), 255, np.uint8)
                    row = np.kron(np.concatenate([r, gap, d, gap, diff], 1), np.ones((4, 4, 1), np.uint8))
                    img = Image.new("RGB", (row.shape[1], row.shape[0] + 62), (20, 20, 24))
                    img.paste(Image.fromarray(row), (0, 26))
                    dr = ImageDraw.Draw(img)
                    for j, label in enumerate(("REALITY", f"DREAM ({run} model)", "DIFFERENCE")):
                        dr.text((j * 264 + 6, 5), label, fill=(235, 235, 235), font=font)
                    dr.text((row.shape[1] - 120, 5), f"{k + 1} steps ahead", fill=(255, 220, 90), font=font)
                    y = row.shape[0] + 30
                    dr.text((6, y), "real:  " + fmt(real_i[v, k]), fill=(170, 200, 255), font=font)
                    ok = (pi[v, k] == real_i[v, k]).all()
                    dr.text((6, y + 16), "model: " + fmt(pi[v, k]), fill=(150, 230, 150) if ok else (255, 150, 120), font=font)
                    frames.append(np.asarray(img))
                write_mp4(os.path.join(vid_dir, f"dream_{v:02d}.mp4"), frames, fps=6)
        results[run] = R
        print(json.dumps({"run": run, "horizon": hz, "creature_psnr": R["creature_psnr"], "night_corr": R["night_corr"], "unlock_recall": R["unlock_recall_all"],
                          "unlock_precision": R["unlock_precision_all"], "inv_after_unlock": R["inv_exact_after_unlock"],
                          "water_memory": R.get("water_memory"),
                          "rare_chain": {a: (round(x["recall"], 2), x["events"]) for a, x in R["rare_chain"].items()}},
                         indent=1), flush=True)
    out = os.path.join(ROOT, "results", "world_model", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=200, help="rollouts / memory cases per run")
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--test_set", default="stage1", help="stage1 (fixed, comparable) | b | all")
    ap.add_argument("--out", default="stage1_eval.json")
    ap.add_argument("--pos_dir", default=None, help="override where real-context positions come from (pos | pos_pix)")
    ap.add_argument("--anchor", type=int, default=0, help="anchored dreaming: re-anchor every n steps (skip models)")
    sys.path.insert(0, ROOT)
    main(ap.parse_args())
