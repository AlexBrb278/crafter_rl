"""Can the world model dream a rare craft? For every real moment where an item count rises (an iron pickaxe
is made, a diamond or iron is mined, ...), start dreams k steps before it (Stage 2a's Dreamer: 64-step
context, map, T = 1 sampling, vitals sampled, item counts most likely), replay the *real* actions and check
whether the dreamed inventory shows the rise by the step it really happened (+ `--slack` steps).
Also records the inventory head's probability that the count rose, at the step it really did.

  .venv/bin/python methods/08_world_model/craft_check.py v4:model_refit.pt --samples 32
  .venv/bin/python methods/08_world_model/craft_check.py v4:model_refit.pt --craft   # with the craft layer

Splits: the world model's own train and test episodes (tokens and offline positions as it was trained),
and "probe": held-out iron-tool episodes from record_probe.py, tokenized and positioned online with the
pixel odometry, as Stage 2a's RealTracker does.

Output: results/world_model/craft_check.json (craft_check_layer.json with --craft; head_p_rise is then the
craft layer's p(works) on make actions)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import DATA, ROOT, load_episodes  # noqa: E402
from dream_env import CONTEXT, INVENTORY, Dreamer, Track  # noqa: E402
from evaluate import load  # noqa: E402

RARE = ("iron_pickaxe", "iron_sword", "diamond")
CONTROLS = ("iron", "stone_pickaxe", "coal")


def probe_episodes(tok, dev):
    """runs/wm/probe/*/ep_*.npz with tokens (tokenizer) and positions (online pixel odometry) added."""
    import glob
    sys.path.insert(0, ROOT)
    from common.pixel_map import PixelMap
    eps = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "wm", "probe", "*", "ep_*.npz"))):
        d = np.load(p)
        e = {k: d[k] for k in ("actions", "inventory")}
        e["path"] = p
        with th.no_grad():
            e["tokens"] = th.cat([tok.encode(th.from_numpy(d["frames"][i:i + 512]).to(dev)).cpu()
                                  for i in range(0, len(d["frames"]), 512)]).numpy().astype(np.int16)
        m = PixelMap()
        m.reset_episode()
        m.observe(d["frames"][0])
        pos = [m.pos.copy()]
        for t, a in enumerate(e["actions"]):
            m.observe(d["frames"][t + 1], int(a))
            pos.append(m.pos.copy())
        e["pos"] = np.stack(pos)
        eps.append(e)
    return eps


def track_of(e):
    tr = Track()
    for t in range(len(e["tokens"])):
        tr.add(e["tokens"][t], e["inventory"][t].clip(0, 255).astype(np.uint8), t, e["pos"][t])
    tr.act = [int(a) for a in e["actions"]] + [-1]
    return tr


def events(eps, items, need, rng, per_item):
    """[(item, episode index, t)]: count of `item` rises from frame t to t + 1, with t >= need."""
    out = []
    for item in items:
        i = INVENTORY.index(item)
        found = [(item, j, int(t)) for j, e in enumerate(eps)
                 for t in np.where(np.diff(e["inventory"][:, i].astype(int)) > 0)[0] if t >= need]
        if per_item and len(found) > per_item:
            found = [found[k] for k in rng.choice(len(found), per_item, replace=False)]
        out += found
    return out


@th.no_grad()
def run(wm, tok, eps, tracks, evs, ks, samples, T, slack, dev, batch=512, craft=None):
    inv_logits = {}
    heads = wm.heads

    def spy(h):  # keep the inventory logits of the last step
        out = heads(h)
        inv_logits["last"] = out[0].float()
        return out
    wm.heads = spy
    res = []
    jobs = [(ev, k) for ev in evs for k in ks if ev[2] - k >= CONTEXT - 1 and ev[2] + slack < len(eps[ev[1]]["actions"])]
    flat = [(ev, k) for ev, k in jobs for _ in range(samples)]
    for b in range(0, len(flat), batch):
        chunk = flat[b:b + batch]
        dv = Dreamer(wm, tok, len(chunk), dev, temperature=T, craft=craft)
        dv.load(range(len(chunk)), [(tracks[ev[1]], ev[2] - k) for ev, k in chunk])
        ii = th.tensor([INVENTORY.index(ev[0]) for ev, _ in chunk], device=dev)
        base = dv.inv[th.arange(len(chunk)), -1, ii].clone()
        steps = max(k for _, k in chunk) + 1 + slack
        hit = th.zeros(len(chunk), dtype=th.bool, device=dev)
        p_true = th.zeros(len(chunk), device=dev)
        for s in range(steps):
            acts = th.tensor([tracks[ev[1]].act[ev[2] - k + s] if ev[2] - k + s < len(eps[ev[1]]["actions"]) else 0
                              for ev, k in chunk], device=dev)
            dv.last_craft_p = None
            _, inv, _, _ = dv.step(acts)
            live = th.tensor([s <= k + slack for _, k in chunk], device=dev)
            hit |= live & (inv[th.arange(len(chunk)), ii] > base)
            # the step on which the count really rose: s == k (action at frame t gives frame t + 1)
            at = th.tensor([s == k for _, k in chunk], device=dev)
            if at.any():
                p = inv_logits["last"].softmax(-1)[th.arange(len(chunk)), ii]  # (N, 10) over counts
                cls = th.arange(p.shape[-1], device=dev)[None]
                rise = (p * (cls > base[:, None]).float()).sum(-1)
                if dv.last_craft_p is not None:  # make actions: the craft layer decides
                    rows, pc = dv.last_craft_p
                    rise[rows] = pc
                p_true = th.where(at, rise, p_true)
        for (ev, k), h, pt in zip(chunk, hit.cpu().numpy(), p_true.cpu().numpy()):
            res.append({"item": ev[0], "ep": ev[1], "t": ev[2], "k": k, "hit": bool(h), "p_rise": float(pt)})
    wm.heads = heads
    return res


def summarise(res, ks):
    out = {}
    for item in sorted({r["item"] for r in res}):
        out[item] = {}
        for k in ks:
            rr = [r for r in res if r["item"] == item and r["k"] == k]
            if rr:
                out[item][f"k{k}"] = {"cases": len({(r['ep'], r['t']) for r in rr}), "dreams": len(rr),
                                      "dream_shows_it": round(float(np.mean([r["hit"] for r in rr])), 3),
                                      "head_p_rise": round(float(np.mean([r["p_rise"] for r in rr])), 3)}
    return out


def main(args):
    dev = th.device("cuda")
    th.manual_seed(0)
    rng = np.random.default_rng(0)
    wm, tok, tokens_dir = load(args.run, dev)
    craft = None
    if args.craft:
        from craft_layer import load_layer
        craft = load_layer(args.run, dev, untied=args.untied)
    R = {"run": args.run, "craft_layer": ("untied" if args.untied else "tied") if args.craft else None, "ks": args.ks, "samples": args.samples, "slack": args.slack, "splits": {}}
    for split in args.splits:
        if split == "probe":
            eps = probe_episodes(tok, dev)
        else:
            eps = load_episodes(split, keys=("actions", "inventory"), extra_dir=tokens_dir, sources="all")
            eps = [e for e in eps if "tokens" in e]
            for e in eps:
                e["pos"] = np.load(os.path.join(ROOT, "runs", "wm", wm.pos_dir, os.path.relpath(e["path"], DATA)))["pos"]
        need = CONTEXT - 1 + max(args.ks)
        evs = events(eps, RARE, need, rng, 0) + (events(eps, CONTROLS, need, rng, args.controls) if split in ("test", "probe") else [])
        if not evs:
            continue
        tracks = {j: track_of(eps[j]) for j in {ev[1] for ev in evs}}
        R["splits"][split] = {"events": [(ev[0], os.path.relpath(eps[ev[1]]["path"], ROOT), ev[2]) for ev in evs]}
        for T in args.temps:
            n = args.samples if T > 0 else 1
            res = run(wm, tok, eps, tracks, evs, args.ks, n, T, args.slack, dev, craft=craft)
            R["splits"][split][f"T{T:g}"] = summarise(res, args.ks)
            R["splits"][split][f"T{T:g}_cases"] = [r for r in res if r["item"] in RARE]
            print(split, f"T={T:g}", json.dumps(R["splits"][split][f"T{T:g}"]))
    name = "craft_check" + ("_layer" + ("_untied" if args.untied else "") if args.craft else "") + ".json"
    out = os.path.join(ROOT, "results", "world_model", name)
    json.dump(R, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 3, 7, 15])
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--controls", type=int, default=100, help="control events per item (test and probe splits)")
    ap.add_argument("--splits", nargs="+", default=["train", "test", "probe"])
    ap.add_argument("--craft", action="store_true", help="dream with the craft layer (craft_layer.py)")
    ap.add_argument("--untied", action="store_true", help="with --craft: the untied layer")
    main(ap.parse_args())
