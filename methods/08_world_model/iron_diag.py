"""Why do so few iron-in-hand dreams end in an iron tool? Play the final weights of a dream run in the real
game and look at every moment with iron in hand and an iron tool missing (dream_env.iron_in_hand):

  funnel   what those moments have: wood, coal, stone, a table / furnace in reach (3 x 3, the craft rule) or
           in view, everything for the craft ("craftable"); and the policy's p(make iron tool) there
  futures  from a sample of them (t >= 63), K real futures of H steps (copies of the game) and K dreams of H
           steps (v4 + craft layer, as in training), same policy: tools made, tables / furnaces placed,
           actions taken. A gap between the two is the dream's fault; no tools in either is the policy's.

  .venv/bin/python methods/08_world_model/iron_diag.py ad-rarity2-dream-iron-s0 ad-rarity2-dream-iron-s1 ...
  ... --near 4 --p_start 0.3 --H 32 --out iron_diag_ready.json   # ready-to-craft starts only
  ... --reach --p_start 1 --episodes 600 --out iron_diag_reach.json   # craftable starts only

Output: results/world_model/iron_diag.json
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
AD = os.path.join(ROOT, "methods", "05_achievement_distillation")
sys.path[:0] = [HERE, AD, ROOT, os.path.join(AD, "upstream"), os.path.join(ROOT, "scripts")]
import gymnasium  # noqa: E402

sys.modules.setdefault("gym", gymnasium)
from dream_env import CONTEXT, INVENTORY, Dreamer, Track  # noqa: E402

ACTIONS = ["noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep", "place_stone", "place_table",
           "place_furnace", "place_plant", "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
           "make_wood_sword", "make_stone_sword", "make_iron_sword"]
MAKE_IP, MAKE_IS = ACTIONS.index("make_iron_pickaxe"), ACTIONS.index("make_iron_sword")


def inv_vec(inv):
    return np.array([inv[k] for k in INVENTORY]).clip(0, 255).astype(np.uint8)


def look(env):
    """Real-game facts at this moment: inventory, stations in reach (3 x 3) and in view (9 x 7)."""
    p, w = env._player, env._world
    near, _ = w.nearby(p.pos, 1)
    x, y = p.pos
    view = set(w._mat_names[i] for i in np.unique(w._mat_map[max(x - 4, 0):x + 5, max(y - 3, 0):y + 4]))
    inv = p.inventory
    f = {"wood": inv["wood"] >= 1, "coal": inv["coal"] >= 1, "stone4": inv["stone"] >= 4, "wood2": inv["wood"] >= 2,
         "table_reach": "table" in near, "furnace_reach": "furnace" in near,
         "table_view": "table" in view, "furnace_view": "furnace" in view}
    f["craftable"] = f["wood"] and f["coal"] and f["table_reach"] and f["furnace_reach"]
    for name in ("table", "furnace"):  # Chebyshev distance to the nearest one in the world (99 = none placed)
        xs, ys = np.where(w._mat_map == w._mat_ids[name])
        f[name + "_dist"] = int(np.max(np.abs(np.stack([xs - x, ys - y])), 0).min()) if len(xs) else 99
    return f


def in_hand(inv):
    return inv["iron"] >= 1 and (inv["iron_pickaxe"] == 0 or inv["iron_sword"] == 0)


def ready(f, near):
    """A "ready-to-craft" moment: wood and coal in hand, a table and a furnace both within `near` tiles."""
    return f["wood"] and f["coal"] and max(f["table_dist"], f["furnace_dist"]) <= near


def play(run, model, device, episodes, n_envs, per_ep, max_starts, rng, model_hid, near=None, p_start=0.05,
         reach=False):
    """Real episodes with the final weights. Returns funnel rows and start snapshots (only ready-to-craft
    moments with `near`, only craftable ones with `reach`)."""
    import crafter
    rows, starts = [], []
    envs, obs, states, hist, ep_starts = [None] * n_envs, [None] * n_envs, [None] * n_envs, [None] * n_envs, [0] * n_envs
    begun = 0

    def start(i):
        nonlocal begun
        env = crafter.Env(seed=int(rng.integers(2**31 - 1)))
        o = env.reset()
        envs[i], obs[i], states[i] = env, o, th.zeros(model_hid, device=device)
        inv = inv_vec(env._player.inventory)
        hist[i] = {"frames": [o], "inv": [inv], "act": []}
        ep_starts[i] = 0
        begun += 1

    for i in range(n_envs):
        start(i)
    active, done_n = list(range(n_envs)), 0
    while active:
        x = th.from_numpy(np.stack([obs[i] for i in active])).to(device).permute(0, 3, 1, 2).float() / 255.0
        with th.no_grad():
            out = model.act(x, states=th.stack([states[i] for i in active]))
        probs = out["pi_logits"].float().softmax(-1)
        acts = out["actions"].squeeze(-1).cpu().numpy()
        nxt = []
        for j, i in enumerate(active):
            env, p = envs[i], envs[i]._player
            if in_hand(p.inventory):
                f = look(env)
                f.update(t=env._step, run=run, pressed=int(acts[j]) in (MAKE_IP, MAKE_IS),
                         p_make=float(probs[j, MAKE_IP] + probs[j, MAKE_IS]))
                rows.append(f)
                t = env._step
                if (t >= CONTEXT - 1 and ep_starts[i] < per_ep and len(starts) < max_starts
                        and (near is None or ready(f, near)) and (not reach or f["craftable"])
                        and rng.random() < p_start):
                    starts.append({"run": run, "env": copy.deepcopy(env), "obs": obs[i].copy(),
                                   "state": states[i].clone(), "t": t, "frames": list(hist[i]["frames"]),
                                   "inv": list(hist[i]["inv"]), "act": list(hist[i]["act"]),
                                   "unlocked": {k for k, v in p.achievements.items() if v > 0}, "look": f})
                    ep_starts[i] += 1
            prev = {k for k, v in p.achievements.items() if v > 0}
            o, r, done, info = env.step(int(acts[j]))
            obs[i] = o
            hist[i]["frames"].append(o)
            hist[i]["inv"].append(inv_vec(info["inventory"]))
            hist[i]["act"].append(int(acts[j]))
            if {k for k, v in info["achievements"].items() if v > 0} - prev:
                with th.no_grad():
                    lat = model.encode(th.from_numpy(o).to(device).permute(2, 0, 1)[None].float() / 255.0)
                states[i] = F.normalize(lat[0] - out["latents"][j], dim=-1)
            if not done:
                nxt.append(i)
                continue
            done_n += 1
            if begun < episodes:
                start(i)
                nxt.append(i)
        active = nxt
    return rows, starts


def summary_of(fut, H):
    """fut: list of per-rollout dicts -> rates."""
    n = max(len(fut), 1)
    by = {f"made_by_{h}": sum(0 <= f["made_at"] < h for f in fut) / n for h in (4, 8, 16, 32, 64) if h <= H}
    acts = np.sum([f["acts"] for f in fut], 0) if fut else np.zeros(len(ACTIONS))
    return {"rollouts": len(fut),
            "iron_tool_made": sum(f["made"] for f in fut) / n,
            "table_placed": sum(f["table"] for f in fut) / n,
            "furnace_placed": sum(f["furnace"] for f in fut) / n,
            "died": sum(f["died"] for f in fut) / n,
            "make_iron_pressed": sum(f["pressed"] for f in fut) / n,
            "pressed_when_craftable": sum(f["pressed_ok"] for f in fut) / max(sum(f["craftable_steps"] for f in fut), 1),
            "craftable_any": sum(f["craftable_steps"] > 0 for f in fut) / n, **by,
            "actions": {a: round(float(c) / max(acts.sum(), 1), 4) for a, c in zip(ACTIONS, acts)}}


def real_futures(model, starts, K, H, device):
    fut = []
    jobs = [(s, copy.deepcopy(s["env"])) for s in starts for _ in range(K)]
    obs = [s["obs"] for s, _ in jobs]
    st = th.stack([s["state"] for s, _ in jobs])
    rec = [{"made": False, "table": False, "furnace": False, "died": False, "pressed": False, "pressed_ok": 0,
            "craftable_steps": 0, "made_at": -1, "acts": np.zeros(len(ACTIONS)), "start": k // K} for k in range(len(jobs))]
    alive = np.ones(len(jobs), bool)
    for h in range(H):
        x = th.from_numpy(np.stack(obs)).to(device).permute(0, 3, 1, 2).float() / 255.0
        with th.no_grad():
            out = model.act(x, states=st)
        acts = out["actions"].squeeze(-1).cpu().numpy()
        for k, (s, env) in enumerate(jobs):
            if not alive[k]:
                continue
            r = rec[k]
            a = int(acts[k])
            ok = look(env)["craftable"]
            r["craftable_steps"] += ok
            r["acts"][a] += 1
            if a in (MAKE_IP, MAKE_IS):
                r["pressed"] = True
                r["pressed_ok"] += ok
            before = dict(env._player.inventory)
            prev = {kk for kk, v in env._player.achievements.items() if v > 0}
            o, _, done, info = env.step(a)
            inv = info["inventory"]
            r["made"] |= inv["iron_pickaxe"] > before["iron_pickaxe"] or inv["iron_sword"] > before["iron_sword"]
            if r["made"] and r["made_at"] < 0:
                r["made_at"] = h
            r["table"] |= a == ACTIONS.index("place_table") and inv["wood"] < before["wood"]
            r["furnace"] |= a == ACTIONS.index("place_furnace") and inv["stone"] < before["stone"]
            obs[k] = o
            if {kk for kk, v in info["achievements"].items() if v > 0} - prev:
                with th.no_grad():
                    lat = model.encode(th.from_numpy(o).to(device).permute(2, 0, 1)[None].float() / 255.0)
                st[k] = F.normalize(lat[0] - out["latents"][k], dim=-1)
            if done:
                r["died"] = info["discount"] == 0
                alive[k] = False
    return rec


def track_from(s, tok, device):
    from common.pixel_map import PixelMap
    frames = np.stack(s["frames"])
    with th.no_grad():
        codes = th.cat([tok.encode(th.from_numpy(frames[i:i + 512]).to(device)).cpu()
                        for i in range(0, len(frames), 512)]).numpy().astype(np.int16)
    m = PixelMap()
    m.reset_episode()
    m.observe(frames[0])
    tr = Track()
    tr.add(codes[0], s["inv"][0], 0, m.pos.copy())
    for t, a in enumerate(s["act"]):
        m.observe(frames[t + 1], a)
        tr.add(codes[t + 1], s["inv"][t + 1], t + 1, m.pos.copy())
    tr.act = list(s["act"]) + [-1]
    return tr


def dream_futures(model, wm, tok, craft, starts, K, H, device):
    tracks = [track_from(s, tok, device) for s in starts]
    D = len(starts) * K
    dv = Dreamer(wm, tok, D, device, temperature=1.0, craft=craft)
    dv.load(range(D), [(tracks[k // K], len(tracks[k // K].tok) - 1) for k in range(D)])
    obs = th.from_numpy(np.stack([starts[k // K]["obs"] for k in range(D)])).to(device)
    st = th.stack([starts[k // K]["state"] for k in range(D)])
    rec = [{"made": False, "table": False, "furnace": False, "died": False, "pressed": False, "pressed_ok": 0,
            "craftable_steps": 0, "made_at": -1, "acts": np.zeros(len(ACTIONS)), "start": k // K} for k in range(D)]
    alive = th.ones(D, dtype=th.bool, device=device)
    i = INVENTORY.index
    for h in range(H):
        x = obs.permute(0, 3, 1, 2).float() / 255.0
        with th.no_grad():
            out = model.act(x, states=st)
        a = out["actions"].squeeze(-1)
        cur = dv.inv[:, -1]
        win = dv.tokens[:, -1, 2:5, 3:6].reshape(D, -1)
        reach = craft.station[:, win].any(-1).T  # (D, 2): table, furnace in the 3 x 3
        ok = (cur[:, i("wood")] >= 1) & (cur[:, i("coal")] >= 1) & reach[:, 0] & reach[:, 1]
        frames, inv, _, done_p = dv.step(a)
        made = (inv[:, i("iron_pickaxe")] > cur[:, i("iron_pickaxe")]) | (inv[:, i("iron_sword")] > cur[:, i("iron_sword")])
        table = (a == ACTIONS.index("place_table")) & (inv[:, i("wood")] < cur[:, i("wood")])
        furnace = (a == ACTIONS.index("place_furnace")) & (inv[:, i("stone")] < cur[:, i("stone")])
        dead = th.bernoulli(done_p).bool() | (inv[:, 0] <= 0)
        an, okn, alv = a.cpu().numpy(), ok.cpu().numpy(), alive.cpu().numpy()
        for k in np.where(alv)[0]:
            r = rec[k]
            r["acts"][an[k]] += 1
            r["craftable_steps"] += int(okn[k])
            if an[k] in (MAKE_IP, MAKE_IS):
                r["pressed"] = True
                r["pressed_ok"] += int(okn[k])
            r["made"] |= bool(made[k])
            if r["made"] and r["made_at"] < 0:
                r["made_at"] = h
            r["table"] |= bool(table[k])
            r["furnace"] |= bool(furnace[k])
            if dead[k]:
                r["died"] = True
        # the AD state follows the unlocks the dream pays (collect / make: count rises; places: count falls)
        rose = (inv[:, 5:16] > cur[:, 5:16]).any(-1) | table | furnace
        new = rose & alive
        if new.any():
            with th.no_grad():
                lat = model.encode(frames[new].permute(0, 3, 1, 2).float() / 255.0)
            st[new] = F.normalize(lat - out["latents"][new], dim=-1)
        alive &= ~dead
        obs = frames
    return rec


def main(args):
    from diagnose_deaths import load_model
    from evaluate import load
    from craft_layer import load_layer
    dev = th.device("cuda")
    th.manual_seed(0)
    rng = np.random.default_rng(args.seed)
    wm, tok, _ = load(args.wm, dev)
    craft = load_layer(args.wm, dev)
    R = {"wm": args.wm, "K": args.K, "H": args.H, "near": args.near, "reach": args.reach, "runs": {}}
    all_rows, all_real, all_dream, all_looks = [], [], [], []
    for run in args.runs:
        t0 = time.time()
        model, cfg = load_model(run, args.ckpt)
        model.to(dev)
        rows, starts = play(run, model, dev, args.episodes, args.envs, args.per_ep, args.starts, rng,
                            cfg["model_kwargs"]["hidsize"], args.near, args.p_start, args.reach)
        real = real_futures(model, starts, args.K, args.H, dev) if starts else []
        dream = dream_futures(model, wm, tok, craft, starts, args.K, args.H, dev) if starts else []
        R["runs"][run] = {"in_hand_steps": len(rows), "starts": len(starts), "funnel": funnel(rows),
                          "start_funnel": funnel([s["look"] for s in starts]),
                          "real": summary_of(real, args.H), "dream": summary_of(dream, args.H)}
        all_rows += rows
        all_real += real
        all_dream += dream
        all_looks += [s["look"] for s in starts]
        print(run, f"{time.time() - t0:.0f} s", json.dumps({k: v for k, v in R["runs"][run].items()
                                                              if k in ("in_hand_steps", "starts", "funnel")}), flush=True)
        print("  real ", json.dumps({k: v for k, v in R["runs"][run]["real"].items() if k != "actions"}))
        print("  dream", json.dumps({k: v for k, v in R["runs"][run]["dream"].items() if k != "actions"}), flush=True)
    R["all"] = {"in_hand_steps": len(all_rows), "funnel": funnel(all_rows), "start_funnel": funnel(all_looks),
                "real": summary_of(all_real, args.H), "dream": summary_of(all_dream, args.H)}
    out = os.path.join(ROOT, "results", "world_model", args.out)
    json.dump(R, open(out, "w"), indent=1, default=float)
    print(json.dumps(R["all"], indent=1, default=float))
    print("wrote", out)


def funnel(rows):
    if not rows:
        return {}
    keys = ["wood", "wood2", "coal", "stone4", "table_view", "furnace_view", "table_reach", "furnace_reach", "craftable"]
    f = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    f["both_reach"] = round(float(np.mean([r["table_reach"] and r["furnace_reach"] for r in rows])), 4)
    f["items_ok"] = round(float(np.mean([r["wood"] and r["coal"] for r in rows])), 4)
    pm = [r["p_make"] for r in rows if r.get("p_make") is not None]
    cr = [r for r in rows if r["craftable"] and r.get("p_make") is not None]
    f["p_make_mean"] = round(float(np.mean(pm)), 4) if pm else None
    f["p_make_when_craftable"] = round(float(np.mean([r["p_make"] for r in cr])), 4) if cr else None
    f["craftable_steps"] = len(cr)
    for name in ("table", "furnace"):
        d = np.array([r[name + "_dist"] for r in rows])
        f[name + "_dist"] = {"none": round(float((d == 99).mean()), 4), "le1": round(float((d <= 1).mean()), 4),
                             "le4": round(float((d <= 4).mean()), 4), "le10": round(float((d <= 10).mean()), 4),
                             "median": float(np.median(d))}
    both = np.array([max(r["table_dist"], r["furnace_dist"]) for r in rows])
    f["both_within_4"] = round(float((both <= 4).mean()), 4)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--wm", default="v4:model_refit.pt")
    ap.add_argument("--ckpt", default="agent-e250.pt")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--per_ep", type=int, default=3, help="most start snapshots per episode")
    ap.add_argument("--starts", type=int, default=60, help="start snapshots per run")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--near", type=int, default=None,
                    help="only ready-to-craft starts: wood and coal in hand, table and furnace within this many tiles")
    ap.add_argument("--p_start", type=float, default=0.05, help="chance an eligible moment becomes a start")
    ap.add_argument("--reach", action="store_true", help="only craftable starts: wood and coal in hand, table and "
                                                         "furnace both in the 3 x 3")
    ap.add_argument("--out", default="iron_diag.json")
    main(ap.parse_args())
