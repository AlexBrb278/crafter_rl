"""Record a world-model dataset by playing saved agents (Stage 1, paper/designs/world_model_stage1.md).

Each episode is saved as runs/wm/data/<source>/ep_<k>.npz with, for T steps:
  frames    (T+1, 64, 64, 3) uint8   the agent's observation before each step and after the last
  actions   (T,) uint8
  rewards   (T,) float32             the game's reward
  inventory (T+1, 16) uint8          INVENTORY order below
  unlocked  (T+1, 22) bool           achievements unlocked so far in the episode (ACHIEVEMENTS order)
  daylight  (T+1,) float16
  sem       (T+1, 9, 7) uint8        semantic ids of the 9x7 world tiles in view (255 = outside the map)
  died      ()   bool
runs/wm/data/meta.json records the semantic ids, inventory order and sources.

  .venv/bin/python methods/08_world_model/record.py              # Stage 1 data
  .venv/bin/python methods/08_world_model/record.py --plan b     # Stage 1b: ~4,000 more episodes
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AD = os.path.join(ROOT, "methods", "05_achievement_distillation")
OUT = os.path.join(ROOT, "runs", "wm", "data")
INVENTORY = ["health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond",
             "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"]

# (run, checkpoint, episodes): strong, mid-training and early policies, plus two other algorithms
PLAN = (
    [(f"ad-rarity2-s{s}", "agent-e250.pt", 330) for s in range(3)]
    + [(f"ad-rarity2-vit1u-s{s}", "agent-e250.pt", 330) for s in range(3)]
    + [(f"{r}-s{s}", f"agent-e{e:03d}.pt", 160) for r in ("ad-rarity2", "ad-rarity2-vit1u")
       for s in (1, 2) for e in (100, 150)]
    + [(f"{r}-s{s}", "agent-e050.pt", 100) for r in ("ad-rarity2", "ad-rarity2-vit1u") for s in (1, 2)]
    + [("ppo-rarity2-s0", "agent-e250.pt", 140), ("ad-ppo_ad-s0", "agent-e250.pt", 140)]
)

# Stage 1b: about 4,000 more episodes with new seeds, adding agents plan "default" did not use
# (GRU, survival v1, gamma 0.99, plain PPO, other rarity strengths). Saved as <source>_b.
PLAN_B = (
    [(f"ad-rarity2-gru-s{s}", "agent-e250.pt", 250) for s in range(3)]
    + [(f"ad-rarity2-gru-s{s}", "agent-e150.pt", 100) for s in range(3)]
    + [(f"ad-rarity2-vit1-s{s}", "agent-e250.pt", 200) for s in range(3)]
    + [(f"ad-rarity2-vit1-s{s}", "agent-e100.pt", 80) for s in range(3)]
    + [(f"ad-rarity2-s{s}", "agent-e250.pt", 250) for s in range(3)]
    + [(f"ad-rarity2-vit1u-s{s}", "agent-e250.pt", 150) for s in range(3)]
    + [("ppo-rarity2-g99-s0", "agent-e250.pt", 150), ("ppo-rarity2-5090-s0", "agent-e250.pt", 150)]
    + [(r, "agent-e250.pt", 100) for r in ("ppo-s1", "ppo-rarity1-s0", "ppo-rarity4-s0", "ad-rarity1-s0")]
    + [(r, "agent-e050.pt", 100) for r in ("ad-rarity2-gru-s1", "ad-rarity2-vit1-s1")]
)
PLANS = {"default": (PLAN, "", 11), "b": (PLAN_B, "_b", 1000)}


def local_sem(env):
    """Semantic ids of the 9x7 tiles the agent sees, centred on the player (x is the long axis)."""
    sem = env._sem_view()
    x, y = env._player.pos
    out = np.full((9, 7), 255, np.uint8)
    for i, dx in enumerate(range(-4, 5)):
        for j, dy in enumerate(range(-3, 4)):
            u, v = x + dx, y + dy
            if 0 <= u < sem.shape[0] and 0 <= v < sem.shape[1]:
                out[i, j] = sem[u, v]
    return out


def record(run, ckpt, episodes, n_envs, device, seed0, tag="", out_root=OUT, keep_if=None):
    """keep_if: optional fn(inventory (T+1, 16)) -> bool; other episodes are played but not saved."""
    import torch as th
    import torch.nn.functional as F
    from diagnose_deaths import load_model
    import crafter
    from common.crafter_score import ACHIEVEMENTS

    model, cfg = load_model(run, ckpt)
    model.to(device)
    is_ad = cfg["model_cls"] in ("PPOADModel", "PPOADGRUModel")
    gru = cfg["model_cls"] == "PPOADGRUModel"
    hid = cfg["model_kwargs"]["hidsize"]
    src = f"{run}_{ckpt[:-3]}{tag}"
    out_dir = os.path.join(out_root, src)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed0)
    started = done_n = 0
    slots = []

    def snapshot(env):
        inv = env._player.inventory
        return (np.array([inv[k] for k in INVENTORY], np.uint8),
                np.array([env._player.achievements[a] > 0 for a in ACHIEVEMENTS]),
                np.float16(env._world.daylight), local_sem(env))

    def start():
        nonlocal started
        env = crafter.Env(seed=int(rng.integers(2**31 - 1)))
        obs = env.reset()
        inv, unl, day, sem = snapshot(env)
        slot = {"env": env, "obs": obs, "k": started, "state": th.zeros(hid, device=device),
                "h": model.initial_hiddens(1, device)[0] if gru else None,
                "frames": [obs], "actions": [], "rewards": [], "inv": [inv], "unl": [unl], "day": [day], "sem": [sem]}
        started += 1
        return slot

    for _ in range(min(n_envs, episodes)):
        slots.append(start())
    t0 = time.time()
    while slots:
        x = th.from_numpy(np.stack([s["obs"] for s in slots])).to(device).permute(0, 3, 1, 2).float() / 255.0
        kw = {"states": th.stack([s["state"] for s in slots])}
        if gru:
            kw["hiddens"] = th.stack([s["h"] for s in slots])
        with th.no_grad():
            out = model.act(x, **kw)
        acts = out["actions"].squeeze(-1).cpu().numpy()
        keep = []
        for j, s in enumerate(slots):
            env = s["env"]
            prev = s["unl"][-1]
            obs, r, done, info = env.step(int(acts[j]))
            s["obs"] = obs
            inv, unl, day, sem = snapshot(env)
            for key, val in (("frames", obs), ("actions", acts[j]), ("rewards", r), ("inv", inv),
                             ("unl", unl), ("day", day), ("sem", sem)):
                s[key].append(val)
            if gru:
                s["h"] = out["hiddens"][j]
            if is_ad and (unl & ~prev).any():  # AD's goal state, as in training
                with th.no_grad():
                    nxt = model.encode(th.from_numpy(obs).to(device).permute(2, 0, 1)[None].float() / 255.0)
                s["state"] = F.normalize(nxt[0] - out["latents"][j], dim=-1)
            if not done:
                keep.append(s)
                continue
            if keep_if is None or keep_if(np.stack(s["inv"])):
                np.savez_compressed(
                    os.path.join(out_dir, f"ep_{s['k']:05d}.npz"),
                    frames=np.stack(s["frames"]), actions=np.array(s["actions"], np.uint8),
                    rewards=np.array(s["rewards"], np.float32), inventory=np.stack(s["inv"]),
                    unlocked=np.stack(s["unl"]), daylight=np.array(s["day"], np.float16),
                    sem=np.stack(s["sem"]), died=np.array(info["discount"] == 0))
            done_n += 1
            if started < episodes:
                keep.append(start())
        slots = keep
    print(f"{src}: {done_n} episodes in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="default")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--only", nargs="*", help="run names to record (default: whole plan)")
    ap.add_argument("--seed", type=int, default=None, help="default: the plan's own seed base")
    args = ap.parse_args()
    sys.path[:0] = [AD, ROOT, os.path.join(AD, "upstream"), os.path.join(ROOT, "scripts")]
    import gymnasium
    sys.modules.setdefault("gym", gymnasium)
    import crafter
    import torch as th
    th.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    env = crafter.Env()
    env.reset()
    plan, tag, seed_base = PLANS[args.plan]
    meta_path = os.path.join(OUT, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({"inventory": INVENTORY, "mat_ids": env._world._mat_ids,
                 "obj_ids": {c.__name__: i for c, i in env._sem_view._obj_ids.items()}})
    meta.setdefault("plans", {})[args.plan] = plan
    json.dump(meta, open(meta_path, "w"), indent=1)
    for i, (run, ckpt, n) in enumerate(plan):
        if args.only and run not in args.only:
            continue
        record(run, ckpt, n, args.envs, th.device("cuda"), (args.seed or seed_base) + i, tag)


if __name__ == "__main__":
    main()
