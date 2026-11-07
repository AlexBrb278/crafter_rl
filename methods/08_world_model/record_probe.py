"""Record held-out iron-tool episodes for craft_check.py: play a saved agent and keep only episodes in which
an iron pickaxe or iron sword is made. Saved under runs/wm/probe/<source>/ (not runs/wm/data, so no world
model trains on them).

  .venv/bin/python methods/08_world_model/record_probe.py ad-rarity2-dream-s0 --episodes 3000 --seed 5000
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import record  # noqa: E402

PROBE = os.path.join(record.ROOT, "runs", "wm", "probe")
IRON_TOOLS = [record.INVENTORY.index(k) for k in ("iron_pickaxe", "iron_sword")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="agent-e250.pt")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=5000)
    args = ap.parse_args()
    sys.path[:0] = [record.AD, record.ROOT, os.path.join(record.AD, "upstream"), os.path.join(record.ROOT, "scripts")]
    import gymnasium
    sys.modules.setdefault("gym", gymnasium)
    import torch as th
    th.set_num_threads(2)
    record.record(args.run, args.ckpt, args.episodes, args.envs, th.device("cuda"), args.seed, out_root=PROBE,
                  keep_if=lambda inv: bool((inv[:, IRON_TOOLS] > 0).any()))
