"""Spectator: periodically film the *latest checkpoint* of a training run, without touching it.

Every `--every` minutes it loads runs/<run>/checkpoints/latest.pt on the CPU, plays one
fresh episode (new world seed, stochastic policy as in training) for up to `--steps`
steps and writes runs/<run>/videos/spectator_XXX_stepYYYYYYY.mp4 plus a line in
runs/<run>/videos/spectator.jsonl (shown by scripts/dashboard.py). Stops when the run
writes final.json.

  .venv/Scripts/python scripts/spectator.py ad-ppo_ad-s0 --every 15 --steps 600
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AD = os.path.join(ROOT, "methods", "05_achievement_distillation")


def main(args):
    sys.path[:0] = [AD, ROOT, os.path.join(AD, "upstream")]
    import gymnasium
    sys.modules.setdefault("gym", gymnasium)
    import numpy as np
    import torch as th
    from achievement_distillation.model import PPOADModel, PPOModel
    from recurrent import PPOADGRUModel
    from train import record_episode

    th.set_num_threads(args.threads)
    run_dir = os.path.join(ROOT, "runs", args.run)
    vid_dir = os.path.join(run_dir, "videos")
    os.makedirs(vid_dir, exist_ok=True)
    index = os.path.join(vid_dir, "spectator.jsonl")
    n = sum(1 for _ in open(index)) if os.path.exists(index) else 0
    last_epoch = None
    rng = np.random.default_rng()
    while not os.path.exists(os.path.join(run_dir, "final.json")):
        ck_path = os.path.join(run_dir, "checkpoints", "latest.pt")
        if os.path.exists(ck_path):
            try:
                ck = th.load(ck_path, map_location="cpu", weights_only=False)
            except Exception as e:  # being written right now
                print("skip:", e, flush=True)
                time.sleep(30)
                continue
            if ck["epoch"] != last_epoch:
                cfg = json.load(open(os.path.join(run_dir, "config.json")))["config"]
                cls = {"PPOADModel": PPOADModel, "PPOModel": PPOModel, "PPOADGRUModel": PPOADGRUModel}[cfg["model_cls"]]
                model = cls(observation_space=gymnasium.spaces.Box(0, 1, (3, 64, 64)),
                            action_space=gymnasium.spaces.Discrete(17), **cfg["model_kwargs"])
                model.load_state_dict(ck["model"])
                step = int(ck["global_step"])
                name = f"spectator_{n:03d}_step{step:07d}.mp4"
                t = time.time()
                res = record_episode(model, cfg, th.device("cpu"), os.path.join(vid_dir, name),
                                     seed=int(rng.integers(2**31 - 1)), max_steps=args.steps,
                                     label=f"{args.run} checkpoint @ {step:,} steps")
                res.update(file=name, step=step, epoch=int(ck["epoch"]), seconds=round(time.time() - t, 1))
                with open(index, "a") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"{name}: {res['length']} steps, return {res['return']:.1f}, "
                      f"{len(res['achievements'])} achievements ({res['seconds']} s)", flush=True)
                n += 1
                last_epoch = ck["epoch"]
        for _ in range(int(args.every * 6)):
            if os.path.exists(os.path.join(run_dir, "final.json")):
                break
            time.sleep(10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--every", type=float, default=15, help="minutes between clips")
    ap.add_argument("--steps", type=int, default=600, help="max steps per clip")
    ap.add_argument("--threads", type=int, default=2)
    main(ap.parse_args())
