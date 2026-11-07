"""Evaluate a trained agent (or the random agent) on fresh Crafter episodes.

    python evaluate.py runs/ad-rarity2-s0/agent.pt --episodes 200
    python evaluate.py --random --episodes 1000
    python evaluate.py runs/ad-rarity2-s0/agent.pt --video episode.mp4   # also film one episode

The policy samples actions, as during training. Each env plays the same number of episodes,
so short episodes are not over-represented.
"""
import argparse
import json

import numpy as np
import torch as th
import torch.nn.functional as F

from envs import ACHIEVEMENTS, NUM_ACTIONS, VecCrafter, crafter_score
from model import Agent


@th.no_grad()
def evaluate(model, episodes=100, num_envs=8, seed=10_000, device="cpu"):
    """model=None plays uniformly random actions. Returns score, reward, length, success rates."""
    venv = VecCrafter(num_envs, seed)
    rng = np.random.RandomState(seed)
    quota = -(-episodes // num_envs)
    done_eps = [[] for _ in range(num_envs)]
    to_t = lambda o: th.from_numpy(o).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    obs = to_t(venv.reset())
    if model is not None:
        model.eval()
        states = th.zeros(num_envs, model.hidsize, device=device)
    prev = th.zeros(num_envs, 22, dtype=th.bool, device=device)
    while min(len(d) for d in done_eps) < quota:
        if model is None:
            actions = rng.randint(NUM_ACTIONS, size=num_envs)
        else:
            out = model.act(obs, states)
            actions = out["actions"].squeeze(-1).cpu().numpy()
        o, _, dones, infos = venv.step(actions)
        obs = to_t(o)
        succ = th.tensor([[info["achievements"][a] > 0 for a in ACHIEVEMENTS] for info in infos], device=device)
        if model is not None and model.ad:  # same achievement-state update as in training
            new = (succ != prev).any(-1, keepdim=True)
            if new.any():
                states = th.where(new, F.normalize(model.encode(obs) - out["latents"], dim=-1), states)
        d = th.from_numpy(dones).to(device).unsqueeze(-1)
        if model is not None:
            states = th.where(d, 0, states)
        prev = th.where(d, False, succ)
        for i, info in enumerate(infos):
            if "episode" in info and len(done_eps[i]) < quota:
                done_eps[i].append({"reward": info["episode"]["r"], "length": info["episode"]["l"],
                                    "achievements": [a for a in ACHIEVEMENTS if info["achievements"][a] > 0]})
    venv.close()
    eps = [e for d in done_eps for e in d]
    rates = [100 * np.mean([a in e["achievements"] for e in eps]) for a in ACHIEVEMENTS]
    return {"episodes": len(eps), "score": crafter_score(rates),
            "reward_mean": float(np.mean([e["reward"] for e in eps])),
            "reward_std": float(np.std([e["reward"] for e in eps])),
            "length_mean": float(np.mean([e["length"] for e in eps])),
            "success_rate": dict(zip(ACHIEVEMENTS, rates))}


@th.no_grad()
def record_video(model, path, seed=0, size=384, device="cpu"):
    """Play one episode in-process and save it as an MP4 (needs imageio + imageio-ffmpeg)."""
    import crafter
    import imageio
    env = crafter.Env(seed=seed)
    obs, done, frames, ret = env.reset(), False, [], 0.0
    states = th.zeros(1, model.hidsize, device=device)
    prev = set()
    model.eval()
    while not done:
        frames.append(env.render((size, size)))
        x = th.from_numpy(obs).to(device).permute(2, 0, 1)[None].float() / 255.0
        out = model.act(x, states)
        obs, r, done, info = env.step(int(out["actions"].item()))
        ret += r
        got = {a for a, c in info["achievements"].items() if c > 0}
        if model.ad and got - prev:
            nxt = th.from_numpy(obs).to(device).permute(2, 0, 1)[None].float() / 255.0
            states = F.normalize(model.encode(nxt) - out["latents"], dim=-1)
        prev = got
    imageio.mimwrite(path, frames, fps=15, macro_block_size=8)
    return {"length": len(frames), "return": ret, "achievements": sorted(prev)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", help="agent.pt written by train.py")
    p.add_argument("--random", action="store_true", help="evaluate the uniformly random agent")
    p.add_argument("--no_ad", action="store_true", help="the checkpoint is a plain PPO network")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--video", default="", help="also save one episode as this MP4")
    p.add_argument("--out", default="", help="write the result as JSON here")
    args = p.parse_args()
    dev = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = None
    if not args.random:
        model = Agent(ad=not args.no_ad).to(dev)
        model.load_state_dict(th.load(args.checkpoint, map_location=dev))
    res = evaluate(model, args.episodes, args.num_envs, args.seed, dev)
    if args.video and model is not None:
        res["video"] = record_video(model, args.video, seed=args.seed, device=dev)
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
