"""Train a Crafter agent: PPO + Achievement Distillation + rarity bonus (our best setting).

    python train.py                        # best configuration, seed 0, 1M steps (~250 x 4096)
    python train.py --seed 1               # another seed
    python train.py --no_ad                # PPO only (+ rarity bonus); --rarity_bonus 0 for plain PPO

Logs go to runs/<run_name>/: metrics.jsonl (one line per update), episodes.jsonl (one line per
finished training episode), eval.json (fresh episodes with the final policy), agent.pt, checkpoint.pt.
The Crafter score is computed over all training episodes, as in Hafner (2022).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch as th

from algo import PPO, Rollout
from envs import ACHIEVEMENTS, RarityBonus, VecCrafter, VitalsBonus, crafter_score
from model import Agent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--seed", type=int, default=0)
    a("--run_name", default=None, help="default: <method>-s<seed>")
    a("--logdir", default="runs")
    # budget and rollouts
    a("--steps", type=int, default=1_000_000, help="environment steps (rounded up to whole rollouts)")
    a("--num_envs", type=int, default=8)
    a("--nstep", type=int, default=512, help="steps per env per rollout (8 x 512 = 4096 per update)")
    # returns
    a("--gamma", type=float, default=0.95)
    a("--gae_lambda", type=float, default=0.65)
    # PPO
    a("--lr", type=float, default=3e-4)
    a("--ppo_epochs", type=int, default=3)
    a("--nbatch", type=int, default=8, help="minibatches per epoch (4096 / 8 = 512 samples)")
    a("--clip", type=float, default=0.2)
    a("--vf_coef", type=float, default=0.5)
    a("--ent_coef", type=float, default=0.01)
    a("--max_grad_norm", type=float, default=0.5)
    a("--hidsize", type=int, default=1024)
    # Achievement Distillation
    a("--no_ad", action="store_true", help="plain PPO network and update (no auxiliary phase)")
    a("--aux_every", type=int, default=8, help="PPO updates between auxiliary phases")
    a("--aux_epochs", type=int, default=6)
    a("--temperature", type=float, default=0.1)
    a("--pi_dist_coef", type=float, default=1.0)
    a("--vf_dist_coef", type=float, default=1.0)
    # reward changes
    a("--rarity_bonus", type=float, default=2.0, help="beta of the rarity bonus (0 = game reward)")
    a("--rarity_window", type=int, default=200, help="episodes used to estimate p_i")
    a("--vitals", type=float, default=0.0, help="alpha of the vitals shaping (ablation; 0 = off)")
    a("--vitals_gamma", type=float, default=1.0, help="gamma inside the vitals shaping term")
    # misc
    a("--micro_batch", type=int, default=0, help="gradient accumulation chunk size for small GPUs (0 = off)")
    a("--eval_episodes", type=int, default=100, help="fresh episodes with the final policy (0 = skip)")
    a("--recent", type=int, default=100, help="episodes in the 'recent' score window")
    a("--resume", action="store_true", help="continue from runs/<run_name>/checkpoint.pt")
    return p.parse_args(argv)


def main(args):
    method = ("ppo" if args.no_ad else "ad") + (f"-rarity{args.rarity_bonus:g}" if args.rarity_bonus else "")
    run_dir = os.path.join(args.logdir, args.run_name or f"{method}-s{args.seed}")
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    if os.path.exists(os.path.join(run_dir, "metrics.jsonl")) and not (args.resume and os.path.exists(ckpt_path)):
        sys.exit(f"{run_dir} already has logs; pick another --run_name or pass --resume")
    os.makedirs(run_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.set_num_threads(2)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    venv = VecCrafter(args.num_envs, args.seed)
    model = Agent(args.hidsize, ad=not args.no_ad, temperature=args.temperature).to(device)
    algo = PPO(model, lr=args.lr, epochs=args.ppo_epochs, nbatch=args.nbatch, clip=args.clip,
               vf_coef=args.vf_coef, ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm,
               aux_every=args.aux_every, aux_epochs=args.aux_epochs, pi_dist_coef=args.pi_dist_coef,
               vf_dist_coef=args.vf_dist_coef, micro_batch=args.micro_batch)
    shapers = []
    if args.rarity_bonus > 0:
        shapers.append(RarityBonus(args.num_envs, args.rarity_bonus, args.rarity_window))
    if args.vitals > 0:
        shapers.append(VitalsBonus(args.num_envs, args.vitals, args.vitals_gamma))

    roll = Rollout(args.nstep, args.num_envs, args.hidsize, device)
    to_t = lambda o: th.from_numpy(o).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    roll.obs[0] = to_t(venv.reset())

    nupdates = -(-args.steps // (args.nstep * args.num_envs))
    start, successes, elapsed0, step = 1, np.zeros((0, 22), np.int8), 0.0, 0
    if args.resume and os.path.exists(ckpt_path):
        ck = th.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        for k, opt in algo.optimizers().items():
            opt.load_state_dict(ck[k])
        algo.updates, start, successes = ck["update"], ck["update"] + 1, ck["successes"]
        elapsed0, step = ck["elapsed"], ck["step"]
        if args.rarity_bonus > 0:
            shapers[0].hist.extend(ck["rarity_hist"])
        print(f"resumed at update {ck['update']}", flush=True)
    nparams = sum(p.numel() for p in model.parameters())
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "params": nparams, "device": str(device)}, f, indent=2)
    print(f"{run_dir}: {nparams / 1e6:.2f}M parameters on {device}, {nupdates} updates", flush=True)

    metrics_f = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    episodes_f = open(os.path.join(run_dir, "episodes.jsonl"), "a")
    t0 = time.time() - elapsed0
    for update in range(start, nupdates + 1):
        # ---- collect one rollout with the current policy ----
        model.eval()
        ep_rew, ep_len, ep_succ = [], [], []
        for t in range(args.nstep):
            out = model.act(roll.obs[t], roll.states[t])
            obs, rews, dones, infos = venv.step(out["actions"].squeeze(-1).cpu().numpy())
            step += args.num_envs
            succ = np.array([[info["achievements"][a] > 0 for a in ACHIEVEMENTS] for info in infos])
            for i, info in enumerate(infos):
                if "episode" in info:  # logged reward = the game's reward, without our shaping
                    ep_rew.append(info["episode"]["r"])
                    ep_len.append(info["episode"]["l"])
                    ep_succ.append(succ[i])
                    rec = {"step": step, "length": info["episode"]["l"], "reward": round(info["episode"]["r"], 1)}
                    rec.update({f"achievement_{a}": int(info["achievements"][a]) for a in ACHIEVEMENTS})
                    episodes_f.write(json.dumps(rec) + "\n")
            for shape in shapers:
                rews = shape(rews, dones, infos)
            roll.insert(t, model, out, to_t(obs), th.from_numpy(rews).to(device).unsqueeze(-1),
                        th.from_numpy(1.0 - dones.astype(np.float32)).to(device).unsqueeze(-1),
                        th.from_numpy(succ).long().to(device))
        roll.vpreds[-1] = model.act(roll.obs[-1], roll.states[-1])["vpreds"]
        t_roll = time.time()

        # ---- learn ----
        roll.compute_returns(args.gamma, args.gae_lambda)
        stats = algo.update(roll)
        roll.reset()

        # ---- log ----
        if ep_succ:
            successes = np.concatenate([successes, np.array(ep_succ, np.int8)])
        rates = 100 * successes.mean(0) if len(successes) else np.zeros(22)
        recent = 100 * successes[-args.recent:].mean(0) if len(successes) else np.zeros(22)
        m = {"update": update, "step": step, "time": round(time.time() - t0, 1), "episodes": len(successes),
             "score": crafter_score(rates), "recent_score": crafter_score(recent),
             "reward_mean": float(np.mean(ep_rew)) if ep_rew else None,
             "length_mean": float(np.mean(ep_len)) if ep_len else None,
             "update_s": round(time.time() - t_roll, 1), **stats,
             **({"rarity_bonus_total": shapers[0].total} if args.rarity_bonus > 0 else {}),
             "success_rate": dict(zip(ACHIEVEMENTS, rates.tolist())),
             "recent_success_rate": dict(zip(ACHIEVEMENTS, recent.tolist()))}
        metrics_f.write(json.dumps(m) + "\n")
        metrics_f.flush()
        episodes_f.flush()
        rw = f"{m['reward_mean']:.2f}" if ep_rew else "-"
        print(f"[{update:3d}/{nupdates}] {step / 1e6:.3f}M steps | score {m['score']:5.2f} (recent {m['recent_score']:5.2f})"
              f" | reward {rw} | {m['time'] / 60:.0f} min", flush=True)

        if update % 10 == 0 or update == nupdates:
            ck = {"model": model.state_dict(), "update": update, "successes": successes,
                  "elapsed": time.time() - t0, "step": step, **{k: o.state_dict() for k, o in algo.optimizers().items()}}
            if args.rarity_bonus > 0:
                ck["rarity_hist"] = list(shapers[0].hist)
            th.save(ck, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    venv.close()
    th.save(model.state_dict(), os.path.join(run_dir, "agent.pt"))

    rates = 100 * successes.mean(0)
    rewards = [json.loads(line)["reward"] for line in open(os.path.join(run_dir, "episodes.jsonl"))]
    final = {"run": os.path.basename(run_dir), "steps": step, "episodes": len(successes),
             "score": crafter_score(rates), "reward_mean": float(np.mean(rewards)),
             "success_rate": dict(zip(ACHIEVEMENTS, rates.tolist())),
             "wall_clock_h": round((time.time() - t0) / 3600, 2)}
    if args.eval_episodes:
        from evaluate import evaluate
        final["eval"] = evaluate(model, args.eval_episodes, device=device, seed=10_000 + args.seed)
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps({k: v for k, v in final.items() if k not in ("success_rate", "eval")}, indent=2))


if __name__ == "__main__":
    main(parse_args())
