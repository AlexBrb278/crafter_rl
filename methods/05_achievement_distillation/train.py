"""PPO / PPO + Achievement Distillation on Crafter (1M steps), Windows + small-GPU runner.

Model, storage, rollout and loss code is the official one in ./upstream
(snu-mllab/Achievement-Distillation @ 9abbce6), configs are the paper's (configs/*.yaml).
Local changes, none of which change the algorithm:
  * env: common/crafter_vec_env.py instead of SB3 SubprocVecEnv + VecMonitor (no gym/SB3);
  * memory: optional gradient micro-batching (algos.py) so batch 512 fits in 4 GB VRAM;
  * logging: TensorBoard + JSONL + status.json instead of W&B; random MP4 clips while
    training, a full-episode video at the end; resumable checkpoints.

Usage (from the repo root):
  .venv/Scripts/python methods/05_achievement_distillation/train.py --config ppo_ad --seed 0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, os.path.join(HERE, "upstream")):
    if p not in sys.path:
        sys.path.insert(0, p)


class Tee:
    """Mirror a console stream into runs/<run>/train.log (tracebacks included)."""

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class RarityBonus:
    """Improvement experiment (not in the paper): extra reward for *rare* achievements.

    The Crafter score is a geometric mean of success rates, so it is dominated by the
    achievements the agent almost never gets (diamond, iron tools, eat plant, ...). On the
    step where achievement i is first unlocked in an episode, the agent receives
    beta * (1 - p_i) on top of the environment's +1, where p_i is the fraction of the last
    `window` finished episodes (all envs) that unlocked i. Common achievements keep ~+1,
    never-seen ones are worth 1 + beta. Logged episode rewards stay the environment's.
    """

    def __init__(self, num_envs, beta, window=200):
        from collections import deque
        from common.crafter_score import ACHIEVEMENTS
        self.names = ACHIEVEMENTS
        self.beta = beta
        self.prev = [set() for _ in range(num_envs)]
        self.hist = deque(maxlen=window)
        self.total = 0.0

    def rates(self):
        if not self.hist:
            return dict.fromkeys(self.names, 0.0)
        m = np.mean(self.hist, axis=0)
        return dict(zip(self.names, m))

    def __call__(self, rews, dones, infos):
        p = self.rates()
        rews = rews.copy()
        for i, info in enumerate(infos):
            unlocked = {a for a, c in info["achievements"].items() if c > 0}
            for a in unlocked - self.prev[i]:
                rews[i] += self.beta * (1.0 - p[a])
                self.total += self.beta * (1.0 - p[a])
            if dones[i]:
                self.hist.append([a in unlocked for a in self.names])
                self.prev[i] = set()
            else:
                self.prev[i] = unlocked
        return rews


class VitalsShaping:
    """Improvement experiment (not in the paper): potential-based survival signal.

    Most deaths happen with food, drink or energy at 0 (results/deaths/). With the potential
        Phi(s) = alpha * sum_{v in food, drink, energy} (1 - ((9 - v) / 9) ** 2)
    the learner gets F = gamma * Phi(s') - Phi(s) on top of the reward, with Phi = 0 after death.
    With gamma equal to the learner's discount this is potential-based shaping (Ng, Harada & Russell
    1999), which leaves the optimal policy unchanged, but it then also costs (1 - gamma) * Phi(s') per
    step alive. `--vitals_gamma 1` uses Phi(s') - Phi(s) instead: no per-step cost, at the price of the
    formal guarantee (relative to the invariant form it adds a bonus of (1 - gamma) * Phi(s') per step). It
    only rewards topping up a low vital (most when it is nearly empty), costs a little as vitals
    run down, and costs Phi(s) on death. Timeouts are not deaths. Logged rewards stay the game's.
    """
    VITALS = ("food", "drink", "energy")

    def __init__(self, num_envs, alpha, gamma, vmax=9):
        self.alpha, self.gamma, self.vmax = alpha, gamma, vmax
        self.full = self.phi(dict.fromkeys(self.VITALS, vmax))  # every episode starts at 9/9/9
        self.prev = np.full(num_envs, self.full)
        self.total = 0.0

    def phi(self, inv):
        return self.alpha * sum(1.0 - ((self.vmax - inv[v]) / self.vmax) ** 2 for v in self.VITALS)

    def __call__(self, rews, dones, infos):
        rews = rews.copy()
        for i, info in enumerate(infos):
            nxt = 0.0 if info["discount"] == 0 else self.phi(info["inventory"])
            f = self.gamma * nxt - self.prev[i]
            rews[i] += f
            self.total += f
            self.prev[i] = self.full if dones[i] else nxt
        return rews


class TorchVecEnv:
    """Adapts CrafterVecEnv to the interface of upstream's VecPyTorch (+ clip recording)."""

    def __init__(self, venv, device, recorder=None, on_episode=None, reward_fn=None):
        import torch as th
        self.th = th
        self.venv = venv
        self.device = device
        self.recorder = recorder
        self.on_episode = on_episode
        self.reward_fn = reward_fn  # optional shaping of the reward fed to the learner
        self.global_step = 0

    def _obs(self, obs):
        th = self.th
        return th.from_numpy(obs).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)

    def reset(self):
        return self._obs(self.venv.reset())

    def step(self, actions):
        from common.crafter_score import ACHIEVEMENTS
        th = self.th
        rec = self.recorder
        if rec is not None:
            rec.maybe_start(self.global_step)
        render_idx = rec.active_env if rec is not None else None
        obs, rews, dones, infos, frame = self.venv.step(
            actions.squeeze(-1).cpu().numpy(), render_idx=render_idx,
            render_size=rec.size if rec is not None else 0)
        self.global_step += self.venv.num_envs
        if rec is not None and render_idx is not None:
            rec.add(frame, rews[render_idx], dones[render_idx], infos[render_idx], self.global_step)

        n = len(infos)
        ep_len = np.zeros(n, np.int64)
        ep_rew = np.zeros(n, np.float32)
        for i, info in enumerate(infos):
            if "episode" in info:
                ep_len[i] = info["episode"]["l"]
                ep_rew[i] = info["episode"]["r"]
                if self.on_episode is not None:
                    self.on_episode(info, self.global_step)
        if self.reward_fn is not None:
            rews = self.reward_fn(rews, dones, infos)
        ach = np.array([[info["achievements"][a] for a in ACHIEVEMENTS] for info in infos])
        ach_t = th.from_numpy(ach).long().to(self.device)
        out_infos = {
            "episode_lengths": th.from_numpy(ep_len).to(self.device),
            "episode_rewards": th.from_numpy(ep_rew).to(self.device),
            "achievements": ach_t,
            "successes": (ach_t > 0).long(),
        }
        return (self._obs(obs),
                th.from_numpy(rews).float().to(self.device).unsqueeze(-1),
                th.from_numpy(dones.astype(np.float32)).to(self.device).unsqueeze(-1),
                out_infos)


def record_episode(model, cfg, device, path, seed=123, max_steps=3000, label=""):
    """Play one episode (stochastic policy, as in training) and save it as MP4."""
    import crafter
    import torch as th
    import torch.nn.functional as F
    from common.video import HUD, write_mp4

    env = crafter.Env(seed=seed)
    obs = env.reset()
    hud = HUD(384)
    model.eval()
    states = th.zeros(1, cfg["model_kwargs"]["hidsize"], device=device)
    mem = {"hiddens": model.initial_hiddens(1, device)} if hasattr(model, "initial_hiddens") else {}
    frames, ret, prev, info = [], 0.0, set(), None

    def to_t(o):
        return th.from_numpy(o).to(device).permute(2, 0, 1)[None].float() / 255.0

    for t in range(1, max_steps + 1):
        frame = env.render((384, 384))
        out = model.act(to_t(obs), states=states, **mem)
        if mem:
            mem["hiddens"] = out["hiddens"]
        obs, r, done, info = env.step(int(out["actions"].item()))
        ret += r
        ach = {k for k, v in info["achievements"].items() if v > 0}
        new = sorted(ach - prev)
        prev = ach
        inv = info["inventory"]
        frames.append(hud.draw(
            frame, f"{label}  evaluation episode   t={t}   return {ret:.1f}",
            f"health {inv['health']}  food {inv['food']}  drink {inv['drink']}  "
            f"energy {inv['energy']}   achievements: {len(ach)}/22", new))
        if new and cfg["model_cls"] in ("PPOADModel", "PPOADGRUModel"):
            with th.no_grad():
                next_lat = model.encode(to_t(obs))
            states = F.normalize(next_lat - out["latents"], dim=-1)
        if done:
            break
    write_mp4(path, frames, fps=15)
    return {"length": t, "return": ret, "died": bool(info["discount"] == 0),
            "achievements": sorted(prev)}


def main(args):
    import random
    import yaml
    import torch as th
    import gymnasium
    sys.modules.setdefault("gym", gymnasium)  # upstream does `from gym import spaces`
    # upstream sample.py imports its SB3-based VecPyTorch only as a type hint
    import types
    stub = types.ModuleType("achievement_distillation.wrapper")
    stub.VecPyTorch = TorchVecEnv
    sys.modules.setdefault("achievement_distillation.wrapper", stub)
    from torch.utils.tensorboard import SummaryWriter

    from achievement_distillation.model import PPOModel, PPOADModel
    from achievement_distillation.sample import sample_rollouts
    from achievement_distillation.storage import RolloutStorage
    from algos import MBPPOAlgorithm, MBPPOADAlgorithm
    from common.crafter_score import ACHIEVEMENTS, crafter_score
    from common.crafter_vec_env import CrafterVecEnv
    from common.video import ClipRecorder

    with open(os.path.join(HERE, "configs", f"{args.config}.yaml")) as f:
        cfg = yaml.safe_load(f)
    if args.nepoch:
        cfg["nepoch"] = args.nepoch
    if args.gamma is not None:  # improvement experiment: longer horizon (paper: 0.95)
        cfg["gamma"] = args.gamma
    if args.gae_lambda is not None:
        cfg["gae_lambda"] = args.gae_lambda

    run_name = args.run_name or f"ad-{args.config}-s{args.seed}"
    run_dir = os.path.join(ROOT, "runs", run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    latest = os.path.join(ckpt_dir, "latest.pt")
    # The logs are appended to, so a fresh start in a used folder would mix two runs (and
    # overwrite a committed one). Only --resume from a checkpoint may reuse a folder.
    used = [f for f in ("metrics.jsonl", "episodes.jsonl", "final.json")
            if os.path.isfile(os.path.join(run_dir, f)) and os.path.getsize(os.path.join(run_dir, f))]
    if used and not (args.resume and os.path.exists(latest)):
        why = "there is no checkpoints/latest.pt to resume from" if args.resume else "--resume was not given"
        sys.exit(f"runs/{run_name} already has {', '.join(used)}, but {why}. "
                 "Pick a new --run_name, or delete the folder to start over.")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    sys.stdout = Tee(sys.stdout, os.path.join(run_dir, "train.log"))
    sys.stderr = Tee(sys.stderr, os.path.join(run_dir, "train.log"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    th.backends.cudnn.benchmark = False
    th.set_num_threads(2)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.vram_frac < 1:
        # Past the card's dedicated memory, Windows (WDDM) pages VRAM to host RAM instead of
        # raising OOM, which is ~10x slower. A cap makes PyTorch free its cache and retry first.
        th.cuda.set_per_process_memory_fraction(args.vram_frac)

    obs_space = gymnasium.spaces.Box(low=0, high=1, shape=(3, 64, 64))
    act_space = gymnasium.spaces.Discrete(17)

    # --- env + recorder ---
    seeds = np.random.randint(0, 2**31 - 1, size=cfg["nproc"])
    venv = CrafterVecEnv(seeds, envs_per_worker=1)
    recorder = None
    if args.clip_every > 0:
        recorder = ClipRecorder(os.path.join(run_dir, "videos"), cfg["nproc"],
                                clip_len=args.clip_len, mean_gap=args.clip_every,
                                seed=args.seed, label=run_name)
    ep_log = open(os.path.join(run_dir, "episodes.jsonl"), "a")

    def on_episode(info, step):
        rec = {"step": step, "length": info["episode"]["l"],
               "reward": round(info["episode"]["r"], 1)}
        rec.update({f"achievement_{a}": int(info["achievements"][a]) for a in ACHIEVEMENTS})
        ep_log.write(json.dumps(rec) + "\n")

    bonus = RarityBonus(cfg["nproc"], args.rarity_bonus, args.rarity_window) if args.rarity_bonus > 0 else None
    vgamma = cfg["gamma"] if args.vitals_gamma is None else args.vitals_gamma
    vitals = VitalsShaping(cfg["nproc"], args.vitals, vgamma) if args.vitals > 0 else None
    if vitals is not None and cfg["algorithm_cls"] == "PPOADAlgorithm":
        # AD must find achievement steps from the success flags, not from the shaped reward
        from achievement_distillation.algorithm.ppo_ad import Buffer
        from algos import goals_from_successes
        Buffer.get_goals = goals_from_successes
    shapers = [f for f in (bonus, vitals) if f is not None]

    def reward_fn(rews, dones, infos):
        for f in shapers:
            rews = f(rews, dones, infos)
        return rews
    env = TorchVecEnv(venv, device, recorder, on_episode, reward_fn=reward_fn if shapers else None)
    obs = env.reset()

    # --- model / algorithm (upstream) ---
    from recurrent import PPOADGRUModel, RecurrentBuffer, RecurrentRolloutStorage
    model_cls = {"PPOModel": PPOModel, "PPOADModel": PPOADModel, "PPOADGRUModel": PPOADGRUModel}[cfg["model_cls"]]
    recurrent = cfg["model_cls"] == "PPOADGRUModel"
    model = model_cls(observation_space=obs_space, action_space=act_space,
                      **cfg["model_kwargs"]).to(device)
    algo_cls = {"PPOAlgorithm": MBPPOAlgorithm, "PPOADAlgorithm": MBPPOADAlgorithm}[cfg["algorithm_cls"]]
    extra = {"aux_micro_batch": args.aux_micro_batch, "fast_aux": bool(args.fast_aux),
             "check_fast_aux": args.check_fast_aux} if cfg["algorithm_cls"] == "PPOADAlgorithm" else {}
    algorithm = algo_cls(model=model, micro_batch=args.micro_batch, amp=args.amp,
                         **extra, **cfg["algorithm_kwargs"])
    batch = cfg["nstep"] * cfg["nproc"] // cfg["algorithm_kwargs"]["ppo_nbatch"]
    if args.micro_batch and args.micro_batch < batch:
        # Keep the value normaliser's EWMA time constant identical despite
        # k forward passes per optimizer step instead of one.
        k = math.ceil(batch / args.micro_batch)
        model.vf_head.normalizer.beta = model.vf_head.normalizer.beta ** (1.0 / k)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"{cfg['model_cls']}: {nparams/1e6:.2f}M params on {device}", flush=True)

    storage_kw = dict(nstep=cfg["nstep"], nproc=cfg["nproc"], observation_space=obs_space,
                      action_space=act_space, hidsize=cfg["model_kwargs"]["hidsize"], device=device)
    if recurrent:
        assert not args.micro_batch, "micro-batching splits sequences; not implemented for the GRU model"
        storage = RecurrentRolloutStorage(gru_size=cfg["model_kwargs"]["gru_size"],
                                          seq_len=cfg["seq_len"], **storage_kw)
        algorithm.buffer = RecurrentBuffer(maxlen=cfg["algorithm_kwargs"]["aux_freq"])
        if args.aux_micro_batch:  # see algos._fast_aux_phase
            algorithm.aux_norm_k = math.ceil(512 / args.aux_micro_batch)
        assert args.fast_aux, "the GRU model's aux losses are implemented in fast_aux.py"
    else:
        storage = RolloutStorage(**storage_kw)
    storage.obs[0].copy_(obs)

    # --- resume ---
    start_epoch, total_successes, t_offset = 1, np.zeros((0, 22), np.int32), 0.0
    if args.resume and os.path.exists(latest):
        ck = th.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        for name in ("optimizer", "match_optimizer", "pred_optimizer"):
            if hasattr(algorithm, name) and name in ck:
                getattr(algorithm, name).load_state_dict(ck[name])
        if hasattr(algorithm, "ppo_count"):
            algorithm.ppo_count = ck["ppo_count"]
        start_epoch = ck["epoch"] + 1
        total_successes = ck["total_successes"]
        t_offset = ck["elapsed"]
        env.global_step = ck["global_step"]
        if recorder is not None:
            recorder.n_clips = ck.get("n_clips", 0)
            recorder.next_start = env.global_step + 2_000
        if bonus is not None and "rarity_hist" in ck:
            # Without this the window restarts empty and every first unlock pays the full beta.
            bonus.hist.extend(ck["rarity_hist"])
            bonus.total = ck["rarity_total"]
        print(f"resumed from epoch {ck['epoch']} ({env.global_step} steps"
              f"{f', {len(bonus.hist)} rarity episodes' if bonus is not None else ''})", flush=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"config": cfg, "args": vars(args), "params": nparams,
                   "device": th.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                   "upstream_commit": "9abbce6"}, f, indent=2)

    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    metrics_f = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    t0 = time.time() - t_offset
    t_session = time.time()
    steps_per_epoch = cfg["nstep"] * cfg["nproc"]

    for epoch in range(start_epoch, cfg["nepoch"] + 1):
        t_ep = time.time()
        rollout_stats = sample_rollouts(env, model, storage)
        t_roll = time.time() - t_ep
        storage.compute_returns(cfg["gamma"], cfg["gae_lambda"])
        if device.type == "cuda":
            th.cuda.reset_peak_memory_stats()
        train_stats = algorithm.update(storage)
        storage.reset()
        t_upd = time.time() - t_ep - t_roll

        succ = rollout_stats["successes"]
        total_successes = np.concatenate([total_successes, succ], axis=0)
        rates = 100 * total_successes.mean(axis=0) if len(total_successes) else np.zeros(22)
        score = crafter_score(rates)
        recent = total_successes[-args.recent:]
        recent_rates = 100 * recent.mean(axis=0) if len(recent) else np.zeros(22)
        recent_score = crafter_score(recent_rates)

        step = epoch * steps_per_epoch
        elapsed = time.time() - t0
        m = {
            "epoch": epoch, "step": step, "time": round(elapsed, 1),
            "episodes": int(len(total_successes)), "score": score, "recent_score": recent_score,
            "reward_mean": float(np.mean(rollout_stats["episode_rewards"])) if len(succ) else None,
            "length_mean": float(np.mean(rollout_stats["episode_lengths"])) if len(succ) else None,
            "new_episodes": int(len(succ)),
            "rollout_s": round(t_roll, 2), "update_s": round(t_upd, 2),
            "sps": round(steps_per_epoch / (time.time() - t_ep), 1),
            "gpu_mem_gb": round(th.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0,
            "gpu_reserved_gb": round(th.cuda.max_memory_reserved() / 1e9, 2) if device.type == "cuda" else 0,
            **{k: float(v) for k, v in train_stats.items()},
            **({"rarity_bonus_total": bonus.total} if bonus is not None else {}),
            **({"vitals_shaping_total": vitals.total} if vitals is not None else {}),
            "success_rate": {a: float(r) for a, r in zip(ACHIEVEMENTS, rates)},
            "recent_success_rate": {a: float(r) for a, r in zip(ACHIEVEMENTS, recent_rates)},
        }
        metrics_f.write(json.dumps(m) + "\n")
        metrics_f.flush()
        ep_log.flush()

        writer.add_scalar("score/crafter_score_cumulative", score, step)
        writer.add_scalar("score/crafter_score_recent", recent_score, step)
        if m["reward_mean"] is not None:
            writer.add_scalar("episode/reward", m["reward_mean"], step)
            writer.add_scalar("episode/length", m["length_mean"], step)
        for k, v in train_stats.items():
            writer.add_scalar(f"train/{k}", v, step)
        writer.add_scalar("perf/sps", m["sps"], step)
        writer.add_scalar("perf/gpu_mem_gb", m["gpu_mem_gb"], step)
        for a, r in zip(ACHIEVEMENTS, recent_rates):
            writer.add_scalar(f"success_recent/{a}", r, step)
        for a, r in zip(ACHIEVEMENTS, rates):
            writer.add_scalar(f"success_cumulative/{a}", r, step)

        eta = (time.time() - t_session) / (epoch - start_epoch + 1) * (cfg["nepoch"] - epoch)
        rw = f"{m['reward_mean']:.2f}" if m["reward_mean"] is not None else "  -  "
        extra = f" match {train_stats['match_loss']:.3f} pred {train_stats['pred_loss']:.3f}" \
            if "match_loss" in train_stats else ""
        print(f"[{run_name}] ep {epoch:3d}/{cfg['nepoch']} step {step/1e6:.3f}M | score {score:5.2f} "
              f"(recent {recent_score:5.2f}) | reward {rw} | eps {len(total_successes):4d} | "
              f"{m['sps']:.0f} sps (roll {t_roll:.1f}s upd {t_upd:.1f}s) | mem {m['gpu_mem_gb']}GB | "
              f"{elapsed/60:.0f}m, eta {eta/60:.0f}m{extra}", flush=True)

        with open(os.path.join(run_dir, "status.json"), "w") as f:
            json.dump({"run": run_name, "epoch": epoch, "nepoch": cfg["nepoch"], "step": step,
                       "elapsed_min": round(elapsed / 60, 1), "eta_min": round(eta / 60, 1),
                       "score": score, "recent_score": recent_score,
                       "clips": recorder.n_clips if recorder else 0, "updated": time.time()}, f)

        if epoch % 10 == 0 or epoch == cfg["nepoch"]:
            ck = {"model": model.state_dict(), "epoch": epoch, "total_successes": total_successes,
                  "elapsed": elapsed, "global_step": env.global_step,
                  "n_clips": recorder.n_clips if recorder else 0}
            for name in ("optimizer", "match_optimizer", "pred_optimizer"):
                if hasattr(algorithm, name):
                    ck[name] = getattr(algorithm, name).state_dict()
            if hasattr(algorithm, "ppo_count"):
                ck["ppo_count"] = algorithm.ppo_count
            if bonus is not None:
                ck["rarity_hist"], ck["rarity_total"] = list(bonus.hist), bonus.total
            th.save(ck, latest + ".tmp")
            os.replace(latest + ".tmp", latest)
        if epoch % cfg["save_freq"] == 0 or epoch == cfg["nepoch"]:
            th.save(model.state_dict(), os.path.join(ckpt_dir, f"agent-e{epoch:03d}.pt"))

    venv.close()
    writer.close()

    # --- final summary + full-episode video ---
    final = {"run": run_name, "config": args.config, "seed": args.seed,
             "steps": int(cfg["nepoch"] * steps_per_epoch), "episodes": int(len(total_successes)),
             "score": crafter_score(100 * total_successes.mean(axis=0)),
             "success_rate": {a: float(r) for a, r in zip(ACHIEVEMENTS, 100 * total_successes.mean(axis=0))},
             "wall_clock_h": round((time.time() - t0) / 3600, 2)}
    rewards = [json.loads(l)["reward"] for l in open(os.path.join(run_dir, "episodes.jsonl"))]
    final["reward_mean_all"] = float(np.mean(rewards))
    final["reward_mean_last100"] = float(np.mean(rewards[-100:]))
    if args.final_video:
        final["eval_episode"] = record_episode(
            model, cfg, device, os.path.join(run_dir, "videos", "final_episode.mp4"),
            label=run_name)
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ppo_ad", help="configs/<name>.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--nepoch", type=int, default=None, help="override (debug)")
    ap.add_argument("--gamma", type=float, default=None, help="override the config's discount (paper: 0.95)")
    ap.add_argument("--gae_lambda", type=float, default=None, help="override the config's GAE lambda (paper: 0.65)")
    ap.add_argument("--micro_batch", type=int, default=None,
                    help="split minibatches into chunks of this size (gradient accumulation)")
    ap.add_argument("--aux_micro_batch", type=int, default=None,
                    help="micro-batch for the AD aux phase (defaults to --micro_batch)")
    ap.add_argument("--fast_aux", type=int, default=1,
                    help="1: fast_aux.py (identical math, no redundant passes); 0: upstream aux loop")
    ap.add_argument("--check_fast_aux", action="store_true",
                    help="compare fast vs upstream aux losses/grads on the first aux phase")
    ap.add_argument("--vram_frac", type=float, default=0.85,
                    help="cap on this process' share of GPU memory (see comment in main)")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (not used for paper runs)")
    ap.add_argument("--rarity_bonus", type=float, default=0.0,
                    help="beta of RarityBonus (improvement experiment; 0 = paper reward)")
    ap.add_argument("--rarity_window", type=int, default=200)
    ap.add_argument("--vitals", type=float, default=0.0,
                    help="alpha of VitalsShaping (survival-signal experiment; 0 = off)")
    ap.add_argument("--vitals_gamma", type=float, default=None,
                    help="gamma inside the shaping term (default: the config's). 1.0 gives Phi(s') - Phi(s), "
                         "which has no per-step cost while alive (not strictly policy-invariant under gamma < 1)")
    ap.add_argument("--recent", type=int, default=100, help="episodes in the 'recent' window")
    ap.add_argument("--clip_every", type=int, default=40_000, help="mean env steps between clips (0=off)")
    ap.add_argument("--clip_len", type=int, default=400)
    ap.add_argument("--final_video", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    main(ap.parse_args())
