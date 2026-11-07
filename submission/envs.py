"""Crafter environments, the Crafter score, and our two reward changes.

VecCrafter   runs N Crafter games in N worker processes (one game each) and auto-resets them.
RarityBonus  our main contribution: extra reward for unlocking achievements the agent rarely gets.
VitalsBonus  an ablation: potential-based reward for keeping food / drink / energy up.
"""
import multiprocessing as mp
from collections import deque

import numpy as np

ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron", "collect_sapling",
    "collect_stone", "collect_wood", "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
    "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant", "place_stone",
    "place_table", "wake_up",
]
NUM_ACTIONS = 17


def crafter_score(rates):
    """Crafter score (Hafner, 2022): geometric mean of the 22 success rates s_i in percent,
    score = exp(mean_i ln(1 + s_i)) - 1. Rare achievements dominate it."""
    s = np.asarray(rates, dtype=np.float64)
    return float(np.exp(np.mean(np.log1p(s))) - 1.0)


def _worker(remote, seed):
    import crafter  # imported in the worker only
    env = crafter.Env(seed=int(seed))
    ret, length = 0.0, 0
    while True:
        cmd, action = remote.recv()
        if cmd == "reset":
            ret, length = 0.0, 0
            remote.send(env.reset())
        elif cmd == "step":
            obs, rew, done, info = env.step(int(action))
            ret, length = ret + rew, length + 1
            info = {k: info[k] for k in ("achievements", "inventory", "discount")}
            if done:  # auto-reset: return the next episode's first frame, keep the final info
                info["episode"] = {"r": ret, "l": length}
                ret, length = 0.0, 0
                obs = env.reset()
            remote.send((obs, rew, done, info))
        else:
            remote.close()
            return


class VecCrafter:
    """`num_envs` Crafter games in parallel. Observations are uint8 (N, 64, 64, 3)."""

    def __init__(self, num_envs, seed):
        seeds = np.random.RandomState(seed).randint(0, 2**31 - 1, size=num_envs)
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for s in seeds:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, s), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.num_envs = num_envs

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        obs, rews, dones, infos = zip(*[r.recv() for r in self.remotes])
        return np.stack(obs), np.array(rews, np.float32), np.array(dones, bool), list(infos)

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)


class RarityBonus:
    """Reward shaping for rare achievements (our change).

    The first time achievement i is unlocked in an episode, the learner gets
        r' = r + beta * (1 - p_i)
    where p_i is the fraction of the last `window` finished episodes (all envs) that unlocked i.
    Common achievements stay worth ~+1, never-seen ones are worth 1 + beta. The bonus fades as an
    achievement becomes common. Logged rewards and the Crafter score always use the game's reward.
    """

    def __init__(self, num_envs, beta, window=200):
        self.beta = beta
        self.hist = deque(maxlen=window)          # one 22-bool row per finished episode
        self.prev = [set() for _ in range(num_envs)]
        self.total = 0.0

    def rates(self):
        return np.mean(self.hist, axis=0) if self.hist else np.zeros(len(ACHIEVEMENTS))

    def __call__(self, rews, dones, infos):
        p = dict(zip(ACHIEVEMENTS, self.rates()))
        rews = rews.copy()
        for i, info in enumerate(infos):
            unlocked = {a for a, c in info["achievements"].items() if c > 0}
            for a in unlocked - self.prev[i]:
                rews[i] += self.beta * (1.0 - p[a])
                self.total += self.beta * (1.0 - p[a])
            if dones[i]:
                self.hist.append([a in unlocked for a in ACHIEVEMENTS])
                unlocked = set()
            self.prev[i] = unlocked
        return rews


class VitalsBonus:
    """Ablation: potential-based shaping on the three vitals (food, drink, energy, each 0..9).

    Phi(s) = alpha * sum_v (1 - ((9 - v) / 9)^2), Phi = 0 after death, F = gamma * Phi(s') - Phi(s).
    With gamma = the learner's discount this is policy-invariant (Ng et al., 1999); gamma = 1 removes
    the -(1 - gamma) * Phi cost per step alive, which worked better in our runs.
    """
    VITALS = ("food", "drink", "energy")

    def __init__(self, num_envs, alpha, gamma):
        self.alpha, self.gamma = alpha, gamma
        self.full = self.phi(dict.fromkeys(self.VITALS, 9))
        self.prev = np.full(num_envs, self.full)
        self.total = 0.0

    def phi(self, inv):
        return self.alpha * sum(1.0 - ((9 - inv[v]) / 9) ** 2 for v in self.VITALS)

    def __call__(self, rews, dones, infos):
        rews = rews.copy()
        for i, info in enumerate(infos):
            nxt = 0.0 if info["discount"] == 0 else self.phi(info["inventory"])
            f = self.gamma * nxt - self.prev[i]
            rews[i] += f
            self.total += f
            self.prev[i] = self.full if dones[i] else nxt
        return rews
