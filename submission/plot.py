"""Learning curves and per-achievement success rates from train.py logs.

    python plot.py "AD + rarity=runs/ad-rarity2-s0,runs/ad-rarity2-s1" "PPO=runs/ppo-s0" --out plots/

Each argument is LABEL=run_dir[,run_dir...]; seeds of one label are averaged (band = sd).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs import ACHIEVEMENTS


def load(run):
    metrics = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
    eps = [json.loads(l) for l in open(os.path.join(run, "episodes.jsonl"))]
    return metrics, eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("groups", nargs="+")
    p.add_argument("--out", default="plots")
    p.add_argument("--bin", type=int, default=25_600, help="steps per reward bin")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    groups = [(g.split("=", 1)[0], g.split("=", 1)[1].split(",")) for g in args.groups]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    rates = {}
    for label, runs in groups:
        data = [load(r) for r in runs]
        n = min(len(m) for m, _ in data)
        step = np.array([m["step"] for m in data[0][0][:n]])
        score = np.array([[m["score"] for m in md[:n]] for md, _ in data])
        line, = ax1.plot(step, score.mean(0), label=label)
        ax1.fill_between(step, score.mean(0) - score.std(0), score.mean(0) + score.std(0), alpha=0.15, color=line.get_color())
        edges = np.arange(0, step[-1] + args.bin, args.bin)
        rew = []
        for _, eps in data:
            st = np.array([e["step"] for e in eps])
            rw = np.array([e["reward"] for e in eps])
            idx = np.digitize(st, edges) - 1
            rew.append([rw[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(edges) - 1)])
        rew = np.array(rew)
        mid = (edges[:-1] + edges[1:]) / 2
        ax2.plot(mid, np.nanmean(rew, 0), color=line.get_color(), label=label)
        ax2.fill_between(mid, np.nanmean(rew, 0) - np.nanstd(rew, 0), np.nanmean(rew, 0) + np.nanstd(rew, 0),
                         alpha=0.15, color=line.get_color())
        rates[label] = np.mean([[md[n - 1]["success_rate"][a] for a in ACHIEVEMENTS] for md, _ in data], 0)
    ax1.set(xlabel="environment steps", ylabel="Crafter score (%)", title="Score over all training episodes")
    ax2.set(xlabel="environment steps", ylabel="episode reward", title="Average episodic reward")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "curves.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    w = 0.8 / len(rates)
    for i, (label, r) in enumerate(rates.items()):
        ax.bar(np.arange(22) - 0.4 + w * (i + 0.5), np.maximum(r, 1e-3), w, label=label)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.set_xticks(range(22), [a.replace("_", " ") for a in ACHIEVEMENTS], rotation=45, ha="right")
    ax.set_ylabel("success rate (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "success_rates.png"), dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
