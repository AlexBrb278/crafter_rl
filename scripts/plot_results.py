"""Turn runs/<name>/{metrics,episodes}.jsonl into committed summaries under results/.

  .venv/Scripts/python scripts/plot_results.py ad-ppo_ad-s0 ad-ppo-s0 ... [--out results]

Writes:
  results/summary.csv               one row per run: score, reward, 22 success rates, wall-clock
  results/plots/score.png           Crafter score (cumulative protocol + last-100 window) vs steps
  results/plots/reward_length.png   episode reward and length vs steps (smoothed per episode)
  results/plots/achievements.png    per-achievement success rates (all training episodes)
  results/plots/achievement_curves.png  success rate over training, one panel per achievement
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.crafter_score import ACHIEVEMENTS, crafter_score  # noqa: E402

# Categorical slots (fixed order, light surface) from the shared chart palette.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#ecebe7"
PAPER = {"Achievement Distillation (paper)": 21.79, "PPO, Moon et al. (paper)": 15.60,
         "Human experts": 50.5}


def load(run):
    d = os.path.join(ROOT, "runs", run)
    rows = lambda f: [json.loads(l) for l in open(os.path.join(d, f)) if l.strip()]
    final = os.path.join(d, "final.json")
    return {"name": run, "metrics": rows("metrics.jsonl"), "episodes": rows("episodes.jsonl"),
            "config": json.load(open(os.path.join(d, "config.json"))),
            "final": json.load(open(final)) if os.path.exists(final) else None}


def style(ax, ylabel=None):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)


def steps_fmt(ax):
    import matplotlib.ticker as mt
    ax.xaxis.set_major_formatter(mt.FuncFormatter(
        lambda v, _: "0" if v == 0 else f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))


def smooth(y, k):
    if len(y) < k:
        return np.asarray(y, float)
    c = np.cumsum(np.insert(np.asarray(y, float), 0, 0))
    return np.concatenate([c[1:k] / np.arange(1, k), (c[k:] - c[:-k]) / k])


def label_of(r):
    return r.get("label") or r["name"]


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "text.color": INK, "axes.labelcolor": INK2})
    runs = [load(r) for r in args.runs]
    for r, lab in zip(runs, args.labels or [None] * len(runs)):
        r["label"] = lab
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)

    # ---- summary table ----
    lines = ["run,label,steps,episodes,score,reward_mean_all,reward_mean_last100,length_mean_last100,"
             "wall_clock_h," + ",".join(ACHIEVEMENTS)]
    for r in runs:
        eps = r["episodes"]
        succ = np.array([[e[f"achievement_{a}"] > 0 for a in ACHIEVEMENTS] for e in eps], float)
        rates = 100 * succ.mean(0)
        r["rates"] = rates
        rew = [e["reward"] for e in eps]
        lens = [e["length"] for e in eps]
        m = r["metrics"][-1]
        wall = r["final"]["wall_clock_h"] if r["final"] else round(m["time"] / 3600, 2)
        lines.append(",".join(map(str, [
            r["name"], label_of(r), m["step"], len(eps), round(crafter_score(rates), 3),
            round(np.mean(rew), 3), round(np.mean(rew[-100:]), 3), round(np.mean(lens[-100:]), 1), wall,
            *[round(x, 2) for x in rates]])))
    with open(os.path.join(out, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- score curves ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for i, r in enumerate(runs):
        x = [m["step"] for m in r["metrics"]]
        axes[0].plot(x, [m["score"] for m in r["metrics"]], color=COLORS[i % 8], lw=2, label=label_of(r))
        axes[1].plot(x, [m["recent_score"] for m in r["metrics"]], color=COLORS[i % 8], lw=2, label=label_of(r))
    for name, v in PAPER.items():
        if v < 30:
            axes[0].axhline(v, color=INK2, lw=1, ls=(0, (4, 3)))
            axes[0].text(axes[0].get_xlim()[0], v, f" {name}: {v}", va="bottom", color=INK2, fontsize=8)
    axes[0].set_title("Crafter score: all training episodes (1M protocol)")
    axes[1].set_title("Crafter score: last 100 episodes")
    for ax in axes:
        style(ax, "score (%)")
        steps_fmt(ax)
        ax.set_xlabel("environment steps", color=INK2)
        ax.set_ylim(bottom=0)
    if len(runs) > 1:
        axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "score.png"), dpi=150)
    plt.close(fig)

    # ---- reward / length ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, r in enumerate(runs):
        x = [e["step"] for e in r["episodes"]]
        axes[0].plot(x, smooth([e["reward"] for e in r["episodes"]], 50), color=COLORS[i % 8], lw=2, label=label_of(r))
        axes[1].plot(x, smooth([e["length"] for e in r["episodes"]], 50), color=COLORS[i % 8], lw=2, label=label_of(r))
    axes[0].set_title("Episode reward (50-episode moving average)")
    axes[1].set_title("Episode length (50-episode moving average)")
    style(axes[0], "reward")
    style(axes[1], "steps")
    for ax in axes:
        steps_fmt(ax)
        ax.set_xlabel("environment steps", color=INK2)
        ax.set_ylim(bottom=0)
    if len(runs) > 1:
        axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "reward_length.png"), dpi=150)
    plt.close(fig)

    # ---- per-achievement bars ----
    order = np.argsort(-np.mean([r["rates"] for r in runs], axis=0))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    n = len(runs)
    w = 0.8 / n
    xs = np.arange(len(ACHIEVEMENTS))
    for i, r in enumerate(runs):
        ax.bar(xs + (i - (n - 1) / 2) * w, r["rates"][order], width=w * 0.92, color=COLORS[i % 8],
               label=f"{label_of(r)} (score {crafter_score(r['rates']):.2f})")
    ax.set_xticks(xs, [ACHIEVEMENTS[j].replace("_", " ") for j in order], rotation=50, ha="right")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([0, 1, 10, 100], ["0%", "1%", "10%", "100%"])
    ax.set_ylim(0, 110)
    style(ax, "success rate (log scale)")
    ax.set_title("Success rate per achievement, over all training episodes")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=min(n, 4))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "achievements.png"), dpi=150)
    plt.close(fig)

    # ---- per-achievement curves (small multiples) ----
    fig, axes = plt.subplots(4, 6, figsize=(16, 10), sharex=True)
    for k, a in enumerate(ACHIEVEMENTS):
        ax = axes.flat[k]
        for i, r in enumerate(runs):
            eps = r["episodes"]
            x = [e["step"] for e in eps]
            y = smooth([100.0 * (e[f"achievement_{a}"] > 0) for e in eps], 100)
            ax.plot(x, y, color=COLORS[i % 8], lw=1.6)
        ax.set_title(a.replace("_", " "), fontsize=9)
        ax.set_ylim(0, 100)
        style(ax)
        steps_fmt(ax)
    for ax in list(axes.flat)[len(ACHIEVEMENTS):]:
        ax.axis("off")
    if len(runs) > 1:
        handles = [plt.Line2D([], [], color=COLORS[i % 8], lw=2) for i in range(len(runs))]
        axes.flat[-1].legend(handles, [label_of(r) for r in runs], frameon=False, loc="center")
    fig.suptitle("Success rate over training (100-episode moving average, %)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "achievement_curves.png"), dpi=130)
    plt.close(fig)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(open(os.path.join(out, "summary.csv"), encoding="utf-8").read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="results")
    main(ap.parse_args())
