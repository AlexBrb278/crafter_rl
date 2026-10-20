"""Aggregate results/deaths/<run>.jsonl (from diagnose_deaths.py) per configuration.

Writes results/deaths/summary.csv, results/deaths/causes.png, results/deaths/timing.png and
runs/report/deaths.json (read by make_report.py).

  .venv/bin/python scripts/analyze_deaths.py
"""
import collections
import csv
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "deaths")
NIGHT = 0.3

# config prefix -> label, in display order
# a configuration's runs are <prefix>-s<seed> (exact match, so ppo-rarity2 does not swallow ppo-rarity2-g99)
CONFIGS = [
    ("ppo", "PPO"),
    ("ppo-rarity1", "PPO + rarity β=1"),
    ("ppo-rarity2", "PPO + rarity β=2"),
    ("ppo-rarity4", "PPO + rarity β=4"),
    ("ppo-rarity2-5090", "PPO + rarity β=2 (rerun, RTX 5090)"),
    ("ppo-rarity2-g99", "PPO + rarity β=2, γ=0.99"),
    ("ad-ppo_ad", "AD"),
    ("ad-rarity1", "AD + rarity β=1"),
    ("ad-rarity2", "AD + rarity β=2"),
    ("ad-rarity2-gru", "AD + rarity β=2 + GRU"),
]
# raw cause -> display group
GROUPS = ["zombie, awake", "zombie, asleep", "skeleton arrow", "thirst", "hunger / fatigue", "lava"]


def group(cause):
    if cause == "zombie":
        return "zombie, awake"
    if cause == "zombie (asleep)":
        return "zombie, asleep"
    if cause == "arrow":
        return "skeleton arrow"
    if cause == "lava":
        return "lava"
    if "thirst" in cause:
        return "thirst"
    return "hunger / fatigue"


def daylight(t):
    p = (t / 300) % 1 + 0.3
    return 1 - abs(np.cos(np.pi * p)) ** 3


def main():
    rows = {}
    for path in sorted(glob.glob(os.path.join(OUT, "*.jsonl"))):
        run = os.path.basename(path)[:-6]
        rows[run] = [json.loads(l) for l in open(path) if l.strip()]
    configs = []
    for prefix, label in CONFIGS:
        runs = sorted(r for r in rows if re.fullmatch(re.escape(prefix) + r"-s\d+", r))
        eps = [e for r in runs for e in rows[r]]
        if not eps:
            continue
        died = [e for e in eps if e["died"]]
        n = len(died)
        cnt = collections.Counter(group(e["cause"]) for e in died)
        L = np.array([e["length"] for e in eps])
        configs.append({
            "label": label, "runs": runs, "episodes": len(eps), "died_pct": 100 * n / len(eps),
            "causes": {g: 100 * cnt[g] / max(n, 1) for g in GROUPS},
            "median_length": float(np.median(L)),
            "survive_300_pct": 100 * float((L > 300).mean()),
            "night_pct": 100 * np.mean([e["night"] for e in died]),
            "asleep_pct": 100 * np.mean([e["asleep_at_death"] for e in died]),
            "drink0_pct": 100 * np.mean([e["inventory"]["drink"] == 0 for e in died]),
            "food0_pct": 100 * np.mean([e["inventory"]["food"] == 0 for e in died]),
            "energy0_pct": 100 * np.mean([e["inventory"]["energy"] == 0 for e in died]),
            "slept_pct": 100 * np.mean([e["sleep_steps"] > 0 for e in eps]),
            "night_asleep_share": 100 * sum(e["night_sleep_steps"] for e in eps) / max(1, sum(e["night_steps"] for e in eps)),
            "open_when_zombie_asleep": float(np.mean([e["open_neighbours"] for e in died if e["cause"] == "zombie (asleep)"]))
            if any(e["cause"] == "zombie (asleep)" for e in died) else None,
            "lengths": L.tolist(),
        })

    os.makedirs(OUT, exist_ok=True)
    keys = ["label", "episodes", "died_pct", "median_length", "survive_300_pct", "night_pct", "asleep_pct",
            "drink0_pct", "food0_pct", "energy0_pct", "slept_pct", "night_asleep_share"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["cause: " + g for g in GROUPS])
        for c in configs:
            w.writerow([c[k] if isinstance(c[k], str) else round(c[k], 1) for k in keys] +
                       [round(c["causes"][g], 1) for g in GROUPS])
    for c in configs:
        print(f"{c['label']:18s} n={c['episodes']:4d} median len {c['median_length']:4.0f}  "
              f">300 steps {c['survive_300_pct']:4.1f}%  night {c['night_pct']:4.1f}%  asleep {c['asleep_pct']:4.1f}%  "
              f"drink0 {c['drink0_pct']:4.1f}%  | " + "  ".join(f"{g} {c['causes'][g]:.0f}%" for g in GROUPS))

    # report data: causes per config + length histogram per config with a daylight curve
    allep = [e for r in rows.values() for e in r]
    dead = [e for e in allep if e["died"]]
    zomb = [e for e in dead if e["cause"].startswith("zombie")]
    vital0 = lambda e: min(e["inventory"][k] for k in ("food", "drink", "energy")) == 0
    overall = {
        "episodes": len(allep), "runs": len(rows), "died_pct": 100 * len(dead) / len(allep),
        "median_length": float(np.median([e["length"] for e in allep])),
        "zombie_pct": 100 * len(zomb) / len(dead),
        "zombie_night_pct": 100 * np.mean([e["night"] for e in zomb]),
        "vital0_pct": 100 * np.mean([vital0(e) for e in dead]),
        "drink0_pct": 100 * np.mean([e["inventory"]["drink"] == 0 for e in dead]),
        "night_asleep_share": 100 * sum(e["night_sleep_steps"] for e in allep) / max(1, sum(e["night_steps"] for e in allep)),
        "asleep_death_pct": 100 * np.mean([e["asleep_at_death"] for e in dead]),
        "iron_pickaxe": sum("make_iron_pickaxe" in e["achievements"] for e in allep),
        "diamond": sum("collect_diamond" in e["achievements"] for e in allep),
    }
    print("overall:", {k: round(v, 1) for k, v in overall.items()})
    bins = list(range(0, 660, 20))
    report = {
        "overall": overall,
        "groups": GROUPS, "night_threshold": NIGHT,
        "configs": [{k: v for k, v in c.items() if k not in ("lengths", "runs")} | {
            "hist": np.histogram(c["lengths"], bins=bins + [10001])[0].tolist()} for c in configs],
        "bins": bins,
        "daylight": [[t, round(daylight(t), 3)] for t in range(0, 661, 5)],
    }
    json.dump(report, open(os.path.join(ROOT, "runs", "report", "deaths.json"), "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#184f95", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = np.zeros(len(configs))
    for g, col in zip(GROUPS, colors):
        v = np.array([c["causes"][g] for c in configs])
        ax.barh([c["label"] for c in configs], v, left=left, color=col, label=g)
        left += v
    ax.invert_yaxis()
    ax.set_xlabel("% of deaths")
    ax.set_xlim(0, 100)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.set_title("Cause of death (final weights, 200 episodes per run)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "causes.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    t = np.arange(0, 660)
    ax2 = ax.twinx()
    ax2.fill_between(t, 0, [1 - daylight(x) for x in t], color="#1a2550", alpha=0.12, step="mid")
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)
    for c, col in zip([c for c in configs if c["label"] in ("PPO", "PPO + rarity β=2", "AD + rarity β=2")],
                      ["#76826f", "#2a78d6", "#eb6834"]):
        h = np.histogram(c["lengths"], bins=bins + [10001])[0]
        ax.plot(np.array(bins) + 10, 100 * h / h.sum(), color=col, label=c["label"], lw=2)
    ax.set_xlim(0, 640)
    ax.set_xlabel("episode step at death (shaded: night)")
    ax.set_ylabel("% of episodes")
    ax.legend(frameon=False)
    ax.set_title("When the agent dies")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "timing.png"), dpi=150)
    print(f"wrote {OUT}/summary.csv, causes.png, timing.png and runs/report/deaths.json")


if __name__ == "__main__":
    main()
