"""Summarise PPO-RNN runs from runs/ppo_rnn/*/metrics.jsonl.

Prints final reward (% of max), per-achievement success rates averaged over the
last 1% of training updates, and a Crafter-style score for the 22 Classic
achievements. Writes a learning-curve plot and a CSV to results/ppo_rnn/.

Note: upstream logs per-update averages over episodes that ended in that update,
so the "score" here averages over updates, not over all training episodes.

    python methods/01_ppo_rnn/summarize.py [runs_dir]   # default runs/ppo_rnn
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from common.crafter_score import crafter_score  # noqa: E402

MAX_REWARD = {"classic": 22.0, "craftax": 226.0}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "ppo_rnn")
    out_dir = os.path.join(ROOT, "results", os.path.basename(os.path.normpath(runs_dir)))
    os.makedirs(out_dir, exist_ok=True)
    rows_csv = []
    curves = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        rows = load(path)
        if not rows:
            continue
        kind = "classic" if name.startswith("classic") else "craftax"
        steps_per_update = 1024 * 64
        tail = rows[-max(1, len(rows) // 100):]
        ret = np.nanmean([r["episode_return"] for r in tail])
        ach_keys = sorted(k for k in rows[-1] if k.startswith("Achievements/"))
        rates = {k.split("/", 1)[1]: np.nanmean([r[k] for r in tail if r.get(k) is not None]) for k in ach_keys}
        score = crafter_score(list(rates.values())) if len(rates) == 22 else float("nan")
        done = os.path.exists(os.path.join(os.path.dirname(path), "DONE"))
        steps = len(rows) * steps_per_update
        rows_csv.append((name, steps, done, ret, 100 * ret / MAX_REWARD[kind], score))
        curves[name] = (
            np.arange(1, len(rows) + 1) * steps_per_update,
            [100 * r["episode_return"] / MAX_REWARD[kind] for r in rows],
        )

        print(f"\n== {name}  ({steps/1e6:.0f}M steps, {'done' if done else 'RUNNING/INCOMPLETE'})")
        print(f"   reward {ret:.2f}  = {100*ret/MAX_REWARD[kind]:.1f}% of max   score {score:.1f}")
        for k, v in sorted(rates.items(), key=lambda kv: -kv[1]):
            print(f"   {v:6.1f}%  {k}")

    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.write("run,steps,done,reward,reward_pct_max,crafter_score\n")
        for r in rows_csv:
            f.write(",".join(str(x) for x in r) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (x, y) in curves.items():
        ax.plot(x / 1e6, y, label=name, linewidth=1)
    ax.set_xlabel("environment steps (M)")
    ax.set_ylabel("episode reward (% of max)")
    ax.set_title("PPO-RNN (Craftax_Baselines defaults)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=150)
    print(f"\nwrote {out_dir}/summary.csv and learning_curves.png")


if __name__ == "__main__":
    main()
