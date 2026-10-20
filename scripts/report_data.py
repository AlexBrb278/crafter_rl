"""Collect everything the final report needs from runs/<name>/ into one compact JSON.

  .venv/Scripts/python scripts/report_data.py ad-ppo_ad-s0:"PPO + AD" ppo-s0:"PPO" --out runs/report/data.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.crafter_score import ACHIEVEMENTS, crafter_score  # noqa: E402


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def moving(x, k):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def thin(xs, ys, n=300):
    if len(xs) <= n:
        return [[int(a), round(float(b), 3)] for a, b in zip(xs, ys)]
    idx = np.linspace(0, len(xs) - 1, n).astype(int)
    return [[int(xs[i]), round(float(ys[i]), 3)] for i in idx]


def run_data(name, label):
    d = os.path.join(ROOT, "runs", name)
    metrics = rows(os.path.join(d, "metrics.jsonl"))
    eps = rows(os.path.join(d, "episodes.jsonl"))
    final = json.load(open(os.path.join(d, "final.json"))) if os.path.exists(os.path.join(d, "final.json")) else None
    cfg = json.load(open(os.path.join(d, "config.json")))
    succ = np.array([[e[f"achievement_{a}"] > 0 for a in ACHIEVEMENTS] for e in eps], float)
    rates = 100 * succ.mean(0)
    last = succ[-300:]
    steps = [e["step"] for e in eps]
    out = {
        "name": name, "label": label, "config": cfg["config"], "args": cfg["args"],
        "finished": final is not None, "steps": metrics[-1]["step"] if metrics else 0,
        "episodes": len(eps), "score": crafter_score(rates),
        "score_last300": crafter_score(100 * last.mean(0)),
        "reward_all": float(np.mean([e["reward"] for e in eps])),
        "reward_last300": float(np.mean([e["reward"] for e in eps[-300:]])),
        "length_last300": float(np.mean([e["length"] for e in eps[-300:]])),
        "hours": final["wall_clock_h"] if final else round(metrics[-1]["time"] / 3600, 2),
        "rates": dict(zip(ACHIEVEMENTS, map(float, rates))),
        "rates_last300": dict(zip(ACHIEVEMENTS, map(float, 100 * last.mean(0)))),
        "score_curve": [[m["step"], round(m["score"], 3)] for m in metrics],
        "recent_curve": [[m["step"], round(m["recent_score"], 3)] for m in metrics],
        "reward_curve": thin(steps, moving([e["reward"] for e in eps], 100)),
        "length_curve": thin(steps, moving([e["length"] for e in eps], 100)),
        "ach_curves": {a: thin(steps, 100 * moving(succ[:, i], 200), 120) for i, a in enumerate(ACHIEVEMENTS)},
        "aux_s": [[m["epoch"], round(m["aux_s"])] for m in metrics if "aux_s" in m],
        "eval_episode": final.get("eval_episode") if final else None,
        "spectator": rows(os.path.join(d, "videos", "spectator.jsonl")),
        "clips": rows(os.path.join(d, "videos", "index.jsonl")),
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name:label")
    ap.add_argument("--out", default="runs/report/data.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = [run_data(*r.split(":", 1)) for r in args.runs]
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(data, open(out, "w"))
    for r in data:
        print(f"{r['label']:20s} steps {r['steps']:>8,}  score {r['score']:6.2f}  last300 {r['score_last300']:6.2f}  "
              f"reward {r['reward_all']:.2f} / last300 {r['reward_last300']:.2f}  {r['hours']} h")
    print(f"wrote {out} ({os.path.getsize(out)/1e3:.0f} kB)")
