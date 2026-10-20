"""Side-by-side table of configurations: training score across seeds plus survival from the
death diagnosis (results/deaths/<run>.jsonl, see scripts/diagnose_deaths.py).

  .venv/bin/python scripts/compare_configs.py ppo-rarity2 ppo-rarity2-5090 ppo-rarity2-g99 \
      --html runs/report/gamma.html --csv results/gamma_summary.csv

A configuration is a run-name prefix; its runs are <prefix>-s<seed> with a final.json.
Training columns use all training episodes within 1M steps (the Crafter protocol). Survival
columns use the final weights (200 fresh episodes per run).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NIGHT = 0.3
LABELS = {"ppo": "PPO, γ = 0.95", "ppo-rarity2": "PPO + rarity β=2, γ = 0.95 (home laptop)",
          "ppo-rarity2-5090": "PPO + rarity β=2, γ = 0.95 (rerun here)",
          "ppo-rarity2-g99": "PPO + rarity β=2, γ = 0.99",
          "ad-ppo_ad": "AD", "ad-rarity2": "AD + rarity β=2", "ad-rarity2-gru": "AD + rarity β=2 + GRU memory"}
KEY = ["make_stone_pickaxe", "collect_iron", "make_iron_pickaxe", "wake_up"]


def runs_of(prefix):
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", f"{prefix}-s*"))):
        m = re.fullmatch(re.escape(prefix) + r"-s(\d+)", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "final.json")):
            out.append(os.path.basename(d))
    return out


def train_length(run, last=300):
    eps = [json.loads(l) for l in open(os.path.join(ROOT, "runs", run, "episodes.jsonl")) if l.strip()]
    return float(np.mean([e["length"] for e in eps[-last:]]))


def main(args):
    rows = []
    for p in args.prefixes:
        runs = runs_of(p)
        if not runs:
            print(f"{p}: no finished runs")
            continue
        fin = [json.load(open(os.path.join(ROOT, "runs", r, "final.json"))) for r in runs]
        sc = np.array([f["score"] for f in fin])
        deaths = [json.loads(l) for r in runs
                  for l in open(os.path.join(ROOT, "results", "deaths", f"{r}.jsonl"))
                  if os.path.exists(os.path.join(ROOT, "results", "deaths", f"{r}.jsonl")) and l.strip()]
        died = [e for e in deaths if e["died"]]
        row = {
            "config": p, "label": LABELS.get(p, p), "n": len(runs), "seeds": [f["seed"] for f in fin],
            "score_mean": float(sc.mean()), "score_std": float(sc.std(ddof=1)) if len(sc) > 1 else None,
            "scores": sc.tolist(), "reward": float(np.mean([f["reward_mean_all"] for f in fin])),
            "train_len_last300": float(np.mean([train_length(r) for r in runs])),
            **{k: float(np.mean([f["success_rate"][k] for f in fin])) for k in KEY},
            "eval_episodes": len(deaths),
            "eval_median_len": float(np.median([e["length"] for e in deaths])) if deaths else None,
            "alive_300": 100 * float(np.mean([e["length"] > 300 for e in deaths])) if deaths else None,
            "zombie_deaths": 100 * float(np.mean([e["cause"].startswith("zombie") for e in died])) if died else None,
            "night_deaths": 100 * float(np.mean([e["night"] for e in died])) if died else None,
            "night_asleep": 100 * sum(e["night_sleep_steps"] for e in deaths) / max(1, sum(e["night_steps"] for e in deaths))
            if deaths else None,
        }
        rows.append(row)
        print(f"{row['label']:45s} n={row['n']} score {row['score_mean']:6.2f}"
              + (f" ± {row['score_std']:.2f}" if row["score_std"] is not None else "")
              + f"  ({', '.join(f'{x:.2f}' for x in sc)})  train len {row['train_len_last300']:.0f}"
              + (f"  eval median {row['eval_median_len']:.0f}  alive@300 {row['alive_300']:.1f}%"
                 f"  zombie {row['zombie_deaths']:.0f}%  night-asleep {row['night_asleep']:.1f}%" if deaths else ""))

    if args.csv:
        cols = [k for k in rows[0] if k not in ("scores", "seeds")]
        with open(os.path.join(ROOT, args.csv), "w", encoding="utf-8") as f:
            f.write(",".join(cols + ["scores"]) + "\n")
            for r in rows:
                f.write(",".join("" if r[c] is None else (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]))
                                 for c in cols) + "," + "/".join(f"{x:.2f}" for x in r["scores"]) + "\n")
    if args.html:
        fmt = lambda v, d=1, u="": "–" if v is None else f"{v:.{d}f}{u}"
        head = ("<tr><th>configuration</th><th>score (mean ± sd)</th><th>per seed</th><th>episode length*</th>"
                "<th>alive at step 300†</th><th>zombie deaths†</th><th>night spent asleep†</th>"
                + "".join(f"<th>{k.replace('_', ' ')}</th>" for k in KEY) + "</tr>")
        body = "".join(
            f"<tr><td>{r['label']}</td><td class=\"val\">{r['score_mean']:.2f}"
            + (f" ± {r['score_std']:.2f}" if r["score_std"] is not None else "") + "</td>"
            f"<td class=\"val\">{' · '.join(f'{x:.1f}' for x in r['scores'])}</td>"
            f"<td class=\"val\">{r['train_len_last300']:.0f}</td><td class=\"val\">{fmt(r['alive_300'], 1, '%')}</td>"
            f"<td class=\"val\">{fmt(r['zombie_deaths'], 0, '%')}</td><td class=\"val\">{fmt(r['night_asleep'], 1, '%')}</td>"
            + "".join(f"<td class=\"val\">{r[k]:.2f}%</td>" if 0 < r[k] < 1 else f"<td class=\"val\">{r[k]:.1f}%</td>"
                      for k in KEY) + "</tr>" for r in rows)
        open(os.path.join(ROOT, args.html), "w", encoding="utf-8").write(head + body)
        print("wrote", args.html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--html")
    ap.add_argument("--csv")
    main(ap.parse_args())
