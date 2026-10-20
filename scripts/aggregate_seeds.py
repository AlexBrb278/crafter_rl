"""Mean ± std of the Crafter score (and key success rates) per configuration across seeds.

  .venv/Scripts/python scripts/aggregate_seeds.py ppo ppo-rarity1 ppo-rarity2 ppo-rarity4 ad-ppo_ad ad-rarity1 ad-rarity2       --html runs/report/seeds.html --beta ppo=0 ppo-rarity1=1 ppo-rarity2=2 ppo-rarity4=4 --plot

A configuration is a run-name prefix; its runs are <prefix>-s<seed> with a final.json.
Writes results/seeds_summary.csv and optionally an HTML table fragment for the report.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.crafter_score import ACHIEVEMENTS  # noqa: E402

KEY = ["make_stone_pickaxe", "collect_iron", "place_furnace", "defeat_skeleton", "wake_up"]
LABELS = {"ppo": "PPO", "ppo-rarity1": "PPO + rarity bonus (β = 1)", "ppo-rarity2": "PPO + rarity bonus (β = 2)", "ppo-rarity4": "PPO + rarity bonus (β = 4)",
          "ad-ppo_ad": "Achievement Distillation", "ad-rarity1": "AD + rarity bonus (β = 1)", "ad-rarity2": "AD + rarity bonus (β = 2)"}


def runs_of(prefix):
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", f"{prefix}-s*"))):
        m = re.fullmatch(re.escape(prefix) + r"-s(\d+)", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "final.json")):
            out.append((int(m.group(1)), json.load(open(os.path.join(d, "final.json")))))
    return out


def plot_beta(pts, path):
    """Dose-response figure: PPO score vs rarity-bonus strength (dots: seeds, line: mean)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, grid, c1 = "#52514e", "#ecebe7", "#2a78d6"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for p in pts:
        ax.scatter([p["beta"]] * len(p["scores"]), p["scores"], s=46, color=c1, edgecolor="white", linewidth=1.5, zorder=3)
    ax.plot([p["beta"] for p in pts], [p["mean"] for p in pts], color=ink2, lw=2, zorder=2, label="mean of seeds")
    for p in pts:
        ax.annotate(f"{p['mean']:.2f}", (p["beta"], p["mean"]), textcoords="offset points", xytext=(-14, 9),
                    ha="right", color=ink2, fontsize=9)
    ax.axhline(21.79, color=ink2, lw=1, ls=(0, (4, 3)))
    ax.text(max(p["beta"] for p in pts) + 0.05, 21.79, "AD (paper) 21.79", va="bottom", ha="right", color=ink2, fontsize=8)
    ax.set_xticks([p["beta"] for p in pts])
    ax.set_xlim(-0.6, max(p["beta"] for p in pts) + 0.3)
    ax.set_xlabel("β (extra reward for a never-seen achievement)", color=ink2)
    ax.set_ylabel("Crafter score (1M steps)", color=ink2)
    ax.set_title("PPO + rarity bonus: score vs bonus strength", loc="left", fontweight="bold")
    ax.grid(True, color=grid)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=ink2, length=0)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp125x
    rows, html_rows = ["config,label,n_seeds,seeds,score_mean,score_std,scores,reward_mean,hours_mean,"
                       + ",".join(f"{k}_mean" for k in KEY)], []
    for p in args.prefixes:
        rs = runs_of(p)
        if not rs:
            continue
        sc = np.array([f["score"] for _, f in rs])
        rew = np.mean([f["reward_mean_all"] for _, f in rs])
        hrs = np.mean([f["wall_clock_h"] for _, f in rs])
        key = {k: np.mean([f["success_rate"][k] for _, f in rs]) for k in KEY}
        std = sc.std(ddof=1) if len(sc) > 1 else float("nan")
        lab = LABELS.get(p, p)
        rows.append(",".join(map(str, [p, lab, len(rs), "/".join(str(s) for s, _ in rs), round(sc.mean(), 2),
                                        round(std, 2), "/".join(f"{x:.2f}" for x in sc), round(rew, 2), round(hrs, 2),
                                        *[round(v, 2) for v in key.values()]])))
        html_rows.append(
            f"<tr><td>{lab}</td><td class=\"val\">{sc.mean():.2f}" + (f" ± {std:.2f}" if len(sc) > 1 else "")
            + f"</td><td class=\"val\">{' · '.join(f'{x:.1f}' for x in sc)}</td>"
            + "".join(f"<td class=\"val\">{v:.1f}%</td>" for v in key.values()) + f"<td class=\"val\">{hrs:.1f} h</td></tr>")
        print(f"{lab:28s} n={len(rs)}  score {sc.mean():6.2f} ± {std:5.2f}   ({', '.join(f'{x:.2f}' for x in sc)})")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    open(os.path.join(ROOT, "results", "seeds_summary.csv"), "w", encoding="utf-8").write("\n".join(rows) + "\n")
    if args.beta and args.html:
        pts = []
        for item in args.beta:
            prefix, beta = item.split("=")
            sc = [f["score"] for _, f in runs_of(prefix)]
            if sc:
                pts.append({"beta": float(beta), "scores": sc, "mean": float(np.mean(sc))})
        json.dump(pts, open(os.path.join(ROOT, os.path.dirname(args.html), "beta.json"), "w"))
    if args.beta and args.plot:
        plot_beta(json.load(open(os.path.join(ROOT, os.path.dirname(args.html), "beta.json"))),
                  os.path.join(ROOT, "results", "plots", "rarity_beta.png"))
    if args.html:
        head = ("<tr><th>configuration</th><th>score (mean ± sd)</th><th>per seed</th>"
                + "".join(f"<th>{k.replace('_', ' ')}</th>" for k in KEY) + "<th>time / run</th></tr>")
        open(os.path.join(ROOT, args.html), "w", encoding="utf-8").write(head + "".join(html_rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--html", default=None)
    ap.add_argument("--beta", nargs="*", default=None, help="prefix=beta pairs for the dose-response chart")
    ap.add_argument("--plot", action="store_true", help="also write results/plots/rarity_beta.png")
    main(ap.parse_args())
