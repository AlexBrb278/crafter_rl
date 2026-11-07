"""Figures and numbers for the slide deck, computed from the run logs in runs/ and results/.

    .venv/bin/python paper/deck/make_figures.py     # writes paper/deck/figs/*.svg and paper/deck/numbers.json
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "paper", "deck", "figs")
os.makedirs(OUT, exist_ok=True)

ACH = ["collect_coal", "collect_diamond", "collect_drink", "collect_iron", "collect_sapling", "collect_stone",
       "collect_wood", "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
       "make_iron_sword", "make_stone_pickaxe", "make_stone_sword", "make_wood_pickaxe", "make_wood_sword",
       "place_furnace", "place_plant", "place_stone", "place_table", "wake_up"]
# roughly by depth in the tech tree, as in the Crafter paper's bar charts
ORDER = ["collect_sapling", "place_plant", "wake_up", "collect_drink", "collect_wood", "place_table", "eat_cow",
         "make_wood_sword", "make_wood_pickaxe", "defeat_zombie", "collect_stone", "place_stone", "eat_plant",
         "collect_coal", "defeat_skeleton", "make_stone_sword", "place_furnace", "make_stone_pickaxe",
         "collect_iron", "make_iron_sword", "make_iron_pickaxe", "collect_diamond"]

# --- style: reference palette, light mode (validated adjacent-pair order) ---
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8985", "#e5e4e0", "#ffffff"
C = {"ours": "#2a78d6", "ppo_r": "#eb6834", "ad": "#1baf7a", "ppo": "#eda100", "random": "#8a8985",
     "violet": "#6250d6", "magenta": "#e87ba4", "green": "#008300", "red": "#e34948"}
plt.rcParams.update({
    "font.family": "Noto Sans", "font.size": 11, "svg.fonttype": "none",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 10, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "lines.linewidth": 2,
})

CONFIGS = {  # label -> (color key, runs)
    "PPO": ("ppo", ["ppo-s0", "ppo-s1", "ppo-s2"]),
    "PPO + rarity β=2": ("ppo_r", ["ppo-rarity2-s0", "ppo-rarity2-s1", "ppo-rarity2-s2"]),
    "AD (1 seed)": ("ad", ["ad-ppo_ad-s0"]),
    "AD + rarity β=2 (ours)": ("ours", ["ad-rarity2-s0", "ad-rarity2-s1", "ad-rarity2-s2"]),
}
SUB_RUNS = sorted(r for r in (os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "runs", "sub-ad-rarity2-s*")))
                  if os.path.exists(os.path.join(ROOT, "runs", r, "final.json")))


def jl(path):
    return [json.loads(line) for line in open(path)]


def metrics(run):
    return jl(os.path.join(ROOT, "runs", run, "metrics.jsonl"))


def final(run):
    return json.load(open(os.path.join(ROOT, "runs", run, "final.json")))


def episodes(run):
    return jl(os.path.join(ROOT, "runs", run, "episodes.jsonl"))


def ms(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def score(rates):
    return float(np.exp(np.mean(np.log1p(np.asarray(rates, float)))) - 1)


RANDOM = json.load(open(os.path.join(ROOT, "results", "random", "random_agent.json")))
numbers = {"random": {"score": RANDOM["score"], "reward": RANDOM["reward_mean"], "length": RANDOM["length_mean"],
                      "episodes": RANDOM["episodes"]}}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fmt_steps(ax):
    ax.set_xlim(0, 1.03e6)
    ax.set_xticks([0, 2.5e5, 5e5, 7.5e5, 1e6], ["0", "250k", "500k", "750k", "1M"])
    ax.set_xlabel("environment steps")


# ------------------------------------------------------------------ 1. score over training
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for label, (ck, runs) in CONFIGS.items():
    rows = [metrics(r) for r in runs]
    n = min(len(r) for r in rows)
    step = np.array([m["step"] for m in rows[0][:n]])
    s = np.array([[m["score"] for m in r[:n]] for r in rows])
    ax.plot(step, s.mean(0), color=C[ck], label=label)
    if len(runs) > 1:
        ax.fill_between(step, s.mean(0) - s.std(0, ddof=1), s.mean(0) + s.std(0, ddof=1), color=C[ck], alpha=0.15, lw=0)
    ax.annotate(f"{s.mean(0)[-1]:.1f}", (step[-1], s.mean(0)[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=10, color=INK2)
ax.axhline(RANDOM["score"], color=MUTED, ls="--", lw=1.5)
ax.text(1.0e6, RANDOM["score"] + 0.6, f"random agent {RANDOM['score']:.2f}", color=INK2, fontsize=10, ha="right")
fmt_steps(ax)
ax.set_ylabel("Crafter score (%)")
ax.set_ylim(0, 27)
ax.set_title("Crafter score over training (all episodes so far; mean ± sd, 3 seeds)")
ax.legend(loc="upper left")
save(fig, "score_curves")

# ------------------------------------------------------------------ 2. average episodic reward (mandatory)
BIN = 25_600
edges = np.arange(0, 1.024e6 + BIN, BIN)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for label, (ck, runs) in CONFIGS.items():
    curves = []
    for r in runs:
        ep = episodes(r)
        st = np.array([e["step"] for e in ep])
        rw = np.array([e["reward"] for e in ep])
        idx = np.digitize(st, edges) - 1
        curves.append([rw[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(edges) - 1)])
    curves = np.array(curves)
    mid = (edges[:-1] + edges[1:]) / 2
    m = np.nanmean(curves, 0)
    ax.plot(mid, m, color=C[ck], label=label)
    if len(runs) > 1:
        sd = np.nanstd(curves, 0, ddof=1)
        ax.fill_between(mid, m - sd, m + sd, color=C[ck], alpha=0.15, lw=0)
ax.axhline(RANDOM["reward_mean"], color=MUTED, ls="--", lw=1.5)
ax.text(1.0e6, RANDOM["reward_mean"] + 0.35, f"random agent {RANDOM['reward_mean']:.2f}", color=INK2, fontsize=10, ha="right")
ax.axhline(22, color=GRID, lw=0)
fmt_steps(ax)
ax.set_ylabel("episode reward (game reward)")
ax.set_ylim(0, 14.5)
ax.set_title("Average episodic reward (episodes binned per 25.6k steps; mean ± sd)")
ax.legend(loc="upper left")
save(fig, "reward_curves")

# ------------------------------------------------------------------ 3. success rates over all training episodes
def train_rates(runs):
    return np.array([[final(r)["success_rate"][a] for a in ORDER] for r in runs])


series = [("Random", "random", np.array([[RANDOM["success_rate"][a] for a in ORDER]]))]
series += [(label, ck, train_rates(runs)) for label, (ck, runs) in CONFIGS.items()]


def rate_bars(series, title, name, ylabel="success rate (%, log scale)"):
    fig, ax = plt.subplots(figsize=(12.6, 3.9))
    k = len(series)
    w = 0.84 / k
    x = np.arange(len(ORDER))
    for i, (label, ck, r) in enumerate(series):
        m = r.mean(0)
        ax.bar(x - 0.42 + w * (i + 0.5), np.maximum(m, 1e-9), w * 0.88, color=C[ck], label=label, bottom=0)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 130)
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.set_xticks(x, [a.replace("_", " ") for a in ORDER], rotation=38, ha="right", fontsize=9.5)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=len(series), loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2)
    save(fig, name)


rate_bars(series, "Success rate per achievement over all 1M training steps (mean over seeds)", "success_train")

# ------------------------------------------------------------------ 4. evaluation: final policies, fresh episodes
def eval_eps(run):
    path = os.path.join(ROOT, "results", "deaths", run + ".jsonl")
    return jl(path) if os.path.exists(path) else None


def eval_stats(run):
    eps = eval_eps(run)
    rates = [100 * np.mean([a in e["achievements"] for e in eps]) for a in ACH]
    # episode reward = unlocks + (final health - 9) / 10 (Crafter's reward, summed over the episode)
    rew = [len(e["achievements"]) + (e["inventory"]["health"] - 9) / 10 for e in eps]
    return {"n": len(eps), "score": score(rates), "reward": float(np.mean(rew)),
            "length": float(np.mean([e["length"] for e in eps])), "rates": dict(zip(ACH, rates))}


eval_series = [("Random", "random", np.array([[RANDOM["success_rate"][a] for a in ORDER]]))]
numbers["eval"] = {}
for label, (ck, runs) in CONFIGS.items():
    st = [eval_stats(r) for r in runs]
    eval_series.append((label, ck, np.array([[s["rates"][a] for a in ORDER] for s in st])))
    numbers["eval"][label] = {"score": ms([s["score"] for s in st]), "reward": ms([s["reward"] for s in st]),
                              "length": ms([s["length"] for s in st]), "episodes": sum(s["n"] for s in st)}
rate_bars(eval_series, "Final policies on fresh worlds: success rate per achievement (200 episodes per seed)",
          "success_eval")

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
labels = ["Random"] + list(CONFIGS)
cols = [C["random"]] + [C[ck] for ck, _ in CONFIGS.values()]
for ax, key, title in ((axes[0], "reward", "episode reward"), (axes[1], "score", "Crafter score (%)")):
    vals = [(RANDOM["reward_mean"] if key == "reward" else RANDOM["score"], 0.0)] + \
           [numbers["eval"][l][key] for l in CONFIGS]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, [v[0] for v in vals], 0.62, color=cols, xerr=[v[1] for v in vals], error_kw={"ecolor": INK2, "lw": 1})
    for yi, v in zip(y, vals):
        ax.text(v[0] + v[1] + 0.3, yi, f"{v[0]:.1f}", va="center", fontsize=10, color=INK2)
    ax.set_yticks(y, [l.replace(" (ours)", "") .replace(" (1 seed)", "") for l in labels] if key == "reward" else [""] * len(y),
                  fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, max(v[0] + v[1] for v in vals) * 1.22)
save(fig, "eval_bars")

# ------------------------------------------------------------------ 5. rarity bonus dose-response (PPO)
betas = {0: ["ppo-s0", "ppo-s1", "ppo-s2"], 1: ["ppo-rarity1-s0", "ppo-rarity1-s1", "ppo-rarity1-s2"],
         2: ["ppo-rarity2-s0", "ppo-rarity2-s1", "ppo-rarity2-s2"], 4: ["ppo-rarity4-s0", "ppo-rarity4-s1", "ppo-rarity4-s2"]}
fig, ax = plt.subplots(figsize=(4.2, 3.4))
numbers["beta"] = {}
for i, (b, runs) in enumerate(betas.items()):
    sc = [final(r)["score"] for r in runs]
    m, sd = ms(sc)
    numbers["beta"][b] = (m, sd)
    ax.bar(i, m, 0.6, color=C["ppo_r"] if b == 2 else "#f6b89a" if b else C["ppo"], yerr=sd,
           error_kw={"ecolor": INK2, "lw": 1})
    ax.scatter([i] * 3, sc, s=22, color=INK, zorder=3, edgecolor=SURF, lw=1.5)
    ax.text(i, m + sd + 0.6, f"{m:.1f}", ha="center", fontsize=10, color=INK)
ax.set_xticks(range(4), [f"β={b}" for b in betas])
ax.set_ylim(0, 26)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Crafter score (%)")
ax.set_title("PPO + rarity bonus: strength β")
save(fig, "beta")

# ------------------------------------------------------------------ 6. the stone-pickaxe wall (recent 100 episodes)
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3), sharey=False)
for ax, a, ttl in zip(axes, ["place_furnace", "make_stone_pickaxe", "collect_iron"],
                      ["place furnace", "make stone pickaxe", "collect iron"]):
    for label, (ck, runs) in CONFIGS.items():
        rows = [metrics(r) for r in runs]
        n = min(len(r) for r in rows)
        step = np.array([m["step"] for m in rows[0][:n]])
        v = np.array([[m["recent_success_rate"][a] for m in r[:n]] for r in rows]).mean(0)
        v = np.convolve(v, np.ones(5) / 5, mode="valid")  # 5-update moving average
        ax.plot(step[4:], v, color=C[ck], label=label, lw=1.8)
    fmt_steps(ax)
    ax.set_title(ttl + " (last 100 episodes, %)", fontsize=11)
axes[0].legend(loc="upper left", fontsize=9)
save(fig, "tech_wall")

# ------------------------------------------------------------------ 7. losses and other training metrics (ours)
runs = CONFIGS["AD + rarity β=2 (ours)"][1]
rows = [metrics(r) for r in runs]
fig, axes = plt.subplots(1, 5, figsize=(12.6, 2.6))
for ax, key, ttl in zip(axes, ["entropy", "vf_loss", "pi_loss", "pred_loss", "match_loss"],
                        ["policy entropy", "value loss (normalised)", "PPO policy loss", "AD prediction loss",
                         "AD matching loss"]):
    for r, rr in zip(runs, rows):
        pts = [(m["step"], m[key]) for m in rr if key in m]
        s, v = np.array(pts).T
        if key in ("entropy", "vf_loss", "pi_loss"):
            v = np.convolve(v, np.ones(5) / 5, mode="valid")
            s = s[2:-2]
        ax.plot(s, v, color=C["ours"], lw=1.4, alpha=0.85)
    fmt_steps(ax)
    ax.set_xlabel("")
    ax.set_title(ttl, fontsize=10.5)
    ax.tick_params(labelsize=9)
save(fig, "losses")

fig, ax = plt.subplots(figsize=(4.4, 3.0))
for label, (ck, runs_) in CONFIGS.items():
    curves = []
    for r in runs_:
        ep = episodes(r)
        st = np.array([e["step"] for e in ep])
        ln = np.array([e["length"] for e in ep])
        idx = np.digitize(st, edges) - 1
        curves.append([ln[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(edges) - 1)])
    ax.plot((edges[:-1] + edges[1:]) / 2, np.nanmean(curves, 0), color=C[ck], label=label, lw=1.6)
ax.axhline(RANDOM["length_mean"], color=MUTED, ls="--", lw=1.2)
ax.text(1.0e6, RANDOM["length_mean"] + 3, "random", color=INK2, fontsize=9, ha="right")
ax.legend(loc="upper left", fontsize=8.5, handlelength=1.2)
fmt_steps(ax)
ax.set_title("episode length (steps)", fontsize=11)
save(fig, "length")

# ------------------------------------------------------------------ 8. causes of death (final policies)
import csv
rows = {r["label"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results", "deaths", "summary.csv")))}
causes = [("cause: zombie, awake", "zombie (awake)"), ("cause: zombie, asleep", "zombie (asleep)"),
          ("cause: skeleton arrow", "skeleton arrow"), ("cause: thirst", "thirst"),
          ("cause: hunger / fatigue", "hunger / fatigue"), ("cause: lava", "lava")]
dlabels = [("PPO", "PPO"), ("PPO + rarity β=2", "PPO + rarity β=2"), ("AD", "AD (1 seed)"),
           ("AD + rarity β=2", "AD + rarity β=2 (ours)")]
ccols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
fig, ax = plt.subplots(figsize=(6.4, 2.7))
for yi, (key, name) in enumerate(dlabels[::-1]):
    left = 0.0
    for (col, cname), cc in zip(causes, ccols):
        v = float(rows[key][col])
        ax.barh(yi, v, 0.62, left=left, color=cc, edgecolor=SURF, lw=1.5, label=cname if yi == 0 else None)
        if v >= 7:
            ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=9, color="#ffffff")
        left += v
ax.set_yticks(range(len(dlabels)), [n for _, n in dlabels[::-1]], fontsize=10)
ax.set_xlim(0, 100)
ax.set_xlabel("% of deaths (600 fresh episodes per config; AD: 200)")
ax.grid(axis="y", visible=False)
ax.legend(ncol=6, loc="upper left", bbox_to_anchor=(-0.32, 1.22), fontsize=9, handlelength=1.2, columnspacing=1)
save(fig, "deaths")
numbers["deaths"] = {k: {c: float(rows[k][c]) for c in rows[k] if c != "label"} for k, _ in dlabels}

# ------------------------------------------------------------------ 9. every configuration we trained
groups = [
    ("baselines", [("Random agent", None, "random"), ("PPO", "ppo-s", "ppo"), ("AD (paper config, 1 seed)", "ad-ppo_ad-s", "ad")]),
    ("rarity bonus", [("PPO + rarity β=1", "ppo-rarity1-s", "x"), ("PPO + rarity β=2", "ppo-rarity2-s", "ppo_r"),
                      ("PPO + rarity β=4", "ppo-rarity4-s", "x"), ("AD + rarity β=1 (1 seed)", "ad-rarity1-s", "x"),
                      ("AD + rarity β=2 (ours)", "ad-rarity2-s", "ours")]),
    ("variations (3 seeds each)", [("PPO + rarity, γ=0.99", "ppo-rarity2-g99-s", "x"), ("+ GRU memory", "ad-rarity2-gru-s", "x"),
                        ("+ survival shaping", "ad-rarity2-vit1-s", "x"), ("+ survival shaping, no drift", "ad-rarity2-vit1u-s", "x"),
                        ("+ own pixel map", "ad-rarity2-map-s", "x"), ("+ map + survival", "ad-rarity2-map-vit1u-s", "x")]),
    ("beyond 1M (world model)", [("+ dreams (frozen world model)", "ad-rarity2-dream-s", "violet"),
                                 ("+ dreams + craft rule", "ad-rarity2-dream-craft-s", "violet"),
                                 ("+ dreams, start pool", "ad-rarity2-dream-pool-s", "violet")]),
]
if SUB_RUNS:
    groups[1][1].append(("ours, re-run with submitted code", "sub-ad-rarity2-s", "ours"))
numbers["configs"] = {}
fig, ax = plt.subplots(figsize=(6.6, 6.0))
y = 0
yt, yl = [], []
for gname, items in groups:
    ax.text(-0.5, y - 0.15, gname.upper(), fontsize=9, color=MUTED, fontweight="bold", ha="left", va="bottom")
    y -= 0.85
    for label, prefix, ck in items:
        if prefix is None:
            sc = [RANDOM["score"]]
        else:
            rs = sorted(r for r in os.listdir(os.path.join(ROOT, "runs")) if r.startswith(prefix) and r[len(prefix):].isdigit()
                        and os.path.exists(os.path.join(ROOT, "runs", r, "final.json")))
            sc = [final(r)["score"] for r in rs]
        m, sd = ms(sc)
        numbers["configs"][label] = {"mean": m, "sd": sd, "seeds": sc}
        col = C.get(ck, "#c9c8c2")
        ax.barh(y, m, 0.66, color=col, xerr=sd if len(sc) > 1 else None, error_kw={"ecolor": INK2, "lw": 1})
        if len(sc) > 1:
            ax.scatter(sc, [y] * len(sc), s=12, color=INK, zorder=3, edgecolor=SURF, lw=1)
        ax.text(max(m + sd, max(sc)) + 0.5, y, f"{m:.1f}" + (f" ± {sd:.1f}" if len(sc) > 1 else ""), va="center",
                fontsize=9.5, color=INK)
        yt.append(y)
        yl.append(label)
        y -= 1
    y -= 0.35
ax.axvline(21.79, color=C["ad"], ls=":", lw=1.4)
ax.text(21.9, 0.25, "AD paper 21.8", fontsize=8.5, color=INK2, va="bottom")
ax.axvline(15.60, color=C["ppo"], ls=":", lw=1.4)
ax.text(15.7, 0.25, "PPO paper 15.6", fontsize=8.5, color=INK2, va="bottom")
ax.set_yticks(yt, yl, fontsize=9.5)
ax.set_ylim(y + 0.6, 1.0)
ax.set_xlim(0, 34)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Crafter score (%), 1M steps; bars = mean ± sd, dots = seeds")
save(fig, "all_configs")

# ------------------------------------------------------------------ numbers for the slides
numbers["train"] = {}
for label, (ck, runs) in CONFIGS.items():
    fs = [final(r) for r in runs]
    numbers["train"][label] = {
        "score": ms([f["score"] for f in fs]), "seeds": [f["score"] for f in fs],
        "reward_all": ms([f["reward_mean_all"] for f in fs]), "reward_last100": ms([f["reward_mean_last100"] for f in fs]),
        "hours": ms([f["wall_clock_h"] for f in fs]),
        "rates": {a: ms([f["success_rate"][a] for f in fs]) for a in ACH}}
if SUB_RUNS:
    fs = [final(r) for r in SUB_RUNS]
    numbers["submitted"] = {"runs": SUB_RUNS, "score": ms([f["score"] for f in fs]), "seeds": [f["score"] for f in fs],
                            "reward": ms([f["reward_mean"] for f in fs]), "hours": ms([f["wall_clock_h"] for f in fs]),
                            "eval_score": ms([f["eval"]["score"] for f in fs if "eval" in f]),
                            "eval_reward": ms([f["eval"]["reward_mean"] for f in fs if "eval" in f])}
json.dump(numbers, open(os.path.join(ROOT, "paper", "deck", "numbers.json"), "w"), indent=1)
print("figures in", OUT)
print(json.dumps({k: v for k, v in numbers.items() if k in ("random", "eval", "beta", "submitted")}, indent=1))
print({k: (round(v["mean"], 2), round(v["sd"], 2)) for k, v in numbers["configs"].items()})
