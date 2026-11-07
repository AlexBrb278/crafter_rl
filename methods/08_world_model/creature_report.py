"""Report pieces for creature_eval.json: a table (rows only, the report wraps them) and an inline SVG of
health lost and zombies per frame against dream step, by day and night: reality vs greedy vs sampled dreams.

  .venv/bin/python methods/08_world_model/creature_report.py --model v4:model_refit.pt
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MARK = (1, 5, 15, 30, 50)  # steps with a hover point


def table(d, model, nomap):
    real_c, real_h = d["real_graded"], d["health_lost"]["real"]
    rows = [("Reality (real frames, same grader)", real_c, real_h, d["walk_in"]["real_true"], None)]
    for run, name in ((model, "map model (v4)"), (nomap, "no-map model (v3_nomap)")):
        for T, dec in ((0, "greedy"), (0.7, "sampled T = 0.7"), (1, "sampled T = 1")):
            r = d["runs"].get(f"{run}|T={T:g}")
            if r is not None and not (run == nomap and T == 0.7):
                rows.append((f"Dream, {name}, {dec}", r["counts"], r["health_lost"], r["walk_in"], r["terrain_acc"]))
    head = ["", "zombie walks into view†", "zombies per frame, night @15 / @50", "zombies within reach, night @15 / @50",
            "health lost by step 15, day / night", "health lost by step 50, day / night", "terrain right @15 / @50"]
    out = ["<tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for name, c, hl, wi, ter in rows:
        z, zr = c["zombie"]["night"], c["zombie_reach"]["night"]
        cells = [f"{wi:.0%}", f"{z[14]:.2f} / {z[49]:.2f}", f"{zr[14]:.2f} / {zr[49]:.2f}",
                 f"{hl['day'][14]:.2f} / {hl['night'][14]:.2f}", f"{hl['day'][49]:.2f} / {hl['night'][49]:.2f}",
                 f"{ter['15']:.0%} / {ter['50']:.0%}" if ter else "&ndash;"]
        out.append(f"<tr><td>{name}</td>" + "".join(f'<td class="val">{x}</td>' for x in cells) + "</tr>")
    return "".join(out)


def panel(x0, y0, w, h, title, series, ymax, yticks, fmt):
    """One small-multiple panel: series = [(label, colour var, dash, values[50])]."""
    def X(k):
        return x0 + (k - 1) / 49 * w

    def Y(v):
        return y0 + h - v / ymax * h
    s = [f'<text x="{x0}" y="{y0 - 10}" class="pt">{title}</text>']
    for t in yticks:
        s.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="gl"/>'
                 f'<text x="{x0 - 6}" y="{Y(t) + 4:.1f}" class="ax" text-anchor="end">{fmt(t)}</text>')
    for k in (1, 15, 30, 50):
        s.append(f'<text x="{X(k):.1f}" y="{y0 + h + 16}" class="ax" text-anchor="middle">{k}</text>')
    for label, col, dash, v in series:
        pts = " ".join(f"{X(k + 1):.1f},{Y(min(x, ymax)):.1f}" for k, x in enumerate(v))
        s.append(f'<polyline points="{pts}" fill="none" stroke="var({col})" stroke-width="2" '
                 f'stroke-dasharray="{dash}" stroke-linejoin="round"/>')
        for k in MARK:
            s.append(f'<circle cx="{X(k):.1f}" cy="{Y(min(v[k - 1], ymax)):.1f}" r="6" fill="transparent">'
                     f'<title>{label}, step {k}: {fmt(v[k - 1], 2)}</title></circle>')
    return "".join(s)


def chart(d, model):
    T0, T1 = d["runs"][f"{model}|T=0"], d["runs"][f"{model}|T=1"]
    S = [("Reality", "--ink", "", None), ("Dream, greedy", "--s1", "", T0), ("Dream, sampled T = 1", "--s2", "6 4", T1)]

    def ser(metric, when):
        out = []
        for label, col, dash, r in S:
            if metric == "health":
                v = (d["health_lost"]["real"] if r is None else r["health_lost"])[when]
            else:
                v = (d["real_graded"] if r is None else r["counts"])["zombie"][when]
            out.append((label, col, dash, v))
        return out
    W, PW, PH = 760, 300, 120
    f1 = lambda v, n=1: f"{v:.{n}f}"  # noqa: E731
    parts = [panel(70, 40, PW, PH, "Health lost since the dream started: day", ser("health", "day"), 2.5, (0, 1, 2), f1),
             panel(430, 40, PW, PH, "… at night", ser("health", "night"), 2.5, (0, 1, 2), f1),
             panel(70, 230, PW, PH, "Zombies in view per frame: day", ser("zombie", "day"), 1.2, (0, 0.5, 1), f1),
             panel(430, 230, PW, PH, "… at night", ser("zombie", "night"), 1.2, (0, 0.5, 1), f1)]
    legend = []
    for i, (label, col, dash, _) in enumerate(S):
        x = 70 + i * 220
        legend.append(f'<line x1="{x}" x2="{x + 26}" y1="400" y2="400" stroke="var({col})" stroke-width="2" '
                      f'stroke-dasharray="{dash}"/><text x="{x + 34}" y="404" class="lg">{label}</text>')
    style = ("<style>.crv .pt{font:600 12px var(--font-body);fill:var(--ink)}.crv .ax{font:11px var(--font-mono);"
             "fill:var(--muted)}.crv .lg{font:12px var(--font-body);fill:var(--ink-2)}.crv .gl{stroke:var(--line);"
             "stroke-width:1}</style>")
    return (f'<figure style="margin:1.2em 0"><svg class="crv" viewBox="0 0 {W} 420" role="img" '
            f'style="width:100%;max-width:{W}px;height:auto;display:block" aria-label="Health lost and zombies per '
            f'frame against dream step, reality vs greedy and sampled dreams">{style}{"".join(parts)}{"".join(legend)}'
            f'<text x="{70 + PW + 30}" y="{230 + PH + 34}" class="ax" text-anchor="middle">dream step</text></svg>'
            f'<figcaption class="eyebrow">Map model v4, 600 dreams (half started at night), real actions. '
            f'Hover a point for its value.</figcaption></figure>')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/world_model/creature_eval.json")
    ap.add_argument("--model", default="v4:model_refit.pt")
    ap.add_argument("--nomap", default="v3_nomap:model_refit.pt")
    ap.add_argument("--table", default="runs/report/creatures.html")
    ap.add_argument("--svg", default="runs/report/creatures_chart.html")
    args = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, args.json)))
    open(os.path.join(ROOT, args.table), "w").write(table(d, args.model, args.nomap))
    open(os.path.join(ROOT, args.svg), "w").write(chart(d, args.model))
    print("wrote", args.table, args.svg)
