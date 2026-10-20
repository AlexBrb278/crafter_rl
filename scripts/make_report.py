"""Fill scripts/report_template.html with run data, narrative and embedded videos.

  .venv/Scripts/python scripts/make_report.py runs/report/data.json runs/report/narrative.json runs/report/crafter_report.html

narrative.json: {"DATE", "LEDE", "RESULT_TEXT", "FINDINGS" (HTML), "IMPROVE_TITLE", "IMPROVE_TEXT" (HTML),
                 "REPRO", "FOOTER", "DEATHS_TEXT" (HTML), "videos" and "death_videos": [{"run", "file", "title", "desc"}],
                 "experiments": [{"title", "html", "table", "caption", "videos"}] (one section each),
                 "tree_extra": [{"label", "prefix"}] (tech-tree entries pooled over <prefix>-s<seed> runs)}
Optional inputs next to data.json: seeds.html, beta.json, deaths.json (scripts/analyze_deaths.py).
Videos are embedded as base64 data URIs so the page is self-contained.
"""
import base64
import html
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def video_html(v):
    # prefer a re-encoded copy under runs/report/media/ (much smaller), else the original
    small = os.path.join(ROOT, "runs", "report", "media", v["run"], "videos", v["file"])
    path = small if os.path.exists(small) else os.path.join(ROOT, "runs", v["run"], "videos", v["file"])
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return (f'<div class="video"><video controls muted loop playsinline preload="auto" '
            f'src="data:video/mp4;base64,{b64}"></video><span class="t">{html.escape(v["title"])}</span>'
            f'<span class="d">{html.escape(v["desc"])}</span></div>')


def experiment_html(e):
    """One report section per follow-up experiment: {"title", "html", "table" (optional path to an
    HTML <tr> fragment, relative to the repo), "caption", "videos": [...]}."""
    table = ""
    if e.get("table"):
        table = f'<div class="tablewrap"><table>{open(os.path.join(ROOT, e["table"]), encoding="utf-8").read()}</table></div>'
    cap = f'<figcaption>{e["caption"]}</figcaption>' if e.get("caption") else ""
    vids = "\n".join(video_html(v) for v in e.get("videos", []))
    return (f'  <section class="stack">\n    <h2>{html.escape(e["title"])}</h2>\n'
            f'    <div class="prose stack">{e.get("html", "")}</div>\n    {table}{cap}\n'
            f'    <div class="videos">{vids}</div>\n  </section>')


def pooled_rates(prefix):
    """Tech-tree entry pooled over seeds: share of ALL training episodes of <prefix>-s<seed> runs
    that unlocked each achievement (and of the last 300 episodes of each seed), plus raw counts."""
    import glob
    import re
    sys.path.insert(0, ROOT)
    from common.crafter_score import ACHIEVEMENTS
    eps_all, eps_last, seeds = [], [], []
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", f"{prefix}-s*"))):
        m = re.fullmatch(re.escape(prefix) + r"-s(\d+)", os.path.basename(d))
        if not (m and os.path.exists(os.path.join(d, "final.json"))):
            continue
        eps = [json.loads(l) for l in open(os.path.join(d, "episodes.jsonl")) if l.strip()]
        eps_all += eps
        eps_last += eps[-300:]
        seeds.append(int(m.group(1)))
    hit = lambda E, a: sum(e[f"achievement_{a}"] > 0 for e in E)
    return {"rates": {a: 100 * hit(eps_all, a) / len(eps_all) for a in ACHIEVEMENTS},
            "rates_last300": {a: 100 * hit(eps_last, a) / len(eps_last) for a in ACHIEVEMENTS},
            "counts": {a: hit(eps_all, a) for a in ACHIEVEMENTS},
            "counts_last300": {a: hit(eps_last, a) for a in ACHIEVEMENTS},
            "episodes": len(eps_all), "seeds": seeds}


def main(data_path, narrative_path, out_path):
    tpl = open(os.path.join(ROOT, "scripts", "report_template.html"), encoding="utf-8").read()
    data = json.load(open(data_path))
    nar = json.load(open(narrative_path, encoding="utf-8"))
    vids = [video_html(v) for v in nar.get("videos", [])]
    for k in ("ach_curves", "clips"):  # not used by the page
        for r in data:
            r.pop(k, None)
    fills = {k: nar.get(k, "") for k in ("DATE", "LEDE", "RESULT_TEXT", "FINDINGS", "IMPROVE_TITLE",
                                          "IMPROVE_TEXT", "FOOTER")}
    fills["REPRO"] = html.escape(nar.get("REPRO", ""))
    fills["VIDEOS"] = "\n".join(vids)
    seeds = os.path.join(os.path.dirname(data_path), "seeds.html")  # from scripts/aggregate_seeds.py --html
    fills["SEEDS"] = open(seeds, encoding="utf-8").read() if os.path.exists(seeds) else ""
    fills["SEEDS_TEXT"] = nar.get("SEEDS_TEXT", "")
    beta = os.path.join(os.path.dirname(data_path), "beta.json")  # aggregate_seeds.py --beta
    fills["BETA"] = open(beta).read() if os.path.exists(beta) else "[]"
    deaths = os.path.join(os.path.dirname(data_path), "deaths.json")  # scripts/analyze_deaths.py
    fills["DEATHS"] = open(deaths).read().replace("</", "<\\/") if os.path.exists(deaths) else "null"
    fills["DEATHS_TEXT"] = nar.get("DEATHS_TEXT", "")
    fills["DEATH_VIDEOS"] = "\n".join(video_html(v) for v in nar.get("death_videos", []))
    fills["EXPERIMENTS"] = "\n".join(experiment_html(e) for e in nar.get("experiments", []))
    tree_extra = [{"label": t["label"], **pooled_rates(t["prefix"])} for t in nar.get("tree_extra", [])]
    fills["TREE_EXTRA"] = json.dumps(tree_extra)
    fills["DATA"] = json.dumps(data).replace("</", "<\\/")
    for k, v in fills.items():
        tpl = tpl.replace(f"__{k}__", v)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(tpl)
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
