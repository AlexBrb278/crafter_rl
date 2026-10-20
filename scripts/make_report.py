"""Fill scripts/report_template.html with run data, narrative and embedded videos.

  .venv/Scripts/python scripts/make_report.py runs/report/data.json runs/report/narrative.json runs/report/crafter_report.html

narrative.json: {"DATE", "LEDE", "RESULT_TEXT", "FINDINGS" (HTML), "IMPROVE_TITLE", "IMPROVE_TEXT" (HTML),
                 "REPRO", "FOOTER", "videos": [{"run", "file", "title", "desc"}]}
Videos are embedded as base64 data URIs so the page is self-contained.
"""
import base64
import html
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(data_path, narrative_path, out_path):
    tpl = open(os.path.join(ROOT, "scripts", "report_template.html"), encoding="utf-8").read()
    data = json.load(open(data_path))
    nar = json.load(open(narrative_path, encoding="utf-8"))
    vids = []
    for v in nar.get("videos", []):
        # prefer a re-encoded copy under runs/report/media/ (much smaller), else the original
        small = os.path.join(ROOT, "runs", "report", "media", v["run"], "videos", v["file"])
        path = small if os.path.exists(small) else os.path.join(ROOT, "runs", v["run"], "videos", v["file"])
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        vids.append(f'<div class="video"><video controls muted loop playsinline preload="auto" '
                    f'src="data:video/mp4;base64,{b64}"></video><span class="t">{html.escape(v["title"])}</span>'
                    f'<span class="d">{html.escape(v["desc"])}</span></div>')
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
    fills["DATA"] = json.dumps(data).replace("</", "<\\/")
    for k, v in fills.items():
        tpl = tpl.replace(f"__{k}__", v)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(tpl)
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
