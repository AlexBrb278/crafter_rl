"""Live training dashboard: http://localhost:8765

Serves scripts/dashboard.html plus a small JSON API over runs/*:
  /api/runs                 -> [{name, status, final}]
  /api/run/<name>           -> {metrics: [...per-epoch...], clips: [...], config}
  /runs/<name>/videos/<f>   -> MP4 clips (with HTTP Range support for seeking)
Standard library only:  .venv/Scripts/python scripts/dashboard.py [--port 8765]
"""
import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "runs")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
DROP = {"success_rate"}  # cumulative per-achievement rates: only the latest is sent


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line being written right now
    return out


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path in ("/", "/index.html"):
            with open(PAGE, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/runs":
            runs = []
            for name in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
                d = os.path.join(RUNS, name)
                if os.path.isfile(os.path.join(d, "metrics.jsonl")):
                    runs.append({"name": name,
                                 "status": read_json(os.path.join(d, "status.json")),
                                 "final": read_json(os.path.join(d, "final.json")),
                                 "mtime": os.path.getmtime(os.path.join(d, "metrics.jsonl"))})
            return self._json(runs)
        m = re.fullmatch(r"/api/run/([\w.\-]+)", path)
        if m:
            d = os.path.join(RUNS, m.group(1))
            metrics = read_jsonl(os.path.join(d, "metrics.jsonl"))
            latest_rates = metrics[-1].get("success_rate") if metrics else None
            slim = [{k: v for k, v in r.items() if k not in DROP} for r in metrics]
            return self._json({"metrics": slim, "success_rate": latest_rates,
                               "clips": read_jsonl(os.path.join(d, "videos", "index.jsonl")),
                               "spectator": read_jsonl(os.path.join(d, "videos", "spectator.jsonl")),
                               "has_final_video": os.path.exists(os.path.join(d, "videos", "final_episode.mp4")),
                               "config": read_json(os.path.join(d, "config.json")),
                               "final": read_json(os.path.join(d, "final.json"))})
        m = re.fullmatch(r"/report/([\w.\-]+\.html)", path)  # final report pages (runs/report/)
        if m and os.path.isfile(os.path.join(RUNS, "report", m.group(1))):
            with open(os.path.join(RUNS, "report", m.group(1)), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        m = re.fullmatch(r"/runs/([\w.\-]+)/videos/([\w.\-]+\.mp4)", path)
        if m:
            return self._video(os.path.join(RUNS, m.group(1), "videos", m.group(2)))
        self._send(404, b"not found", "text/plain")

    def _video(self, fpath):
        if not os.path.isfile(fpath):
            return self._send(404, b"not found", "text/plain")
        size = os.path.getsize(fpath)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m and m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else end
        with open(fpath, "rb") as f:
            f.seek(start)
            data = f.read(end - start + 1)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{start + len(data) - 1}/{size}")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"dashboard on http://localhost:{args.port}  (runs: {RUNS})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
