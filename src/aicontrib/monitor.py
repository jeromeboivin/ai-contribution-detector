"""Live training-loss dashboard.

`train.py` appends one JSON line per epoch to a metrics file; `evaluate-repo`
appends one JSON line per run to a sibling known-repo-results file. This module
serves a small self-contained HTML page (no external JS libraries -- it draws
its own line charts on <canvas>) that polls both and redraws live. Can run
either embedded in the training process (started automatically by `train()`)
or standalone via `python -m aicontrib monitor` against an already-running or
finished training run.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_STATIC_DIR = Path(__file__).parent / "static"
KNOWN_REPO_RESULTS_FILENAME = "known_repo_results.jsonl"


class MetricsLogger:
    """Appends one JSON line per epoch. Truncates on construction so stale
    metrics from a previous run don't leak into a fresh dashboard."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def log(self, **fields) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(fields) + "\n")


def append_known_repo_result(models_dir: Path, **fields) -> None:
    """Appends one evaluate-repo run's result. Never truncates -- unlike training
    metrics, results across separate CLI invocations (e.g. one per retrain) are
    meant to accumulate so the dashboard can show a trend over time."""
    path = Path(models_dir) / KNOWN_REPO_RESULTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(fields) + "\n")


def _read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_handler(metrics_path: Path) -> type[BaseHTTPRequestHandler]:
    known_repo_results_path = metrics_path.parent / KNOWN_REPO_RESULTS_FILENAME

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - silence default request logging
            pass

        def _send_json(self, rows):
            body = json.dumps(rows).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/metrics":
                self._send_json(_read_metrics(metrics_path))
            elif self.path == "/known-repo-results":
                self._send_json(_read_metrics(known_repo_results_path))
            else:
                body = (_STATIC_DIR / "dashboard.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    return Handler


def serve_dashboard(metrics_path: Path, port: int, background: bool = False) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(metrics_path))
    if background:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    else:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return server
