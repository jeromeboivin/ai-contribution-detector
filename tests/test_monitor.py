import json
import urllib.request

from aicontrib.monitor import MetricsLogger, append_known_repo_result, serve_dashboard


def test_dashboard_serves_logged_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(metrics_path)
    logger.log(epoch=1, train_loss=1.1, val_macro_f1=0.3)
    logger.log(epoch=2, train_loss=0.9, val_macro_f1=0.4)

    server = serve_dashboard(metrics_path, port=0, background=True)
    try:
        port = server.server_address[1]

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            assert b"<html" in resp.read().lower()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics") as resp:
            rows = json.loads(resp.read())
        assert rows == [
            {"epoch": 1, "train_loss": 1.1, "val_macro_f1": 0.3},
            {"epoch": 2, "train_loss": 0.9, "val_macro_f1": 0.4},
        ]
    finally:
        server.shutdown()


def test_known_repo_results_accumulate_across_calls(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    MetricsLogger(metrics_path)  # just to establish models_dir contents, as train() would

    append_known_repo_result(tmp_path, timestamp=1.0, name="tslint", expected_class="human", accuracy=0.9)
    append_known_repo_result(tmp_path, timestamp=2.0, name="tslint", expected_class="human", accuracy=0.95)
    append_known_repo_result(tmp_path, timestamp=1.5, name="known-ai-repo", expected_class="ai", accuracy=0.8)

    server = serve_dashboard(metrics_path, port=0, background=True)
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/known-repo-results") as resp:
            rows = json.loads(resp.read())
        names = {r["name"] for r in rows}
        assert names == {"tslint", "known-ai-repo"}
        assert len(rows) == 3
    finally:
        server.shutdown()
