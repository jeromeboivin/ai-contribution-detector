import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from aicontrib.config import load_config
from aicontrib.diff import timeline


def _git(repo, *args, date=None):
    env = dict(os.environ)
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, env=env)


def _make_repo(path: Path) -> None:
    """Commits in 2023-01 (x2) and 2023-03 (x1), a gap in 2023-02, plus a merge."""
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@t.com")
    _git(path, "config", "user.name", "t")
    for i, date in enumerate(["2023-01-05T10:00:00", "2023-01-20T10:00:00"]):
        (path / "a.py").write_text(f"x = {i}\n")
        _git(path, "add", "a.py")
        _git(path, "commit", "-q", "-m", f"c{i}", date=date)
    _git(path, "checkout", "-q", "-b", "feature")
    (path / "b.py").write_text("y = 1\n")
    _git(path, "add", "b.py")
    _git(path, "commit", "-q", "-m", "feature", date="2023-03-02T10:00:00")
    _git(path, "checkout", "-q", "main")
    _git(path, "merge", "-q", "--no-ff", "-m", "merge feature", "feature", date="2023-03-03T10:00:00")


@pytest.fixture(autouse=True)
def _scored_languages(monkeypatch):
    # The fake model files below aren't real checkpoints to read the trained languages from.
    monkeypatch.setattr(timeline, "scored_languages", lambda cfg: ["Python"])


def _config_with_model(tmp_path: Path) -> str:
    cfg = load_config()
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["paths"]["timeline_cache_dir"] = str(tmp_path / "cache")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "mlp_classifier.pt").write_bytes(b"model-v1")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


class _StubClassifier:
    calls = 0

    def __init__(self, config_path=None):
        pass

    def classify(self, repo_path, sha):
        _StubClassifier.calls += 1
        agg = {"human": 0.1, "co_authored": 0.2, "ai": 0.7}
        return {"aggregate": agg, "per_language": {"python": agg}}


def test_month_range_crosses_year_boundary():
    assert timeline.month_range("2022-11", "2023-02") == ["2022-11", "2022-12", "2023-01", "2023-02"]


def test_monthly_breakdown_fills_gaps_and_counts_skips_and_errors():
    rows = [
        {"month": "2023-01", "predicted": "ai", "error": None},
        {"month": "2023-01", "predicted": None, "error": None},
        {"month": "2023-03", "predicted": "human", "error": None},
        {"month": "2023-03", "predicted": None, "error": "boom"},
    ]
    months = timeline.monthly_breakdown(rows, ["human", "co_authored", "ai"])
    assert [m["month"] for m in months] == ["2023-01", "2023-02", "2023-03"]
    assert months[0] == {"month": "2023-01", "counts": {"human": 0, "co_authored": 0, "ai": 1}, "skipped": 1, "errors": 0}
    assert months[1]["counts"] == {"human": 0, "co_authored": 0, "ai": 0}
    assert months[2]["counts"]["human"] == 1 and months[2]["errors"] == 1


def test_monthly_breakdown_per_language_counts_each_commit_under_its_languages():
    rows = [
        {"month": "2023-01", "predicted": "ai", "languages": {"Python": "ai", "C++": "human"}, "error": None},
        {"month": "2023-01", "predicted": None, "languages": {}, "error": None},
        {"month": "2023-03", "predicted": "human", "languages": {"C++": "human"}, "error": None},
    ]
    python = timeline.monthly_breakdown(rows, ["human", "ai"], "Python")
    cpp = timeline.monthly_breakdown(rows, ["human", "ai"], "C++")
    assert [m["month"] for m in python] == ["2023-01", "2023-02", "2023-03"]  # same axis for every language
    assert python[0]["counts"] == {"human": 0, "ai": 1} and python[2]["counts"] == {"human": 0, "ai": 0}
    assert cpp[0]["counts"] == {"human": 1, "ai": 0} and cpp[2]["counts"] == {"human": 1, "ai": 0}
    assert python[0]["skipped"] == 0  # a commit without Python code isn't a skipped Python commit


def test_list_commits_excludes_merges_and_uses_author_month(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    commits = timeline.list_commits(str(repo))
    assert [month for _, month in commits] == ["2023-01", "2023-01", "2023-03"]  # merge not listed


def test_analyze_repo_caches_and_invalidates_on_new_model(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_repo(repo)
    config_path = _config_with_model(tmp_path)
    monkeypatch.setattr(timeline, "CommitClassifier", _StubClassifier)

    _StubClassifier.calls = 0
    report = timeline.analyze_repo(str(repo), config_path)
    assert _StubClassifier.calls == 3
    assert report["n_classified"] == 3 and report["totals"]["ai"] == 3
    assert report["languages"]["Python"]["totals"]["ai"] == 3
    assert [m["month"] for m in report["months"]] == ["2023-01", "2023-02", "2023-03"]

    timeline.analyze_repo(str(repo), config_path)
    assert _StubClassifier.calls == 3  # everything served from cache

    (tmp_path / "models" / "mlp_classifier.pt").write_bytes(b"model-v2")
    timeline.analyze_repo(str(repo), config_path)
    assert _StubClassifier.calls == 6  # new model -> cache ignored


def test_rendered_html_embeds_data_without_breaking_script_tag(tmp_path):
    report = {"repo": "evil</script><script>alert(1)</script>", "months": []}
    html = timeline.render_html(report)
    payload = html.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload)["repo"] == report["repo"]
    assert "__REPORT_DATA__" not in html


def test_date_range_limits_the_report_and_reuses_the_cache(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_repo(repo)
    config_path = _config_with_model(tmp_path)
    monkeypatch.setattr(timeline, "CommitClassifier", _StubClassifier)

    assert [m for _, m in timeline.list_commits(str(repo), since="2023-02-01")] == ["2023-03"]

    _StubClassifier.calls = 0
    report = timeline.analyze_repo(str(repo), config_path, until="2023-01-31")
    assert _StubClassifier.calls == 2
    assert report["n_commits_total"] == 2 and [m["month"] for m in report["months"]] == ["2023-01"]
    assert (report["since"], report["until"]) == (None, "2023-01-31")

    full = timeline.analyze_repo(str(repo), config_path)
    assert _StubClassifier.calls == 3  # only the commit outside the first range was new
    assert full["n_classified"] == 3 and full["since"] is None


def test_rows_cached_without_languages_are_reclassified(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_repo(repo)
    config_path = _config_with_model(tmp_path)
    monkeypatch.setattr(timeline, "CommitClassifier", _StubClassifier)
    timeline.analyze_repo(str(repo), config_path)
    cache = next((tmp_path / "cache").glob("*.jsonl"))
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    for row in rows[:2]:
        del row["languages"]  # as written before the per-language breakdown existed
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows))

    _StubClassifier.calls = 0
    report = timeline.analyze_repo(str(repo), config_path)
    assert _StubClassifier.calls == 2
    assert report["languages"]["Python"]["n_classified"] == 3
