"""Full-history authorship timeline for a local repo, rendered as a static HTML page.

Every (non-merge) commit is classified with CommitClassifier and the monthly share
of commits per class (human / AI) is written into a self-contained HTML file
(no external scripts, opens offline). Per-commit results are cached as they're
computed, so an interrupted run resumes where it stopped and a re-run after new
commits only classifies the new ones. The cache is keyed on the model file's
hash, so retraining automatically invalidates stale predictions.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from aicontrib.config import load_config
from aicontrib.diff.commit import CommitClassifier, run_git
from aicontrib.diff.known_repo_eval import evenly_spaced

_TEMPLATE = Path(__file__).resolve().parents[1] / "static" / "timeline.html"
CLASS_LABELS = {"human": "Human", "co_authored": "Co-authored", "ai": "AI"}


def list_commits(repo_path: str) -> list[tuple[str, str]]:
    """(sha, "YYYY-MM" of the author date), oldest first. Merges are excluded: a
    merge's diff re-counts code already attributed to its branch's own commits."""
    out = run_git(repo_path, "log", "--no-merges", "--reverse", "--date=format:%Y-%m", "--format=%H %ad")
    return [tuple(line.split(" ", 1)) for line in out.splitlines() if line.strip()]


def month_range(first: str, last: str) -> list[str]:
    year, month = map(int, first.split("-"))
    last_key = tuple(map(int, last.split("-")))
    months = []
    while (year, month) <= last_key:
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def monthly_breakdown(rows: list[dict], class_names: list[str]) -> list[dict]:
    """Every calendar month from the first to the last commit, including empty ones,
    so gaps in activity show as gaps on the timeline instead of being squeezed out."""
    if not rows:
        return []
    by_month: dict[str, dict] = {}
    for row in rows:
        entry = by_month.setdefault(row["month"], {"counts": Counter(), "skipped": 0, "errors": 0})
        if row.get("error"):
            entry["errors"] += 1
        elif row["predicted"] is None:
            entry["skipped"] += 1
        else:
            entry["counts"][row["predicted"]] += 1

    empty = {"counts": Counter(), "skipped": 0, "errors": 0}
    months = sorted(by_month)
    return [
        {
            "month": month,
            "counts": {c: by_month.get(month, empty)["counts"].get(c, 0) for c in class_names},
            "skipped": by_month.get(month, empty)["skipped"],
            "errors": by_month.get(month, empty)["errors"],
        }
        for month in month_range(months[0], months[-1])
    ]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _cache_path(cfg: dict, repo_path: str, model_digest: str) -> Path:
    resolved = Path(repo_path).resolve()
    repo_id = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    return Path(cfg["paths"]["timeline_cache_dir"]) / f"{resolved.name}-{repo_id}-model-{model_digest}.jsonl"


def analyze_repo(
    repo_path: str, config_path: str | None = None, max_commits: int | None = None, use_cache: bool = True
) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    class_names = cfg["classes"]["names"]
    model_digest = _file_digest(Path(cfg["paths"]["models_dir"]) / "mlp_classifier.pt")

    all_commits = list_commits(repo_path)
    commits = evenly_spaced(all_commits, max_commits) if max_commits else all_commits

    cache_path = _cache_path(cfg, repo_path, model_digest)
    results: dict[str, dict] = {}
    if use_cache and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if not row["error"]:  # failed commits are retried: the cause may have been fixed since
                        results[row["sha"]] = row

    todo = [(sha, month) for sha, month in commits if sha not in results]
    if todo:
        print(f"{len(commits) - len(todo)} commits already cached, classifying {len(todo)} more")
        classifier = CommitClassifier(config_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a" if use_cache else "w", encoding="utf-8") as cache:
            for sha, month in tqdm(todo, desc="classifying commits", unit="commit"):
                row = {"sha": sha, "month": month, "predicted": None, "aggregate": None, "error": None}
                try:
                    aggregate = classifier.classify(repo_path, sha)["aggregate"]
                    if aggregate is not None:
                        row["aggregate"] = aggregate
                        row["predicted"] = max(aggregate, key=aggregate.get)
                except Exception as exc:  # noqa: BLE001 - one odd commit shouldn't abort an hours-long scan
                    row["error"] = f"{type(exc).__name__}: {exc}"
                cache.write(json.dumps(row) + "\n")
                cache.flush()
                results[sha] = row

    months = monthly_breakdown([results[sha] for sha, _ in commits], class_names)
    totals = {c: sum(m["counts"][c] for m in months) for c in class_names}
    return {
        # Folder name only, never the absolute path -- the page is meant to be shareable.
        "repo": Path(repo_path).resolve().name,
        "generated_at": dt.datetime.now().isoformat(timespec="minutes"),
        "model": model_digest,
        "class_names": class_names,
        "class_labels": {c: CLASS_LABELS.get(c, c) for c in class_names},
        "n_commits_total": len(all_commits),
        "n_commits_analyzed": len(commits),
        "n_classified": sum(totals.values()),
        "n_skipped": sum(m["skipped"] for m in months),
        "n_errors": sum(m["errors"] for m in months),
        "totals": totals,
        "months": months,
    }


def render_html(report: dict) -> str:
    # Escape <, >, & so no repo or file name can close the <script> tag the JSON sits in.
    data = json.dumps(report).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return _TEMPLATE.read_text(encoding="utf-8").replace("__REPORT_DATA__", data)


def write_report(
    repo_path: str,
    output: str | None = None,
    config_path: str | None = None,
    max_commits: int | None = None,
    use_cache: bool = True,
) -> Path:
    report = analyze_repo(repo_path, config_path, max_commits, use_cache)
    out_path = Path(output) if output else Path.cwd() / f"{report['repo']}-authorship-timeline.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(report), encoding="utf-8")
    return out_path.resolve()
