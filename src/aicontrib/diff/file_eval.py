"""File-level check of the trained classifier against the repositories under known_repos.

`evaluate-repo` scores sampled commits of one repository at a time. This scores whole files instead
and compares repositories: for each one, the mean P(ai) and the share of files called AI; for every
pair of a `human` and an `ai` repository, the AUC -- the chance that a random file of the AI repository
gets a higher P(ai) than a random file of the human one (0.5 = no signal, 1.0 = perfect). The AUC
needs no decision threshold, so it measures separation even when the probabilities are off-scale.

Files are read whole: at HEAD, or, for an entry with `since`/`until`, the files created by commits in
that date range, as they were at its end -- so a yearly entry holds only that year's new code.
"""
from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from aicontrib.diff.commit import CommitClassifier, run_git
from aicontrib.diff.known_repo_eval import date_range_args, evenly_spaced


def sample_files(repo_path: str, extensions: dict[str, str], exclude: list[str], n: int,
                 since=None, until=None) -> tuple[str, list[str]]:
    """(revision, up to n source file paths to read at that revision), evenly spread over the sorted
    paths, minus excluded patterns (vendored and generated code: its authorship isn't the repo's).

    No date range: every file at HEAD. With one: only files *created* by commits in the range (later
    edits, maybe by someone -- or something -- else, stay out), read at the range's last commit and
    skipped if gone by then."""
    if not since and not until:
        revision, paths = "HEAD", run_git(repo_path, "ls-files", "-z").split("\0")
    else:
        revision = run_git(repo_path, "rev-list", "-1", *date_range_args(None, until), "HEAD").strip()
        if not revision:
            return "HEAD", []
        created = set(run_git(repo_path, "log", "--no-merges", "--diff-filter=A", "--name-only", "--format=", "-z",
                              *date_range_args(since, until), revision).split("\0"))
        present = run_git(repo_path, "ls-tree", "-r", "--name-only", "-z", revision).split("\0")
        paths = [p for p in present if p in created]
    keep = [p for p in paths if p and Path(p).suffix.lower() in extensions
            and not any(fnmatch(p, pat) for pat in exclude)]
    return revision, evenly_spaced(sorted(keep), n)


def auc(human: list[float], ai: list[float]) -> float:
    """P(a random AI file scores above a random human file)."""
    return float(roc_auc_score([0] * len(human) + [1] * len(ai), human + ai))


def evaluate_files(cfg: dict, files_per_repo: int | None = None, log: Callable[[str], None] = print) -> dict:
    fcfg = cfg["file_eval"]
    extensions = cfg["commit_classification"]["supported_extensions"]
    n = files_per_repo or fcfg["files_per_repo"]
    classifier = CommitClassifier()
    tokenizer = classifier.embedder.tokenizer
    ai_index = classifier.class_names.index("ai")

    repos = []
    for entry in cfg.get("known_repos", []):
        name = entry.get("name") or Path(entry["path"]).resolve().name
        since, until = entry.get("since"), entry.get("until")
        files, too_short = [], 0
        revision, paths = sample_files(entry["path"], extensions, fcfg.get("exclude") or [], n, since, until)
        for path in paths:
            text = run_git(entry["path"], "show", f"{revision}:{path}").replace("\r\n", "\n")
            # Too little code for a meaningful prediction.
            if len(tokenizer(text, truncation=True, max_length=fcfg["min_tokens"])["input_ids"]) < fcfg["min_tokens"]:
                too_short += 1
                continue
            files.append({"path": path, "language": extensions[Path(path).suffix.lower()], "text": text})
        log(f"[{name}] {len(files)} files ({too_short} skipped: under {fcfg['min_tokens']} tokens)")
        if files:
            for f, p in zip(files, classifier._probabilities([f["text"] for f in files])):
                f["p_ai"] = float(p[ai_index])
                del f["text"]
        p_ai = [f["p_ai"] for f in files]
        repos.append({"name": name, "repo": entry["path"], "expected_class": entry["expected_class"],
                      "since": str(since) if since else None, "until": str(until) if until else None,
                      "files": files, "n_files_too_short": too_short,
                      "mean_p_ai": float(np.mean(p_ai)) if p_ai else None,
                      "share_called_ai": float(np.mean([p > 0.5 for p in p_ai])) if p_ai else None})

    pairs = [{"human": h["name"], "ai": a["name"],
              "auc": auc([f["p_ai"] for f in h["files"]], [f["p_ai"] for f in a["files"]])}
             for h in repos if h["expected_class"] == "human" and h["files"]
             for a in repos if a["expected_class"] == "ai" and a["files"]]

    result = {"repos": repos, "pairs": pairs}
    out_path = Path(cfg["paths"]["models_dir"]) / "file_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["results_path"] = str(out_path)
    return result
