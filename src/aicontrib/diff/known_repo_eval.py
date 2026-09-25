"""Real-world sanity check: run classify_commit across a sample of commits from
a local repo whose authorship is already known with certainty (e.g. a repo the
user knows was 100% AI-written), and report how well the classifier agrees.

This is a complement to the held-out AICD-Bench/CodeMirage test split, not a
replacement for it -- it catches domain-shift failures (e.g. the TypeScript
JS-proxy approximation, or a codebase style far from the training distribution)
that a benchmark test set drawn from the same sources as training can't reveal.
"""
from __future__ import annotations

from collections import defaultdict

from aicontrib.config import load_config
from aicontrib.diff.commit import CommitClassifier, run_git


def evenly_spaced(items: list, n: int) -> list:
    """Evenly spaced across the full list rather than the first/last n, so early
    scaffolding commits and late feature work are both represented."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _sample_commit_shas(repo_path: str, n_samples: int) -> list[str]:
    # --no-merges: a merge's diff re-counts code already attributed to its branch's commits.
    return evenly_spaced(run_git(repo_path, "log", "--no-merges", "--format=%H").splitlines(), n_samples)


def evaluate_known_repo(
    repo_path: str, expected_class: str, n_samples: int = 50, config_path: str | None = None
) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    class_names = cfg["classes"]["names"]
    if expected_class not in class_names:
        raise ValueError(f"expected_class must be one of {class_names}, got {expected_class!r}")

    shas = _sample_commit_shas(repo_path, n_samples)
    classifier = CommitClassifier(config_path)

    per_commit = []
    skipped = 0
    per_language_probs: dict[str, list[float]] = defaultdict(list)

    for sha in shas:
        result = classifier.classify(repo_path, sha)
        if result["aggregate"] is None:
            skipped += 1
            continue
        predicted = max(result["aggregate"], key=result["aggregate"].get)
        per_commit.append(
            {
                "commit": sha,
                "predicted": predicted,
                "expected_class_probability": result["aggregate"][expected_class],
                "aggregate": result["aggregate"],
            }
        )
        for f in result["files"]:
            per_language_probs[f["language"]].append(f["probabilities"][expected_class])

    n_evaluated = len(per_commit)
    n_correct = sum(1 for c in per_commit if c["predicted"] == expected_class)
    mean_expected_prob = sum(c["expected_class_probability"] for c in per_commit) / n_evaluated if n_evaluated else 0.0

    return {
        "repo": repo_path,
        "expected_class": expected_class,
        "n_commits_sampled": len(shas),
        "n_commits_evaluated": n_evaluated,
        "n_commits_skipped_no_supported_files": skipped,
        "accuracy": n_correct / n_evaluated if n_evaluated else 0.0,
        "mean_expected_class_probability": mean_expected_prob,
        "mean_expected_class_probability_by_language": {
            lang: sum(probs) / len(probs) for lang, probs in per_language_probs.items()
        },
        "per_commit": per_commit,
    }
