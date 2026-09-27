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

from sklearn.metrics import roc_auc_score

from aicontrib.config import load_config
from aicontrib.data.languages import canonical
from aicontrib.diff.commit import CommitClassifier, run_git


def evenly_spaced(items: list, n: int) -> list:
    """Evenly spaced across the full list rather than the first/last n, so early
    scaffolding commits and late feature work are both represented."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def date_range_args(since=None, until=None) -> list[str]:
    """git log options for an optional date range. YAML turns 2021-12-31 into a date object: str() it."""
    return ([f"--since={since}"] if since else []) + ([f"--until={until}"] if until else [])


def _sample_commit_shas(repo_path: str, n_samples: int, since=None, until=None) -> list[str]:
    # --no-merges: a merge's diff re-counts code already attributed to its branch's commits.
    log = run_git(repo_path, "log", "--no-merges", "--format=%H", *date_range_args(since, until))
    return evenly_spaced(log.splitlines(), n_samples)


def evaluate_known_repo(
    repo_path: str, expected_class: str, n_samples: int = 50, config_path: str | None = None,
    since=None, until=None,
) -> dict:
    """Commits are classified exactly as `report` classifies them. since/until: only sample
    commits in that date range (anything `git log --since` accepts, e.g. 2021-12-31)."""
    cfg = load_config(config_path) if config_path else load_config()
    class_names = cfg["classes"]["names"]
    # A repo labelled with a class the model merges into another (co_authored -> human) is scored as that one.
    expected_class = (cfg["classes"].get("remap") or {}).get(expected_class, expected_class)
    if expected_class not in class_names:
        raise ValueError(f"expected_class must be one of {class_names}, got {expected_class!r}")

    shas = _sample_commit_shas(repo_path, n_samples, since, until)
    if not shas:
        raise ValueError(f"no commits in {repo_path}" + (f" between {since or 'the start'} and {until or 'now'}"
                                                          if since or until else ""))
    classifier = CommitClassifier(config_path)

    per_commit = []
    skipped = 0

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
                "per_language": {canonical(lang): agg for lang, agg in result["per_language"].items()},
            }
        )

    n_evaluated = len(per_commit)
    n_correct = sum(1 for c in per_commit if c["predicted"] == expected_class)
    mean_expected_prob = sum(c["expected_class_probability"] for c in per_commit) / n_evaluated if n_evaluated else 0.0
    # All classes, not just the expected one: shows where the missing probability goes
    # (e.g. the opposite class, or co_authored in a 3-class model), which call for different fixes.
    mean_probabilities = {
        cls: sum(c["aggregate"][cls] for c in per_commit) / n_evaluated if n_evaluated else 0.0 for cls in class_names
    }
    predicted_counts = {cls: sum(1 for c in per_commit if c["predicted"] == cls) for cls in class_names}
    by_language = language_breakdown(per_commit, expected_class, class_names)

    return {
        "repo": repo_path,
        "expected_class": expected_class,
        "n_commits_sampled": len(shas),
        "n_commits_evaluated": n_evaluated,
        "n_commits_skipped_no_supported_files": skipped,
        "accuracy": n_correct / n_evaluated if n_evaluated else 0.0,
        "mean_expected_class_probability": mean_expected_prob,
        "mean_probabilities": mean_probabilities,
        "predicted_counts": predicted_counts,
        "by_language": by_language,
        "mean_expected_class_probability_by_language": {
            lang: b["mean_expected_class_probability"] for lang, b in by_language.items()
        },
        "per_commit": per_commit,
    }


def language_breakdown(per_commit: list[dict], expected_class: str, class_names: list[str]) -> dict[str, dict]:
    """Per language: the commits that change code in it, each classified on that code alone (a commit
    changing two languages counts under both), with the same metrics as the whole-commit ones."""
    by_language: dict[str, list[dict]] = defaultdict(list)
    for c in per_commit:
        for lang, agg in c["per_language"].items():
            by_language[lang].append(agg)
    out = {}
    for lang in sorted(by_language):
        aggs = by_language[lang]
        out[lang] = {
            "n_commits": len(aggs),
            "accuracy": sum(max(a, key=a.get) == expected_class for a in aggs) / len(aggs),
            "mean_expected_class_probability": sum(a[expected_class] for a in aggs) / len(aggs),
            "mean_probabilities": {cls: sum(a[cls] for a in aggs) / len(aggs) for cls in class_names},
        }
    return out


def _auc(human: list[float], ai: list[float]) -> float:
    return float(roc_auc_score([0] * len(human) + [1] * len(ai), human + ai))


def pair_aucs(results: list[dict]) -> list[dict]:
    """For every (human repo, AI repo) pair of evaluate_known_repo results: the chance that a random
    commit of the AI repo gets a higher P(ai) than a random commit of the human one (0.5 = no signal,
    1.0 = perfect). Needs no decision threshold, so it measures separation even when the
    probabilities are off-scale. Also per language both repos have commits in, on that language's
    code. `results` items carry a "name" key."""
    def p_ai(result, lang=None):
        if lang is None:
            return [c["aggregate"]["ai"] for c in result["per_commit"]]
        return [c["per_language"][lang]["ai"] for c in result["per_commit"] if lang in c["per_language"]]

    humans = [r for r in results if r["expected_class"] == "human" and r["per_commit"]]
    ais = [r for r in results if r["expected_class"] == "ai" and r["per_commit"]]
    pairs = []
    for h in humans:
        for a in ais:
            by_language = {}
            for lang in sorted(set(h["by_language"]) & set(a["by_language"])):
                ph, pa = p_ai(h, lang), p_ai(a, lang)
                by_language[lang] = {"auc": _auc(ph, pa), "n_human": len(ph), "n_ai": len(pa)}
            pairs.append({"human": h["name"], "ai": a["name"], "auc": _auc(p_ai(h), p_ai(a)),
                          "by_language": by_language})
    return pairs
