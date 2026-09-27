import os
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from aicontrib.config import load_config
from aicontrib.model.classifier import MLPClassifier


def _run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _make_repo(repo_path: Path) -> None:
    repo_path.mkdir()
    _run("git", "init", "-q", cwd=repo_path)
    _run("git", "config", "user.email", "t@t.com", cwd=repo_path)
    _run("git", "config", "user.name", "t", cwd=repo_path)
    for i in range(3):
        (repo_path / "f.py").write_text(f"def f():\n    return {i}\n")
        _run("git", "add", "f.py", cwd=repo_path)
        _run("git", "commit", "-q", "-m", f"commit {i}", cwd=repo_path)


def _make_checkpoint(models_dir: Path, dim: int, class_names: list[str]) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    model = MLPClassifier(input_dim=dim, hidden_dims=[8], num_classes=n, dropout=0.0)
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": dim, "hidden_dims": [8], "num_classes": n, "dropout": 0.0,
         "class_names": class_names},
        models_dir / "mlp_classifier.pt",
    )


@pytest.fixture
def config_path(tmp_path):
    cfg = load_config()  # paths here are already absolute (see config.py), safe to dump as-is
    cfg["paths"]["models_dir"] = str(tmp_path / "models")

    try:
        from aicontrib.features.embed import CodeEmbedder

        embedder = CodeEmbedder(cfg)  # skip if the real encoder can't be downloaded, same as other tests
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"encoder unavailable, skipping known-repo eval test: {exc}")
    _make_checkpoint(tmp_path / "models", embedder.dim, cfg["classes"]["names"])

    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_evaluate_known_repo_structure(tmp_path, config_path):
    from aicontrib.diff.known_repo_eval import evaluate_known_repo

    repo_path = tmp_path / "repo"
    _make_repo(repo_path)

    result = evaluate_known_repo(str(repo_path), "ai", n_samples=10, config_path=config_path)

    assert result["n_commits_sampled"] == 3
    assert result["n_commits_evaluated"] == 3  # all 3 touch a supported .py file
    assert 0.0 <= result["accuracy"] <= 1.0
    assert "Python" in result["mean_expected_class_probability_by_language"]
    assert result["by_language"]["Python"]["n_commits"] == 3
    assert len(result["per_commit"]) == 3
    assert set(result["mean_probabilities"]) == {"human", "ai"}
    assert sum(result["mean_probabilities"].values()) == pytest.approx(1.0)
    assert result["mean_probabilities"]["ai"] == pytest.approx(result["mean_expected_class_probability"])
    assert sum(result["predicted_counts"].values()) == 3


def test_commit_sampling_respects_the_date_range(tmp_path):
    from aicontrib.diff.commit import run_git
    from aicontrib.diff.known_repo_eval import _sample_commit_shas

    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git", "init", "-q", cwd=repo)
    for date in ["2019-06-01", "2020-03-01", "2020-09-01", "2023-05-01"]:
        (repo / "f.ts").write_text(date)
        _run("git", "add", "f.ts", cwd=repo)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.com", "commit", "-q", "-m", date],
                       cwd=repo, check=True, env={**os.environ, "GIT_AUTHOR_DATE": f"{date}T12:00:00",
                                                  "GIT_COMMITTER_DATE": f"{date}T12:00:00"})

    shas = _sample_commit_shas(str(repo), 10, since="2020-01-01", until="2021-12-31")

    messages = [run_git(str(repo), "log", "-1", "--format=%s", sha).strip() for sha in shas]
    assert sorted(messages) == ["2020-03-01", "2020-09-01"]


def test_pair_aucs_compares_every_human_repo_with_every_ai_repo():
    from aicontrib.diff.known_repo_eval import pair_aucs

    from aicontrib.diff.known_repo_eval import language_breakdown

    def repo(name, expected_class, commits):
        """commits: (P(ai), language) per commit, each changing one language."""
        per_commit = [{"aggregate": {"human": 1 - p, "ai": p}, "per_language": {lang: {"human": 1 - p, "ai": p}}}
                      for p, lang in commits]
        return {"name": name, "expected_class": expected_class, "per_commit": per_commit,
                "by_language": language_breakdown(per_commit, expected_class, ["human", "ai"])}

    pairs = pair_aucs([repo("h", "human", [(0.1, "Python"), (0.2, "C++"), (0.6, "Python")]),
                       repo("a", "ai", [(0.5, "Python"), (0.9, "TypeScript")]), repo("empty", "ai", [])])

    assert len(pairs) == 1
    assert (pairs[0]["human"], pairs[0]["ai"]) == ("h", "a")
    assert pairs[0]["auc"] == pytest.approx(5 / 6)  # 0.6 > 0.5: one of 6 pairs misordered
    # Only languages both repos have; Python: 0.5 beats 0.1 but not 0.6.
    assert pairs[0]["by_language"] == {"Python": {"auc": pytest.approx(0.5), "n_human": 2, "n_ai": 1}}


def test_language_breakdown_scores_each_language_on_its_own():
    from aicontrib.diff.known_repo_eval import language_breakdown

    per_commit = [{"per_language": {"Python": {"human": 0.2, "ai": 0.8}, "C++": {"human": 0.7, "ai": 0.3}}},
                  {"per_language": {"Python": {"human": 0.4, "ai": 0.6}}}]

    b = language_breakdown(per_commit, "ai", ["human", "ai"])

    assert b["Python"]["n_commits"] == 2 and b["Python"]["accuracy"] == 1.0
    assert b["Python"]["mean_expected_class_probability"] == pytest.approx(0.7)
    assert b["C++"]["n_commits"] == 1 and b["C++"]["accuracy"] == 0.0
