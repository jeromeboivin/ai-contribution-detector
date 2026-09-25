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


def _make_checkpoint(models_dir: Path, dim: int) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    model = MLPClassifier(input_dim=dim, hidden_dims=[8], num_classes=3, dropout=0.0)
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": dim, "hidden_dims": [8], "num_classes": 3, "dropout": 0.0},
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
    _make_checkpoint(tmp_path / "models", embedder.dim)

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
    assert "python" in result["mean_expected_class_probability_by_language"]
    assert len(result["per_commit"]) == 3
    assert set(result["mean_probabilities"]) == {"human", "co_authored", "ai"}
    assert sum(result["mean_probabilities"].values()) == pytest.approx(1.0)
    assert result["mean_probabilities"]["ai"] == pytest.approx(result["mean_expected_class_probability"])
    assert sum(result["predicted_counts"].values()) == 3


def test_evaluate_known_repo_added_files_only(tmp_path, config_path):
    from aicontrib.diff.known_repo_eval import evaluate_known_repo

    repo_path = tmp_path / "repo"
    _make_repo(repo_path)

    result = evaluate_known_repo(str(repo_path), "ai", n_samples=10, config_path=config_path, added_files_only=True)

    # Only the first commit creates f.py; the other two modify it.
    assert result["added_files_only"] is True
    assert result["n_commits_evaluated"] == 1
    assert result["n_commits_skipped_no_supported_files"] == 2
    assert sum(result["predicted_counts"].values()) == 1
