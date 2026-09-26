import os
import subprocess

import numpy as np

from aicontrib.config import load_config
from aicontrib.diff import file_eval
from aicontrib.diff.commit import run_git
from aicontrib.diff.file_eval import sample_files
from aicontrib.diff.known_repo_eval import _sample_commit_shas


def test_sample_files_filters_extensions_and_excluded_paths(tmp_path):
    repo = tmp_path / "repo"
    for rel in ["src/a.ts", "src/b.tsx", "src/types.d.ts", "vendor/lib/x.js", "pkg/vendor/y.js",
                "README.md", "dist/bundle.js", "src/app.min.js"]:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    exclude = ["vendor/*", "*/vendor/*", "dist/*", "*.min.js", "*.d.ts"]
    revision, files = sample_files(str(repo), {".ts": "javascript", ".tsx": "javascript", ".js": "javascript"}, exclude, 10)

    assert revision == "HEAD"
    assert files == ["src/a.ts", "src/b.tsx"]


def _dated_commit(repo, date, message, write=None, remove=None):
    env = {"GIT_AUTHOR_DATE": f"{date}T12:00:00", "GIT_COMMITTER_DATE": f"{date}T12:00:00",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.com", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t.com", "PATH": os.environ["PATH"]}
    for name, content in (write or {}).items():
        (repo / name).write_text(content)
        subprocess.run(["git", "add", name], cwd=repo, check=True)
    for name in remove or []:
        subprocess.run(["git", "rm", "-q", name], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True, env=env)


def _history(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _dated_commit(repo, "2019-06-01", "before the range", {"old.ts": "old\n"})
    _dated_commit(repo, "2020-03-01", "created in range", {"kept.ts": "v1\n", "gone.ts": "x\n", "notes.md": "n\n"})
    _dated_commit(repo, "2020-09-01", "deleted in range", remove=["gone.ts"])
    _dated_commit(repo, "2023-05-01", "after the range", {"kept.ts": "v2\n", "late.ts": "late\n"})
    return repo


def test_date_range_keeps_files_created_in_it_read_at_its_end(tmp_path):
    repo = _history(tmp_path)

    revision, files = sample_files(str(repo), {".ts": "javascript"}, [], 10, since="2020-01-01", until="2021-12-31")

    assert files == ["kept.ts"]  # not old.ts (created before), gone.ts (deleted), late.ts (after)
    assert run_git(str(repo), "show", f"{revision}:kept.ts") == "v1\n"  # the 2023 edit is left out


def test_until_alone_reads_every_file_as_of_that_date(tmp_path):
    repo = _history(tmp_path)

    _, files = sample_files(str(repo), {".ts": "javascript"}, [], 10, until="2021-12-31")

    assert files == ["kept.ts", "old.ts"]


def test_date_range_before_the_first_commit_finds_nothing(tmp_path):
    repo = _history(tmp_path)
    assert sample_files(str(repo), {".ts": "javascript"}, [], 10, until="2010-01-01")[1] == []


def test_evaluate_repo_commit_sampling_respects_the_date_range(tmp_path):
    repo = _history(tmp_path)

    shas = _sample_commit_shas(str(repo), 10, since="2020-01-01", until="2021-12-31")

    messages = [run_git(str(repo), "log", "-1", "--format=%s", sha).strip() for sha in shas]
    assert sorted(messages) == ["created in range", "deleted in range"]


class _StubClassifier:
    """P(ai) = 0.9 for files containing "ai", else 0.2; a whitespace "tokenizer"."""
    class_names = ["human", "ai"]

    class embedder:
        @staticmethod
        def tokenizer(text, truncation, max_length):
            return {"input_ids": text.split()[:max_length]}

    def _probabilities(self, texts):
        return np.array([[0.1, 0.9] if "ai" in t.split() else [0.8, 0.2] for t in texts])


def _repo_with(tmp_path, name, files):
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    for rel, text in files.items():
        (repo / rel).write_text(text)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    _dated_commit(repo, "2020-01-01", "init")
    return repo


def test_evaluate_files_scores_repos_and_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(file_eval, "CommitClassifier", lambda: _StubClassifier())
    words = " x" * 70
    human = _repo_with(tmp_path, "h", {"a.ts": "human" + words, "b.ts": "human" + words, "tiny.ts": "ai"})
    ai = _repo_with(tmp_path, "a", {"c.ts": "ai" + words, "d.ts": "human" + words})
    cfg = load_config()
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["known_repos"] = [{"name": "H", "path": str(human), "expected_class": "human"},
                          {"name": "A", "path": str(ai), "expected_class": "ai"}]

    result = file_eval.evaluate_files(cfg, log=lambda _: None)

    h, a = result["repos"]
    assert len(h["files"]) == 2 and h["n_files_too_short"] == 1  # tiny.ts: under min_tokens
    assert h["mean_p_ai"] == 0.2 and h["share_called_ai"] == 0.0
    assert a["mean_p_ai"] == 0.55 and a["share_called_ai"] == 0.5
    assert result["pairs"] == [{"human": "H", "ai": "A", "auc": 0.75}]
    assert (tmp_path / "models" / "file_eval_results.json").exists()
