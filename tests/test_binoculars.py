import math
import os
import subprocess

import pytest
import torch

from aicontrib.binoculars import binoculars_scores, sample_files
from aicontrib.diff.commit import run_git
from aicontrib.diff.known_repo_eval import _sample_commit_shas


def _reference_score(observer_logits, performer_logits, input_ids):
    """Unbatched, unchunked, straight from the definition (Hans et al. 2024, Eq. 3-4)."""
    n = input_ids.shape[0] - 1
    perf_logp = performer_logits[:n].log_softmax(-1)
    obs_p = observer_logits[:n].softmax(-1)
    log_ppl = -perf_logp[torch.arange(n), input_ids[1:]].mean()
    log_x_ppl = -(obs_p * perf_logp).sum(-1).mean()
    return (log_ppl / log_x_ppl).item()


def test_uniform_models_score_one():
    # Both models uniform over V tokens: perplexity and cross-perplexity are both log V.
    logits = torch.zeros(1, 10, 50)
    ids = torch.randint(0, 50, (1, 10))
    score = binoculars_scores(logits, logits, ids, torch.ones_like(ids))
    assert score.item() == pytest.approx(1.0)


def test_matches_definition_across_chunks_and_padding():
    torch.manual_seed(0)
    vocab, long_len, short_len = 40, 12, 7
    obs, perf = torch.randn(2, long_len, vocab), torch.randn(2, long_len, vocab)
    ids = torch.randint(0, vocab, (2, long_len))
    mask = torch.ones_like(ids)
    mask[1, short_len:] = 0  # second sequence right-padded

    scores = binoculars_scores(obs, perf, ids, mask, chunk=5)  # chunk boundaries fall mid-sequence

    assert scores[0].item() == pytest.approx(_reference_score(obs[0], perf[0], ids[0]), rel=1e-5)
    assert scores[1].item() == pytest.approx(
        _reference_score(obs[1, :short_len], perf[1, :short_len], ids[1, :short_len]), rel=1e-5)
    assert all(math.isfinite(s) for s in scores.tolist())


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
