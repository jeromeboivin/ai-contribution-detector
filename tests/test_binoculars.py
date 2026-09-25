import math
import subprocess

import pytest
import torch

from aicontrib.binoculars import binoculars_scores, sample_head_files


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


def test_sample_head_files_filters_extensions_and_excluded_paths(tmp_path):
    repo = tmp_path / "repo"
    for rel in ["src/a.ts", "src/b.tsx", "src/types.d.ts", "vendor/lib/x.js", "pkg/vendor/y.js",
                "README.md", "dist/bundle.js", "src/app.min.js"]:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    exclude = ["vendor/*", "*/vendor/*", "dist/*", "*.min.js", "*.d.ts"]
    files = sample_head_files(str(repo), {".ts": "javascript", ".tsx": "javascript", ".js": "javascript"}, exclude, 10)

    assert files == ["src/a.ts", "src/b.tsx"]
