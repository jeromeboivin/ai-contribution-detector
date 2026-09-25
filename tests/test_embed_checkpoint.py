import json

import numpy as np
import pytest

from aicontrib.config import load_config

# Different lengths on purpose: embed_split processes them longest first, so these tests also
# check that results land back in the original row order.
CODES = [
    "x = 1\n",
    "def add(a, b):\n    return a + b\n" * 6,
    "print('hi')\n",
    "class Box:\n    def __init__(self, v):\n        self.v = v\n" * 3,
    "import os\nprint(os.getcwd())\n",
    "for i in range(10):\n    print(i * i)\n" * 9,
]


@pytest.fixture(scope="module")
def embedder():
    try:
        from aicontrib.features.embed import CodeEmbedder

        return CodeEmbedder(load_config())
    except Exception as exc:  # noqa: BLE001 - network/model download can fail offline
        pytest.skip(f"encoder unavailable, skipping checkpoint tests: {exc}")


def _setup(tmp_path, max_tokens=64):
    cfg = load_config()
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    cfg["paths"]["embeddings_dir"] = str(tmp_path / "embeddings")
    cfg["embedding"]["max_tokens_per_batch"] = max_tokens  # small, so several batches are needed
    cfg["embedding"]["checkpoint_every_seconds"] = 0  # save after every batch
    (tmp_path / "processed").mkdir()
    (tmp_path / "embeddings").mkdir()
    with open(tmp_path / "processed" / "train.jsonl", "w", encoding="utf-8") as f:
        for i, code in enumerate(CODES):
            f.write(json.dumps({"code": code, "label": i % 3}) + "\n")
    return cfg


def _cosines(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def test_output_is_in_original_row_order(tmp_path, embedder):
    from aicontrib.features.embed import embed_split

    cfg = _setup(tmp_path)
    data = np.load(embed_split(cfg, embedder, "train"))
    one_by_one = np.concatenate([embedder.embed_batch([c]) for c in CODES])
    assert _cosines(data["embeddings"], one_by_one).min() > 0.9999
    assert data["labels"].tolist() == [i % 3 for i in range(len(CODES))]


def test_interrupted_run_resumes_and_matches_an_uninterrupted_one(tmp_path, embedder, monkeypatch):
    from aicontrib.features import embed

    cfg = _setup(tmp_path)
    real = embed.CodeEmbedder.embed_encoded
    calls = {"n": 0}

    def crash_after_two_batches(self, features):
        calls["n"] += 1
        if calls["n"] > 2:
            raise KeyboardInterrupt
        return real(self, features)

    monkeypatch.setattr(embed.CodeEmbedder, "embed_encoded", crash_after_two_batches)
    with pytest.raises(KeyboardInterrupt):
        embed.embed_split(cfg, embedder, "train")
    partial = tmp_path / "embeddings" / "train.partial"
    assert len(list(partial.glob("shard-*.npz"))) == 2

    monkeypatch.setattr(embed.CodeEmbedder, "embed_encoded", real)
    resumed = np.load(embed.embed_split(cfg, embedder, "train"))["embeddings"]
    one_by_one = np.concatenate([embedder.embed_batch([c]) for c in CODES])
    assert _cosines(resumed, one_by_one).min() > 0.9999
    assert not partial.exists()  # cleaned up once the split is complete


def test_checkpoint_from_different_data_is_discarded(tmp_path, embedder):
    from aicontrib.features.embed import embed_split

    cfg = _setup(tmp_path)
    partial = tmp_path / "embeddings" / "train.partial"
    partial.mkdir()
    (partial / "meta.json").write_text(json.dumps({"data_hash": "some other data"}))
    sentinel = np.full((1, embedder.dim), 7.0, dtype=np.float32)
    np.savez(partial / "shard-00000.npz", positions=np.array([0]), embeddings=sentinel)

    data = np.load(embed_split(cfg, embedder, "train"))
    assert not np.allclose(data["embeddings"][0], 7.0)  # stale shard ignored, row 0 really embedded


def test_cached_embeddings_of_another_representation_are_recomputed(tmp_path, embedder):
    from aicontrib.features.embed import embed_split

    cfg = _setup(tmp_path)
    out = tmp_path / "embeddings" / "train.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    stale = np.full((len(CODES), 1536), 7.0, dtype=np.float32)
    np.savez(out, embeddings=stale, labels=np.zeros(len(CODES)), representation=np.array("hidden:6,12"))

    data = np.load(embed_split(cfg, embedder, "train"))
    assert data["embeddings"].shape == (len(CODES), embedder.dim)
    assert str(data["representation"]) == embedder.representation_id()


def test_cache_without_representation_counts_as_projected(tmp_path, embedder):
    from aicontrib.features.embed import embed_split

    cfg = _setup(tmp_path)
    out = tmp_path / "embeddings" / "train.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = np.full((len(CODES), embedder.dim), 7.0, dtype=np.float32)
    np.savez(out, embeddings=old, labels=np.zeros(len(CODES)))  # written by an older version

    data = np.load(embed_split(cfg, embedder, "train"))
    assert np.allclose(data["embeddings"], 7.0)  # reused, not recomputed
