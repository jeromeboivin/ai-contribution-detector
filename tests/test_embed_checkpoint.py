import json

import numpy as np
import pytest

from aicontrib.config import load_config


@pytest.fixture
def embedder():
    cfg = load_config()
    try:
        from aicontrib.features.embed import CodeEmbedder

        return CodeEmbedder(cfg)
    except Exception as exc:  # noqa: BLE001 - network/model download can fail offline
        pytest.skip(f"encoder unavailable, skipping checkpoint smoke test: {exc}")


def _write_processed(path, n):
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"code": f"def f{i}():\n    return {i}\n", "label": i % 3}) + "\n")


def _cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    cfg["paths"]["embeddings_dir"] = str(tmp_path / "embeddings")
    cfg["embedding"] = dict(cfg["embedding"])
    cfg["embedding"]["batch_size"] = 2
    cfg["embedding"]["checkpoint_every_batches"] = 1
    return cfg


def test_resumes_from_matching_checkpoint(tmp_path, embedder):
    from aicontrib.features.embed import embed_split

    cfg = _cfg(tmp_path)
    processed_dir = tmp_path / "processed"
    embeddings_dir = tmp_path / "embeddings"
    processed_dir.mkdir()
    embeddings_dir.mkdir()
    _write_processed(processed_dir / "train.jsonl", 6)

    labels = np.array([i % 3 for i in range(6)], dtype=np.int64)
    sentinel = np.full((3, cfg["embedding"]["dim"]), -1.0, dtype=np.float32)
    np.savez(embeddings_dir / "train.partial.npz", embeddings=sentinel, labels=labels, n_done=3)

    out_path = embed_split(cfg, embedder, "train")
    data = np.load(out_path)

    assert np.array_equal(data["embeddings"][:3], sentinel)  # came from checkpoint, not recomputed
    assert not np.array_equal(data["embeddings"][3:], np.zeros((3, cfg["embedding"]["dim"])))  # actually embedded
    assert not (embeddings_dir / "train.partial.npz").exists()  # checkpoint cleaned up on completion


def test_discards_checkpoint_with_mismatched_labels(tmp_path, embedder):
    from aicontrib.features.embed import embed_split

    cfg = _cfg(tmp_path)
    processed_dir = tmp_path / "processed"
    embeddings_dir = tmp_path / "embeddings"
    processed_dir.mkdir()
    embeddings_dir.mkdir()
    _write_processed(processed_dir / "train.jsonl", 4)

    wrong_labels = np.array([9, 9, 9, 9], dtype=np.int64)
    sentinel = np.full((2, cfg["embedding"]["dim"]), -1.0, dtype=np.float32)
    np.savez(embeddings_dir / "train.partial.npz", embeddings=sentinel, labels=wrong_labels, n_done=2)

    out_path = embed_split(cfg, embedder, "train")
    data = np.load(out_path)

    assert not np.array_equal(data["embeddings"][:2], sentinel)  # stale checkpoint was ignored
