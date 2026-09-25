import pytest

from aicontrib.config import load_config


def test_embed_batch_shape():
    """Integration smoke test: downloads the real encoder, so skip if offline."""
    cfg = load_config()
    try:
        from aicontrib.features.embed import CodeEmbedder

        embedder = CodeEmbedder(cfg)
    except Exception as exc:  # noqa: BLE001 - network/model download can fail in CI/offline
        pytest.skip(f"encoder unavailable, skipping embedding smoke test: {exc}")

    out = embedder.embed_batch(["def foo():\n    return 1\n", "const x = 1;"])
    assert out.shape == (2, cfg["embedding"]["dim"])
