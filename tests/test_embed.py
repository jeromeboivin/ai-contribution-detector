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
    assert out.shape == (2, embedder.dim)


def test_hidden_representation_shape_and_padding_invariance():
    cfg = load_config()
    cfg["embedding"]["representation"] = "hidden"
    cfg["embedding"]["hidden_layers"] = [6, 12]
    try:
        from aicontrib.features.embed import CodeEmbedder

        embedder = CodeEmbedder(cfg)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"encoder unavailable, skipping embedding smoke test: {exc}")

    short, long = "const x = 1;", "function f(a, b) {\n  return a + b;\n}\n" * 10
    assert embedder.dim == 2 * 768
    assert embedder.representation_id() == "hidden:6,12"
    alone = embedder.embed_batch([short])
    padded = embedder.embed_batch([short, long])  # short gets padded to long's length
    assert padded.shape == (2, 1536)
    assert abs(alone[0] - padded[0]).max() < 1e-4 * max(1.0, abs(alone[0]).max())


def test_invalid_hidden_layers_are_refused():
    cfg = load_config()
    cfg["embedding"]["representation"] = "hidden"
    cfg["embedding"]["hidden_layers"] = [13]
    try:
        from aicontrib.features.embed import CodeEmbedder

        with pytest.raises(ValueError, match="hidden_layers"):
            CodeEmbedder(cfg)
    except OSError as exc:
        pytest.skip(f"encoder unavailable: {exc}")
