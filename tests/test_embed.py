import numpy as np
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


@pytest.fixture(scope="module")
def qwen_cfg():
    cfg = load_config()
    cfg["embedding"].update(model_name="Qwen/Qwen2.5-Coder-0.5B", representation="hidden", hidden_layers=[12, 24])
    return cfg


def test_language_model_hidden_states(qwen_cfg):
    try:
        from aicontrib.features.embed import CodeEmbedder

        embedder = CodeEmbedder(qwen_cfg)
    except OSError as exc:
        pytest.skip(f"Qwen2.5-Coder unavailable: {exc}")

    assert embedder.dim == 2 * 896
    assert embedder.representation_id() == "Qwen/Qwen2.5-Coder-0.5B|hidden:12,24"
    assert embedder.calibrated  # runs in its native precision, no fp32 comparison
    short = "const answer: number = 42;"
    long = "export function add(a: number, b: number): number {\n  return a + b;\n}\n" * 8
    alone = embedder.embed_batch([short])
    padded = embedder.embed_batch([long, short])  # short padded (on the right) to long's length
    assert padded.shape == (2, 1792)
    assert np.allclose(alone[0], padded[1], rtol=1e-3, atol=1e-3 * np.abs(alone[0]).max())


def test_language_model_has_no_projected_embedding(qwen_cfg):
    from aicontrib.features.embed import CodeEmbedder

    cfg = {**qwen_cfg, "embedding": {**qwen_cfg["embedding"], "representation": "projected"}}
    try:
        with pytest.raises(ValueError, match="representation: hidden"):
            CodeEmbedder(cfg)
    except OSError as exc:
        pytest.skip(f"Qwen2.5-Coder unavailable: {exc}")


def test_representation_slug_is_a_safe_file_name():
    from aicontrib.features.embed import representation_slug

    assert representation_slug("hidden:6,12") == "hidden-6_12"  # name of existing caches
    assert representation_slug("Qwen/Qwen2.5-Coder-0.5B|hidden:12,24") == "Qwen-Qwen2.5-Coder-0.5B-hidden-12_24"
