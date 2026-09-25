import numpy as np

from aicontrib.config import load_config
from aicontrib.model.generator_holdout import holdout_aucs, sample_rows


def test_sample_rows_keeps_all_humans_and_caps_each_generator():
    rows = ([{"code": f"h{i}", "language": "JavaScript", "source": "Human"} for i in range(5)]
            + [{"code": f"a{i}", "language": "JavaScript", "source": "gen-a"} for i in range(10)]
            + [{"code": f"b{i}", "language": "Python", "source": "gen-b"} for i in range(10)]
            + [{"code": "  ", "language": "JavaScript", "source": "gen-a"}])
    only_js = lambda lang: lang == "JavaScript"  # noqa: E731

    sampled = sample_rows(rows, per_generator=3, allowed=only_js, seed=0)

    sources = [r["source"] for r in sampled]
    assert sources.count("Human") == 5
    assert sources.count("gen-a") == 3
    assert "gen-b" not in sources  # Python filtered out
    assert all(r["code"].strip() for r in sampled)


def test_unseen_generator_with_a_different_fingerprint_is_missed():
    # Human code at the origin; each generator is shifted along its own feature. A model that never
    # saw gen-a learns only gen-b's direction, so it can't tell gen-a from human code.
    rng = np.random.default_rng(0)

    def make(n_per):
        x = [rng.normal(0, 1, (n_per, 4))]
        src = ["Human"] * n_per
        for g, dim in (("gen-a", 0), ("gen-b", 1)):
            block = rng.normal(0, 1, (n_per, 4))
            block[:, dim] += 4
            x.append(block)
            src += [g] * n_per
        return np.concatenate(x).astype(np.float32), np.array(src)

    cfg = load_config()
    cfg["training"].update(epochs=30, early_stopping_patience=5, batch_size=32)
    results = {r["generator"]: r for r in holdout_aucs(*make(200), *make(100), cfg, seed=0)}

    assert set(results) == {"gen-a", "gen-b"}
    for r in results.values():
        assert r["seen_auc"] > 0.95
        assert r["unseen_auc"] < 0.75
