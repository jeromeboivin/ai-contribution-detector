import json

import numpy as np
import pytest
import yaml

from aicontrib.config import load_config
from aicontrib.model.train import train


def _separable(n, rng):
    # Three well-separated clusters: validation macro-F1 peaks within a few epochs, then stalls.
    # Few noise dimensions: the model standardizes its inputs, which scales near-constant noise
    # dimensions up to the signal's size -- with many of them, 90 rows would just be memorized.
    labels = np.arange(n) % 3
    x = np.eye(3, 16, dtype=np.float32)[labels] * 5 + rng.normal(0, 0.1, (n, 16)).astype(np.float32)
    return x, labels


def test_learning_rate_is_halved_on_plateau_then_training_stops(tmp_path):
    rng = np.random.default_rng(0)
    emb = tmp_path / "embeddings"
    emb.mkdir()
    for split, n in (("train", 90), ("validation", 30)):
        x, y = _separable(n, rng)
        np.savez(emb / f"{split}.npz", embeddings=x, labels=y)

    cfg = load_config()
    cfg["paths"]["embeddings_dir"] = str(emb)
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["monitor"]["port"] = 0
    cfg["training"].update(epochs=100, lr=0.001, early_stopping_patience=7,
                           lr_reduce_patience=2, lr_reduce_factor=0.5, min_lr=1e-6)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    train(str(config_path))

    rows = [json.loads(line) for line in (tmp_path / "models" / "metrics.jsonl").read_text().splitlines()]
    f1s, lrs = [r["val_macro_f1"] for r in rows], [r["lr"] for r in rows]
    best_epoch = f1s.index(max(f1s)) + 1
    assert len(rows) == best_epoch + 7  # stopped exactly `early_stopping_patience` epochs after the best
    assert lrs == sorted(lrs, reverse=True)  # the learning rate only ever goes down
    # 7 stalled epochs with lr_reduce_patience=2: halved after the 3rd and the 6th stalled epoch.
    assert lrs[-1] == pytest.approx(0.001 * 0.5 * 0.5)
    assert (tmp_path / "models" / "mlp_classifier.pt").exists()
