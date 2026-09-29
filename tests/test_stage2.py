import json
import pickle

import numpy as np
import pytest
import torch
import yaml
from sklearn.metrics import roc_auc_score

from aicontrib.config import load_config
from aicontrib.features.handcrafted import FEATURES, feature_matrix


def _features(code, language, *names):
    return dict(zip(names, feature_matrix([code], [language], list(names))[0]))


def test_comments_are_found_outside_strings_only():
    ts = ('const url = "http://example.com//path";  // Fetch the user profile — once.\n'
          "/**\n * Adds two numbers.\n */\n"
          "function add(a: any, b: number) { return a?.x ?? b; }  \n")
    f = _features(ts, "TypeScript", "comment_line_ratio", "mean_comment_words", "typographic_symbols", "any_density",
                  "null_check_density", "trailing_ws_ratio")
    # " * Adds two numbers." only: not the URL line (code + comment), nor "/**" and " */" (no words).
    assert f["comment_line_ratio"] == pytest.approx(1 / 5)
    assert f["mean_comment_words"] == pytest.approx((6 + 3) / 2)  # "Fetch the user profile — once." and "Adds two numbers."
    assert f["typographic_symbols"] == pytest.approx(np.log1p(1))  # the em dash
    assert f["any_density"] > 0 and f["null_check_density"] > 0 and f["trailing_ws_ratio"] == pytest.approx(1 / 5)

    py = 'def f(x):\n    """Return x."""\n    s = "a # not a comment"  # TODO: check\n    return x\n'
    f = _features(py, "Python", "comment_line_ratio", "todo_count")
    assert f["comment_line_ratio"] == pytest.approx(1 / 4)  # the docstring; the inline # comment isn't a whole line
    assert f["todo_count"] == pytest.approx(np.log1p(1))


def test_unknown_feature_names_are_refused():
    with pytest.raises(ValueError, match="Unknown stage2 features"):
        feature_matrix(["x"], ["Python"], ["no_such_feature"])


def _code(label, i):
    # AI rows: long sentence comments and an em dash; human rows: a TODO and trailing whitespace.
    if label == "ai":
        return f"// Compute the value for item {i} — carefully and clearly.\nconst v{i} = compute({i});\n"
    return f"const v{i} = compute({i});   \n// TODO {i}\n"


def _setup(tmp_path, n_repos=(12, 3, 3), per_repo=20):
    rng = np.random.default_rng(0)
    cfg = load_config()
    for key in ("embeddings_dir", "processed_dir", "models_dir"):
        cfg["paths"][key] = str(tmp_path / key)
        (tmp_path / key).mkdir()
    (tmp_path / "agent").mkdir()
    i = 0
    for split, repos in zip(("train", "validation", "test"), n_repos):
        rows = []
        for r in range(repos):
            for k in range(per_repo):
                label = "ai" if k % 2 else "human"
                rows.append({"code": _code(label, i), "label": label, "language": "TypeScript", "repo": f"{split}/r{r}"})
                i += 1
        y = np.array([row["label"] == "ai" for row in rows], dtype=int)
        x = rng.normal(size=(len(rows), 8)).astype(np.float32)
        x[:, 0] += 0.6 * y  # a weak embedding signal: the features must add to it
        np.savez(tmp_path / "embeddings_dir" / f"{split}.npz", embeddings=x, labels=y)
        with open(tmp_path / "processed_dir" / f"{split}.jsonl", "w") as f:
            for row, lab in zip(rows, y):
                f.write(json.dumps({"code": row["code"], "label": int(lab), "language": "TypeScript"}) + "\n")
        with open(tmp_path / "agent" / f"{split}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    source = next(s for s in cfg["dataset"]["sources"] if s["name"] == "agent_commits")
    cfg["dataset"]["sources"] = [{**source, "path": str(tmp_path / "agent"), "archive": []}]
    cfg["classes"] = {"names": ["human", "ai"], "remap": {}}
    cfg["monitor"]["port"] = 0
    cfg["training"].update(epochs=30, early_stopping_patience=5)
    cfg["stage2"].update(folds=3)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path), cfg


def test_train_then_stage2_improves_held_out_repositories(tmp_path, capsys):
    from aicontrib.model.evaluate import load_checkpoint
    from aicontrib.model.stage2 import _p_ai, _split, load_stage2
    from aicontrib.model.train import train

    config_path, cfg = _setup(tmp_path)
    train(config_path)  # trains the MLP, then stage 2

    out = capsys.readouterr().out
    assert "fold 3/3" in out and "[test] AUC" in out
    stage2 = load_stage2(cfg)
    assert stage2 is not None and stage2.features == cfg["stage2"]["features"]
    x, y, codes, langs = _split(cfg, "test")
    p1 = _p_ai(load_checkpoint(tmp_path / "models_dir" / "mlp_classifier.pt", torch.device("cpu")), x, torch.device("cpu"))
    p2 = stage2.apply(p1, codes, langs)
    assert roc_auc_score(y, p2) > roc_auc_score(y, p1)
    assert roc_auc_score(y, p2) > 0.95  # the features separate these rows


def test_stage2_trained_for_another_model_is_not_used(tmp_path):
    from aicontrib.model.stage2 import STAGE2_FILENAME, load_stage2

    cfg = load_config()
    cfg["paths"]["models_dir"] = str(tmp_path)
    (tmp_path / "mlp_classifier.pt").write_bytes(b"model-v1")
    notes = []
    assert load_stage2(cfg, log=notes.append) is None and "train-stage2" in notes[-1]  # none trained yet

    from aicontrib.model.stage2 import file_digest

    with open(tmp_path / STAGE2_FILENAME, "wb") as f:
        pickle.dump({"model": None, "features": [], "mlp_digest": file_digest(tmp_path / "mlp_classifier.pt")}, f)
    assert load_stage2(cfg, log=notes.append) is not None
    (tmp_path / "mlp_classifier.pt").write_bytes(b"model-v2")  # retrained since
    assert load_stage2(cfg, log=notes.append) is None and "earlier model" in notes[-1]


def test_commit_classifier_applies_stage2_to_each_file():
    from aicontrib.diff.commit import CommitClassifier

    class Embedder:
        @staticmethod
        def embed_batch(texts):
            return np.zeros((len(texts), 4))

    class Stage2:
        seen = None

        def apply(self, p_ai, codes, languages):
            Stage2.seen = (list(p_ai), codes, languages)
            return np.full(len(codes), 0.9)

    clf = object.__new__(CommitClassifier)
    clf.cfg = load_config()
    clf.device = torch.device("cpu")
    clf.embedder = Embedder()
    clf.model = lambda x: torch.zeros(len(x), 2)  # P(ai) = 0.5 from the embedding model
    clf.stage2 = Stage2()

    probs = clf._probabilities(["a", "b"], ["typescript", "csharp"])

    assert Stage2.seen == ([0.5, 0.5], ["a", "b"], ["typescript", "csharp"])
    assert probs.tolist() == [[pytest.approx(0.1), 0.9], [pytest.approx(0.1), 0.9]]
    clf.stage2 = None
    assert clf._probabilities(["a"], ["typescript"]).tolist() == [[0.5, 0.5]]


def test_every_configured_feature_exists():
    assert set(load_config()["stage2"]["features"]) <= set(FEATURES)


def test_stage2_applies_only_to_the_configured_languages():
    from aicontrib.model.stage2 import Stage2

    class Tree:
        @staticmethod
        def predict_proba(x):
            return np.c_[np.full(len(x), 0.1), np.full(len(x), 0.9)]

    bundle = {"model": Tree(), "features": ["todo_count"], "mlp_digest": "x", "digest": "d"}
    p = Stage2(bundle, ["TypeScript", "C#"]).apply(np.array([0.3, 0.3, 0.3]), ["a", "b", "c"],
                                                    ["typescript", "cpp", "csharp"])
    assert p.tolist() == [0.9, 0.3, 0.9]  # C++ keeps the embedding model's score
    assert Stage2(bundle).apply(np.array([0.3]), ["a"], ["cpp"]).tolist() == [0.9]
    assert Stage2(bundle, ["TypeScript"]).digest != Stage2(bundle).digest  # a different report cache
