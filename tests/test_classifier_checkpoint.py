import pytest
import torch

from aicontrib.config import load_config
from aicontrib.model.classifier import MLPClassifier
from aicontrib.model.evaluate import load_checkpoint


def _save(path, model, **extra):
    torch.save({"state_dict": model.state_dict(), "input_dim": 4, "hidden_dims": [3], "num_classes": 2,
                "dropout": 0.0, **extra}, path)


def test_input_scaling_standardizes_features():
    model = MLPClassifier(4, [3], 2, 0.0)
    x = torch.randn(500, 4) * torch.tensor([1.0, 10.0, 100.0, 0.1]) + 5
    model.fit_input_scaling(x)
    z = (x - model.input_mean) / model.input_std
    assert torch.allclose(z.mean(0), torch.zeros(4), atol=1e-4)
    assert torch.allclose(z.std(0), torch.ones(4), atol=1e-4)


def test_old_checkpoint_without_scaling_or_representation_still_loads(tmp_path):
    model = MLPClassifier(4, [3], 2, 0.0)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("input_")}
    torch.save({"state_dict": state, "input_dim": 4, "hidden_dims": [3], "num_classes": 2, "dropout": 0.0},
               tmp_path / "old.pt")
    loaded = load_checkpoint(tmp_path / "old.pt", torch.device("cpu"), representation="projected")
    assert torch.equal(loaded.input_std, torch.ones(4))


def test_checkpoint_of_another_representation_is_refused(tmp_path):
    _save(tmp_path / "m.pt", MLPClassifier(4, [3], 2, 0.0), representation="hidden:6,12")
    with pytest.raises(ValueError, match="trained on 'hidden:6,12'"):
        load_checkpoint(tmp_path / "m.pt", torch.device("cpu"), representation="projected")
    load_checkpoint(tmp_path / "m.pt", torch.device("cpu"), representation="hidden:6,12")


def test_old_three_class_checkpoint_is_refused_by_the_binary_config(tmp_path):
    model = MLPClassifier(4, [3], 3, 0.0)
    torch.save({"state_dict": model.state_dict(), "input_dim": 4, "hidden_dims": [3], "num_classes": 3,
                "dropout": 0.0}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="trained for classes \\['human', 'co_authored', 'ai'\\]"):
        load_checkpoint(tmp_path / "old.pt", torch.device("cpu"), class_names=["human", "ai"])
    load_checkpoint(tmp_path / "old.pt", torch.device("cpu"), class_names=["human", "co_authored", "ai"])


def test_checkpoint_with_other_class_names_is_refused(tmp_path):
    _save(tmp_path / "m.pt", MLPClassifier(4, [3], 2, 0.0), class_names=["human", "ai"])
    with pytest.raises(ValueError, match="classes.names"):
        load_checkpoint(tmp_path / "m.pt", torch.device("cpu"), class_names=["ai", "human"])
    load_checkpoint(tmp_path / "m.pt", torch.device("cpu"), class_names=["human", "ai"])


def _cfg_with_checkpoint(tmp_path, **ckpt):
    cfg = load_config()
    cfg["paths"]["models_dir"] = str(tmp_path)
    torch.save(ckpt, tmp_path / "mlp_classifier.pt")
    return cfg


def test_commits_are_scored_only_in_the_languages_the_model_was_trained_on(tmp_path):
    from aicontrib.diff.commit import scored_languages

    cfg = _cfg_with_checkpoint(tmp_path, languages=["C#", "TypeScript", "Go"])  # Go: not a supported extension
    cfg["dataset"]["languages"] = {"include": ["Python"]}  # the checkpoint's list wins over the config

    assert scored_languages(cfg) == ["C#", "TypeScript"]


def test_older_checkpoints_fall_back_to_the_config_languages(tmp_path):
    from aicontrib.diff.commit import scored_languages

    cfg = _cfg_with_checkpoint(tmp_path)
    cfg["dataset"]["languages"] = {"include": ["JavaScript", "C++"]}
    assert scored_languages(cfg) == ["C++", "JavaScript"]

    cfg["dataset"]["languages"] = {}
    assert scored_languages(cfg) == ["C#", "C++", "JavaScript", "Python", "TypeScript"]
