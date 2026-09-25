from aicontrib import config


def test_local_yaml_overrides_any_setting_without_touching_the_rest(tmp_path, monkeypatch):
    local = tmp_path / "local.yaml"
    local.write_text(
        "dataset:\n"
        "  per_class_cap:\n"
        "    train: 10000\n"
        "training:\n"
        "  epochs: 5\n"
        "known_repos:\n"
        "  - {name: mine, path: '/x', expected_class: ai}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "LOCAL_CONFIG_PATH", local)

    cfg = config.load_config()
    assert cfg["dataset"]["per_class_cap"]["train"] == 10000
    assert cfg["dataset"]["per_class_cap"]["validation"] is None  # sibling keys kept from default.yaml
    assert cfg["dataset"]["sources"]  # untouched sections survive
    assert cfg["training"]["epochs"] == 5
    assert cfg["training"]["lr"] == 0.001
    assert cfg["known_repos"] == [{"name": "mine", "path": "/x", "expected_class": "ai"}]
    assert cfg["paths"]["models_dir"].endswith("models")  # still resolved to an absolute path


def test_no_local_yaml_means_plain_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_CONFIG_PATH", tmp_path / "absent.yaml")
    cfg = config.load_config()
    assert cfg["dataset"]["per_class_cap"]["train"] is None
    assert cfg["known_repos"] == []
