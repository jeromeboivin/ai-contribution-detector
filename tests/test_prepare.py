import json

from aicontrib.data import prepare


def _fake_sources(rows_by_source):
    """rows_by_source: {source_name: [(code, class_name), ...]}"""

    def fake_iter_source(source_cfg, out_name):
        return iter((code, cls, None) for code, cls in rows_by_source.get(source_cfg["name"], []))

    return fake_iter_source


def _base_cfg(tmp_path, per_class_cap):
    return {
        "dataset": {
            "sources": [
                {"name": "a", "enabled": True},
                {"name": "b", "enabled": True},
            ],
            "per_class_cap": {"train": per_class_cap, "validation": per_class_cap, "test": per_class_cap},
        },
        "classes": {"names": ["human", "co_authored", "ai"]},
        "paths": {"processed_dir": str(tmp_path)},
    }


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_uncapped_takes_everything_available(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare,
        "iter_source",
        _fake_sources(
            {
                "a": [("h1", "human"), ("h2", "human"), ("a1", "ai")],
                "b": [("h3", "human"), ("c1", "co_authored")],
            }
        ),
    )
    cfg = _base_cfg(tmp_path, per_class_cap=None)
    out_path = prepare.prepare_split(cfg, "train")
    rows = _read_jsonl(out_path)

    assert len(rows) == 5  # nothing capped
    codes = {r["code"] for r in rows}
    assert codes == {"h1", "h2", "a1", "h3", "c1"}


def test_cap_stops_pulling_once_reached_across_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare,
        "iter_source",
        _fake_sources(
            {
                "a": [("h1", "human"), ("h2", "human")],
                "b": [("h3", "human"), ("h4", "human")],
            }
        ),
    )
    cfg = _base_cfg(tmp_path, per_class_cap=3)
    out_path = prepare.prepare_split(cfg, "train")
    rows = _read_jsonl(out_path)

    assert len(rows) == 3  # capped at 3 human rows even though 4 were available
    assert {r["code"] for r in rows} == {"h1", "h2", "h3"}


def test_rows_without_code_are_skipped(tmp_path, monkeypatch):
    # CodeMirage's train split really contains rows whose code is None.
    monkeypatch.setattr(
        prepare,
        "iter_source",
        _fake_sources({"a": [(None, "ai"), ("   \n", "human"), ("real code", "human")]}),
    )
    cfg = _base_cfg(tmp_path, per_class_cap=None)
    rows = _read_jsonl(prepare.prepare_split(cfg, "train"))
    assert [r["code"] for r in rows] == ["real code"]


def test_exact_duplicate_code_is_deduped_across_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare,
        "iter_source",
        _fake_sources(
            {
                "a": [("same code", "human")],
                "b": [("same code", "human"), ("unique", "ai")],
            }
        ),
    )
    cfg = _base_cfg(tmp_path, per_class_cap=None)
    out_path = prepare.prepare_split(cfg, "train")
    rows = _read_jsonl(out_path)

    assert len(rows) == 2
    assert {r["code"] for r in rows} == {"same code", "unique"}


def test_remap_merges_co_authored_into_human(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare,
        "iter_source",
        _fake_sources({"a": [("h1", "human"), ("c1", "co_authored"), ("a1", "ai"), ("c2", "co_authored")]}),
    )
    cfg = _base_cfg(tmp_path, per_class_cap=2)
    cfg["classes"] = {"names": ["human", "ai"], "remap": {"co_authored": "human"}}

    rows = _read_jsonl(prepare.prepare_split(cfg, "train"))

    assert {r["code"]: r["label"] for r in rows} == {"h1": 0, "c1": 0, "a1": 1}  # c2: human cap of 2 reached
