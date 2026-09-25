import json
from collections import Counter

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


def _lang_sources(rows_by_source):
    def fake_iter_source(source_cfg, out_name):
        return iter(rows_by_source.get(source_cfg["name"], []))
    return fake_iter_source


def test_balance_languages_gives_each_language_an_equal_share(tmp_path, monkeypatch):
    # Source "a" has plenty of TypeScript; without balancing it would fill the whole cap.
    a = [(f"ts{i}", cls, "TypeScript") for i in range(10) for cls in ("human", "ai")]
    b = [(f"js{i}", cls, "JavaScript") for i in range(10) for cls in ("human", "ai")]
    monkeypatch.setattr(prepare, "iter_source", _lang_sources({"a": a, "b": b}))
    cfg = _base_cfg(tmp_path, per_class_cap=4)
    cfg["classes"] = {"names": ["human", "ai"]}
    cfg["dataset"].update(languages={"include": ["TypeScript", "js"]}, balance_languages=True)

    rows = _read_jsonl(prepare.prepare_split(cfg, "train"))

    counts = Counter((r["language"], r["label"]) for r in rows)
    assert counts == {("TypeScript", 0): 2, ("TypeScript", 1): 2, ("JavaScript", 0): 2, ("JavaScript", 1): 2}


def test_balance_languages_reports_a_language_that_runs_short(tmp_path, monkeypatch, capsys):
    a = [("ts0", "human", "TypeScript"), ("ts1", "ai", "TypeScript")]
    b = [(f"js{i}", cls, "JavaScript") for i in range(10) for cls in ("human", "ai")]
    monkeypatch.setattr(prepare, "iter_source", _lang_sources({"a": a, "b": b}))
    cfg = _base_cfg(tmp_path, per_class_cap=4)
    cfg["classes"] = {"names": ["human", "ai"]}
    cfg["dataset"].update(languages={"include": ["TypeScript", "JavaScript"]}, balance_languages=True)

    rows = _read_jsonl(prepare.prepare_split(cfg, "train"))

    assert Counter(r["language"] for r in rows) == {"JavaScript": 4, "TypeScript": 2}  # JS isn't topped up
    assert "not enough rows for: TypeScript/human 1, TypeScript/ai 1" in capsys.readouterr().out


def test_balance_languages_needs_an_include_list(tmp_path):
    import pytest

    cfg = _base_cfg(tmp_path, per_class_cap=4)
    cfg["dataset"]["balance_languages"] = True
    with pytest.raises(ValueError, match="languages.include"):
        prepare.prepare_split(cfg, "train")
