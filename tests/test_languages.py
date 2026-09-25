import json

import pytest

from aicontrib.data import languages, prepare
from aicontrib.data.languages import UNKNOWN, AicdLanguageIndex, _fingerprint, language_filter, save_index


def _cfg(spec):
    return {"dataset": {"languages": spec}}


def test_no_filter_configured():
    assert language_filter(_cfg({})) is None
    assert language_filter({"dataset": {}}) is None


def test_include_exclude_and_case_insensitive_aliases():
    keep = language_filter(_cfg({"include": ["python", "JS", "c++", "CSharp"]}))
    assert [lang for lang in languages.LANGUAGES if keep(lang)] == ["C#", "C++", "JavaScript", "Python"]
    drop = language_filter(_cfg({"exclude": ["Rust", "ruby", "HTML"]}))
    assert not drop("Rust") and not drop("Ruby") and drop("Python")
    both = language_filter(_cfg({"include": ["Python", "C"], "exclude": ["C"]}))
    assert both("Python") and not both("C") and not both("Go")
    assert not keep(None)  # unlabelled rows never pass an active filter


def test_unknown_language_name_is_rejected_with_the_valid_list():
    with pytest.raises(ValueError, match="Valid names"):
        language_filter(_cfg({"include": ["Pyhton"]}))


def test_typescript_is_a_language_for_the_agent_commit_data():
    keep = language_filter(_cfg({"include": ["TypeScript", "js"]}))
    assert keep("TypeScript") and keep("JavaScript") and not keep("Python")
    assert languages.canonical("ts") == "TypeScript"


def test_index_roundtrip_and_misalignment_detection(tmp_path):
    path = tmp_path / "idx.npz"
    py, js = languages.LANGUAGES.index("Python"), languages.LANGUAGES.index("JavaScript")
    codes = {"train": [py, UNKNOWN, js], "validation": [js], "test": []}
    ckpts = {"train": ([0, 2], [_fingerprint("a"), _fingerprint("c")]), "validation": ([0], [_fingerprint("x")]),
             "test": ([], [])}
    save_index(path, "rev1", codes, ckpts)

    index = AicdLanguageIndex(path)
    assert index.revision == "rev1"
    assert [index.language("train", i) for i in range(3)] == ["Python", None, "JavaScript"]
    index.verify("train", 0, "a")
    index.verify("train", 1, "anything")  # not a checkpoint row
    with pytest.raises(RuntimeError, match="language-census"):
        index.verify("train", 2, "not c")


def test_prepare_keeps_only_included_languages_and_records_them(tmp_path, monkeypatch):
    rows = [("p1", "human", "Python"), ("r1", "ai", "Rust"), ("j1", "ai", "JavaScript"), ("", "ai", None)]
    monkeypatch.setattr(prepare, "iter_source", lambda source_cfg, out_name: iter(rows))
    cfg = {
        "dataset": {"sources": [{"name": "a"}], "per_class_cap": {"train": None},
                    "languages": {"include": ["Python", "JavaScript"]}},
        "classes": {"names": ["human", "co_authored", "ai"]},
        "paths": {"processed_dir": str(tmp_path)},
    }
    out = prepare.prepare_split(cfg, "train")
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["code"], r["language"]) for r in written] == [("p1", "Python"), ("j1", "JavaScript")]


def test_prepare_refuses_to_filter_rows_without_language(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "iter_source", lambda source_cfg, out_name: iter([("code", "ai", None)]))
    cfg = {
        "dataset": {"sources": [{"name": "aicd"}], "per_class_cap": {"train": None},
                    "languages": {"exclude": ["Rust"]}},
        "classes": {"names": ["human", "co_authored", "ai"]},
        "paths": {"processed_dir": str(tmp_path)},
    }
    with pytest.raises(ValueError, match="language-census"):
        prepare.prepare_split(cfg, "train")
