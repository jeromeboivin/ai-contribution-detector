"""Per-dataset adapters. Each yields (code, class_name, language) triples for one HF split;
language is a canonical name from aicontrib.data.languages.LANGUAGES, or None if unknown.

AICD-Bench and CodeMirage are combined here rather than DroidCollection, because
AICD-Bench is built directly on top of Droid (extended + MinHash-deduplicated
against it -- see the AICD-Bench paper, arXiv:2602.02079, Section 3) so Droid's
content is already folded in; adding it separately would only reintroduce
near-duplicate rows the AICD authors deliberately removed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset

from aicontrib.config import REPO_ROOT
from aicontrib.data.languages import canonical, load_index

# AICD-Bench T3's raw int label -> our class. Not documented by the dataset authors
# (the HF repo card only declares the field as int64); this mapping was inferred by
# manually reading ~450 sample rows spread across the dataset (see `aicontrib audit`).
# Both raw 1 and 3 collapse to "ai" regardless of which is "machine" vs "adversarial",
# so the ambiguity between them doesn't actually matter for our 3-class scheme.
_AICD_BENCH_LABEL_MAP = {0: "human", 2: "co_authored", 1: "ai", 3: "ai"}

Row = tuple[str, str, "str | None"]


def iter_aicd_bench(source_cfg: dict[str, Any], hf_split: str) -> Iterator[Row]:
    revision = source_cfg.get("hf_revision")
    index = load_index()
    if index is not None and index.revision != revision:
        index = None  # the shipped language labels describe a different dataset revision
    ds = load_dataset(source_cfg["hf_repo"], source_cfg["hf_config"], split=hf_split, streaming=True,
                      revision=revision)
    for i, row in enumerate(ds):
        if index is not None:
            index.verify(hf_split, i, row["code"])
        cls = _AICD_BENCH_LABEL_MAP.get(row["label"])
        if cls is not None:
            yield row["code"], cls, index.language(hf_split, i) if index is not None else None


def iter_codemirage(source_cfg: dict[str, Any], hf_split: str) -> Iterator[Row]:
    """CodeMirage is binary only (source == "Human" or a model name) -- it can only
    contribute to the human/ai buckets, never co_authored. CC-BY-NC-ND-4.0 licensed:
    non-commercial, no-derivatives (see README)."""
    ds = load_dataset(source_cfg["hf_repo"], split=hf_split, streaming=True)
    for row in ds:
        cls = "human" if row["source"] == "Human" else "ai"
        yield row["code"], cls, canonical(row["language"])


def iter_agent_commits(source_cfg: dict[str, Any], split: str) -> Iterator[Row]:
    """Local files written by `aicontrib build-agent-commits` (aicontrib/data/agent_commits.py);
    nothing until they exist."""
    path = Path(source_cfg["path"])
    path = (path if path.is_absolute() else REPO_ROOT / path) / f"{split}.jsonl"
    if not path.exists():
        print(f"[{source_cfg['name']}] no {path} -- run `aicontrib build-agent-commits` to use this source")
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row["code"], row["label"], canonical(row["language"])


ADAPTERS = {
    "agent_commits": iter_agent_commits,
    "aicd_bench": iter_aicd_bench,
    "codemirage": iter_codemirage,
}


def iter_source(source_cfg: dict[str, Any], our_split: str) -> Iterator[Row]:
    hf_split = source_cfg["hf_splits"].get(our_split)
    if hf_split is None:
        return iter(())  # this source has no data for this split (e.g. CodeMirage has no validation)
    adapter = ADAPTERS[source_cfg["adapter"]]
    return adapter(source_cfg, hf_split)
