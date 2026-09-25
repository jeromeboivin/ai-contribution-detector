"""Per-dataset adapters. Each yields (code, class_name) pairs for one HF split.

AICD-Bench and CodeMirage are combined here rather than DroidCollection, because
AICD-Bench is built directly on top of Droid (extended + MinHash-deduplicated
against it -- see the AICD-Bench paper, arXiv:2602.02079, Section 3) so Droid's
content is already folded in; adding it separately would only reintroduce
near-duplicate rows the AICD authors deliberately removed.
"""
from __future__ import annotations

from typing import Any, Iterator

from datasets import load_dataset

# AICD-Bench T3's raw int label -> our class. Not documented by the dataset authors
# (the HF repo card only declares the field as int64); this mapping was inferred by
# manually reading ~450 sample rows spread across the dataset (see `aicontrib audit`).
# Both raw 1 and 3 collapse to "ai" regardless of which is "machine" vs "adversarial",
# so the ambiguity between them doesn't actually matter for our 3-class scheme.
_AICD_BENCH_LABEL_MAP = {0: "human", 2: "co_authored", 1: "ai", 3: "ai"}


def iter_aicd_bench(source_cfg: dict[str, Any], hf_split: str) -> Iterator[tuple[str, str]]:
    ds = load_dataset(source_cfg["hf_repo"], source_cfg["hf_config"], split=hf_split, streaming=True)
    for row in ds:
        cls = _AICD_BENCH_LABEL_MAP.get(row["label"])
        if cls is not None:
            yield row["code"], cls


def iter_codemirage(source_cfg: dict[str, Any], hf_split: str) -> Iterator[tuple[str, str]]:
    """CodeMirage is binary only (source == "Human" or a model name) -- it can only
    contribute to the human/ai buckets, never co_authored. CC-BY-NC-ND-4.0 licensed:
    non-commercial, no-derivatives (see README)."""
    ds = load_dataset(source_cfg["hf_repo"], split=hf_split, streaming=True)
    for row in ds:
        cls = "human" if row["source"] == "Human" else "ai"
        yield row["code"], cls


ADAPTERS = {
    "aicd_bench": iter_aicd_bench,
    "codemirage": iter_codemirage,
}


def iter_source(source_cfg: dict[str, Any], our_split: str) -> Iterator[tuple[str, str]]:
    hf_split = source_cfg["hf_splits"].get(our_split)
    if hf_split is None:
        return iter(())  # this source has no data for this split (e.g. CodeMirage has no validation)
    adapter = ADAPTERS[source_cfg["adapter"]]
    return adapter(source_cfg, hf_split)
