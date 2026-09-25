"""Pull from each enabled dataset source in turn, remap to our 3 classes, optionally keep
only some programming languages, cap rows per class per split, and write
data/processed/{split}.jsonl.

Sources are combined rather than concatenated wholesale: each is streamed only
until every class hits its per-split cap (see configs/default.yaml), and exact
duplicate code strings are dropped across sources within a split so the same
snippet can't appear twice (e.g. if AICD-Bench and CodeMirage both scraped the
same GitHub file).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from aicontrib.config import load_config
from aicontrib.data.languages import language_filter
from aicontrib.data.sources import iter_source

_SPLITS = ("train", "validation", "test")


def prepare_split(cfg: dict, out_name: str) -> Path:
    cap = cfg["dataset"]["per_class_cap"][out_name]  # None means unbounded -- take everything available
    class_names = cfg["classes"]["names"]
    num_classes = len(class_names)

    def cap_reached(cls: str) -> bool:
        return cap is not None and counts[cls] >= cap

    counts: dict[str, int] = defaultdict(int)
    languages: Counter = Counter()
    empty_rows = filtered_out = 0
    allowed = language_filter(cfg)
    seen_hashes: set[str] = set()
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.jsonl"

    total_target = cap * num_classes if cap is not None else None
    with open(out_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=total_target, desc=f"prepare[{out_name}]")
        for source_cfg in cfg["dataset"]["sources"]:
            if not source_cfg.get("enabled", True):
                continue
            if all(cap_reached(c) for c in class_names):
                break
            for code, cls, lang in iter_source(source_cfg, out_name):
                if not code or not code.strip():  # upstream data has a few rows with no code (e.g. CodeMirage)
                    empty_rows += 1
                    continue
                if cls not in class_names:
                    continue
                if allowed is not None:
                    if lang is None:
                        raise ValueError(
                            f"Can't filter by language: source {source_cfg['name']!r} has rows with no language "
                            "label (its language index is missing or was built for another dataset revision). "
                            "Run `aicontrib language-census`, or remove dataset.languages from configs/local.yaml."
                        )
                    if not allowed(lang):
                        filtered_out += 1
                        continue
                if cap_reached(cls):
                    continue
                digest = hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                counts[cls] += 1
                languages[lang or "unknown"] += 1
                f.write(json.dumps({"code": code, "label": class_names.index(cls), "language": lang}) + "\n")
                pbar.update(1)
                if all(cap_reached(c) for c in class_names):
                    break
        pbar.close()

    print(f"[{out_name}] wrote {sum(counts.values())} rows, per-class counts: {dict(counts)}, "
          f"skipped {empty_rows} rows with no code"
          + (f", {filtered_out} rows in excluded languages" if allowed is not None else ""))
    print(f"[{out_name}] per-language counts: {dict(languages.most_common())}")
    return out_path


def prepare_all(config_path: str | None = None) -> None:
    cfg = load_config(config_path) if config_path else load_config()
    for out_name in _SPLITS:
        prepare_split(cfg, out_name)


if __name__ == "__main__":
    prepare_all()
