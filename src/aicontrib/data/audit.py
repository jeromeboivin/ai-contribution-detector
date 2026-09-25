"""Print sample rows per raw AICD-Bench T3 label so the label->class mapping
hypothesis in configs/default.yaml can be checked against real content before
`prepare.py` bakes it in. Run this once and eyeball the output; see README.
"""
from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from aicontrib.config import load_config


def audit(n_per_label: int = 15, split: str = "train", source_name: str = "aicd_bench") -> None:
    cfg = load_config()
    source_cfg = next(s for s in cfg["dataset"]["sources"] if s["name"] == source_name)
    ds = load_dataset(source_cfg["hf_repo"], source_cfg["hf_config"], split=split, streaming=True,
                      revision=source_cfg.get("hf_revision"))

    seen: dict[int, int] = defaultdict(int)
    labels_wanted = {0, 1, 2, 3}

    for row in ds:
        label = row["label"]
        if label not in labels_wanted or seen[label] >= n_per_label:
            if all(seen[l] >= n_per_label for l in labels_wanted):
                break
            continue
        seen[label] += 1
        snippet = row["code"][:200].replace("\n", "\\n")
        print(f"[label={label}] {snippet}")
        print("-" * 80)


if __name__ == "__main__":
    audit()
