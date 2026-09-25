"""Batch-embed code strings with a frozen pretrained code encoder.

This is the only CPU-heavy step in the pipeline (a GPU, if available, speeds
it up automatically via aicontrib.device.get_device()). The MLP trained on
top of these fixed-size vectors is cheap to train either way.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from aicontrib.config import load_config
from aicontrib.device import get_device


class CodeEmbedder:
    def __init__(self, cfg: dict):
        self.cfg = cfg["embedding"]
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name"], trust_remote_code=True)
        # codet5p-embedding's remote code predates transformers versions that stopped silently
        # defaulting missing config attributes -- T5Stack.__init__ now hard-fails on the config's
        # missing `is_decoder`. The upstream config never sets it since this is encoder-only.
        model_config = AutoConfig.from_pretrained(self.cfg["model_name"], trust_remote_code=True)
        model_config.is_decoder = False
        self.model = AutoModel.from_pretrained(
            self.cfg["model_name"], config=model_config, trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_batch(self, codes: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=self.cfg["max_length"],
            return_tensors="pt",
        ).to(self.device)
        embeddings = self.model(**inputs)
        return embeddings.cpu().numpy()


def embed_split(cfg: dict, embedder: CodeEmbedder, split: str, force: bool = False) -> Path:
    """Embeds one split, checkpointing every `checkpoint_every_batches` batches to
    {split}.partial.npz so a killed process resumes mid-split instead of restarting
    it from row 0 -- the train split alone can take hours, so losing that would waste
    most of a run. The checkpoint is validated against the current labels before being
    trusted, in case `prepare` was re-run with different data since the checkpoint was
    written.
    """
    processed_path = Path(cfg["paths"]["processed_dir"]) / f"{split}.jsonl"
    out_dir = Path(cfg["paths"]["embeddings_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.npz"
    partial_path = out_dir / f"{split}.partial.npz"

    if out_path.exists() and not force:
        print(f"[{split}] embeddings already cached at {out_path}, skipping (use force=True to recompute)")
        return out_path

    codes, labels = [], []
    with open(processed_path) as f:
        for line in f:
            row = json.loads(line)
            codes.append(row["code"])
            labels.append(row["label"])
    labels_arr = np.array(labels, dtype=np.int64)

    dim = cfg["embedding"]["dim"]
    embeddings = np.zeros((len(codes), dim), dtype=np.float32)
    start_row = 0

    if partial_path.exists() and not force:
        checkpoint = np.load(partial_path)
        n_done = int(checkpoint["n_done"])
        if np.array_equal(checkpoint["labels"], labels_arr):
            embeddings[:n_done] = checkpoint["embeddings"][:n_done]
            start_row = n_done
            print(f"[{split}] resuming from row {start_row}/{len(codes)} (checkpoint found)")
        else:
            print(f"[{split}] checkpoint doesn't match current processed data -- restarting split from row 0")

    batch_size = cfg["embedding"]["batch_size"]
    checkpoint_every = cfg["embedding"]["checkpoint_every_batches"]
    total_batches = (len(codes) + batch_size - 1) // batch_size

    batches_since_checkpoint = 0
    for i in tqdm(
        range(start_row, len(codes), batch_size),
        desc=f"embedding[{split}]",
        initial=start_row // batch_size,
        total=total_batches,
    ):
        batch_codes = codes[i : i + batch_size]
        embeddings[i : i + len(batch_codes)] = embedder.embed_batch(batch_codes)
        batches_since_checkpoint += 1
        if batches_since_checkpoint >= checkpoint_every:
            done = i + len(batch_codes)
            np.savez(partial_path, embeddings=embeddings[:done], labels=labels_arr, n_done=done)
            batches_since_checkpoint = 0

    np.savez(out_path, embeddings=embeddings, labels=labels_arr)
    partial_path.unlink(missing_ok=True)
    print(f"[{split}] embedded {len(codes)} rows -> {out_path}")
    return out_path


def embed_all(config_path: str | None = None, force: bool = False) -> None:
    cfg = load_config(config_path) if config_path else load_config()
    embedder = CodeEmbedder(cfg)
    for split in ("train", "validation", "test"):
        embed_split(cfg, embedder, split, force=force)


if __name__ == "__main__":
    embed_all()
