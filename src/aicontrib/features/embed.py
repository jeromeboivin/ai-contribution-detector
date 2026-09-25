"""Batch-embed code strings with a frozen pretrained code encoder.

This is the slow step of the pipeline. Speedups, none of which change the embeddings:
- Samples are processed longest first and batched by token count, so each batch holds
  similar lengths and almost no compute is spent on padding. (Padding doesn't change a
  sample's embedding -- see tests/test_embed_speedups.py.)
- On a GPU, batches are as large as memory allows (halved and retried on out-of-memory) and
  run in bfloat16/float16 if a startup check shows it matches fp32 on that GPU.
- Resume checkpoints are append-only shards written every few minutes, instead of a growing
  file rewritten every few batches (which, on the full dataset, meant rewriting up to 1 GB).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from aicontrib.config import load_config
from aicontrib.device import describe_device, get_device, require_supported_torch

MIN_COSINE = 0.999  # reduced precision is only used if every check sample stays this close to fp32
TOKENIZE_CHUNK = 2048
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def precision_matches(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[bool, float]:
    if not torch.isfinite(candidate).all():
        return False, float("nan")
    cosine = F.cosine_similarity(reference.float(), candidate.float()).min().item()
    return cosine >= MIN_COSINE, cosine


def plan_batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices into `lengths`, grouped longest first so rows x longest-in-batch <= budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches, k = [], 0
    while k < len(order):
        size = max(1, budget // max(1, lengths[order[k]]))
        batches.append(order[k : k + size])
        k += size
    return batches


class CodeEmbedder:
    def __init__(self, cfg: dict):
        require_supported_torch()
        self.cfg = cfg["embedding"]
        self.device = get_device()
        print(f"Code encoder running on: {describe_device(self.device)}")
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
        self.autocast_dtype: torch.dtype | None = None  # fp32 until calibrate_precision() says otherwise
        self.calibrated = False

    def _forward(self, inputs, dtype: torch.dtype | None) -> torch.Tensor:
        inputs = inputs.to(self.device)
        if dtype is None:
            return self.model(**inputs)
        with torch.autocast(self.device.type, dtype=dtype):
            return self.model(**inputs)

    @torch.no_grad()
    def _embed_inputs(self, inputs) -> np.ndarray:
        out = self._forward(inputs, self.autocast_dtype)
        if self.autocast_dtype is not None and not torch.isfinite(out).all():
            out = self._forward(inputs, None)  # rare reduced-precision overflow: redo this batch in fp32
        return out.float().cpu().numpy()

    def embed_batch(self, codes: list[str]) -> np.ndarray:
        inputs = self.tokenizer(codes, padding=True, truncation=True, max_length=self.cfg["max_length"],
                                return_tensors="pt")
        return self._embed_inputs(inputs)

    def embed_encoded(self, features: list[dict]) -> np.ndarray:
        """Embed already-tokenized samples; on GPU out-of-memory, split the batch in two and retry."""
        try:
            return self._embed_inputs(self.tokenizer.pad(features, return_tensors="pt"))
        except torch.cuda.OutOfMemoryError:
            if len(features) == 1:
                raise
            torch.cuda.empty_cache()
            mid = len(features) // 2
            return np.concatenate([self.embed_encoded(features[:mid]), self.embed_encoded(features[mid:])])

    @torch.no_grad()
    def calibrate_precision(self, sample_codes: list[str], choice: str = "auto") -> None:
        """On a GPU, use bfloat16 (or float16) only if it reproduces fp32 on *this* GPU."""
        self.calibrated = True
        if self.device.type != "cuda" or choice == "fp32" or not sample_codes:
            print("Precision: fp32")
            return
        name = choice if choice in _DTYPES else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
        inputs = self.tokenizer(sample_codes, padding=True, truncation=True, max_length=self.cfg["max_length"],
                                return_tensors="pt")
        ok, cosine = precision_matches(self._forward(inputs, None), self._forward(inputs, _DTYPES[name]))
        if ok:
            self.autocast_dtype = _DTYPES[name]
            print(f"Precision: {name} (matches fp32 on this GPU: min cosine similarity {cosine:.5f})")
        else:
            print(f"Precision: fp32 ({name} drifted from fp32 on this GPU: min cosine similarity {cosine:.4f})")

    def token_budget(self, value: int | str = "auto") -> int:
        if value != "auto":
            return int(value)
        if self.device.type == "cuda":
            memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            return int(min(65536, max(8192, 4096 * memory_gb)))
        return 16384


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_shard(partial_dir: Path, shard_id: int, positions: list[int], vectors: list[np.ndarray]) -> None:
    tmp = partial_dir / f"shard-{shard_id:05d}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, positions=np.array(positions, dtype=np.int64), embeddings=np.concatenate(vectors))
    os.replace(tmp, partial_dir / f"shard-{shard_id:05d}.npz")  # atomic: a crash never leaves half a shard


def embed_split(cfg: dict, embedder: CodeEmbedder, split: str, force: bool = False) -> Path:
    """Embeds one split. Stop it at any time: already-embedded samples are kept in
    {split}.partial/ shards and skipped on the next run, as long as the prepared data
    (checked by hash) hasn't changed since."""
    processed_path = Path(cfg["paths"]["processed_dir"]) / f"{split}.jsonl"
    out_dir = Path(cfg["paths"]["embeddings_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.npz"
    partial_dir = out_dir / f"{split}.partial"
    (out_dir / f"{split}.partial.npz").unlink(missing_ok=True)  # checkpoint file of older versions

    if out_path.exists() and not force:
        print(f"[{split}] embeddings already cached at {out_path}, skipping (use --force to recompute)")
        return out_path

    codes, labels = [], []
    with open(processed_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            codes.append(row["code"])
            labels.append(row["label"])
    labels_arr = np.array(labels, dtype=np.int64)
    embeddings = np.zeros((len(codes), cfg["embedding"]["dim"]), dtype=np.float32)
    done = np.zeros(len(codes), dtype=bool)

    data_hash = _file_digest(processed_path)
    meta_path = partial_dir / "meta.json"
    if partial_dir.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if not force and meta.get("data_hash") == data_hash:
            for shard in sorted(partial_dir.glob("shard-*.npz")):
                data = np.load(shard)
                embeddings[data["positions"]] = data["embeddings"]
                done[data["positions"]] = True
            print(f"[{split}] resuming: {int(done.sum()):,}/{len(codes):,} samples already embedded")
        else:
            if not force:
                print(f"[{split}] checkpoint is from different prepared data -- starting this split over")
            shutil.rmtree(partial_dir)
    partial_dir.mkdir(exist_ok=True)
    meta_path.write_text(json.dumps({"data_hash": data_hash}), encoding="utf-8")
    shard_id = len(list(partial_dir.glob("shard-*.npz")))

    # Longest first: batches of similar length (little padding), out-of-memory surfaces immediately,
    # and the time estimate starts pessimistic and improves as samples get shorter.
    todo = sorted(np.flatnonzero(~done).tolist(), key=lambda i: len(codes[i]), reverse=True)
    if not embedder.calibrated:
        spread = np.linspace(0, len(todo) - 1, 16).astype(int) if todo else []
        embedder.calibrate_precision([codes[todo[k]] for k in spread], cfg["embedding"].get("precision", "auto"))
    budget = embedder.token_budget(cfg["embedding"].get("max_tokens_per_batch", "auto"))
    save_every = cfg["embedding"]["checkpoint_every_seconds"]

    pending_pos, pending_vec, last_save = [], [], time.monotonic()
    pbar = tqdm(total=len(codes), initial=int(done.sum()), desc=f"embedding[{split}]", unit="sample", smoothing=0.05)
    for c in range(0, len(todo), TOKENIZE_CHUNK):
        chunk = todo[c : c + TOKENIZE_CHUNK]
        enc = embedder.tokenizer([codes[i] for i in chunk], truncation=True, max_length=cfg["embedding"]["max_length"])
        features = [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(enc["input_ids"], enc["attention_mask"])]
        for batch in plan_batches([len(f["input_ids"]) for f in features], budget):
            vectors = embedder.embed_encoded([features[j] for j in batch])
            positions = [chunk[j] for j in batch]
            embeddings[positions] = vectors
            pending_pos += positions
            pending_vec.append(vectors)
            pbar.update(len(batch))
            if time.monotonic() - last_save >= save_every:
                _write_shard(partial_dir, shard_id, pending_pos, pending_vec)
                shard_id, pending_pos, pending_vec, last_save = shard_id + 1, [], [], time.monotonic()
    if pending_pos:
        _write_shard(partial_dir, shard_id, pending_pos, pending_vec)
    pbar.close()

    np.savez(out_path, embeddings=embeddings, labels=labels_arr)
    shutil.rmtree(partial_dir, ignore_errors=True)
    print(f"[{split}] embedded {len(codes):,} samples -> {out_path}")
    return out_path


def embed_all(config_path: str | None = None, force: bool = False) -> None:
    cfg = load_config(config_path) if config_path else load_config()
    embedder = CodeEmbedder(cfg)
    for split in ("train", "validation", "test"):
        embed_split(cfg, embedder, split, force=force)


if __name__ == "__main__":
    embed_all()
