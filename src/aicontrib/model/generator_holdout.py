"""Held-out-generator evaluation: how well does the classifier catch code from an AI model it
never saw in training? That's the situation on real repos (esker-cowork was written by a newer
agent than any generator in the training data), and a normal test split can't measure it.

Uses CodeMirage, the one source that records which model generated each row (10 generators:
GPT, Claude, Gemini, DeepSeek, Llama, Qwen...). AICD-Bench T3 has no generator column. The
setup is binary, human vs AI, and self-contained: its own sample of CodeMirage (respecting
dataset.languages), embedded with the configured embedding.representation, so representations
can be compared by running it once per setting.

For each generator G, two MLPs are scored on G's test rows vs the human test rows (AUC):
- seen: trained on every generator (G included) -- the usual in-distribution number;
- unseen: trained on every generator except G -- the generalization number.
The gap between the two is what a new, unseen model costs.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from torch import nn

from aicontrib.data.languages import canonical, language_filter
from aicontrib.features.embed import CodeEmbedder, embed_texts
from aicontrib.model.classifier import MLPClassifier

HUMAN = "Human"


def sample_rows(rows: list[dict], per_generator: int, allowed, seed: int) -> list[dict]:
    """Every human row, plus up to per_generator random rows of each generator, in allowed languages."""
    rows = [r for r in rows if r["code"] and r["code"].strip() and (allowed is None or allowed(canonical(r["language"])))]
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    rng = random.Random(seed)
    out = list(by_source.get(HUMAN, []))
    for source in sorted(s for s in by_source if s != HUMAN):
        group = by_source[source]
        out += rng.sample(group, min(per_generator, len(group)))
    return out


def _embed_cached(cfg: dict, embedder: CodeEmbedder, split: str, rows: list[dict]) -> np.ndarray:
    codes = [r["code"] for r in rows]
    digest = hashlib.sha256("\0".join(codes).encode("utf-8", errors="replace")).hexdigest()
    rep = embedder.representation_id()
    safe = rep.replace(":", "-").replace(",", "_")  # ':' isn't allowed in Windows file names
    path = Path(cfg["paths"]["embeddings_dir"]) / "generator_holdout" / f"{split}-{safe}.npz"
    if path.exists():
        with np.load(path) as cached:
            if str(cached["digest"]) == digest:
                print(f"[{split}] embeddings cached at {path}")
                return cached["embeddings"]
    embeddings = embed_texts(embedder, codes, cfg, desc=f"embedding[{split}]")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=embeddings, digest=np.array(digest))
    return embeddings


def fit_binary(x: np.ndarray, y: np.ndarray, cfg: dict, seed: int) -> MLPClassifier:
    """MLP with the configured architecture and training settings, early-stopped on the AUC of a
    random 10% of the rows. Classes are weighted so the few human rows count as much as the AI ones."""
    tcfg = cfg["training"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_val = max(1, len(y) // 10)
    val, tr = perm[:n_val], perm[n_val:]
    x_tr, y_tr = torch.tensor(x[tr]), torch.tensor(y[tr])
    x_val = torch.tensor(x[val])

    model = MLPClassifier(x.shape[1], cfg["model"]["hidden_dims"], 2, cfg["model"]["dropout"])
    model.fit_input_scaling(x_tr)
    counts = torch.bincount(y_tr, minlength=2).float()
    criterion = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts.clamp_min(1)))
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    best_auc, best_state, since_best = -1.0, None, 0
    for _ in range(tcfg["epochs"]):
        model.train()
        order = torch.randperm(len(y_tr))
        for i in range(0, len(order), tcfg["batch_size"]):
            idx = order[i : i + tcfg["batch_size"]]
            optimizer.zero_grad()
            criterion(model(x_tr[idx]), y_tr[idx]).backward()
            optimizer.step()
        auc = roc_auc_score(y[val], _p_ai(model, x_val))
        if auc > best_auc:
            best_auc, best_state, since_best = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since_best += 1
            if since_best >= tcfg["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def _p_ai(model: MLPClassifier, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.softmax(model(x), dim=-1)[:, 1].numpy()


def holdout_aucs(x_train: np.ndarray, src_train: np.ndarray, x_test: np.ndarray, src_test: np.ndarray,
                 cfg: dict, seed: int, fit=fit_binary) -> list[dict]:
    """Per generator: AUC of G's test rows vs human test rows, for a model that saw G and one that didn't."""
    y_train = (src_train != HUMAN).astype(np.int64)
    human_test = src_test == HUMAN
    generators = sorted(set(src_train.tolist()) - {HUMAN})

    def auc_on(model, generator):
        mask = human_test | (src_test == generator)
        return float(roc_auc_score((src_test[mask] != HUMAN).astype(int), _p_ai(model, torch.tensor(x_test[mask]))))

    seen_model = fit(x_train, y_train, cfg, seed)
    results = []
    for g in generators:
        if not (src_test == g).any():
            continue
        keep = src_train != g
        unseen_model = fit(x_train[keep], y_train[keep], cfg, seed)
        results.append({"generator": g, "n_test": int((src_test == g).sum()),
                        "seen_auc": auc_on(seen_model, g), "unseen_auc": auc_on(unseen_model, g)})
        print(f"  {g}: seen {results[-1]['seen_auc']:.3f}, unseen {results[-1]['unseen_auc']:.3f}")
    return results


def run_generator_holdout(cfg: dict) -> dict:
    hcfg = cfg["generator_holdout"]
    source = next(s for s in cfg["dataset"]["sources"] if s["adapter"] == "codemirage")
    allowed = language_filter(cfg)
    embedder = CodeEmbedder(cfg)

    splits = {}
    for split in ("train", "test"):
        ds = load_dataset(source["hf_repo"], split=source["hf_splits"][split]).select_columns(["code", "language", "source"])
        rows = sample_rows(ds.to_list(), hcfg["per_generator"][split], allowed, hcfg["seed"])
        n_human = sum(r["source"] == HUMAN for r in rows)
        print(f"[{split}] {n_human} human rows, {len(rows) - n_human} AI rows")
        splits[split] = (_embed_cached(cfg, embedder, split, rows), np.array([r["source"] for r in rows]))

    print(f"Training {len(set(splits['train'][1].tolist()))} classifiers (one per held-out generator, plus one seeing all)...")
    results = holdout_aucs(*splits["train"], *splits["test"], cfg, hcfg["seed"])
    summary = {
        "representation": embedder.representation_id(),
        "languages": cfg["dataset"].get("languages") or "all",
        "generators": results,
        "mean_seen_auc": float(np.mean([r["seen_auc"] for r in results])),
        "mean_unseen_auc": float(np.mean([r["unseen_auc"] for r in results])),
    }
    safe = summary["representation"].replace(":", "-").replace(",", "_")
    out = Path(cfg["paths"]["models_dir"]) / f"generator_holdout-{safe}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["results_path"] = str(out)
    return summary
