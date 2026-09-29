"""Second stage: the embedding model's score, corrected with hand-made features of the code.

A small gradient-boosted tree model takes, per text, the logit of the embedding MLP's P(ai) and the
features of aicontrib.features.handcrafted, and outputs a new P(ai). Appending the features to the
1,536-d embedding barely helped (the MLP mostly ignores them); as a second stage they raised AUC by ~0.03
on repositories never seen in training (README, "Second stage").

Training needs P(ai) for training rows from models that never saw them: the training rows it learns from
(agent-commit rows by default, the real-commit data closest to what `report` classifies) are split into
`folds` groups by repository; for each, an MLP is trained on every other training row, exactly as `train`
does (early stopping on the validation split), and scores the held-out group. The tree model is saved in
models/stage2.pkl with the digest of the MLP it was trained for; a retrained MLP needs a new stage 2.
Binary models (human / ai) only.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aicontrib.config import load_config
from aicontrib.data.languages import canonical
from aicontrib.data.sources import agent_commit_rows
from aicontrib.device import get_device
from aicontrib.features.handcrafted import feature_matrix
from aicontrib.model.classifier import MLPClassifier
from aicontrib.model.evaluate import load_checkpoint

STAGE2_FILENAME = "stage2.pkl"
CLASSES = ["human", "ai"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


class Stage2:
    """A trained second stage, applied to the embedding model's P(ai) of each text. languages: the only
    languages it applies to (stage2.languages); texts in others keep the embedding model's P(ai)."""

    def __init__(self, bundle: dict, languages: list[str] | None = None):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.mlp_digest = bundle["mlp_digest"]
        self.languages = sorted({canonical(lang) or lang for lang in languages}) if languages else None
        # Identifies what this second stage does (e.g. in the report cache): the model file and the languages.
        self.digest = bundle.get("digest", "") + (
            "-" + hashlib.sha256(",".join(self.languages).encode()).hexdigest()[:6] if self.languages else "")

    def apply(self, p_ai: np.ndarray, codes: list[str], languages: list[str | None]) -> np.ndarray:
        p_ai = np.asarray(p_ai, dtype=np.float64)
        names = [canonical(lang) for lang in languages]
        use = np.array([self.languages is None or name in self.languages for name in names], dtype=bool)
        out = p_ai.copy()
        if use.any():
            x = np.c_[_logit(p_ai[use]), feature_matrix([c for c, u in zip(codes, use) if u],
                                                        [n for n, u in zip(names, use) if u], self.features)]
            out[use] = self.model.predict_proba(x)[:, 1]
        return out


def load_stage2(cfg: dict, log: Callable[[str], None] = print) -> Stage2 | None:
    """The saved second stage, if there is one and it was trained for the current MLP; else None, with a
    note saying why (commits are then classified by the embedding model alone)."""
    models_dir = Path(cfg["paths"]["models_dir"])
    path = models_dir / STAGE2_FILENAME
    if cfg["classes"]["names"] != CLASSES:
        log(f"Stage 2 skipped: it's for the binary model (classes {CLASSES}).")
        return None
    if not path.exists():
        log("Stage 2 skipped: no trained second stage -- run `aicontrib train-stage2` (or pass --no-stage2).")
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)  # our own file, written by train_stage2
    bundle["digest"] = file_digest(path)
    if bundle["mlp_digest"] != file_digest(models_dir / "mlp_classifier.pt"):
        log("Stage 2 skipped: it was trained for an earlier model -- run `aicontrib train-stage2` again.")
        return None
    return Stage2(bundle, cfg["stage2"].get("languages"))


# ---- Training ----

def _macro_f1(model: nn.Module, x: torch.Tensor, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(-1).cpu().numpy()
    return f1_score(y, pred, average="macro")


def fit_mlp(cfg: dict, x: torch.Tensor, y: torch.Tensor, x_val: torch.Tensor, y_val: np.ndarray,
            device: torch.device) -> MLPClassifier:
    """An MLP trained as `train` trains one (same model, optimizer, learning-rate schedule and early stopping on
    validation macro-F1), kept in memory at its best epoch."""
    tcfg = cfg["training"]
    torch.manual_seed(tcfg["seed"])
    model = MLPClassifier(input_dim=x.shape[1], hidden_dims=cfg["model"]["hidden_dims"], num_classes=len(CLASSES),
                          dropout=cfg["model"]["dropout"])
    model.fit_input_scaling(x)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=tcfg["lr_reduce_factor"], patience=tcfg["lr_reduce_patience"],
        min_lr=tcfg["min_lr"], threshold=0.0)
    loader = DataLoader(TensorDataset(x, y), batch_size=tcfg["batch_size"], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    x_val = x_val.to(device)
    best_f1, best_state, stale = -1.0, None, 0
    for _ in range(tcfg["epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        f1 = _macro_f1(model, x_val, y_val)
        scheduler.step(f1)
        if f1 > best_f1:
            best_f1, stale = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= tcfg["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _p_ai(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x.to(device)), dim=-1)[:, 1].cpu().numpy()


def _split(cfg: dict, split: str) -> tuple[torch.Tensor, np.ndarray, list[str], list[str | None]]:
    """(embeddings, labels, codes, languages) of a split; the prepared file and the embeddings must match."""
    data = np.load(Path(cfg["paths"]["embeddings_dir"]) / f"{split}.npz")
    with open(Path(cfg["paths"]["processed_dir"]) / f"{split}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if len(rows) != len(data["labels"]) or any(r["label"] != int(lab) for r, lab in zip(rows, data["labels"])):
        raise ValueError(f"data/processed/{split}.jsonl doesn't match the {split} embeddings -- re-run `aicontrib embed`.")
    return (torch.tensor(data["embeddings"], dtype=torch.float32), data["labels"].astype(int),
            [r["code"] for r in rows], [r.get("language") for r in rows])


def _agent_repos(cfg: dict) -> dict[str, str]:
    """code -> repository, for every agent-commit row (the prepared rows don't record their repository)."""
    repos = {}
    for source in cfg["dataset"]["sources"]:
        if source["adapter"] != "agent_commits" or not source.get("enabled", True):
            continue
        for split in set((source.get("hf_splits") or {}).values()):
            for row in agent_commit_rows(source, split, quiet=True):
                repos[row["code"]] = row["repo"]
    return repos


def _tree() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          early_stopping=True, random_state=0)


def _auc_table(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, languages: list[str | None]) -> list[tuple]:
    """(language, rows, AUC embedding only, AUC with stage 2) for all rows, then per language."""
    langs = np.array([lang or "unknown" for lang in languages])
    out = []
    for name, mask in [("all", np.ones(len(y), bool))] + [(lang, langs == lang) for lang in sorted(set(langs))]:
        if 0 < y[mask].sum() < mask.sum():
            out.append((name, int(mask.sum()), roc_auc_score(y[mask], p1[mask]), roc_auc_score(y[mask], p2[mask])))
    return out


def train_stage2(config_path: str | None = None, log: Callable[[str], None] = print) -> Path:
    cfg = load_config(config_path) if config_path else load_config()
    scfg = cfg["stage2"]
    if cfg["classes"]["names"] != CLASSES:
        raise ValueError(f"Stage 2 is for the binary model: classes.names must be {CLASSES}.")
    models_dir = Path(cfg["paths"]["models_dir"])
    checkpoint = models_dir / "mlp_classifier.pt"
    if not checkpoint.exists():
        raise ValueError("No trained model -- run `aicontrib train` first.")
    device = get_device()
    x_tr, y_tr, codes_tr, langs_tr = _split(cfg, "train")
    x_val, y_val, codes_val, langs_val = _split(cfg, "validation")

    # The rows stage 2 learns from, grouped by repository (agent-commit rows) or one group per row.
    repos = _agent_repos(cfg)
    if scfg["train_on"] == "agent_commits":
        rows = np.array([i for i, c in enumerate(codes_tr) if c in repos])
        if len(rows) < 200:
            raise ValueError(f"Only {len(rows)} agent-commit rows in the training data: too few for stage 2. "
                             "Set stage2.train_on: all, or add agent-commit data (README).")
    else:
        rows = np.arange(len(codes_tr))
    groups = np.array([repos.get(codes_tr[i], f"row-{i}") for i in rows])
    folds = scfg["folds"]
    log(f"Stage 2: {len(rows)} training rows ({scfg['train_on']}), {len(set(groups))} groups; "
        f"{folds} extra models to score them out of fold...")

    oof = np.zeros(len(rows))
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    for k, (_, held) in enumerate(GroupKFold(n_splits=folds).split(rows, y_tr[rows], groups), 1):
        keep = np.ones(len(codes_tr), bool)
        keep[rows[held]] = False
        model = fit_mlp(cfg, x_tr[keep], y_tr_t[keep], x_val, y_val, device)
        oof[held] = _p_ai(model, x_tr[rows[held]], device)
        log(f"  fold {k}/{folds}: {len(held)} rows scored, AUC {roc_auc_score(y_tr[rows[held]], oof[held]):.3f}")

    features = list(scfg["features"])
    log(f"Computing {len(features)} features...")
    x2 = np.c_[_logit(oof), feature_matrix([codes_tr[i] for i in rows], [langs_tr[i] for i in rows], features)]

    # The most reliable estimate of the gain, per language: the tree model cross-validated over the training
    # repositories (the validation and test splits below hold only a few repositories per language).
    cv = np.zeros(len(rows))
    for fit_idx, held in GroupKFold(n_splits=folds).split(rows, y_tr[rows], groups):
        cv[held] = _tree().fit(x2[fit_idx], y_tr[rows][fit_idx]).predict_proba(x2[held])[:, 1]
    langs_rows = [langs_tr[i] for i in rows]
    log(f"\n[train] AUC cross-validated over the {len(set(groups))} training repositories (embedding only -> "
        "with stage 2):")
    for name, n, a1, a2 in _auc_table(y_tr[rows], oof, cv, langs_rows):
        log(f"  {name:<12}{n:>7} rows   {a1:.3f} -> {a2:.3f}  ({a2 - a1:+.3f})")

    tree = _tree().fit(x2, y_tr[rows])

    path = models_dir / STAGE2_FILENAME
    with open(path, "wb") as f:
        pickle.dump({"model": tree, "features": features, "mlp_digest": file_digest(checkpoint),
                     "train_on": scfg["train_on"], "n_rows": int(len(rows))}, f)

    # Held-out check on the validation and test splits: the saved MLP alone vs with stage 2.
    mlp = load_checkpoint(checkpoint, device, class_names=CLASSES)
    stage2 = load_stage2(cfg, log=lambda _: None)
    for split in ("validation", "test"):
        try:
            x, y, codes, langs = _split(cfg, split)
        except (FileNotFoundError, ValueError) as exc:
            log(f"[{split}] not checked: {exc}")
            continue
        if scfg["train_on"] == "agent_commits":
            mask = np.array([c in repos for c in codes])
            if not mask.any():
                continue
            x, y = x[mask], y[mask]
            codes = [c for c, m in zip(codes, mask) if m]
            langs = [lang for lang, m in zip(langs, mask) if m]
        p1 = _p_ai(mlp, x, device)
        p2 = stage2.apply(p1, codes, langs)
        kind = "agent-commit rows" if scfg["train_on"] == "agent_commits" else "rows"
        log(f"\n[{split}] AUC on {kind} of repositories not used for stage 2 (embedding only -> with stage 2):")
        for name, n, a1, a2 in _auc_table(y, p1, p2, langs):
            log(f"  {name:<12}{n:>7} rows   {a1:.3f} -> {a2:.3f}  ({a2 - a1:+.3f})")
    log(f"\nSaved to {path}")
    return path
