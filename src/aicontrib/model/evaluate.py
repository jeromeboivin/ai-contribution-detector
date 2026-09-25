from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

from aicontrib.config import load_config
from aicontrib.device import get_device
from aicontrib.model.classifier import MLPClassifier


def load_checkpoint(checkpoint_path: Path, device: torch.device, representation: str | None = None) -> MLPClassifier:
    """representation: the embedding representation the caller will feed (CodeEmbedder.representation_id());
    a model trained on another one is refused instead of producing garbage or a shape error."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    trained_on = ckpt.get("representation", "projected")  # checkpoints of older versions: projected
    if representation is not None and representation != trained_on:
        raise ValueError(f"The model at {checkpoint_path} was trained on '{trained_on}' embeddings but is being given "
                         f"'{representation}' ones (embedding.representation / hidden_layers in the config). Re-run "
                         "`aicontrib embed` and `aicontrib train`, or change the config back.")
    model = MLPClassifier(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
        num_classes=ckpt["num_classes"],
        dropout=ckpt["dropout"],
    )
    state = dict(ckpt["state_dict"])
    for name, buffer in model.state_dict().items():  # older checkpoints have no input scaling: identity
        if name.startswith("input_"):
            state.setdefault(name, buffer)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _split_languages(cfg: dict, split: str) -> list[str]:
    """Each row's language, from the prepared JSONL. `embed` keeps its row order, so row i here is
    embedding i. Older prepared files have no language field ("unknown")."""
    path = Path(cfg["paths"]["processed_dir"]) / f"{split}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line).get("language") or "unknown" for line in f]


def per_language_report(y_true: np.ndarray, y_pred: np.ndarray, languages: list[str], names: list[str]) -> str:
    """One line per language: rows, accuracy, macro-F1 and each class's recall."""
    labels = list(range(len(names)))
    langs = np.array(languages)
    header = f"{'language':<12}{'rows':>6}{'accuracy':>10}{'macro-F1':>10}" + "".join(
        f"{'recall ' + n:>20}" for n in names)
    lines = [header]
    for lang in sorted(set(languages)):
        mask = langs == lang
        t, p = y_true[mask], y_pred[mask]
        recalls = recall_score(t, p, labels=labels, average=None, zero_division=0)
        lines.append(f"{lang:<12}{int(mask.sum()):>6}{accuracy_score(t, p):>10.3f}"
                     f"{f1_score(t, p, labels=labels, average='macro', zero_division=0):>10.3f}"
                     + "".join(f"{r:>20.3f}" for r in recalls))
    return "\n".join(lines)


@torch.no_grad()
def evaluate(config_path: str | None = None) -> None:
    cfg = load_config(config_path) if config_path else load_config()
    device = get_device()

    checkpoint_path = Path(cfg["paths"]["models_dir"]) / "mlp_classifier.pt"

    data = np.load(Path(cfg["paths"]["embeddings_dir"]) / "test.npz")
    embedded_as = str(data["representation"]) if "representation" in data else "projected"
    model = load_checkpoint(checkpoint_path, device, representation=embedded_as)
    x = torch.tensor(data["embeddings"], dtype=torch.float32).to(device)
    y_true = data["labels"]

    logits = model(x)
    y_pred = logits.argmax(dim=-1).cpu().numpy()

    names = cfg["classes"]["names"]
    print(classification_report(y_true, y_pred, target_names=names, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(names)
    print(confusion_matrix(y_true, y_pred))

    languages = _split_languages(cfg, "test")
    if len(languages) != len(y_true):
        print("\nNo per-language breakdown: data/processed/test.jsonl doesn't match the test embeddings "
              "(re-run `aicontrib embed` after `prepare`).")
        return
    print("\nPer language (recall = share of that class's rows predicted correctly):")
    print(per_language_report(y_true, y_pred, languages, names))


if __name__ == "__main__":
    evaluate()
