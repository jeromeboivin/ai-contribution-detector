from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from aicontrib.config import load_config
from aicontrib.device import get_device
from aicontrib.model.classifier import MLPClassifier


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> MLPClassifier:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MLPClassifier(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
        num_classes=ckpt["num_classes"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(config_path: str | None = None) -> None:
    cfg = load_config(config_path) if config_path else load_config()
    device = get_device()

    checkpoint_path = Path(cfg["paths"]["models_dir"]) / "mlp_classifier.pt"
    model = load_checkpoint(checkpoint_path, device)

    data = np.load(Path(cfg["paths"]["embeddings_dir"]) / "test.npz")
    x = torch.tensor(data["embeddings"], dtype=torch.float32).to(device)
    y_true = data["labels"]

    logits = model(x)
    y_pred = logits.argmax(dim=-1).cpu().numpy()

    names = cfg["classes"]["names"]
    print(classification_report(y_true, y_pred, target_names=names, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(names)
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()
