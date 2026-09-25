from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aicontrib.config import load_config
from aicontrib.device import get_device
from aicontrib.model.classifier import MLPClassifier
from aicontrib.monitor import MetricsLogger, serve_dashboard


def _load_split(cfg: dict, split: str) -> TensorDataset:
    npz_path = Path(cfg["paths"]["embeddings_dir"]) / f"{split}.npz"
    data = np.load(npz_path)
    x = torch.tensor(data["embeddings"], dtype=torch.float32)
    y = torch.tensor(data["labels"], dtype=torch.long)
    return TensorDataset(x, y)


@torch.no_grad()
def _evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return f1_score(all_labels, all_preds, average="macro")


def train(config_path: str | None = None) -> Path:
    cfg = load_config(config_path) if config_path else load_config()
    device = get_device()
    torch.manual_seed(cfg["training"]["seed"])

    train_ds = _load_split(cfg, "train")
    val_ds = _load_split(cfg, "validation")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])

    input_dim = train_ds.tensors[0].shape[1]
    num_classes = len(cfg["classes"]["names"])
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=cfg["model"]["hidden_dims"],
        num_classes=num_classes,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    epochs_without_improvement = 0
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "mlp_classifier.pt"

    metrics_logger = MetricsLogger(models_dir / "metrics.jsonl")
    port = cfg["monitor"]["port"]
    serve_dashboard(metrics_logger.path, port, background=True)
    print(f"Training dashboard: http://127.0.0.1:{port}")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        val_f1 = _evaluate_loader(model, val_loader, device)
        train_loss = total_loss / len(train_ds)
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val_macro_f1={val_f1:.4f}")
        metrics_logger.log(epoch=epoch + 1, train_loss=train_loss, val_macro_f1=val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dims": cfg["model"]["hidden_dims"],
                    "num_classes": num_classes,
                    "dropout": cfg["model"]["dropout"],
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["training"]["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch + 1} (best val_macro_f1={best_f1:.4f})")
                break

    print(f"best val_macro_f1={best_f1:.4f}, checkpoint saved to {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train()
