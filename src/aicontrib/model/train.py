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
from aicontrib.model.evaluate import load_checkpoint
from aicontrib.monitor import MetricsLogger, serve_dashboard


def _load_split(cfg: dict, split: str) -> TensorDataset:
    npz_path = Path(cfg["paths"]["embeddings_dir"]) / f"{split}.npz"
    data = np.load(npz_path)
    x = torch.tensor(data["embeddings"], dtype=torch.float32)
    y = torch.tensor(data["labels"], dtype=torch.long)
    return TensorDataset(x, y)


def _representation(cfg: dict, split: str) -> str:
    with np.load(Path(cfg["paths"]["embeddings_dir"]) / f"{split}.npz") as data:
        return str(data["representation"]) if "representation" in data else "projected"


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


def _load_for_resume(checkpoint_path: Path, device: torch.device, cfg: dict, input_dim: int,
                     representation: str) -> tuple[MLPClassifier, dict]:
    """The saved model and its checkpoint, if it can continue training on the current embeddings."""
    if not checkpoint_path.exists():
        raise ValueError(f"Nothing to resume: no model at {checkpoint_path}. Run `aicontrib train` first.")
    # Refuses other classes or another embedding representation.
    model = load_checkpoint(checkpoint_path, device, representation=representation, class_names=cfg["classes"]["names"])
    ckpt = torch.load(checkpoint_path, map_location=device)
    mismatches = [f"{name}: saved {saved}, now {now}" for name, saved, now in (
        ("embedding size", ckpt["input_dim"], input_dim),
        ("model.hidden_dims", ckpt["hidden_dims"], cfg["model"]["hidden_dims"]),
        ("model.dropout", ckpt["dropout"], cfg["model"]["dropout"]),
    ) if saved != now]
    if mismatches:
        raise ValueError(f"Can't resume from {checkpoint_path} ({'; '.join(mismatches)}). "
                         "Run `aicontrib train` without --resume to start over.")
    return model, ckpt


def train(config_path: str | None = None, resume: bool = False) -> Path:
    """resume: continue from the saved best model (weights, optimizer state and learning rate) instead of
    starting over -- e.g. after raising early_stopping_patience. training.epochs then counts from there."""
    cfg = load_config(config_path) if config_path else load_config()
    device = get_device()
    torch.manual_seed(cfg["training"]["seed"])

    train_ds = _load_split(cfg, "train")
    val_ds = _load_split(cfg, "validation")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])

    input_dim = train_ds.tensors[0].shape[1]
    num_classes = len(cfg["classes"]["names"])
    for split, ds in (("train", train_ds), ("validation", val_ds)):
        if int(ds.tensors[1].max()) >= num_classes:
            raise ValueError(f"The {split} embeddings have labels for more classes than classes.names "
                             f"{cfg['classes']['names']} -- they were prepared for other classes. "
                             "Re-run `aicontrib prepare` and `aicontrib embed`.")
    representation = _representation(cfg, "train")
    if _representation(cfg, "validation") != representation:
        raise ValueError("train and validation embeddings use different representations -- re-run `aicontrib embed`")

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "mlp_classifier.pt"
    tcfg = cfg["training"]

    if resume:
        model, ckpt = _load_for_resume(checkpoint_path, device, cfg, input_dim, representation)
    else:
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=cfg["model"]["hidden_dims"],
            num_classes=num_classes,
            dropout=cfg["model"]["dropout"],
        )
        model.fit_input_scaling(train_ds.tensors[0])
        model.to(device)
        if checkpoint_path.exists():
            print(f"Training from scratch: the model saved at {checkpoint_path} will be replaced "
                  "(`aicontrib train --resume` continues from it instead).")

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    start_epoch, best_f1 = 0, -1.0
    if resume:
        if "optimizer" in ckpt:  # checkpoints of older versions: weights only, fresh optimizer at training.lr
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        # The bar to beat, measured on the current validation data.
        best_f1 = _evaluate_loader(model, val_loader, device)
        print(f"Resuming from epoch {start_epoch}: validation macro-F1 {best_f1:.4f}, "
              f"learning rate {optimizer.param_groups[0]['lr']:.2e}")

    # Halve the learning rate when validation macro-F1 stalls: smaller steps often find further gains
    # before early stopping gives up. threshold=0 -> same "strictly better" rule as early stopping.
    # Built fresh on resume too, so changed lr_reduce_* settings apply.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=tcfg["lr_reduce_factor"], patience=tcfg["lr_reduce_patience"],
        min_lr=tcfg["min_lr"], threshold=0.0,
    )
    if resume:
        scheduler.best = best_f1
    criterion = nn.CrossEntropyLoss()

    epochs_without_improvement = 0
    # Resuming keeps the dashboard's history up to the resumed epoch; epochs after it are replaced.
    metrics_logger = MetricsLogger(models_dir / "metrics.jsonl", keep_until_epoch=start_epoch if resume else None)
    port = cfg["monitor"]["port"]
    serve_dashboard(metrics_logger.path, port, background=True)
    print(f"Training dashboard: http://127.0.0.1:{port}")

    for epoch in range(start_epoch, start_epoch + tcfg["epochs"]):
        model.train()
        total_loss = 0.0
        lr = optimizer.param_groups[0]["lr"]
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
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val_macro_f1={val_f1:.4f} lr={lr:.2e}")
        metrics_logger.log(epoch=epoch + 1, train_loss=train_loss, val_macro_f1=val_f1, lr=lr)
        scheduler.step(val_f1)
        if optimizer.param_groups[0]["lr"] < lr:
            print(f"  validation macro-F1 stalled: learning rate reduced to {optimizer.param_groups[0]['lr']:.2e}")

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
                    "representation": representation,
                    "class_names": cfg["classes"]["names"],
                    # For --resume:
                    "epoch": epoch + 1,
                    "val_macro_f1": val_f1,
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= tcfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch + 1}: no better validation macro-F1 for "
                      f"{epochs_without_improvement} epochs (best {best_f1:.4f})")
                break

    print(f"best val_macro_f1={best_f1:.4f}, checkpoint saved to {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train()
