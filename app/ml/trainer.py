from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.metrics import accuracy_from_cm, confusion_matrix, per_class_prf

ProgressCb = Callable[[float, str], None]
LogCb = Callable[[str], None]


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    use_class_weights: bool,
    device: torch.device,
    seed: int,
    progress_cb: ProgressCb | None = None,
    log_cb: LogCb | None = None,
) -> dict:
    """Train with per-epoch validation and early stopping on val_loss.

    On return the model is loaded with the best (lowest val-loss) weights.
    num_workers=0 on purpose: safe on Windows and inside service threads.
    """
    log = log_cb or (lambda m: None)
    progress = progress_cb or (lambda frac, step: None)

    torch.manual_seed(seed)
    model = model.to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    class_weights = None
    if use_class_weights:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
        weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        log(f"class weights: {np.round(weights, 4).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss, batches = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if has_batchnorm and x_batch.shape[0] == 1:
                continue  # BatchNorm cannot compute stats from one sample
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
        train_loss = running_loss / max(batches, 1)

        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            val_loss = float(criterion(logits, y_val_t).item())
            val_pred = logits.argmax(dim=1).cpu().numpy()
        cm = confusion_matrix(y_val, val_pred, num_classes)
        _, _, val_f1, _ = per_class_prf(cm)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(accuracy_from_cm(cm), 6),
            "val_f1_macro": round(float(val_f1.mean()), 6),
            "epoch_time_sec": round(time.perf_counter() - epoch_start, 3),
        })
        log(f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {history[-1]['val_accuracy']:.4f}")
        progress(epoch / epochs, f"training epoch {epoch}/{epochs}")

        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= early_stopping_patience:
                log(f"early stopping at epoch {epoch} (no val_loss improvement for {stale} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6) if best_val_loss != float("inf") else None,
        "best_val_metrics": history[best_epoch - 1] if history else None,
        "duration_sec": round(time.perf_counter() - started, 3),
        "num_train_samples": int(len(X_train)),
        "num_val_samples": int(len(X_val)),
    }