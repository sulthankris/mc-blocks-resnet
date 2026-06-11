"""Training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics collected for one epoch or split."""

    loss: float
    accuracy: float
    macro_f1: float | None = None


def _autocast(enabled: bool):
    return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    epoch: int,
) -> EpochMetrics:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
        total_samples += batch_size
        progress.set_postfix(loss=total_loss / max(total_samples, 1), acc=total_correct / max(total_samples, 1))

    return EpochMetrics(loss=total_loss / total_samples, accuracy=total_correct / total_samples)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    split_name: str = "val",
) -> tuple[EpochMetrics, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a model and return metrics, targets, predictions, and probabilities."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[list[float]] = []
    progress = tqdm(loader, desc=split_name, leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((predictions == targets).sum().detach().cpu())
        total_samples += batch_size
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        y_scores.extend(probabilities.cpu().tolist())
        progress.set_postfix(loss=total_loss / max(total_samples, 1), acc=total_correct / max(total_samples, 1))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics = EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        macro_f1=macro_f1,
    )
    return metrics, np.asarray(y_true), np.asarray(y_pred), np.asarray(y_scores)


def save_checkpoint(
    path: str,
    model: nn.Module,
    model_name: str,
    num_classes: int,
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    """Save a project-standard model checkpoint."""
    torch.save(
        {
            "model_name": model_name,
            "num_classes": num_classes,
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(key): value for key, value in idx_to_class.items()},
            "state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )
