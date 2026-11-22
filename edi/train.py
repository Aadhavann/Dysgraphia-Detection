"""Training loop for the CNN backbone."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from .config import PipelineConfig
from .model import DysgraphiaClassifier


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_auc: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    best_val_auc: float = 0.0
    best_epoch: int = -1
    stopped_epoch: int = -1

    @property
    def epochs_run(self) -> int:
        return len(self.val_auc)


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches, so a config identifies a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_classifier(
    model: DysgraphiaClassifier,
    train_loader,
    val_loader,
    device: torch.device,
    config: PipelineConfig,
    checkpoint_path: Path | str,
    verbose: bool = True,
) -> tuple[DysgraphiaClassifier, TrainingHistory]:
    """Train with early stopping on validation AUC, restoring the best weights.

    Model selection is on AUC rather than accuracy because the two classes are
    not exactly balanced and AUC is the metric the paper reports.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config.scheduler_patience, factor=config.scheduler_factor
    )

    history = TrainingHistory()
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for data, target, _ in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_correct += int(output.argmax(dim=1).eq(target).sum().item())

        model.eval()
        val_loss, val_correct = 0.0, 0
        val_probs: List[float] = []
        val_labels: List[int] = []

        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += float(criterion(output, target).item())
                val_correct += int(output.argmax(dim=1).eq(target).sum().item())
                val_probs.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy().tolist())
                val_labels.extend(target.cpu().numpy().tolist())

        train_acc = train_correct / max(len(train_loader.dataset), 1)
        val_acc = val_correct / max(len(val_loader.dataset), 1)
        val_auc = float(roc_auc_score(val_labels, val_probs)) if len(set(val_labels)) > 1 else float("nan")

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.train_loss.append(train_loss / max(len(train_loader), 1))
        history.val_loss.append(val_loss / max(len(val_loader), 1))
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)
        history.val_auc.append(val_auc)
        history.learning_rate.append(current_lr)

        if not np.isnan(val_auc):
            scheduler.step(val_auc)

        if verbose:
            print(
                f"  epoch {epoch + 1:>3d}/{config.epochs}  "
                f"train_acc {train_acc:.4f}  val_acc {val_acc:.4f}  "
                f"val_auc {val_auc:.4f}  lr {current_lr:.2e}"
            )

        if not np.isnan(val_auc) and val_auc > history.best_val_auc:
            history.best_val_auc = val_auc
            history.best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                history.stopped_epoch = epoch + 1
                if verbose:
                    print(f"  early stopping at epoch {epoch + 1}")
                break

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif verbose:
        print("  WARNING: no checkpoint was written; keeping final-epoch weights")

    return model, history
