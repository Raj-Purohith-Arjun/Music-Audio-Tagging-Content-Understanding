import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from typing import Optional


class Trainer:
    """Configurable training loop with checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        lr = config.get("learning_rate", 1e-3)
        wd = config.get("weight_decay", 1e-4)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = self._build_scheduler(config)
        self.criterion = nn.BCEWithLogitsLoss()

        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.patience = config.get("patience", 10)
        self.history = {"train_loss": [], "val_loss": [], "val_f1": []}

    def _build_scheduler(self, config):
        sched = config.get("scheduler", "cosine")
        epochs = config.get("num_epochs", 50)
        if sched == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif sched == "step":
            return StepLR(self.optimizer, step_size=epochs // 3, gamma=0.5)
        return None

    def train_epoch(self, loader) -> float:
        """Single training pass."""
        self.model.train()
        total_loss = 0.0
        for features, labels, _ in tqdm(loader, leave=False, desc="Train"):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader) -> tuple:
        """Validation pass; returns (loss, f1)."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for features, labels, _ in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(features)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        preds_t = torch.cat(all_preds)
        labels_t = torch.cat(all_labels)
        f1 = self._f1_score(preds_t, labels_t)

        return total_loss / len(loader), f1

    def _f1_score(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        tp = (preds * labels).sum().item()
        fp = (preds * (1 - labels)).sum().item()
        fn = ((1 - preds) * labels).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        path = os.path.join(self.checkpoint_dir, f"model_{tag}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt["epoch"], ckpt["val_loss"]

    def fit(self, train_loader, val_loader) -> dict:
        """Full training loop with early stopping."""
        num_epochs = self.config.get("num_epochs", 50)

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_f1 = self.eval_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"f1={val_f1:.4f} | {elapsed:.1f}s"
            )

            # checkpoint best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, tag="best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            self.save_checkpoint(epoch, val_loss, tag="latest")

        return self.history
