import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.evaluation.metrics import compute_metrics, per_class_metrics


class Evaluator:
    """Run inference and compute evaluation metrics."""

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        """Return (probas, labels, track_ids)."""
        self.model.eval()
        all_proba, all_labels, all_ids = [], [], []

        for features, labels, track_ids in loader:
            features = features.to(self.device)
            logits = self.model(features)
            proba = torch.sigmoid(logits).cpu().numpy()
            all_proba.append(proba)
            all_labels.append(labels.numpy())
            all_ids.extend(track_ids if isinstance(track_ids, list) else track_ids.tolist())

        return (
            np.concatenate(all_proba),
            np.concatenate(all_labels),
            all_ids,
        )

    def evaluate(self, loader: DataLoader, class_names: list = None) -> dict:
        """Full evaluation with all metrics."""
        proba, labels, _ = self.predict(loader)
        overall = compute_metrics(labels, proba)
        per_class = per_class_metrics(labels, proba, class_names)
        return {"overall": overall, "per_class": per_class}
