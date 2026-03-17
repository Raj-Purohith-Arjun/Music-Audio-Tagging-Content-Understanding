import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from typing import Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    average: str = "macro",
) -> dict:
    """Compute classification metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba, average=average)
        metrics["avg_precision"] = average_precision_score(y_true, y_pred_proba, average=average)
    except ValueError:
        metrics["roc_auc"] = 0.0
        metrics["avg_precision"] = 0.0

    return metrics


def per_class_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: Optional[list] = None,
    threshold: float = 0.5,
) -> dict:
    """Per-class F1, precision, recall."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    n_classes = y_true.shape[1]
    names = class_names or [str(i) for i in range(n_classes)]

    results = {}
    for i, name in enumerate(names):
        results[name] = {
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
        }
    return results
