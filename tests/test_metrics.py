import numpy as np
import pytest
from src.evaluation.metrics import compute_metrics, per_class_metrics


def make_data(n=100, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(n, n_classes)).astype(float)
    y_prob = rng.random((n, n_classes))
    return y_true, y_prob


def test_compute_metrics_keys():
    y_true, y_prob = make_data()
    m = compute_metrics(y_true, y_prob)
    assert "f1_macro" in m
    assert "roc_auc" in m
    assert "precision" in m
    assert "recall" in m


def test_f1_perfect():
    y_true = np.eye(4)
    y_prob = np.eye(4)
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["f1_macro"] == pytest.approx(1.0)


def test_per_class_metrics_keys():
    y_true, y_prob = make_data(n_classes=3)
    classes = ["rock", "jazz", "pop"]
    pc = per_class_metrics(y_true, y_prob, class_names=classes)
    assert set(pc.keys()) == {"rock", "jazz", "pop"}
    assert "f1" in pc["rock"]
