import torch
import pytest
from src.models.cnn import CNNModel
from src.models.crnn import CRNNModel
from src.models.factory import build_model


@pytest.fixture
def dummy_batch():
    # (batch, channels, freq, time)
    return torch.randn(4, 1, 128, 128)


def test_cnn_forward(dummy_batch):
    model = CNNModel(num_classes=8)
    out = model(dummy_batch)
    assert out.shape == (4, 8)


def test_cnn_embedding(dummy_batch):
    model = CNNModel(num_classes=8, embedding_dim=64)
    emb = model.get_embedding(dummy_batch)
    assert emb.shape == (4, 64)


def test_crnn_forward(dummy_batch):
    model = CRNNModel(num_classes=8, hidden_size=64)
    out = model(dummy_batch)
    assert out.shape == (4, 8)


def test_crnn_embedding(dummy_batch):
    model = CRNNModel(num_classes=8, hidden_size=64, embedding_dim=64)
    emb = model.get_embedding(dummy_batch)
    assert emb.shape == (4, 64)


def test_factory_cnn():
    cfg = {"architecture": "cnn", "num_classes": 4, "embedding_dim": 32, "dropout": 0.1}
    model = build_model(cfg)
    assert isinstance(model, CNNModel)


def test_factory_crnn():
    cfg = {
        "architecture": "crnn",
        "num_classes": 4,
        "embedding_dim": 32,
        "dropout": 0.1,
        "hidden_size": 64,
    }
    model = build_model(cfg)
    assert isinstance(model, CRNNModel)


def test_factory_unknown():
    with pytest.raises(ValueError):
        build_model({"architecture": "transformer"})
