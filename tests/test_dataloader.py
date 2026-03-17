import torch
import numpy as np
import pytest
import pandas as pd
from torch.utils.data import Dataset
from src.data.dataloader import create_dataloaders


class DummyDataset(Dataset):
    def __init__(self, n=100):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randn(1, 64, 64), torch.zeros(4), idx


def test_split_sizes():
    ds = DummyDataset(100)
    train, val, test = create_dataloaders(
        ds, train_ratio=0.7, val_ratio=0.15, batch_size=8, num_workers=0
    )
    # check approximate splits
    assert len(train.dataset) == 70
    assert len(val.dataset) == 15
    assert len(test.dataset) == 15


def test_batch_shapes():
    ds = DummyDataset(64)
    train, _, _ = create_dataloaders(
        ds, train_ratio=0.8, val_ratio=0.1, batch_size=16, num_workers=0
    )
    batch = next(iter(train))
    features, labels, ids = batch
    assert features.shape == (16, 1, 64, 64)
    assert labels.shape == (16, 4)
