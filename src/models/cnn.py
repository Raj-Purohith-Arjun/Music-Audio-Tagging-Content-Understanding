import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Pool."""

    def __init__(self, in_ch: int, out_ch: int, pool_size=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class CNNModel(nn.Module):
    """CNN baseline for spectrogram classification."""

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        emb = self.embedding(x)
        logits = self.classifier(emb)
        return logits

    def get_embedding(self, x):
        """Extract embedding before classifier head."""
        x = self.features(x)
        x = self.gap(x)
        return self.embedding(x)
