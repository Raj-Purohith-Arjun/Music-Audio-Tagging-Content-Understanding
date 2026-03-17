import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNModel(nn.Module):
    """CNN + LSTM for temporal audio modeling."""

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        # convolutional frontend
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # compress freq, keep time

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        # recurrent backend
        self.rnn = nn.GRU(
            input_size=128 * 8,  # freq_bins after pooling
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        rnn_out_dim = hidden_size * 2  # bidirectional

        self.embedding = nn.Sequential(
            nn.Linear(rnn_out_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: (B, C, F, T)
        b = x.size(0)
        x = self.cnn(x)                     # (B, 128, F', T)
        x = x.permute(0, 3, 1, 2)           # (B, T, 128, F')
        x = x.reshape(b, x.size(1), -1)     # (B, T, 128*F')

        out, _ = self.rnn(x)                # (B, T, 2*H)
        x = out[:, -1, :]                   # last timestep

        emb = self.embedding(x)
        logits = self.classifier(emb)
        return logits

    def get_embedding(self, x):
        """Return embedding vector."""
        b = x.size(0)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, x.size(1), -1)
        out, _ = self.rnn(x)
        x = out[:, -1, :]
        return self.embedding(x)
