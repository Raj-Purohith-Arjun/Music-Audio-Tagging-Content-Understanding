import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


def load_fma_metadata(metadata_dir: str) -> pd.DataFrame:
    """Load FMA tracks metadata and genre labels."""
    tracks_path = os.path.join(metadata_dir, "tracks.csv")
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    # flatten multi-level columns
    keep = tracks["set", "split"].notna()
    tracks = tracks[keep]

    genre_col = tracks["track", "genre_top"]
    split_col = tracks["set", "split"]

    df = pd.DataFrame({
        "track_id": tracks.index,
        "genre": genre_col.values,
        "split": split_col.values,
    }).dropna()

    return df


def build_label_encoder(genres: pd.Series) -> dict:
    """Map genre strings to integer indices."""
    unique = sorted(genres.unique())
    return {g: i for i, g in enumerate(unique)}


class AudioTaggingDataset(Dataset):
    """PyTorch dataset for audio tag prediction."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        features_dir: str,
        label_encoder: dict,
        num_classes: int,
        transform=None,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.features_dir = Path(features_dir)
        self.label_encoder = label_encoder
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        track_id = int(row["track_id"])
        genre = row["genre"]

        # load precomputed feature tensor
        feat_path = self.features_dir / f"{track_id:06d}.npy"
        features = np.load(str(feat_path))
        features = torch.from_numpy(features).float()

        if self.transform:
            features = self.transform(features)

        # one-hot label
        label = torch.zeros(self.num_classes)
        if genre in self.label_encoder:
            label[self.label_encoder[genre]] = 1.0

        return features, label, track_id


class MagnaTagAtuneDataset(Dataset):
    """Dataset wrapper for MagnaTagATune multi-label tagging."""

    def __init__(
        self,
        annotations_file: str,
        features_dir: str,
        top_tags: list,
        transform=None,
    ):
        self.df = pd.read_csv(annotations_file, sep="\t")
        self.features_dir = Path(features_dir)
        self.top_tags = top_tags
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_id = row["clip_id"]

        feat_path = self.features_dir / f"{clip_id}.npy"
        features = np.load(str(feat_path))
        features = torch.from_numpy(features).float()

        if self.transform:
            features = self.transform(features)

        # multi-label vector
        label = torch.tensor(
            [float(row.get(tag, 0)) for tag in self.top_tags],
            dtype=torch.float,
        )

        return features, label, clip_id
