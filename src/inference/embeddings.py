import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


class EmbeddingExtractor:
    """Extract embeddings from trained model."""

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, loader: DataLoader):
        """Return (embeddings, track_ids) arrays."""
        all_embeds, all_ids = [], []

        for features, _, track_ids in tqdm(loader, desc="Extracting"):
            features = features.to(self.device)
            emb = self.model.get_embedding(features)
            all_embeds.append(emb.cpu().numpy())
            ids = track_ids if isinstance(track_ids, list) else track_ids.tolist()
            all_ids.extend(ids)

        return np.concatenate(all_embeds), all_ids

    def save(self, embeddings: np.ndarray, track_ids: list, output_path: str):
        """Persist embeddings as .npy."""
        np.save(output_path, embeddings)
        pd.DataFrame({"track_id": track_ids}).to_csv(
            output_path.replace(".npy", "_ids.csv"), index=False
        )
