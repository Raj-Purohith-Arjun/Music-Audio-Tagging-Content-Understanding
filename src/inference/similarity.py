import numpy as np
import faiss
import pandas as pd
from pathlib import Path


class MusicSimilarityIndex:
    """FAISS-based nearest-neighbor music retrieval."""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.track_ids = []

    def build(self, embeddings: np.ndarray, track_ids: list):
        """Add embeddings to FAISS index."""
        vecs = embeddings.astype(np.float32)
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.track_ids = list(track_ids)

    def search(self, query: np.ndarray, top_k: int = 10) -> list:
        """Return top-k similar track ids."""
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.track_ids):
                results.append({"track_id": self.track_ids[idx], "distance": float(dist)})
        return results

    def save(self, path: str):
        """Persist FAISS index to disk."""
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, track_ids: list, embedding_dim: int) -> "MusicSimilarityIndex":
        """Restore index from disk."""
        obj = cls(embedding_dim)
        obj.index = faiss.read_index(path)
        obj.track_ids = track_ids
        return obj
