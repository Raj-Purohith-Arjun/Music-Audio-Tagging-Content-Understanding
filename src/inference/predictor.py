import torch
import numpy as np
from src.features.extractor import AudioFeatureExtractor
from src.inference.similarity import MusicSimilarityIndex


class AudioTagger:
    """End-to-end inference: audio file -> tags + similar tracks."""

    def __init__(
        self,
        model,
        label_encoder: dict,
        similarity_index: MusicSimilarityIndex,
        device: torch.device = None,
        extractor_config: dict = None,
    ):
        self.model = model
        self.label_encoder = label_encoder
        self.id_to_label = {v: k for k, v in label_encoder.items()}
        self.similarity_index = similarity_index
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        cfg = extractor_config or {}
        self.extractor = AudioFeatureExtractor(**cfg)

    @torch.no_grad()
    def predict(self, audio_path: str, threshold: float = 0.5) -> dict:
        """Predict tags and return similar tracks."""
        feat = self.extractor.extract(audio_path, feature_type="mel_spectrogram")
        tensor = self.extractor.to_tensor(feat).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        proba = torch.sigmoid(logits).cpu().numpy()[0]

        tags = [
            {"tag": self.id_to_label[i], "score": float(p)}
            for i, p in enumerate(proba)
            if p >= threshold and i in self.id_to_label
        ]
        tags.sort(key=lambda x: -x["score"])

        embedding = self.model.get_embedding(tensor).cpu().numpy()[0]
        similar = self.similarity_index.search(embedding, top_k=5)

        return {"tags": tags, "similar_tracks": similar, "embedding": embedding.tolist()}
