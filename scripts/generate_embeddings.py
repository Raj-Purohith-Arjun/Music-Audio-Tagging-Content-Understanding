#!/usr/bin/env python3
"""Generate embeddings and build FAISS index."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from src.data.dataset import AudioTaggingDataset, load_fma_metadata, build_label_encoder
from src.data.dataloader import create_dataloaders
from src.models.factory import build_model
from src.inference.embeddings import EmbeddingExtractor
from src.inference.similarity import MusicSimilarityIndex


def parse_args():
    parser = argparse.ArgumentParser(description="Generate track embeddings")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--output-dir", default="embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    meta = load_fma_metadata(cfg["dataset"]["metadata_dir"])
    label_enc = build_label_encoder(meta["genre"])
    num_classes = len(label_enc)
    cfg["model"]["num_classes"] = num_classes

    dataset = AudioTaggingDataset(
        metadata=meta,
        features_dir=args.features_dir,
        label_encoder=label_enc,
        num_classes=num_classes,
    )

    loader = create_dataloaders(
        dataset,
        train_ratio=1.0,
        val_ratio=0.0,
        batch_size=cfg["training"]["batch_size"],
    )[0]

    model = build_model(cfg["model"])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    extractor = EmbeddingExtractor(model)
    embeddings, track_ids = extractor.extract(loader)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = str(out_dir / "track_embeddings.npy")
    extractor.save(embeddings, track_ids, emb_path)

    # build FAISS index
    embedding_dim = embeddings.shape[1]
    idx = MusicSimilarityIndex(embedding_dim)
    idx.build(embeddings, track_ids)
    idx.save(str(out_dir / "faiss_index.bin"))

    print(f"Saved {len(track_ids)} embeddings to {out_dir}")


if __name__ == "__main__":
    main()
