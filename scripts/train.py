#!/usr/bin/env python3
"""Train audio tagging model."""

import argparse
import yaml
import torch
from pathlib import Path

from src.data.dataset import AudioTaggingDataset, load_fma_metadata, build_label_encoder
from src.data.dataloader import create_dataloaders
from src.models.factory import build_model
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train music tagger")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # load metadata
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

    train_cfg = cfg["dataset"]
    train_loader, val_loader, _ = create_dataloaders(
        dataset,
        train_ratio=train_cfg["train_split"],
        val_ratio=train_cfg["val_split"],
        batch_size=cfg["training"]["batch_size"],
    )

    model = build_model(cfg["model"])
    trainer = Trainer(model, cfg["training"])

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.fit(train_loader, val_loader)
    print(f"Best val loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
