#!/usr/bin/env python3
"""Evaluate trained model on test set."""

import argparse
import yaml
import json
import torch
from pathlib import Path

from src.data.dataset import AudioTaggingDataset, load_fma_metadata, build_label_encoder
from src.data.dataloader import create_dataloaders
from src.models.factory import build_model
from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate music tagger")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--output", default="results/metrics.json")
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

    _, _, test_loader = create_dataloaders(
        dataset,
        train_ratio=cfg["dataset"]["train_split"],
        val_ratio=cfg["dataset"]["val_split"],
        batch_size=cfg["training"]["batch_size"],
    )

    model = build_model(cfg["model"])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    evaluator = Evaluator(model)
    results = evaluator.evaluate(test_loader, class_names=list(label_enc.keys()))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")
    print("Overall metrics:")
    for k, v in results["overall"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
