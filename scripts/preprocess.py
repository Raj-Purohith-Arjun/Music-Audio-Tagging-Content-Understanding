#!/usr/bin/env python3
"""Preprocess raw audio files into feature tensors."""

import argparse
import yaml

from src.data.dataset import load_fma_metadata
from src.features.preprocess import preprocess_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess audio features")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", required=True, help="FMA audio root dir")
    parser.add_argument("--output-dir", default="data/features")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    meta = load_fma_metadata(cfg["dataset"]["metadata_dir"])
    track_ids = meta["track_id"].tolist()

    feat_cfg = {
        "sample_rate": cfg["dataset"]["sample_rate"],
        "n_mels": cfg["features"]["n_mels"],
        "n_mfcc": cfg["features"]["n_mfcc"],
        "n_fft": cfg["features"]["n_fft"],
        "hop_length": cfg["features"]["hop_length"],
        "fmax": cfg["features"]["fmax"],
        "duration": cfg["dataset"]["duration"],
    }

    results = preprocess_dataset(
        track_ids=track_ids,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        extractor_config=feat_cfg,
        num_workers=args.workers,
    )
    print(f"Done: {results['success']} success, {results['failed']} failed")


if __name__ == "__main__":
    main()
