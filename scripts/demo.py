#!/usr/bin/env python3
"""Flask demo: upload audio, get tags + similar tracks."""

import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify

from src.models.factory import build_model
from src.inference.predictor import AudioTagger
from src.inference.similarity import MusicSimilarityIndex

app = Flask(__name__)

# globals loaded at startup
tagger = None


def load_tagger():
    global tagger
    config_path = os.environ.get("CONFIG", "configs/default.yaml")
    checkpoint_path = os.environ.get("CHECKPOINT", "checkpoints/model_best.pt")
    index_path = os.environ.get("INDEX_PATH", "embeddings/faiss_index.bin")
    ids_path = os.environ.get("IDS_PATH", "embeddings/track_embeddings_ids.csv")
    label_path = os.environ.get("LABELS_PATH", "embeddings/label_encoder.json")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    with open(label_path) as f:
        label_enc = json.load(f)

    cfg["model"]["num_classes"] = len(label_enc)
    model = build_model(cfg["model"])
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    import pandas as pd
    track_ids = pd.read_csv(ids_path)["track_id"].tolist()
    embedding_dim = cfg["model"]["embedding_dim"]
    sim_index = MusicSimilarityIndex.load(index_path, track_ids, embedding_dim)

    extractor_cfg = {
        "sample_rate": cfg["dataset"]["sample_rate"],
        "n_mels": cfg["features"]["n_mels"],
        "n_fft": cfg["features"]["n_fft"],
        "hop_length": cfg["features"]["hop_length"],
        "duration": cfg["dataset"]["duration"],
    }

    tagger = AudioTagger(model, label_enc, sim_index, extractor_config=extractor_cfg)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Accept audio file upload, return tags and similar tracks."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    tmp_path = f"/tmp/{audio_file.filename}"
    audio_file.save(tmp_path)

    try:
        result = tagger.predict(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    # only load tagger if required files exist
    try:
        load_tagger()
    except Exception as e:
        print(f"Tagger not loaded: {e}")
    app.run(host="0.0.0.0", port=5000, debug=False)
