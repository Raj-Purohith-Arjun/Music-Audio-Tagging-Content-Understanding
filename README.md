# 🎵 Music Audio Tagging & Content Understanding

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

A production-style deep learning system that automatically predicts music attributes — **genre, mood, and instrumentation** — directly from raw audio files. It also generates compact audio embeddings that power a FAISS-backed music recommendation engine.

---

## 📑 Table of Contents

1. [What This Project Does](#-what-this-project-does)
2. [System Architecture](#-system-architecture)
3. [Project Structure](#-project-structure)
4. [Prerequisites](#-prerequisites)
5. [Installation](#-installation)
6. [Dataset Setup](#-dataset-setup)
7. [Feature Extraction](#-feature-extraction)
8. [Models](#-models)
9. [How to Run — Step by Step](#-how-to-run--step-by-step)
   - [Step 1 — Preprocess Audio](#step-1--preprocess-audio)
   - [Step 2 — Train the Model](#step-2--train-the-model)
   - [Step 3 — Evaluate the Model](#step-3--evaluate-the-model)
   - [Step 4 — Generate Embeddings & FAISS Index](#step-4--generate-embeddings--faiss-index)
   - [Step 5 — Run the Demo Server](#step-5--run-the-demo-server)
10. [API Reference](#-api-reference)
11. [Embedding-Based Recommendation](#-embedding-based-recommendation)
12. [Configuration Reference](#-configuration-reference)
13. [Experiment Results](#-experiment-results)
14. [Docker](#-docker)
15. [Running Tests](#-running-tests)
16. [Project Workflow Diagram](#-project-workflow-diagram)

---

## 🎯 What This Project Does

Given any audio track (MP3, WAV, etc.), this system:

1. **Extracts audio features** — mel spectrograms, MFCCs, spectral contrast, and chroma using `librosa`.
2. **Classifies the track** — predicts genre labels (e.g., Rock, Hip-Hop, Electronic) using a CNN or CRNN deep learning model.
3. **Generates a 128-dimensional embedding** — a compact numerical fingerprint of the track's sonic character.
4. **Finds similar tracks** — uses FAISS (a fast approximate nearest-neighbour library) to retrieve the most sonically similar tracks from an indexed library.
5. **Serves predictions** — exposes a Flask REST API so any application can query predictions in real time.

---

## 🏗 System Architecture

```
Raw Audio (.mp3 / .wav)
        │
        ▼
┌─────────────────────┐
│  Feature Extraction  │  ← librosa: mel-spectrogram, MFCC, chroma, spectral contrast
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Deep Learning      │  ← CNN baseline  OR  CRNN (CNN + Bidirectional GRU)
│   Model              │
└─────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
Genre     128-dim
Labels    Embedding
             │
             ▼
     ┌──────────────┐
     │  FAISS Index  │  ← nearest-neighbour similarity search
     └──────────────┘
             │
             ▼
     Similar Track IDs
```

---

## 📁 Project Structure

```
Music-Audio-Tagging-Content-Understanding/
├── src/
│   ├── data/
│   │   ├── dataset.py          # AudioTaggingDataset, FMA metadata loader, label encoder
│   │   └── dataloader.py       # Train/val/test DataLoader splits
│   ├── features/
│   │   ├── extractor.py        # AudioFeatureExtractor — mel, MFCC, chroma, contrast
│   │   └── preprocess.py       # Batch feature extraction utilities
│   ├── models/
│   │   ├── cnn.py              # 4-block CNN baseline
│   │   ├── crnn.py             # CNN + Bidirectional GRU model
│   │   └── factory.py          # build_model() — picks model from config
│   ├── training/
│   │   └── trainer.py          # Training loop, early stopping, checkpointing
│   ├── evaluation/
│   │   └── metrics.py          # F1 (macro), ROC-AUC, precision, recall
│   └── inference/
│       ├── embeddings.py       # Batch embedding extraction
│       ├── predictor.py        # AudioTagger — audio file → tags + similar tracks
│       └── similarity.py       # MusicSimilarityIndex wrapping FAISS
├── scripts/
│   ├── preprocess.py           # Convert audio files → saved feature tensors
│   ├── train.py                # Run model training
│   ├── evaluate.py             # Run test-set evaluation, save metrics JSON
│   ├── generate_embeddings.py  # Build FAISS index from trained model
│   └── demo.py                 # Flask REST API server
├── configs/
│   ├── default.yaml            # CRNN training config (recommended)
│   └── cnn.yaml                # CNN-only config (faster, lighter)
├── notebooks/                  # Jupyter notebooks for exploration & visualisation
├── tests/                      # Unit tests (pytest)
│   ├── test_models.py
│   ├── test_features.py
│   ├── test_dataloader.py
│   ├── test_metrics.py
│   └── test_similarity.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ✅ Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher |
| pip | 23+ |
| ffmpeg | any recent version (for audio decoding) |
| libsndfile | any recent version |
| CUDA (optional) | 11.8+ for GPU training |
| Docker (optional) | 20+ |

> **macOS**: Install system dependencies with `brew install ffmpeg libsndfile`
>
> **Ubuntu/Debian**: `sudo apt-get install ffmpeg libsndfile1`

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Raj-Purohith-Arjun/Music-Audio-Tagging-Content-Understanding.git
cd Music-Audio-Tagging-Content-Understanding
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `torch` / `torchaudio` | Deep learning framework |
| `librosa` | Audio feature extraction |
| `numpy`, `pandas` | Numerical processing & metadata |
| `scikit-learn` | Metrics, label encoding |
| `faiss-cpu` | Fast similarity search |
| `flask` | REST API server |
| `matplotlib`, `seaborn` | Visualisation |
| `tqdm` | Progress bars |
| `pyyaml` | YAML config loading |
| `pytest` | Unit testing |

---

## 📦 Dataset Setup

This project uses the [FMA (Free Music Archive)](https://github.com/mdeff/fma) dataset — a publicly licensed, genre-balanced benchmark for music information retrieval.

### Download

```bash
mkdir -p data
cd data

# ~7.2 GB — 8,000 tracks × 8 genres (30-second clips)
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

# ~342 MB — CSV metadata (genres, moods, track info)
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

unzip fma_small.zip
unzip fma_metadata.zip
cd ..
```

### Expected directory layout after extraction

```
data/
├── fma_small/
│   ├── 000/
│   │   ├── 000002.mp3
│   │   └── ...
│   ├── 001/
│   └── ...
└── fma_metadata/
    ├── tracks.csv
    ├── genres.csv
    └── ...
```

### Dataset summary

| Split | Size |
|-------|------|
| Training (70%) | ~5,600 tracks |
| Validation (15%) | ~1,200 tracks |
| Test (15%) | ~1,200 tracks |

The 8 genres are: **Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock**.

---

## 🔊 Feature Extraction

Four complementary feature types are computed with `librosa` for every audio clip:

| Feature | Shape | What it captures |
|---------|-------|-----------------|
| **Mel Spectrogram** | 128 × T | Log-frequency energy distribution — primary input for genre detection |
| **MFCC** | 20 × T | Timbral texture — captures voice/instrument timbre |
| **Spectral Contrast** | 7 × T | Peak-to-valley ratio across frequency bands — highlights bright vs. dull sounds |
| **Chroma** | 12 × T | Pitch class distribution — captures harmonic and tonal content |

All features are:
- **Normalised** per clip to zero mean and unit variance.
- **Padded or trimmed** to a fixed time dimension T = `(sample_rate × duration) / hop_length` ≈ 1293 frames for the default settings (22050 Hz × 30 s / 512).

The default model uses mel spectrogram only. Set `feature_type: mel_spectrogram` in `configs/default.yaml` to change this.

---

## 🧠 Models

### CNN Baseline (`configs/cnn.yaml`)

A lightweight 4-block convolutional network operating on mel spectrograms.

```
Input (1 × 128 × T)
  └─ ConvBlock ×4  (Conv2d → BatchNorm → ReLU → MaxPool)
       └─ Global Average Pooling
            └─ Linear → ReLU → Dropout  →  128-dim embedding
                 └─ Linear classifier  →  N genre logits
```

- **Parameters**: ~1.2 M
- Trains fast; good starting point.

### CRNN (`configs/default.yaml`) — *recommended*

A CNN frontend feeds into a bidirectional GRU to model long-range temporal structure.

```
Input (1 × 128 × T)
  └─ CNN frontend  (4 × Conv2d → BN → ReLU → MaxPool along frequency axis)
       └─ Reshape → (T, 128 × F')
            └─ Bidirectional GRU (2 layers, hidden=256)
                 └─ Last hidden state
                      └─ Linear → ReLU → Dropout  →  128-dim embedding
                           └─ Linear classifier  →  N genre logits
```

- **Parameters**: ~3.1 M
- Better accuracy; captures rhythm, phrase, and development.

Both models expose `get_embedding(x)` to extract the 128-d audio fingerprint used by the recommendation engine.

---

## 🚀 How to Run — Step by Step

> **Make sure you are in the repository root** and your virtual environment is active before running any command.

### Step 1 — Preprocess Audio

Convert raw `.mp3` files into saved NumPy feature tensors. This only needs to run once.

```bash
python scripts/preprocess.py \
  --data-dir   data/fma_small \
  --output-dir data/features \
  --config     configs/default.yaml
```

After this step, `data/features/` will contain `.npy` files — one per track.

---

### Step 2 — Train the Model

```bash
# Train the recommended CRNN model
python scripts/train.py \
  --config       configs/default.yaml \
  --features-dir data/features

# Alternatively, train the faster CNN baseline
python scripts/train.py \
  --config       configs/cnn.yaml \
  --features-dir data/features
```

**Resume training from a checkpoint**

```bash
python scripts/train.py \
  --config   configs/default.yaml \
  --features-dir data/features \
  --resume   checkpoints/model_latest.pt
```

**What you will see during training**

```
Epoch   1/50 | train=0.6823 | val=0.5941 | f1=0.4812 | 42.3s
Epoch   2/50 | train=0.5102 | val=0.4873 | f1=0.5634 | 41.8s
...
Early stopping at epoch 38
Best val loss: 0.3241
```

Checkpoints are saved to `checkpoints/`:
- `checkpoints/model_best.pt` — best validation loss
- `checkpoints/model_latest.pt` — most recent epoch

---

### Step 3 — Evaluate the Model

Run evaluation on the held-out test set:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/model_best.pt \
  --config     configs/default.yaml \
  --output     results/metrics.json
```

This writes a JSON file with per-genre and macro-averaged metrics:

```json
{
  "f1_macro": 0.763,
  "roc_auc": 0.931,
  "precision": 0.771,
  "recall": 0.756,
  "per_class": {
    "Electronic": {"f1": 0.81, "precision": 0.84, "recall": 0.78},
    "Hip-Hop":    {"f1": 0.79, ...},
    ...
  }
}
```

---

### Step 4 — Generate Embeddings & FAISS Index

Extract embeddings for every track in the dataset and build a FAISS index for fast nearest-neighbour search.

```bash
python scripts/generate_embeddings.py \
  --checkpoint   checkpoints/model_best.pt \
  --config       configs/default.yaml \
  --features-dir data/features \
  --output-dir   embeddings
```

This creates:

```
embeddings/
├── faiss_index.bin              # FAISS L2 index
├── track_embeddings.npy         # (N × 128) embedding matrix
├── track_embeddings_ids.csv     # track_id ↔ row mapping
└── label_encoder.json           # genre label → class index
```

---

### Step 5 — Run the Demo Server

Start the Flask inference API:

```bash
python scripts/demo.py
```

The server reads paths from environment variables (or uses the defaults below):

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | `configs/default.yaml` | Model config |
| `CHECKPOINT` | `checkpoints/model_best.pt` | Trained weights |
| `INDEX_PATH` | `embeddings/faiss_index.bin` | FAISS index |
| `IDS_PATH` | `embeddings/track_embeddings_ids.csv` | Track ID mapping |
| `LABELS_PATH` | `embeddings/label_encoder.json` | Label encoder |

Override defaults:

```bash
CONFIG=configs/cnn.yaml \
CHECKPOINT=checkpoints/model_best.pt \
python scripts/demo.py
```

The server starts on **http://localhost:5000**.

---

## 📡 API Reference

### `GET /health`

Check that the server is up.

```bash
curl http://localhost:5000/health
```

```json
{"status": "ok"}
```

---

### `POST /predict`

Upload an audio file and receive predicted genre tags plus similar tracks.

```bash
curl -X POST \
  -F "audio=@/path/to/your/song.mp3" \
  http://localhost:5000/predict
```

**Response**

```json
{
  "tags": [
    {"tag": "Electronic", "score": 0.92},
    {"tag": "Instrumental", "score": 0.74}
  ],
  "similar_tracks": [
    {"track_id": "012345", "distance": 0.18},
    {"track_id": "067890", "distance": 0.22},
    {"track_id": "034512", "distance": 0.31},
    {"track_id": "089012", "distance": 0.35},
    {"track_id": "056789", "distance": 0.41}
  ],
  "embedding": [0.012, -0.341, 0.892, ...]
}
```

| Field | Description |
|-------|-------------|
| `tags` | Predicted genre labels with confidence scores ≥ 0.5 |
| `similar_tracks` | Top-5 nearest tracks from the FAISS index (lower distance = more similar) |
| `embedding` | 128-dimensional audio fingerprint of the uploaded track |

---

## 🎧 Embedding-Based Recommendation

The model's penultimate layer produces a **128-dimensional dense embedding** for each track. These vectors encode semantic audio properties (genre, mood, instrumentation) in a continuous space, enabling:

| Use Case | How |
|----------|-----|
| **Similar track retrieval** | FAISS L2 nearest-neighbour search over indexed embeddings |
| **Cold-start recommendations** | Tag a new track and find its nearest neighbours — no user history needed |
| **Collaborative filtering input** | Use embeddings as content features alongside interaction signals |
| **Playlist generation** | Chain similar-track lookups to build a cohesive playlist |

### Python usage

```python
from src.inference.similarity import MusicSimilarityIndex
import numpy as np

# Load the pre-built index
# track_ids should be loaded from embeddings/track_embeddings_ids.csv:
#   import pandas as pd
#   track_ids = pd.read_csv("embeddings/track_embeddings_ids.csv")["track_id"].tolist()
track_ids = [...]  # list of track IDs in index order
index = MusicSimilarityIndex.load(
    "embeddings/faiss_index.bin",
    track_ids,
    dim=128
)

# Query with a single embedding vector
query_embedding = np.load("my_track_embedding.npy")   # shape: (128,)
results = index.search(query_embedding, top_k=10)
# returns: [{"track_id": "...", "distance": 0.xx}, ...]

print(results)
```

### End-to-end Python inference

```python
import torch, yaml, json
from src.models.factory import build_model
from src.inference.predictor import AudioTagger
from src.inference.similarity import MusicSimilarityIndex
import pandas as pd

# Load config and model
with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

with open("embeddings/label_encoder.json") as f:
    label_enc = json.load(f)

cfg["model"]["num_classes"] = len(label_enc)
model = build_model(cfg["model"])
ckpt = torch.load("checkpoints/model_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])

# Build similarity index
track_ids = pd.read_csv("embeddings/track_embeddings_ids.csv")["track_id"].tolist()
sim_index = MusicSimilarityIndex.load(
    "embeddings/faiss_index.bin", track_ids, cfg["model"]["embedding_dim"]
)

# Create tagger
tagger = AudioTagger(
    model, label_enc, sim_index,
    extractor_config={
        "sample_rate": cfg["dataset"]["sample_rate"],
        "n_mels":      cfg["features"]["n_mels"],
        "n_fft":       cfg["features"]["n_fft"],
        "hop_length":  cfg["features"]["hop_length"],
        "duration":    cfg["dataset"]["duration"],
    }
)

# Predict
result = tagger.predict("my_song.mp3")
print("Tags:", result["tags"])
print("Similar tracks:", result["similar_tracks"])
```

---

## ⚙️ Configuration Reference

Both YAML configs (`configs/default.yaml` and `configs/cnn.yaml`) share the same schema:

```yaml
dataset:
  name: fma_small
  data_dir: data/fma_small          # path to raw audio
  metadata_dir: data/fma_metadata   # path to FMA CSV metadata
  sample_rate: 22050                # Hz — all audio resampled to this
  duration: 30                      # seconds per clip
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15

features:
  n_mels: 128           # mel frequency bins
  n_mfcc: 20            # MFCC coefficients
  n_fft: 2048           # FFT window size
  hop_length: 512       # frames between FFT windows
  fmax: 8000            # maximum frequency (Hz)
  feature_type: mel_spectrogram   # mel_spectrogram | mfcc | chroma | spectral_contrast

model:
  architecture: crnn    # crnn | cnn
  num_classes: 8        # set automatically from metadata
  dropout: 0.3
  hidden_size: 256      # GRU hidden units (CRNN only)
  embedding_dim: 128    # size of the audio fingerprint

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: cosine     # cosine | step
  patience: 10          # early stopping patience (epochs)
  checkpoint_dir: checkpoints
  log_dir: logs

evaluation:
  threshold: 0.5        # sigmoid threshold for multi-label prediction
  top_k: 5

inference:
  index_path: embeddings/faiss_index.bin
  embeddings_path: embeddings/track_embeddings.npy
  metadata_path: embeddings/track_metadata.csv
```

---

## 📊 Experiment Results

Results on **FMA-small** (8-genre classification, 30-second clips):

| Model | F1 (macro) | ROC-AUC | Parameters | Training time* |
|-------|:---------:|:-------:|:----------:|:--------------:|
| CNN baseline | ~0.72 | ~0.91 | 1.2 M | ~25 min |
| CRNN | ~0.76 | ~0.93 | 3.1 M | ~55 min |

*\*Approximate — single NVIDIA V100 GPU, 50 epochs.*

The CRNN outperforms the CNN baseline by ~4 pp F1 because the bidirectional GRU captures temporal evolution across the 30-second clip — information that is lost by the CNN's global average pooling.

---

## 🐳 Docker

### Build the image

```bash
docker build -t music-tagger .
```

### Run the server

```bash
docker run -p 5000:5000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/embeddings:/app/embeddings \
  music-tagger
```

The container:
- Installs `ffmpeg` and `libsndfile` at build time (required for audio decoding).
- Exposes port **5000**.
- Reads checkpoints and embeddings from the mounted host directories so you do not need to re-train inside the container.

### Run a one-off prediction

```bash
docker run --rm \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/embeddings:/app/embeddings \
  -v $(pwd)/data:/app/data \
  music-tagger \
  python scripts/evaluate.py \
    --checkpoint checkpoints/model_best.pt \
    --output results/metrics.json
```

---

## 🧪 Running Tests

The test suite covers feature extraction, model forward passes, dataloaders, metrics, and FAISS similarity:

```bash
# Run all tests with verbose output
pytest tests/ -v
```

```
tests/test_features.py::test_mel_spectrogram_shape    PASSED
tests/test_features.py::test_normalization             PASSED
tests/test_models.py::test_cnn_forward                 PASSED
tests/test_models.py::test_crnn_forward                PASSED
tests/test_models.py::test_embedding_shape             PASSED
tests/test_dataloader.py::test_split_sizes             PASSED
tests/test_metrics.py::test_f1_perfect                 PASSED
tests/test_metrics.py::test_roc_auc                    PASSED
tests/test_similarity.py::test_faiss_search            PASSED
```

Run a specific test file:

```bash
pytest tests/test_models.py -v
```

---

## 🔄 Project Workflow Diagram

```
1. Download FMA dataset
         │
         ▼
2. python scripts/preprocess.py     →  data/features/*.npy
         │
         ▼
3. python scripts/train.py          →  checkpoints/model_best.pt
         │
         ▼
4. python scripts/evaluate.py       →  results/metrics.json
         │
         ▼
5. python scripts/generate_embeddings.py  →  embeddings/faiss_index.bin
         │
         ▼
6. python scripts/demo.py           →  http://localhost:5000
         │
         ▼
7. curl -X POST -F "audio=@song.mp3" http://localhost:5000/predict
```
