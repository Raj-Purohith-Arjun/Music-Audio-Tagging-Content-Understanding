# Music Audio Tagging & Content Understanding

A production-style deep learning system for predicting music attributes (genre, mood, instrumentation) from raw audio, with embedding-based recommendation.

## Architecture

```
project/
├── src/
│   ├── data/          # Dataset loaders, dataloaders
│   ├── features/      # Librosa feature extraction pipeline
│   ├── models/        # CNN, CRNN architectures
│   ├── training/      # Training loop with checkpointing
│   ├── evaluation/    # Metrics: F1, ROC-AUC, precision/recall
│   └── inference/     # Embedding extraction, FAISS similarity
├── scripts/
│   ├── preprocess.py  # Convert audio → feature tensors
│   ├── train.py       # Model training
│   ├── evaluate.py    # Test-set evaluation
│   ├── generate_embeddings.py  # Build FAISS index
│   └── demo.py        # Flask inference server
├── configs/           # YAML experiment configs
├── notebooks/         # Exploration and visualization
├── tests/             # Unit tests
├── Dockerfile
└── requirements.txt
```

## Dataset

This project is designed for the [FMA (Free Music Archive)](https://github.com/mdeff/fma) dataset. The FMA dataset provides:
- **fma_small**: 8,000 tracks × 8 balanced genres (30s clips, ~7.2 GB)
- **fma_metadata**: CSV files with genre, mood, and track metadata

FMA is ideal for music tagging because it is publicly licensed, balanced across genres, and has been used as a benchmark in numerous music information retrieval studies.

To download:
```bash
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
```

## Feature Extraction

Four feature types are extracted via `librosa`:

| Feature | Dimensions | Purpose |
|---------|-----------|---------|
| Mel Spectrogram | 128 × T | Primary frequency representation |
| MFCC | 20 × T | Timbral characteristics |
| Spectral Contrast | 7 × T | Spectral peak/valley ratio |
| Chroma | 12 × T | Harmonic and pitch content |

All features are zero-mean unit-variance normalized and padded/trimmed to a fixed time dimension.

## Models

### CNN Baseline
4-block ConvNet on mel-spectrograms. Each block: Conv2d → BatchNorm → ReLU → MaxPool. Global average pooling → linear embedding → classifier.

### CRNN
CNN frontend compresses frequency dimension while preserving time. Bidirectional GRU captures long-range temporal patterns. Final hidden state → linear embedding → classifier.

## Experiments

| Model | F1 (macro) | ROC-AUC | Params |
|-------|-----------|---------|--------|
| CNN | ~0.72 | ~0.91 | 1.2M |
| CRNN | ~0.76 | ~0.93 | 3.1M |

*(Results on FMA-small 8-genre classification, 30-second clips)*

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess audio
```bash
python scripts/preprocess.py --data-dir data/fma_small --output-dir data/features
```

### 3. Train
```bash
python scripts/train.py --config configs/default.yaml
```

### 4. Evaluate
```bash
python scripts/evaluate.py --checkpoint checkpoints/model_best.pt --output results/metrics.json
```

### 5. Generate embeddings & FAISS index
```bash
python scripts/generate_embeddings.py --checkpoint checkpoints/model_best.pt
```

### 6. Run demo server
```bash
python scripts/demo.py
# POST /predict with audio file
curl -X POST -F "audio=@song.mp3" http://localhost:5000/predict
```

## Embeddings for Recommendation

The model's penultimate layer produces a dense 128-dimensional embedding per track. These embeddings capture semantic audio features (genre, mood, instrumentation) and can be used for:

1. **Candidate generation**: Retrieve top-K similar tracks using FAISS L2 search
2. **Cold-start**: Tag new tracks with predicted genres and find neighbors
3. **Downstream models**: Feed embeddings as features into collaborative filtering

```python
from src.inference.similarity import MusicSimilarityIndex

index = MusicSimilarityIndex.load("embeddings/faiss_index.bin", track_ids, dim=128)
similar = index.search(query_embedding, top_k=10)
```

## Docker

```bash
docker build -t music-tagger .
docker run -p 5000:5000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/embeddings:/app/embeddings \
  music-tagger
```

## Tests

```bash
pytest tests/ -v
```
