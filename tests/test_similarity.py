import numpy as np
import pytest
from src.inference.similarity import MusicSimilarityIndex


def make_embeddings(n=50, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, dim)).astype(np.float32)


def test_build_and_search():
    embs = make_embeddings()
    track_ids = list(range(50))
    idx = MusicSimilarityIndex(embedding_dim=64)
    idx.build(embs, track_ids)

    results = idx.search(embs[0], top_k=5)
    assert len(results) == 5
    assert "track_id" in results[0]
    assert "distance" in results[0]


def test_top_k_bound():
    embs = make_embeddings(n=10)
    idx = MusicSimilarityIndex(embedding_dim=64)
    idx.build(embs, list(range(10)))
    results = idx.search(embs[0], top_k=100)
    assert len(results) == 10


def test_save_load(tmp_path):
    embs = make_embeddings()
    track_ids = list(range(50))
    idx = MusicSimilarityIndex(embedding_dim=64)
    idx.build(embs, track_ids)

    index_path = str(tmp_path / "test.bin")
    idx.save(index_path)

    loaded = MusicSimilarityIndex.load(index_path, track_ids, embedding_dim=64)
    results = loaded.search(embs[0], top_k=3)
    assert len(results) == 3
