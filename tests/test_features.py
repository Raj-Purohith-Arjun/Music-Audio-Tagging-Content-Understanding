import numpy as np
import pytest
import tempfile
import soundfile as sf
from src.features.extractor import AudioFeatureExtractor


def make_sine_wav(path: str, duration: float = 5.0, sr: int = 22050):
    """Write a synthetic sine wave to disk."""
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(path, y, sr)


@pytest.fixture
def extractor():
    return AudioFeatureExtractor(sample_rate=22050, duration=5.0, n_mels=64)


@pytest.fixture
def sine_wav(tmp_path):
    p = tmp_path / "test.wav"
    make_sine_wav(str(p))
    return str(p)


def test_mel_spectrogram_shape(extractor, sine_wav):
    feat = extractor.extract(sine_wav, "mel_spectrogram")
    assert feat.ndim == 2
    assert feat.shape[0] == 64  # n_mels


def test_mfcc_shape(extractor, sine_wav):
    feat = extractor.extract(sine_wav, "mfcc")
    assert feat.ndim == 2
    assert feat.shape[0] == 20  # n_mfcc


def test_spectral_contrast_shape(extractor, sine_wav):
    feat = extractor.extract(sine_wav, "spectral_contrast")
    assert feat.ndim == 2
    assert feat.shape[0] == 7  # 7 bands


def test_chroma_shape(extractor, sine_wav):
    feat = extractor.extract(sine_wav, "chroma")
    assert feat.ndim == 2
    assert feat.shape[0] == 12


def test_normalization(extractor, sine_wav):
    feat = extractor.extract(sine_wav, "mel_spectrogram")
    assert abs(feat.mean()) < 1.0


def test_to_tensor(extractor, sine_wav):
    import torch
    feat = extractor.extract(sine_wav, "mel_spectrogram")
    tensor = extractor.to_tensor(feat)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 3  # (1, freq, time)


def test_unknown_feature_raises(extractor, sine_wav):
    with pytest.raises(ValueError):
        extractor.extract(sine_wav, "unknown_feature")
