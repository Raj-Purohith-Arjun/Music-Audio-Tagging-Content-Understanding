import numpy as np
import librosa
import torch
from typing import Optional


class AudioFeatureExtractor:
    """Modular audio feature extraction pipeline."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_mfcc: int = 20,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmax: Optional[float] = 8000,
        duration: Optional[float] = 30.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmax = fmax
        self.duration = duration
        self.max_frames = int(duration * sample_rate / hop_length) + 1 if duration else None

    def load_audio(self, path: str) -> np.ndarray:
        """Load and resample audio file."""
        y, _ = librosa.load(path, sr=self.sample_rate, duration=self.duration, mono=True)
        # pad if shorter than expected
        if self.duration:
            expected = int(self.duration * self.sample_rate)
            if len(y) < expected:
                y = np.pad(y, (0, expected - len(y)))
        return y

    def mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Log-mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmax=self.fmax,
        )
        return librosa.power_to_db(mel, ref=np.max)

    def mfcc(self, y: np.ndarray) -> np.ndarray:
        """MFCC features."""
        return librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def spectral_contrast(self, y: np.ndarray) -> np.ndarray:
        """Spectral contrast across 7 bands."""
        return librosa.feature.spectral_contrast(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def chroma(self, y: np.ndarray) -> np.ndarray:
        """Chromagram features."""
        return librosa.feature.chroma_stft(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def extract(self, path: str, feature_type: str = "mel_spectrogram") -> np.ndarray:
        """Extract named feature from audio file."""
        y = self.load_audio(path)

        extractors = {
            "mel_spectrogram": self.mel_spectrogram,
            "mfcc": self.mfcc,
            "spectral_contrast": self.spectral_contrast,
            "chroma": self.chroma,
        }

        if feature_type not in extractors:
            raise ValueError(f"Unknown feature type: {feature_type}")

        feat = extractors[feature_type](y)
        feat = self._normalize(feat)
        feat = self._pad_or_trim(feat)
        return feat  # shape: (freq_bins, time_frames)

    def extract_all(self, path: str) -> np.ndarray:
        """Stack all features into single array."""
        y = self.load_audio(path)
        mel = self._normalize(self._pad_or_trim(self.mel_spectrogram(y)))
        mfcc = self._normalize(self._pad_or_trim(self.mfcc(y)))
        sc = self._normalize(self._pad_or_trim(self.spectral_contrast(y)))
        chroma = self._normalize(self._pad_or_trim(self.chroma(y)))
        return np.concatenate([mel, mfcc, sc, chroma], axis=0)

    def _normalize(self, feat: np.ndarray) -> np.ndarray:
        """Zero-mean unit-variance normalization."""
        mean = feat.mean()
        std = feat.std() + 1e-8
        return (feat - mean) / std

    def _pad_or_trim(self, feat: np.ndarray) -> np.ndarray:
        """Ensure fixed time dimension."""
        if self.max_frames is None:
            return feat
        t = feat.shape[1]
        if t < self.max_frames:
            feat = np.pad(feat, ((0, 0), (0, self.max_frames - t)))
        else:
            feat = feat[:, : self.max_frames]
        return feat

    def to_tensor(self, feat: np.ndarray) -> torch.Tensor:
        """Convert numpy array to 3D tensor (1, freq, time)."""
        return torch.from_numpy(feat).float().unsqueeze(0)
