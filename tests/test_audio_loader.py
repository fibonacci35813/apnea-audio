"""Unit tests for the audio loader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from apnea_screen.audio_loader import duration_seconds, load_audio


def _write_test_wav(path: str, sr: int = 44100, duration_s: float = 2.0) -> None:
    """Write a short sine-wave WAV for testing."""
    t = np.arange(int(sr * duration_s)) / sr
    data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, data, sr)


def test_load_wav():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_test_wav(f.name)
        audio, sr = load_audio(f.name)
    assert sr == 16_000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) > 0


def test_peak_normalised():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_test_wav(f.name)
        audio, _ = load_audio(f.name)
    assert np.max(np.abs(audio)) == pytest.approx(1.0, abs=0.01)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_audio("/nonexistent/audio.wav")


def test_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not audio")
    with pytest.raises(ValueError, match="Unsupported"):
        load_audio(f.name)


def test_duration():
    assert duration_seconds(np.zeros(16000, dtype=np.float32), 16000) == 1.0
