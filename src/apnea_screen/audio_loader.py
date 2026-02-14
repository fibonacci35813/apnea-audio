"""Load and normalise audio files (WAV / MP3) for downstream processing."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

TARGET_SR = 16_000  # 16 kHz – standard for speech/breath analysis


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an audio file, convert to mono float32, and resample.

    Parameters
    ----------
    path : str | Path
        Path to a WAV or MP3 file.
    target_sr : int
        Target sample rate in Hz.

    Returns
    -------
    audio : np.ndarray  (float32, shape ``(n_samples,)``)
        Mono waveform normalised to [-1, 1].
    sr : int
        Sample rate (== *target_sr*).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".ogg"}:
        raise ValueError(f"Unsupported audio format: {suffix}")

    # librosa handles WAV, MP3, FLAC, OGG transparently
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    audio = audio.astype(np.float32)

    # peak-normalise so downstream energy calculations are consistent
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio, sr


def duration_seconds(audio: np.ndarray, sr: int) -> float:
    """Return duration in seconds."""
    return len(audio) / sr
