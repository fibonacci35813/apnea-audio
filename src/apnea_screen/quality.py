"""Recording quality checks for clinical robustness.

Computes two metrics from the raw audio and separated streams:

- **clipping_ratio**: fraction of samples within 1% of full-scale (|x| > 0.99).
  High values indicate the microphone was overdriven.
- **residual_energy_ratio**: ratio of residual energy to breathing-band energy.
  A high ratio means the recording is dominated by non-respiratory sounds
  (background noise, TV, fan, etc.).

Returns a ``QualityResult`` with per-metric values and an overall flag:
  ``OK`` / ``WARN_CLIPPED`` / ``WARN_NOISY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

QualityFlag = Literal["OK", "WARN_CLIPPED", "WARN_NOISY"]

# Thresholds (empirically reasonable defaults)
CLIPPING_THRESHOLD = 0.99  # |sample| above this counts as clipped
CLIPPING_WARN = 0.01       # > 1 % clipped → WARN_CLIPPED
NOISE_WARN = 2.0           # residual energy > 2× breathing energy → WARN_NOISY


@dataclass
class QualityResult:
    """Recording quality assessment."""

    clipping_ratio: float
    residual_energy_ratio: float
    flag: QualityFlag


def compute_clipping_ratio(audio: np.ndarray, threshold: float = CLIPPING_THRESHOLD) -> float:
    """Fraction of samples near full-scale."""
    if len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(audio) > threshold))


def compute_residual_energy_ratio(
    breathing: np.ndarray, residual: np.ndarray
) -> float:
    """Ratio of residual RMS energy to breathing RMS energy."""
    breath_rms = float(np.sqrt(np.mean(breathing**2))) + 1e-12
    resid_rms = float(np.sqrt(np.mean(residual**2)))
    return resid_rms / breath_rms


def assess_quality(
    audio: np.ndarray,
    breathing: np.ndarray,
    residual: np.ndarray,
) -> QualityResult:
    """Run all quality checks and return a ``QualityResult``.

    Parameters
    ----------
    audio : np.ndarray
        Raw mono waveform (peak-normalised).
    breathing : np.ndarray
        Separated breathing stream.
    residual : np.ndarray
        Separation residual.
    """
    clip = compute_clipping_ratio(audio)
    noise = compute_residual_energy_ratio(breathing, residual)

    # Clipping is the more actionable warning, so check it first
    if clip > CLIPPING_WARN:
        flag: QualityFlag = "WARN_CLIPPED"
    elif noise > NOISE_WARN:
        flag = "WARN_NOISY"
    else:
        flag = "OK"

    return QualityResult(
        clipping_ratio=round(clip, 4),
        residual_energy_ratio=round(noise, 2),
        flag=flag,
    )
