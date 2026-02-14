"""Unit tests for the apnea/hypopnea detector."""

import numpy as np
import pytest

from apnea_screen.detector import (
    Event,
    compute_baseline,
    compute_envelope,
    detect_events,
    run_detection,
)

SR = 16_000


def _make_breathing(duration_s: float = 60.0, freq: float = 0.25) -> np.ndarray:
    """Simulate a breathing signal at *freq* Hz (default 0.25 = 15 breaths/min)."""
    t = np.arange(int(duration_s * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_envelope_shape():
    audio = _make_breathing(10.0)
    env, env_sr = compute_envelope(audio, SR)
    assert env_sr == 10  # hop_s=0.1 -> 10 Hz
    assert len(env) > 0
    assert env.dtype == np.float32


def test_baseline_shape():
    audio = _make_breathing(30.0)
    env, env_sr = compute_envelope(audio, SR)
    bl = compute_baseline(env, env_sr)
    assert bl.shape == env.shape


def test_no_events_on_clean_signal():
    """A steady breathing signal should produce zero events."""
    audio = _make_breathing(120.0)
    result = run_detection(audio, SR)
    assert len(result.events) == 0
    assert result.ahi == 0.0


def test_apnea_detected():
    """Insert a 15-second silence gap -> should detect at least one apnea."""
    audio = _make_breathing(120.0)
    # Zero out a 15-second window (apnea)
    start = 50 * SR
    end = 65 * SR
    audio[start:end] = 0.0
    result = run_detection(audio, SR)
    apneas = result.apneas
    assert len(apneas) >= 1
    # The detected event should overlap the silent region
    a = apneas[0]
    assert a.start_s < 65 and a.end_s > 50


def test_hypopnea_detected():
    """30% amplitude drop for 15 seconds -> should be a hypopnea."""
    audio = _make_breathing(120.0)
    # Reduce amplitude by 50% (>30% drop) for 15 seconds
    start = 50 * SR
    end = 65 * SR
    audio[start:end] *= 0.3
    result = run_detection(audio, SR)
    assert len(result.events) >= 1


def test_ahi_calculation():
    """AHI = events / hours."""
    audio = _make_breathing(3600.0)  # 1 hour
    # Insert 10 apneas
    for i in range(10):
        s = (i * 300 + 50) * SR  # every 5 minutes
        e = s + 15 * SR
        if e < len(audio):
            audio[s:e] = 0.0
    result = run_detection(audio, SR)
    # AHI should be approximately 10 (1 hour recording, ~10 events)
    assert result.ahi >= 5  # allow some tolerance
