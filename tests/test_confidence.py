"""Unit tests for event confidence scoring."""

import numpy as np
import pytest

from apnea_screen.detector import (
    Event,
    _score_depth,
    _score_duration,
    compute_confidence,
    compute_envelope,
    compute_baseline,
    run_detection,
)

SR = 16_000
ENV_SR = 10  # hop_s=0.1 default


# ---------------------------------------------------------------------------
# _score_depth
# ---------------------------------------------------------------------------
def test_depth_apnea_full_drop():
    """Complete cessation (ratio=0) → depth score = 1.0."""
    assert _score_depth(0.0, "apnea") == 1.0


def test_depth_apnea_threshold():
    """ratio=0.10 (edge of apnea) → score around 0.3."""
    score = _score_depth(0.10, "apnea")
    assert 0.2 <= score <= 0.4


def test_depth_hypopnea_deep():
    """Deep hypopnea (ratio=0.2) → high score."""
    score = _score_depth(0.2, "hypopnea")
    assert score >= 0.7


def test_depth_hypopnea_shallow():
    """Shallow hypopnea (ratio=0.65) → low score."""
    score = _score_depth(0.65, "hypopnea")
    assert score <= 0.5


# ---------------------------------------------------------------------------
# _score_duration
# ---------------------------------------------------------------------------
def test_duration_minimum():
    """10 s event → floor score (0.3)."""
    assert _score_duration(10.0) == pytest.approx(0.3)


def test_duration_maximum():
    """30+ s event → score = 1.0."""
    assert _score_duration(30.0) == 1.0
    assert _score_duration(60.0) == 1.0


def test_duration_midpoint():
    """20 s → halfway between 0.3 and 1.0 → 0.65."""
    assert _score_duration(20.0) == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------
def test_confidence_apnea_no_gasp():
    """Apnea with full drop and 15s duration, no gasp data."""
    n = 1200  # 120 s at 10 Hz
    ratio = np.ones(n, dtype=np.float32)
    # Zero out 50-65 s
    ratio[500:650] = 0.0

    ev = Event("apnea", 50.0, 65.0)
    conf = compute_confidence(ev, ratio, ENV_SR, gasp_envelope=None)

    # depth=1.0 (ratio=0), dur=_score_duration(15)≈0.475
    # conf = 0.5*1.0 + 0.3*0.475 + 0.2*0 = 0.6425
    assert 0.5 <= conf <= 0.8
    assert isinstance(conf, float)


def test_confidence_with_gasp_boost():
    """Gasp burst after event should increase confidence."""
    n = 1200
    ratio = np.ones(n, dtype=np.float32)
    ratio[500:650] = 0.0

    gasp_env = np.zeros(n, dtype=np.float32)
    # Insert a gasp burst right after the event ends (at frame 650-660)
    gasp_env[650:660] = 1.0  # big spike vs median=0

    ev = Event("apnea", 50.0, 65.0)
    conf_with = compute_confidence(ev, ratio, ENV_SR, gasp_envelope=gasp_env)
    conf_without = compute_confidence(ev, ratio, ENV_SR, gasp_envelope=None)

    assert conf_with > conf_without


def test_confidence_bounded():
    """Confidence should always be in [0, 1]."""
    ev = Event("apnea", 0.0, 60.0)
    ratio = np.zeros(6000, dtype=np.float32)
    gasp = np.ones(6000, dtype=np.float32) * 10.0
    conf = compute_confidence(ev, ratio, ENV_SR, gasp_envelope=gasp)
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Integration: run_detection returns events with confidence
# ---------------------------------------------------------------------------
def test_run_detection_events_have_confidence():
    """Events from run_detection should have confidence > 0."""
    t = np.arange(int(120.0 * SR)) / SR
    audio = (0.5 * np.sin(2 * np.pi * 0.25 * t)).astype(np.float32)
    audio[50 * SR : 65 * SR] = 0.0

    result = run_detection(audio, SR)
    assert len(result.events) >= 1
    for ev in result.events:
        assert ev.confidence > 0.0
        assert ev.confidence <= 1.0
