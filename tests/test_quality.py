"""Unit tests for recording quality checks."""

import numpy as np
import pytest

from apnea_screen.quality import (
    QualityResult,
    assess_quality,
    compute_clipping_ratio,
    compute_residual_energy_ratio,
)

SR = 16_000


def test_clipping_ratio_clean():
    """A sine wave at 0.5 amplitude has zero clipping."""
    t = np.arange(SR * 2) / SR
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert compute_clipping_ratio(audio) == 0.0


def test_clipping_ratio_clipped():
    """A hard-clipped signal should report a high ratio."""
    audio = np.ones(SR, dtype=np.float32)  # all samples at 1.0
    ratio = compute_clipping_ratio(audio)
    assert ratio == 1.0


def test_clipping_ratio_partial():
    """50% of samples at full-scale → ratio ~ 0.5."""
    audio = np.zeros(1000, dtype=np.float32)
    audio[:500] = 1.0
    ratio = compute_clipping_ratio(audio)
    assert ratio == pytest.approx(0.5, abs=0.01)


def test_clipping_ratio_empty():
    assert compute_clipping_ratio(np.array([], dtype=np.float32)) == 0.0


def test_residual_energy_ratio_quiet_residual():
    """Low residual relative to breathing → ratio near 0."""
    breathing = np.random.randn(SR).astype(np.float32) * 0.5
    residual = np.random.randn(SR).astype(np.float32) * 0.01
    ratio = compute_residual_energy_ratio(breathing, residual)
    assert ratio < 0.1


def test_residual_energy_ratio_loud_residual():
    """Residual louder than breathing → ratio > 1."""
    breathing = np.random.randn(SR).astype(np.float32) * 0.1
    residual = np.random.randn(SR).astype(np.float32) * 0.5
    ratio = compute_residual_energy_ratio(breathing, residual)
    assert ratio > 1.0


def test_assess_quality_ok():
    """Clean signal → OK flag."""
    t = np.arange(SR * 2) / SR
    audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    breathing = audio.copy()
    residual = np.zeros_like(audio)
    result = assess_quality(audio, breathing, residual)
    assert isinstance(result, QualityResult)
    assert result.flag == "OK"


def test_assess_quality_clipped():
    """Clipped signal → WARN_CLIPPED."""
    audio = np.ones(SR * 2, dtype=np.float32)  # everything at 1.0
    breathing = audio * 0.5
    residual = np.zeros_like(audio)
    result = assess_quality(audio, breathing, residual)
    assert result.flag == "WARN_CLIPPED"


def test_assess_quality_noisy():
    """Residual much louder than breathing → WARN_NOISY."""
    t = np.arange(SR * 2) / SR
    audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    breathing = audio * 0.05
    residual = np.random.randn(len(audio)).astype(np.float32) * 0.5
    result = assess_quality(audio, breathing, residual)
    assert result.flag == "WARN_NOISY"
