"""Tests for SAM Audio separation backend.

These tests skip gracefully if audiosep / torch are not installed,
so they will not break CI in environments without GPU or SAM deps.
"""

import numpy as np
import pytest

from apnea_screen.separator import SeparatedStreams, separate

SR = 16_000

# Check whether the SAM Audio stack is available
try:
    import torch  # noqa: F401
    import torchaudio  # noqa: F401
    from audiosep import AudioSep  # type: ignore[import-untyped]  # noqa: F401

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

requires_sam = pytest.mark.skipif(not SAM_AVAILABLE, reason="audiosep not installed")


@requires_sam
def test_sam_separation_returns_streams():
    """SAM Audio backend should return SeparatedStreams with correct shape."""
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.3
    streams = separate(audio, SR, backend="sam_audio")
    assert isinstance(streams, SeparatedStreams)
    assert streams.backend_used == "sam_audio"
    assert streams.sr == SR
    assert len(streams.breathing) == len(audio)
    assert len(streams.snoring) == len(audio)
    assert len(streams.gasps) == len(audio)
    assert len(streams.residual) == len(audio)


@requires_sam
def test_sam_separation_dtypes():
    audio = np.random.randn(SR * 3).astype(np.float32)
    streams = separate(audio, SR, backend="sam_audio")
    assert streams.breathing.dtype == np.float32
    assert streams.snoring.dtype == np.float32
    assert streams.gasps.dtype == np.float32
    assert streams.residual.dtype == np.float32


@requires_sam
def test_sam_residual_is_populated():
    """Residual from SAM should have non-zero energy."""
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.3
    streams = separate(audio, SR, backend="sam_audio")
    assert np.any(streams.residual != 0)


# ---------------------------------------------------------------------------
# Tests that always run (no SAM dependency)
# ---------------------------------------------------------------------------
def test_auto_falls_back_without_sam():
    """When SAM is not installed, auto backend should still produce results."""
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.3
    streams = separate(audio, SR, backend="auto")
    assert isinstance(streams, SeparatedStreams)
    # It should have fallen back to dsp (or openunmix if installed)
    assert streams.backend_used in ("sam_audio", "openunmix", "dsp")
    assert len(streams.breathing) == len(audio)


def test_explicit_sam_backend_falls_back_gracefully():
    """Requesting sam_audio when not installed should not crash —
    it returns None internally and the chain ends, but since sam_audio
    is the only item in the chain, we need to verify behavior."""
    if SAM_AVAILABLE:
        pytest.skip("SAM is installed; can't test fallback")
    # When sam_audio is requested directly but not installed, the chain
    # only has ["sam_audio"], and _try_sam returns None, so the loop ends
    # and hits the final _dsp_separation fallback.
    audio = np.random.randn(SR * 3).astype(np.float32)
    streams = separate(audio, SR, backend="sam_audio")
    assert isinstance(streams, SeparatedStreams)
    assert streams.backend_used == "dsp"


def test_backend_dsp_has_residual():
    """DSP backend should now populate the residual field."""
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.3
    streams = separate(audio, SR, backend="dsp")
    assert streams.backend_used == "dsp"
    assert len(streams.residual) == len(audio)


def test_legacy_prefer_neural_false():
    """prefer_neural=False should map to DSP backend."""
    audio = np.random.randn(SR * 3).astype(np.float32)
    streams = separate(audio, SR, prefer_neural=False)
    assert streams.backend_used == "dsp"


def test_legacy_prefer_neural_true():
    """prefer_neural=True should map to auto backend (and fall through to dsp)."""
    audio = np.random.randn(SR * 3).astype(np.float32)
    streams = separate(audio, SR, prefer_neural=True)
    assert isinstance(streams, SeparatedStreams)
