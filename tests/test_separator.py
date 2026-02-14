"""Unit tests for the DSP source separator."""

import numpy as np

from apnea_screen.separator import SeparatedStreams, separate

SR = 16_000


def test_dsp_separation_returns_streams():
    """DSP fallback should always succeed and return the right shape."""
    audio = np.random.randn(SR * 10).astype(np.float32) * 0.1
    streams = separate(audio, SR, prefer_neural=False)
    assert isinstance(streams, SeparatedStreams)
    assert streams.sr == SR
    assert len(streams.breathing) == len(audio)
    assert len(streams.snoring) == len(audio)
    assert len(streams.gasps) == len(audio)


def test_separation_dtypes():
    audio = np.random.randn(SR * 5).astype(np.float32)
    streams = separate(audio, SR, prefer_neural=False)
    assert streams.breathing.dtype == np.float32
    assert streams.snoring.dtype == np.float32
    assert streams.gasps.dtype == np.float32


def test_dsp_backend_kwarg():
    """Using backend='dsp' should work identically to prefer_neural=False."""
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    streams = separate(audio, SR, backend="dsp")
    assert isinstance(streams, SeparatedStreams)
    assert streams.backend_used == "dsp"
    assert len(streams.breathing) == len(audio)


def test_dsp_produces_residual():
    """DSP backend should populate the residual field."""
    audio = np.random.randn(SR * 5).astype(np.float32) * 0.1
    streams = separate(audio, SR, backend="dsp")
    assert len(streams.residual) == len(audio)
    assert streams.residual.dtype == np.float32
