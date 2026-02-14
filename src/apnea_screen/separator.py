"""Audio source separation for breathing / snoring / gasp extraction.

Three backends are available, selected via the ``backend`` parameter:

``"sam_audio"``
    Meta's text-prompted audio separation (AudioSep).  Runs three prompted
    passes ("human breathing", "snoring", "gasping for air") to produce
    clinically-targeted stems.  Requires ``pip install -e ".[sam]"``.

``"openunmix"``
    Open-Unmix music separator repurposed via frequency-band heuristics.
    Requires ``pip install -e ".[neural]"``.

``"dsp"``
    Lightweight bandpass-filter + energy-gating pipeline.  No extra deps.

``"auto"`` (default)
    Tries SAM Audio → Open-Unmix → DSP, using the first that succeeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.signal as sig

from .audio_loader import TARGET_SR

log = logging.getLogger(__name__)

Backend = Literal["auto", "sam_audio", "openunmix", "dsp"]

# ---------------------------------------------------------------------------
# Frequency-band definitions (Hz)
# ---------------------------------------------------------------------------
BREATHING_LOW = 100
BREATHING_HIGH = 1500
SNORING_LOW = 80
SNORING_HIGH = 400
GASP_LOW = 500
GASP_HIGH = 4000

# SAM Audio text prompts
SAM_PROMPT_BREATHING = "human breathing"
SAM_PROMPT_SNORING = "snoring"
SAM_PROMPT_GASP = "gasping for air"


@dataclass
class SeparatedStreams:
    """Container for the separated audio streams."""

    breathing: np.ndarray
    snoring: np.ndarray
    gasps: np.ndarray
    sr: int
    residual: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    backend_used: str = "dsp"


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------
def _bandpass(audio: np.ndarray, sr: int, low: float, high: float, order: int = 5) -> np.ndarray:
    sos = sig.butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sig.sosfiltfilt(sos, audio).astype(np.float32)


def _envelope(audio: np.ndarray, sr: int, cutoff: float = 2.0) -> np.ndarray:
    """Amplitude envelope via full-wave rectification + low-pass."""
    rect = np.abs(audio)
    sos = sig.butter(2, cutoff, btype="low", fs=sr, output="sos")
    return sig.sosfiltfilt(sos, rect).astype(np.float32)


# ---------------------------------------------------------------------------
# Backend: SAM Audio (text-prompted separation)
# ---------------------------------------------------------------------------
def _try_sam_separation(audio: np.ndarray, sr: int) -> SeparatedStreams | None:
    """Attempt separation via AudioSep (text-prompted sound separation).

    AudioSep API per prompt: ``model.separate(audio_tensor, prompt)``
    returns ``(target, residual)`` tensors.

    We run three prompted passes and use the breathing-pass residual as the
    overall residual stream.
    """
    try:
        import torch
        import torchaudio  # noqa: F401
        from audiosep import AudioSep  # type: ignore[import-untyped]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("SAM Audio: using device=%s", device)

        model = AudioSep.from_pretrained("audiosep-base", device=device)

        # AudioSep expects (batch, channels, samples) at its native rate
        # (typically 16 kHz or 32 kHz).  Resample if needed.
        model_sr: int = getattr(model, "sample_rate", 16000)
        tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float().to(device)

        if sr != model_sr:
            resampler = torchaudio.transforms.Resample(sr, model_sr).to(device)
            tensor = resampler(tensor)

        def _sep(prompt: str) -> tuple[np.ndarray, np.ndarray]:
            target, residual = model.separate(tensor, prompt)
            # Squeeze to 1-D
            t_np = target.squeeze().cpu().numpy().astype(np.float32)
            r_np = residual.squeeze().cpu().numpy().astype(np.float32)
            # Resample back if needed
            if sr != model_sr:
                down = torchaudio.transforms.Resample(model_sr, sr).to(device)
                t_np = down(torch.from_numpy(t_np).unsqueeze(0).to(device)).squeeze().cpu().numpy()
                r_np = down(torch.from_numpy(r_np).unsqueeze(0).to(device)).squeeze().cpu().numpy()
            return t_np.astype(np.float32), r_np.astype(np.float32)

        breathing, breath_residual = _sep(SAM_PROMPT_BREATHING)
        snoring, _ = _sep(SAM_PROMPT_SNORING)
        gasps, _ = _sep(SAM_PROMPT_GASP)

        # Use the breathing-pass residual as the overall residual — it
        # captures everything *except* breathing, which is the most useful
        # residual for clinical review.
        residual = breath_residual

        # Ensure all arrays are the same length as the input
        n = len(audio)
        breathing = _pad_or_trim(breathing, n)
        snoring = _pad_or_trim(snoring, n)
        gasps = _pad_or_trim(gasps, n)
        residual = _pad_or_trim(residual, n)

        log.info("SAM Audio separation complete")
        return SeparatedStreams(
            breathing=breathing,
            snoring=snoring,
            gasps=gasps,
            sr=sr,
            residual=residual,
            backend_used="sam_audio",
        )

    except ImportError:
        log.debug("SAM Audio (audiosep) not installed")
        return None
    except Exception:
        log.warning("SAM Audio separation failed, falling back", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Backend: Open-Unmix
# ---------------------------------------------------------------------------
def _try_umx_separation(audio: np.ndarray, sr: int) -> SeparatedStreams | None:
    """Attempt GPU/CPU neural separation via Open-Unmix."""
    try:
        import torch
        import torchaudio  # noqa: F401
        from openunmix import predict

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Open-Unmix: using device=%s", device)

        tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float()  # (1,1,T)

        if sr != 44100:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100).to(device)
            tensor = resampler(tensor.to(device))
        else:
            tensor = tensor.to(device)

        estimates = predict.separate(tensor, rate=44100, device=device)

        if sr != 44100:
            down = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sr).to(device)

        def _to_np(key: str) -> np.ndarray:
            t = estimates[key].squeeze()
            if t.ndim > 1:
                t = t.mean(dim=0)
            if sr != 44100:
                t = down(t.unsqueeze(0)).squeeze()
            return t.cpu().numpy().astype(np.float32)

        vocals = _to_np("vocals")
        other = _to_np("other")

        breathing = _bandpass(other, sr, BREATHING_LOW, BREATHING_HIGH)
        snoring = _bandpass(vocals, sr, SNORING_LOW, SNORING_HIGH)

        n = len(audio)
        residual = audio[:n] - _pad_or_trim(breathing, n) - _pad_or_trim(snoring, n)
        gasps = _bandpass(residual, sr, GASP_LOW, GASP_HIGH)

        breathing = _pad_or_trim(breathing, n)
        snoring = _pad_or_trim(snoring, n)
        gasps = _pad_or_trim(gasps, n)
        residual = _pad_or_trim(residual, n)

        log.info("Open-Unmix separation complete")
        return SeparatedStreams(
            breathing=breathing,
            snoring=snoring,
            gasps=gasps,
            sr=sr,
            residual=residual,
            backend_used="openunmix",
        )

    except ImportError:
        log.debug("Open-Unmix not installed")
        return None
    except Exception:
        log.warning("Open-Unmix separation failed, falling back", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Backend: DSP fallback
# ---------------------------------------------------------------------------
def _dsp_separation(audio: np.ndarray, sr: int) -> SeparatedStreams:
    """Pure DSP fallback: bandpass filters + energy gating."""
    breathing = _bandpass(audio, sr, BREATHING_LOW, BREATHING_HIGH)
    snoring = _bandpass(audio, sr, SNORING_LOW, SNORING_HIGH)
    gasps = _bandpass(audio, sr, GASP_LOW, GASP_HIGH)

    # Gate gasps to transient-only regions
    gasp_env = _envelope(gasps, sr, cutoff=5.0)
    med = np.median(gasp_env)
    std = np.std(gasp_env) + 1e-10
    gasp_mask = (gasp_env > med + 2.5 * std).astype(np.float32)
    gasps = gasps * gasp_mask

    residual = audio - breathing - snoring - gasps

    return SeparatedStreams(
        breathing=breathing,
        snoring=snoring,
        gasps=gasps,
        sr=sr,
        residual=residual,
        backend_used="dsp",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pad_or_trim(arr: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros or trim *arr* to exactly *n* samples."""
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)), mode="constant").astype(arr.dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_BACKEND_CHAIN = {
    "auto": ["sam_audio", "openunmix", "dsp"],
    "sam_audio": ["sam_audio"],
    "openunmix": ["openunmix"],
    "dsp": ["dsp"],
}

_BACKEND_FN = {
    "sam_audio": _try_sam_separation,
    "openunmix": _try_umx_separation,
}


def separate(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    *,
    backend: Backend = "auto",
    # Legacy parameter — mapped to backend for backward compat
    prefer_neural: bool | None = None,
) -> SeparatedStreams:
    """Separate an audio recording into breathing, snoring, and gasp streams.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 waveform.
    sr : int
        Sample rate.
    backend : ``"auto"`` | ``"sam_audio"`` | ``"openunmix"`` | ``"dsp"``
        Which separation backend to use.
    prefer_neural : bool | None
        **Deprecated.** If given, ``True`` maps to ``backend="auto"`` and
        ``False`` maps to ``backend="dsp"``.
    """
    # Legacy compat
    if prefer_neural is not None:
        backend = "auto" if prefer_neural else "dsp"

    chain = _BACKEND_CHAIN.get(backend, ["dsp"])

    for name in chain:
        if name == "dsp":
            return _dsp_separation(audio, sr)
        fn = _BACKEND_FN[name]
        result = fn(audio, sr)
        if result is not None:
            return result

    # Should never reach here, but just in case
    return _dsp_separation(audio, sr)
