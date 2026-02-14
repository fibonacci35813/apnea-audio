"""End-to-end pipeline: load -> separate -> quality -> detect -> report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .audio_loader import load_audio
from .detector import DetectionResult, run_detection, compute_envelope
from .quality import QualityResult, assess_quality
from .report import build_summary, create_timeline, save_summary, save_timeline
from .separator import Backend, SeparatedStreams, separate


@dataclass
class PipelineResult:
    """Everything produced by a single run."""

    streams: SeparatedStreams
    detection: DetectionResult
    quality: QualityResult
    summary: dict
    figure: plt.Figure

    # Envelopes for snoring/gasp panels (same rate as detection envelope)
    snoring_env: np.ndarray
    gasp_env: np.ndarray


def _save_stems(streams: SeparatedStreams, output_dir: Path) -> None:
    """Write separated stems as WAV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [
        ("breathing", streams.breathing),
        ("snoring", streams.snoring),
        ("gasp", streams.gasps),
        ("residual", streams.residual),
    ]:
        if len(data) > 0:
            sf.write(str(output_dir / f"{name}.wav"), data, streams.sr, subtype="FLOAT")


def run_pipeline(
    audio_path: str | Path,
    *,
    backend: Backend = "auto",
    output_dir: str | Path | None = None,
    # Legacy parameter — mapped to backend for backward compat
    prefer_neural: bool | None = None,
) -> PipelineResult:
    """Run the full screening pipeline.

    Parameters
    ----------
    audio_path : str | Path
        Path to the input audio file.
    backend : ``"auto"`` | ``"sam_audio"`` | ``"openunmix"`` | ``"dsp"``
        Which separation backend to use.
    output_dir : str | Path | None
        If given, save timeline plot, JSON summary, and separated stems.
    prefer_neural : bool | None
        **Deprecated.** Kept for backward compatibility.
    """
    audio, sr = load_audio(audio_path)

    # --- Separation ---
    streams = separate(audio, sr, backend=backend, prefer_neural=prefer_neural)

    # --- Quality check ---
    residual = streams.residual if len(streams.residual) > 0 else np.zeros_like(audio)
    quality = assess_quality(audio, streams.breathing, residual)

    # --- Build gasp envelope for confidence boosting ---
    gasp_env, _ = compute_envelope(streams.gasps, sr)

    # --- Detection (with gasp boost) ---
    detection = run_detection(streams.breathing, sr, gasp_envelope=gasp_env)

    # --- Build snoring envelope for the timeline ---
    snoring_env, _ = compute_envelope(streams.snoring, sr)

    # Trim envelopes to same length
    min_len = min(len(detection.envelope), len(snoring_env), len(gasp_env))
    snoring_env = snoring_env[:min_len]
    gasp_env = gasp_env[:min_len]

    # --- Report ---
    summary = build_summary(
        detection,
        filename=Path(audio_path).name,
        quality=quality,
        backend_used=streams.backend_used,
    )
    fig = create_timeline(detection, snoring_env=snoring_env, gasp_env=gasp_env)

    if output_dir is not None:
        output_dir = Path(output_dir)
        save_timeline(fig, output_dir / "timeline.png")
        save_summary(summary, output_dir / "summary.json")
        _save_stems(streams, output_dir)

    return PipelineResult(
        streams=streams,
        detection=detection,
        quality=quality,
        summary=summary,
        figure=fig,
        snoring_env=snoring_env,
        gasp_env=gasp_env,
    )
