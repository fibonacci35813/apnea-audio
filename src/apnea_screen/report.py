"""Generate timeline plots and JSON summary reports."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .detector import DetectionResult
from .quality import QualityResult


# ---------------------------------------------------------------------------
# Timeline plot
# ---------------------------------------------------------------------------
def create_timeline(
    result: DetectionResult,
    *,
    snoring_env: np.ndarray | None = None,
    gasp_env: np.ndarray | None = None,
    title: str = "Sleep-Apnea Screening \u2013 Timeline",
) -> plt.Figure:
    """Create a multi-panel timeline figure.

    Event shading alpha is scaled by confidence (higher confidence = more
    opaque), with a floor of 0.15 so low-confidence events are still visible.
    """
    env = result.envelope
    env_sr = result.envelope_sr
    t = np.arange(len(env)) / env_sr  # time in seconds

    # Convert to minutes for readability on long recordings
    t_min = t / 60.0
    x_label = "Time (min)"

    n_panels = 1 + (snoring_env is not None) + (gasp_env is not None)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3 * n_panels + 1), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: breathing envelope + events ---
    ax = axes[0]
    ax.plot(t_min, env, color="#2196F3", linewidth=0.5, label="Breathing envelope")
    ax.plot(t_min, result.baseline, color="gray", linewidth=0.8, linestyle="--", label="Baseline")

    for ev in result.events:
        base_color = "#F44336" if ev.kind == "apnea" else "#FF9800"
        alpha = 0.15 + 0.45 * ev.confidence  # range [0.15, 0.60]
        ax.axvspan(ev.start_s / 60, ev.end_s / 60, alpha=alpha, color=base_color)

    apnea_patch = mpatches.Patch(color="#F44336", alpha=0.4, label="Apnea")
    hypopnea_patch = mpatches.Patch(color="#FF9800", alpha=0.4, label="Hypopnea")
    ax.legend(handles=[apnea_patch, hypopnea_patch, *ax.get_legend_handles_labels()[0][:2]],
              loc="upper right", fontsize=8)
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=12)

    # --- Panel 2 (optional): snoring ---
    idx = 1
    if snoring_env is not None:
        ax2 = axes[idx]
        t2 = np.arange(len(snoring_env)) / env_sr / 60.0
        ax2.plot(t2, snoring_env, color="#9C27B0", linewidth=0.5)
        ax2.set_ylabel("Snoring")
        ax2.set_title("Snoring activity", fontsize=10)
        idx += 1

    # --- Panel 3 (optional): gasps ---
    if gasp_env is not None:
        ax3 = axes[idx]
        t3 = np.arange(len(gasp_env)) / env_sr / 60.0
        ax3.plot(t3, gasp_env, color="#E91E63", linewidth=0.5)
        ax3.set_ylabel("Gasps")
        ax3.set_title("Gasp / obstruction events", fontsize=10)

    axes[-1].set_xlabel(x_label)
    fig.tight_layout()
    return fig


def figure_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 150) -> bytes:
    """Render a matplotlib figure to bytes."""
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------
def build_summary(
    result: DetectionResult,
    filename: str = "",
    *,
    quality: QualityResult | None = None,
    backend_used: str = "",
) -> dict:
    """Build a JSON-serialisable summary dict."""
    ahi = result.ahi
    if ahi < 5:
        severity = "Normal"
    elif ahi < 15:
        severity = "Mild"
    elif ahi < 30:
        severity = "Moderate"
    else:
        severity = "Severe"

    summary: dict = {
        "filename": filename,
        "duration_hours": round(result.total_hours, 2),
        "duration_minutes": round(result.total_hours * 60, 1),
        "backend": backend_used,
        "total_events": len(result.events),
        "apneas": len(result.apneas),
        "hypopneas": len(result.hypopneas),
        "ahi": round(ahi, 1),
        "severity": severity,
        "events": [
            {
                "type": e.kind,
                "start_s": round(e.start_s, 1),
                "end_s": round(e.end_s, 1),
                "duration_s": round(e.duration_s, 1),
                "confidence": e.confidence,
            }
            for e in result.events
        ],
    }

    if quality is not None:
        summary["quality"] = {
            "flag": quality.flag,
            "clipping_ratio": quality.clipping_ratio,
            "residual_energy_ratio": quality.residual_energy_ratio,
        }

    return summary


def save_summary(summary: dict, path: str | Path) -> None:
    """Write summary dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def save_timeline(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save timeline figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
