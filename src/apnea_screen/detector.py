"""Apnea & hypopnea event detection from separated breathing audio.

Clinical definitions (AASM simplified):
  - **Apnea**:   >= 10 s where breathing energy drops below a *low* threshold
                  (< 10 % of baseline amplitude).
  - **Hypopnea**: >= 10 s where breathing energy drops by >= 30 % from the
                   local baseline.
  - **AHI** (Apnea–Hypopnea Index): (apneas + hypopneas) / hours of recording.

Each event carries a **confidence** score in [0, 1] derived from:
  1. Drop depth relative to baseline (deeper → higher confidence).
  2. Duration scaling (10 s = low, 30+ s = high).
  3. Optional post-event gasp boost (if gasp data is available).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal as sig


@dataclass
class Event:
    """A single detected respiratory event."""

    kind: str  # "apnea" or "hypopnea"
    start_s: float
    end_s: float
    confidence: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class DetectionResult:
    """Full detection output."""

    envelope: np.ndarray  # breathing amplitude envelope (same sr)
    envelope_sr: int
    baseline: np.ndarray  # rolling baseline
    events: list[Event] = field(default_factory=list)
    total_hours: float = 0.0

    @property
    def apneas(self) -> list[Event]:
        return [e for e in self.events if e.kind == "apnea"]

    @property
    def hypopneas(self) -> list[Event]:
        return [e for e in self.events if e.kind == "hypopnea"]

    @property
    def ahi(self) -> float:
        if self.total_hours <= 0:
            return 0.0
        return len(self.events) / self.total_hours


def compute_envelope(
    breathing: np.ndarray, sr: int, *, frame_len_s: float = 0.5, hop_s: float = 0.1
) -> tuple[np.ndarray, int]:
    """RMS amplitude envelope at a reduced sample rate.

    Returns
    -------
    envelope : np.ndarray, float32
    envelope_sr : int  (samples per second of the envelope signal)
    """
    frame_len = int(frame_len_s * sr)
    hop = int(hop_s * sr)
    n_frames = 1 + (len(breathing) - frame_len) // hop

    env = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = breathing[start : start + frame_len]
        env[i] = np.sqrt(np.mean(frame**2))

    env_sr = int(1.0 / hop_s)
    return env, env_sr


def compute_baseline(
    envelope: np.ndarray, env_sr: int, *, window_s: float = 120.0
) -> np.ndarray:
    """Rolling median baseline over *window_s* seconds.

    The baseline is floored at 50% of the global median so that silent
    regions do not drag the baseline down to zero (which would mask apneas).
    """
    win = max(3, int(window_s * env_sr))
    if win % 2 == 0:
        win += 1  # medfilt needs odd kernel
    win = min(win, len(envelope))
    if win % 2 == 0:
        win -= 1
    baseline = sig.medfilt(envelope, kernel_size=win).astype(np.float32)

    # Floor: prevent baseline from dropping below half the global median.
    global_median = float(np.median(envelope[envelope > 0])) if np.any(envelope > 0) else 0.0
    floor = 0.5 * global_median
    baseline = np.maximum(baseline, floor).astype(np.float32)

    return baseline


def _find_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Find contiguous True runs in *mask* of at least *min_len* frames."""
    runs: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------
# Duration scaling: linear ramp from 0.3 at 10 s to 1.0 at 30 s
_DUR_MIN_S = 10.0
_DUR_MAX_S = 30.0
_DUR_FLOOR = 0.3
# Gasp boost: if a gasp is detected within this window after an event ends
_GASP_LOOKAHEAD_S = 5.0
_GASP_BOOST = 0.15


def _score_depth(mean_ratio: float, kind: str) -> float:
    """Score based on how far the envelope dropped below baseline.

    For apneas (ratio near 0): depth score → 1.0.
    For hypopneas (ratio ~ 0.5–0.7): depth score ~ 0.4–0.7.
    """
    if kind == "apnea":
        # ratio is in [0, 0.10); map 0→1.0, 0.10→0.3
        return max(0.0, min(1.0, 1.0 - 7.0 * mean_ratio))
    else:
        # hypopnea: ratio in [0.10, 0.70); map 0.10→0.9, 0.70→0.3
        return max(0.0, min(1.0, 1.05 - 1.25 * mean_ratio))


def _score_duration(duration_s: float) -> float:
    """Longer events are more clinically significant."""
    if duration_s >= _DUR_MAX_S:
        return 1.0
    if duration_s <= _DUR_MIN_S:
        return _DUR_FLOOR
    # Linear interpolation
    frac = (duration_s - _DUR_MIN_S) / (_DUR_MAX_S - _DUR_MIN_S)
    return _DUR_FLOOR + frac * (1.0 - _DUR_FLOOR)


def compute_confidence(
    event: Event,
    ratio: np.ndarray,
    env_sr: int,
    gasp_envelope: np.ndarray | None = None,
) -> float:
    """Compute confidence in [0, 1] for a single event.

    Components (weighted average):
      - depth:    0.5 weight
      - duration: 0.3 weight
      - gasp:     0.2 weight (boost if gasp detected right after event)
    """
    s_frame = int(event.start_s * env_sr)
    e_frame = int(event.end_s * env_sr)
    s_frame = max(0, min(s_frame, len(ratio) - 1))
    e_frame = max(s_frame + 1, min(e_frame, len(ratio)))

    mean_ratio = float(np.mean(ratio[s_frame:e_frame]))

    depth = _score_depth(mean_ratio, event.kind)
    dur = _score_duration(event.duration_s)

    gasp_score = 0.0
    if gasp_envelope is not None:
        # Check for a gasp burst in the window right after the event
        look_start = e_frame
        look_end = min(len(gasp_envelope), e_frame + int(_GASP_LOOKAHEAD_S * env_sr))
        if look_end > look_start:
            window = gasp_envelope[look_start:look_end]
            gasp_median = float(np.median(gasp_envelope))
            gasp_std = float(np.std(gasp_envelope)) + 1e-12
            if float(np.max(window)) > gasp_median + 2.0 * gasp_std:
                gasp_score = 1.0

    conf = 0.5 * depth + 0.3 * dur + 0.2 * gasp_score
    return round(max(0.0, min(1.0, conf)), 2)


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------
def detect_events(
    envelope: np.ndarray,
    baseline: np.ndarray,
    env_sr: int,
    *,
    apnea_thresh: float = 0.10,
    hypopnea_drop: float = 0.30,
    min_duration_s: float = 10.0,
    gasp_envelope: np.ndarray | None = None,
) -> list[Event]:
    """Scan the envelope for apnea / hypopnea events.

    Uses a two-pass approach:
      1. Find apneas (ratio < apnea_thresh for >= min_duration).
      2. Find hypopneas (ratio < 1-hypopnea_drop for >= min_duration)
         in regions not already marked as apnea.

    Parameters
    ----------
    apnea_thresh : float
        Fraction of baseline below which an apnea is flagged.
    hypopnea_drop : float
        Fractional drop from baseline that flags a hypopnea.
    min_duration_s : float
        Minimum event duration in seconds (AASM: 10 s).
    gasp_envelope : np.ndarray | None
        If provided, used to boost confidence when a gasp occurs
        immediately after an event.
    """
    n = len(envelope)
    min_frames = int(min_duration_s * env_sr)

    safe_baseline = np.where(baseline > 0, baseline, 1.0)
    ratio = envelope / safe_baseline

    events: list[Event] = []

    # Pass 1: apneas
    apnea_mask = ratio < apnea_thresh
    apnea_runs = _find_runs(apnea_mask, min_frames)
    claimed = np.zeros(n, dtype=bool)
    for s, e in apnea_runs:
        ev = Event("apnea", s / env_sr, e / env_sr)
        ev.confidence = compute_confidence(ev, ratio, env_sr, gasp_envelope)
        events.append(ev)
        claimed[s:e] = True

    # Pass 2: hypopneas in unclaimed regions
    hypo_mask = (ratio < (1.0 - hypopnea_drop)) & ~claimed
    hypo_runs = _find_runs(hypo_mask, min_frames)
    for s, e in hypo_runs:
        ev = Event("hypopnea", s / env_sr, e / env_sr)
        ev.confidence = compute_confidence(ev, ratio, env_sr, gasp_envelope)
        events.append(ev)

    # Sort by start time
    events.sort(key=lambda ev: ev.start_s)
    return events


def run_detection(
    breathing: np.ndarray,
    sr: int,
    *,
    gasp_envelope: np.ndarray | None = None,
) -> DetectionResult:
    """Full detection pipeline: envelope -> baseline -> events -> AHI.

    Parameters
    ----------
    breathing : np.ndarray
        Mono float32 breathing-stream waveform.
    sr : int
        Sample rate.
    gasp_envelope : np.ndarray | None
        Envelope of the gasp stream (at envelope sample rate) for
        confidence boosting.  Computed externally so that the detector
        doesn't need to know about separation details.
    """
    envelope, env_sr = compute_envelope(breathing, sr)
    baseline = compute_baseline(envelope, env_sr)

    # Trim gasp envelope to match if provided
    trimmed_gasp = None
    if gasp_envelope is not None:
        min_len = min(len(envelope), len(gasp_envelope))
        trimmed_gasp = gasp_envelope[:min_len]

    events = detect_events(envelope, baseline, env_sr, gasp_envelope=trimmed_gasp)
    total_hours = len(breathing) / sr / 3600.0

    return DetectionResult(
        envelope=envelope,
        envelope_sr=env_sr,
        baseline=baseline,
        events=events,
        total_hours=total_hours,
    )
