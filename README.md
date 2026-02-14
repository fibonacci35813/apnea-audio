# Sleep-Apnea Audio Screener

Screen for obstructive sleep apnea from overnight audio recordings. Upload a WAV or MP3 file and get an estimated Apnea-Hypopnea Index (AHI), event timeline, and JSON report.

> **Disclaimer:** This tool is for educational and screening purposes only. It is **not** a certified medical diagnostic device. Consult a qualified sleep physician for diagnosis and treatment.

## How It Works

```
WAV/MP3 -> Audio Loader -> Source Separation -> Quality Check -> Detector -> Report
             (16 kHz mono)   (3 backends)        (clipping,      (envelope,   (timeline,
                                                   noise)          events,     JSON + AHI,
                                                                   confidence) stems)
```

1. **Audio Loading** -- Reads WAV/MP3/FLAC/OGG, resamples to 16 kHz mono, peak-normalises.
2. **Source Separation** -- Extracts three streams (breathing, snoring, gasps) + a residual. Three backends:

   | Backend       | Method                                           | Install extra   |
   |---------------|--------------------------------------------------|-----------------|
   | `sam_audio`   | Text-prompted neural separation (AudioSep)       | `pip install -e ".[sam]"` |
   | `openunmix`   | Music-source separator repurposed via bandpass    | `pip install -e ".[neural]"` |
   | `dsp`         | Bandpass filters + energy gating (no GPU needed)  | *(included)*    |
   | `auto`        | Tries sam_audio -> openunmix -> dsp               | --              |

3. **Quality Check** -- Flags clipped or noisy recordings before analysis.
4. **Detection** -- Computes a breathing amplitude envelope, estimates a rolling baseline, then scans for:
   - **Apnea**: >= 10 s where energy < 10% of baseline
   - **Hypopnea**: >= 10 s where energy drops >= 30% from baseline
   - Each event gets a **confidence score** (0-1) based on drop depth, duration, and post-event gasp presence.
5. **Reporting** -- Multi-panel timeline plot (shading intensity = confidence), JSON summary with AHI, severity, quality, and per-event confidence.

## Installation

```bash
# Clone and install (DSP backend works out of the box)
git clone <repo-url> && cd apnea-audio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### SAM Audio backend (recommended for best separation)

```bash
pip install -e ".[sam]"
```

### Open-Unmix backend

```bash
pip install -e ".[neural]"

# Or CPU-only torch (smaller download):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openunmix
```

## Usage

### Streamlit UI

```bash
streamlit run src/apnea_screen/app.py
```

Then open http://localhost:8501, upload an audio file, select a backend, and view results.

### Command Line

```bash
# Auto backend (tries SAM Audio -> Open-Unmix -> DSP)
python -m apnea_screen recording.wav -o output/

# Force a specific backend
python -m apnea_screen recording.wav --backend sam_audio -o output/
python -m apnea_screen recording.wav --backend dsp -o output/

# Legacy flag (still works)
python -m apnea_screen recording.wav --no-neural -o output/
```

Output files in `output/`:
- `timeline.png` -- multi-panel event timeline
- `summary.json` -- AHI, severity, per-event confidence, quality metrics
- `breathing.wav`, `snoring.wav`, `gasp.wav`, `residual.wav` -- separated stems

### Python API

```python
from apnea_screen.pipeline import run_pipeline

result = run_pipeline("recording.wav", backend="sam_audio")
print(result.summary)          # JSON-serialisable dict
print(result.quality.flag)     # "OK", "WARN_CLIPPED", or "WARN_NOISY"
print(result.streams.backend_used)  # which backend actually ran
```

## Project Structure

```
apnea-audio/
├── src/apnea_screen/
│   ├── __init__.py        # Package metadata
│   ├── __main__.py        # CLI entry point (--backend flag)
│   ├── app.py             # Streamlit UI
│   ├── audio_loader.py    # WAV/MP3 loading & normalisation
│   ├── separator.py       # Source separation (SAM Audio + Open-Unmix + DSP)
│   ├── quality.py         # Recording quality checks
│   ├── detector.py        # Apnea/hypopnea detection + confidence scoring
│   ├── pipeline.py        # End-to-end orchestration
│   └── report.py          # Timeline plots & JSON reports
├── tests/
│   ├── test_audio_loader.py
│   ├── test_detector.py
│   ├── test_separator.py
│   ├── test_quality.py
│   ├── test_confidence.py
│   └── test_sam_separator.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests for SAM Audio skip gracefully if `audiosep` is not installed.

## AHI Severity Scale

| AHI          | Classification |
|--------------|----------------|
| < 5          | Normal         |
| 5 -- 14.9   | Mild           |
| 15 -- 29.9  | Moderate       |
| >= 30        | Severe         |

## License

MIT
