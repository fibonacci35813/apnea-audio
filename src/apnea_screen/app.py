"""Streamlit UI for sleep-apnea screening."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Sleep Apnea Screener", layout="wide")


def main() -> None:
    st.title("Sleep Apnea Screener")
    st.markdown(
        "Upload an overnight audio recording (WAV or MP3) to screen for "
        "obstructive sleep-apnea events."
    )

    st.warning(
        "**Disclaimer:** This tool is for *screening purposes only* and is "
        "**not** a medical diagnostic device.  Always consult a qualified "
        "sleep physician for diagnosis.",
    )

    uploaded = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "flac", "ogg"],
        help="Mono or stereo, any sample rate. Longer recordings give more reliable results.",
    )

    col1, col2 = st.columns(2)
    with col1:
        backend = st.selectbox(
            "Separation backend",
            options=["auto", "sam_audio", "openunmix", "dsp"],
            index=0,
            help=(
                "**auto** — tries SAM Audio, then Open-Unmix, then DSP.  "
                "**sam_audio** — text-prompted neural separation (requires audiosep).  "
                "**openunmix** — music-source separator repurposed for breath sounds.  "
                "**dsp** — lightweight bandpass filters (always available)."
            ),
        )
    with col2:
        save_output = st.checkbox("Save outputs to disk", value=False)

    if uploaded is None:
        st.info("Waiting for file upload ...")
        return

    # Write uploaded bytes to a temp file so librosa can read it
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    st.markdown("---")

    with st.spinner("Running pipeline ... (this may take a while for long recordings)"):
        # Lazy import so Streamlit starts quickly
        from apnea_screen.pipeline import run_pipeline

        output_dir = "output" if save_output else None
        result = run_pipeline(tmp_path, backend=backend, output_dir=output_dir)

    # --- Quality warnings ---
    qf = result.quality.flag
    if qf == "WARN_CLIPPED":
        st.error(
            f"**Recording quality: clipped** — {result.quality.clipping_ratio:.1%} of "
            "samples are near full-scale. Results may be unreliable. "
            "Re-record with lower microphone gain if possible."
        )
    elif qf == "WARN_NOISY":
        st.warning(
            f"**Recording quality: noisy** — residual-to-breathing energy ratio is "
            f"{result.quality.residual_energy_ratio:.1f}x. Background noise may "
            "reduce detection accuracy."
        )

    # --- Results ---
    st.success(f"Analysis complete!  (backend: **{result.streams.backend_used}**)")

    # Summary metrics
    summary = result.summary
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AHI", f"{summary['ahi']}")
    m2.metric("Severity", summary["severity"])
    m3.metric("Apneas", summary["apneas"])
    m4.metric("Hypopneas", summary["hypopneas"])

    st.markdown(f"**Recording duration:** {summary['duration_minutes']} min")

    # Timeline
    st.subheader("Timeline")
    st.caption("Shading intensity reflects event confidence (darker = higher).")
    st.pyplot(result.figure)

    # Events table
    if summary["events"]:
        st.subheader("Detected Events")
        st.dataframe(summary["events"], use_container_width=True)
    else:
        st.info("No apnea or hypopnea events detected.")

    # JSON download
    st.subheader("JSON Report")
    json_str = json.dumps(summary, indent=2)
    st.download_button(
        "Download JSON",
        data=json_str,
        file_name="apnea_summary.json",
        mime="application/json",
    )
    with st.expander("View raw JSON"):
        st.code(json_str, language="json")

    if save_output:
        st.info(
            "Outputs saved to `output/` — timeline.png, summary.json, "
            "breathing.wav, snoring.wav, gasp.wav, residual.wav"
        )


main()
