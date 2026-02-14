"""CLI entry point: ``python -m apnea_screen <audio_file>``."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sleep-apnea screening from overnight audio."
    )
    parser.add_argument("audio", help="Path to WAV / MP3 audio file")
    parser.add_argument(
        "-o", "--output-dir", default="output", help="Directory for results (default: output/)"
    )

    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--backend",
        choices=["auto", "sam_audio", "openunmix", "dsp"],
        default="auto",
        help="Separation backend (default: auto)",
    )
    backend_group.add_argument(
        "--no-neural", action="store_true",
        help="(deprecated) Skip neural separation, use DSP only",
    )

    args = parser.parse_args()

    from .pipeline import run_pipeline

    # Legacy compat
    backend = args.backend
    if args.no_neural:
        backend = "dsp"

    print(f"Processing: {args.audio}  (backend={backend})")
    result = run_pipeline(
        args.audio,
        backend=backend,
        output_dir=args.output_dir,
    )

    # Quality warnings
    if result.quality.flag != "OK":
        print(f"\n  Quality warning: {result.quality.flag}")
        if result.quality.flag == "WARN_CLIPPED":
            print(f"  Clipping ratio: {result.quality.clipping_ratio:.2%}")
        else:
            print(f"  Residual/breathing energy ratio: {result.quality.residual_energy_ratio:.1f}")

    print(json.dumps(result.summary, indent=2))
    print(f"\nBackend used: {result.streams.backend_used}")
    print(f"Outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
