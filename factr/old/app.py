# factr/app.py

from pathlib import Path

import streamlit as st

from factr.config import FactrConfig
from factr.ingest import ingest_youtube


def main():
    st.set_page_config(page_title="FACTR – Debate Analyzer", layout="centered")

    st.title("FACTR – Debate Analyzer (Step 1: Ingest)")

    st.write(
        "Paste a **YouTube debate URL** below and click "
        "**Download & normalise audio**.\n\n"
        "This step just downloads the audio and converts it to 16 kHz mono WAV."
    )

    youtube_url = st.text_input("YouTube URL:", "")

    cfg = FactrConfig()  # root is auto-detected from package location

    if st.button("Download & normalise audio", type="primary"):
        if not youtube_url.strip():
            st.error("Please enter a YouTube URL first.")
        else:
            with st.spinner("Downloading and processing audio…"):
                try:
                    snap = ingest_youtube(youtube_url.strip(), cfg=cfg)
                except Exception as e:
                    st.error(f"Error during ingest: {e}")
                else:
                    st.success("Audio ingest complete ✅")

                    st.write("**Run ID:**", snap["run_id"])
                    st.write("**YouTube URL:**", snap["youtube_url"])
                    st.write("**Audio path (local):**", snap["audio_path"])
                    st.write("**Duration (seconds):**", f"{snap['duration_sec']:.1f}")
                    st.write("**Sample rate:**", snap["sample_rate"])
                    st.write("**Channels:**", snap["channels"])

                    st.info(
                        "The WAV file is stored under "
                        "`data/processed/` in your FATCR project. "
                        "Next steps will use this file for transcription, "
                        "claim extraction, and verification."
                    )


if __name__ == "__main__":
    main()
