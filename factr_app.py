# factr_app.py  (project root)

from pathlib import Path
import json

import streamlit as st

from factr.config import FactrConfig
from factr.ingest import ingest_youtube
from factr.asr import transcribe_audio
from factr.claims import extract_claims
from factr.verify import verify_claims
import pandas as pd



def main():
    st.set_page_config(page_title="FACTR – Debate Analyzer", layout="centered")

    st.title("FACTR – Debate Analyzer")

    cfg = FactrConfig()
    
    # ======================================================
    # One-click end-to-end debate analysis (beta)
    # ======================================================
    st.header("Analyse debate (one-click)")

    st.write(
        "Paste a YouTube debate URL and run the entire pipeline: "
        "audio ingest → transcription → claim extraction → verification "
        "against the Islamic and Christian KB."
    )

    url_oneclick = st.text_input(
        "YouTube URL for full analysis",
        key="oneclick_url",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        max_chunks_all = st.number_input(
            "Max transcript chunks (0 = all)",
            min_value=0,
            value=2,
            step=1,
            help="Limit chunks for claim extraction while testing.",
        )
    with col2:
        max_claims_all = st.number_input(
            "Max claims to verify (0 = all)",
            min_value=0,
            value=5,
            step=1,
            help="Limit number of claims analysed to control API cost.",
        )
    with col3:
        top_k_all = st.number_input(
            "Top-K evidence per tradition",
            min_value=1,
            value=5,
            step=1,
            help="How many KB passages to retrieve for each claim.",
        )

    if st.button("Analyse debate (end-to-end)"):
        if not url_oneclick.strip():
            st.error("Please enter a YouTube URL.")
        else:
            progress = st.progress(0.0)
            status = st.empty()

            # ---------------------------
            # Step 1: Ingest YouTube audio
            # ---------------------------
            status.text("Step 1/4: Downloading & normalising audio…")
            try:
                result = ingest_youtube(url_oneclick, cfg=cfg)
            except Exception as e:
                st.error(f"Error during ingest: {e}")
                st.stop()

            # Normalise whatever ingest_youtube returned
            run_id = None
            audio_path = None
            ingest_meta = {}

            if isinstance(result, tuple):
                # Common patterns:
                #   (run_id, audio_path, meta)
                #   (run_id, meta)
                if len(result) == 3:
                    run_id, audio_path, ingest_meta = result
                elif len(result) == 2:
                    run_id, ingest_meta = result
                    if isinstance(ingest_meta, dict):
                        audio_path = ingest_meta.get("audio_path")
                else:
                    # Fallback: treat first element as meta dict
                    ingest_meta = result[0]
            elif isinstance(result, dict):
                ingest_meta = result

            # Try to fill in missing bits from meta
            if run_id is None and isinstance(ingest_meta, dict):
                run_id = ingest_meta.get("run_id")
            if audio_path is None and isinstance(ingest_meta, dict):
                audio_path = ingest_meta.get("audio_path")

            if not audio_path:
                st.error("Could not determine audio_path from ingest_youtube result.")
                st.stop()

            progress.progress(0.15)


            # ---------------------------
            # Step 2: Transcribe audio
            # ---------------------------
            def asr_progress(frac: float) -> None:
                # Map [0,1] ASR progress into [0.15, 0.60] of the main bar
                progress.progress(0.15 + 0.45 * frac)
                status.text(f"Step 2/4: Transcribing audio… {frac*100:.1f}%")

            try:
                utterances, asr_meta = transcribe_audio(
                    audio_path,
                    cfg=cfg,
                    # uncomment if you switched model size:
                    # model_size="medium.en",
                    progress_callback=asr_progress,
                )
            except Exception as e:
                st.error(f"Error during transcription: {e}")
                st.stop()

            progress.progress(0.60)

            # ---------------------------
            # Step 3: Extract claims
            # ---------------------------
            status.text("Step 3/4: Extracting claims from transcript…")
            mc_chunks = None if max_chunks_all == 0 else int(max_chunks_all)
            try:
                claims_path, claims_meta = extract_claims(
                    cfg=cfg,
                    max_chunks=mc_chunks,
                )
            except Exception as e:
                st.error(f"Error during claim extraction: {e}")
                st.stop()

            progress.progress(0.75)

            # ---------------------------
            # Step 4: Verify claims vs KB
            # ---------------------------
            status.text("Step 4/4: Verifying claims against the KB…")
            mc_claims = None if max_claims_all == 0 else int(max_claims_all)
            try:
                ver_path, ver_meta = verify_claims(
                    cfg=cfg,
                    max_claims=mc_claims,
                    top_k_evidence=int(top_k_all),
                )
            except Exception as e:
                st.error(f"Error during verification: {e}")
                st.stop()

            progress.progress(1.0)
            status.text("Analysis complete ✅")
            st.success("End-to-end debate analysis complete ✅")

            # -----------------------------------------
            # Show final verification results + evidence
            # -----------------------------------------
            rows = []
            with open(ver_meta["verification_path"], "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 100:   # just show a sample
                        break
                    rows.append(json.loads(line))

            if not rows:
                st.info("No verification records were produced.")
            else:
                st.subheader("Debate analysis – claims and verdicts")

                df = pd.DataFrame(rows)
                show_cols = [
                    "claim_id",
                    "side",
                    "verdict_overall",
                    "verdict_islam",
                    "verdict_christian",
                    "confidence",
                    "claim_text",
                    "explanation",
                ]
                df_basic = df[[c for c in show_cols if c in df.columns]]
                st.dataframe(df_basic)

                st.subheader("Evidence details (first few claims)")
                for _, row in df.head(5).iterrows():
                    header = (
                        f"{row.get('claim_id')} – "
                        f"{row.get('verdict_overall')} – "
                        f"{row.get('claim_text')[:80]}..."
                    )
                    with st.expander(header):
                        st.markdown("**Islamic evidence:**")
                        ev_islam = row.get("evidence_islam") or []
                        if not ev_islam:
                            st.write("_No Islamic evidence selected._")
                        else:
                            for ev in ev_islam:
                                st.write(
                                    f"[{ev.get('id')}] "
                                    f"{ev.get('ref') or ''} – "
                                    f"{ev.get('text')}"
                                )

                        st.markdown("---")
                        st.markdown("**Christian evidence:**")
                        ev_christ = row.get("evidence_christian") or []
                        if not ev_christ:
                            st.write("_No Christian evidence selected._")
                        else:
                            for ev in ev_christ:
                                st.write(
                                    f"[{ev.get('id')}] "
                                    f"{ev.get('ref') or ''} – "
                                    f"{ev.get('text')}"
                                )

    # -----------------------
    # Step 1 – Ingest
    # -----------------------
    st.header("Step 1 – Download & normalise audio")

    st.write(
        "Paste a **YouTube debate URL** below and click "
        "**Download & normalise audio**.\n\n"
        "This step just downloads the audio and converts it to 16 kHz mono WAV."
    )

    youtube_url = st.text_input("YouTube URL:", "")

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

    st.markdown("---")

    # -----------------------
    # Step 2 – Transcribe
    # -----------------------
    st.header("Step 2 – Transcribe latest audio")

    st.write(
        "This will transcribe the most recently ingested audio file using "
        "Whisper (faster-whisper). For now, diarisation is not enabled; "
        "all segments are marked as SPEAKER_00."
    )

    if st.button("Transcribe latest audio"):
        last_ingest_path = cfg.processed_dir / "LAST_INGEST.json"
        if not last_ingest_path.exists():
            st.error(
                "No ingested audio found. Please run Step 1 first "
                "(Download & normalise audio)."
            )
        else:
            ingest_meta = json.loads(last_ingest_path.read_text(encoding="utf-8"))
            audio_path = ingest_meta["audio_path"]

            # --- Streamlit progress widgets ---
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(frac: float) -> None:
                progress_bar.progress(frac)
                status_text.text(f"Transcribing… {frac * 100:.1f}%")

            with st.spinner("Transcribing audio… this may take a while for long videos."):
                try:
                    utterances, asr_meta = transcribe_audio(
                        audio_path,
                        cfg=cfg,
                        progress_callback=update_progress,  # <-- NEW
                    )
                except Exception as e:
                    st.error(f"Error during transcription: {e}")
                else:
                    progress_bar.progress(1.0)
                    status_text.text("Transcription 100% complete.")
                    st.success("Transcription complete ✅")

                    st.write("**Audio path:**", asr_meta["audio_path"])
                    st.write("**Language detected:**", asr_meta["language"])
                    st.write("**Duration (seconds):**", f"{asr_meta['duration_sec']:.1f}")
                    st.write("**Number of segments:**", asr_meta["num_utterances"])
                    st.write("**Model size:**", asr_meta["model_size"])
                    st.write("**Device:**", asr_meta["device"])

                    st.subheader("Sample of utterances")
                    st.dataframe(utterances.head(20))

    st.markdown("---")

    # -----------------------
    # Step 3 – Extract claims
    # -----------------------
    st.header("Step 3 – Extract claims from transcript")

    st.write(
        "This will call the OpenAI API on the transcribed debate to extract "
        "atomic claims in JSON form. Make sure you have run Step 2 first."
    )

    max_chunks = st.number_input(
        "Max chunks to process (0 = all)",
        min_value=0,
        value=2, # set to '0' for no limit (set to 1-3 for testing)
        step=1,
        help="Use a small number while testing to control cost.",
    )

    if st.button("Extract claims"):
        # quick existence check
        utt_path = cfg.processed_dir / "UTTERANCES.parquet"
        if not utt_path.exists():
            st.error("UTTERANCES.parquet not found. Please run Step 2 first.")
        else:
            with st.spinner("Extracting claims with OpenAI…"):
                try:
                    mc = None if max_chunks == 0 else int(max_chunks)
                    claims_path, meta = extract_claims(cfg=cfg, max_chunks=mc)
                except Exception as e:
                    st.error(f"Error during claim extraction: {e}")
                else:
                    st.success("Claim extraction complete ✅")

                    st.write("**Claims file:**", meta["claims_path"])
                    st.write("**Number of claims:**", meta["num_claims"])
                    st.write("**Chunks processed:**", meta["num_chunks"])
                    st.write("**Model:**", meta["model"])

                    # Load a small sample to show
                    rows = []
                    with open(meta["claims_path"], "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i >= 50:
                                break
                            rows.append(json.loads(line))
                    if rows:
                        st.subheader("Sample of extracted claims")
                        st.dataframe(pd.DataFrame(rows))
                    else:
                        st.info("No claims were extracted (model returned empty lists).")

    st.markdown("---")

    # -----------------------
    # Step 4 – Analyse claims against KB
    # -----------------------
    st.header("Step 4 – Analyse claims against KB")

    st.write(
        "This step compares each extracted claim to your Islamic and Christian "
        "knowledge base, then asks GPT to label it as supported, contradicted, "
        "mixed, or insufficient."
    )

    max_claims = st.number_input(
        "Max claims to verify (0 = all)",
        min_value=0,
        value=5,
        step=1,
        help="Use a small number while testing to control API cost.",
    )

    top_k_evidence = st.number_input(
        "Top-K evidence passages per tradition",
        min_value=1,
        value=5,
        step=1,
        help="How many passages to retrieve from each tradition for each claim.",
    )

    if st.button("Analyse claims vs KB"):
        ver_path = cfg.processed_dir / "VERIFICATION.jsonl"
        claims_path = cfg.processed_dir / "CLAIMS.jsonl"

        if not claims_path.exists():
            st.error("CLAIMS.jsonl not found. Please run Step 3 first.")
        else:
            with st.spinner("Verifying claims against the KB…"):
                try:
                    mc = None if max_claims == 0 else int(max_claims)
                    ver_path, meta = verify_claims(
                        cfg=cfg,
                        max_claims=mc,
                        top_k_evidence=int(top_k_evidence),
                    )
                except Exception as e:
                    st.error(f"Error during verification: {e}")
                else:
                    st.success("Verification complete ✅")

                    st.write("**Verification file:**", meta["verification_path"])
                    st.write("**Number of claims verified:**", meta["num_verifications"])
                    st.write("**Model:**", meta["model"])

                    # Show a sample table of results
                    rows = []
                    with open(meta["verification_path"], "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i >= 50:
                                break
                            rows.append(json.loads(line))
                    if rows:
                        st.subheader("Sample of verification results")
                        df = pd.DataFrame(rows)

                        # Basic table
                        show_cols = [
                            "claim_id",
                            "side",
                            "verdict_overall",
                            "verdict_islam",
                            "verdict_christian",
                            "confidence",
                            "claim_text",
                            "explanation",
                        ]
                        df_basic = df[[c for c in show_cols if c in df.columns]]
                        st.dataframe(df_basic)

                        # Detailed evidence view for first few claims
                        st.subheader("Evidence details (first few claims)")
                        for _, row in df.head(5).iterrows():
                            header = (
                                f"{row.get('claim_id')} – "
                                f"{row.get('verdict_overall')} – "
                                f"{row.get('claim_text')[:80]}..."
                            )
                            with st.expander(header):
                                st.markdown("**Islamic evidence:**")
                                ev_islam = row.get("evidence_islam") or []
                                if not ev_islam:
                                    st.write("_No Islamic evidence selected._")
                                else:
                                    for ev in ev_islam:
                                        st.write(
                                            f"[{ev.get('id')}] "
                                            f"{ev.get('ref') or ''} – "
                                            f"{ev.get('text')}"
                                        )

                                st.markdown("---")
                                st.markdown("**Christian evidence:**")
                                ev_christ = row.get("evidence_christian") or []
                                if not ev_christ:
                                    st.write("_No Christian evidence selected._")
                                else:
                                    for ev in ev_christ:
                                        st.write(
                                            f"[{ev.get('id')}] "
                                            f"{ev.get('ref') or ''} – "
                                            f"{ev.get('text')}"
                                        )
                    else:
                        st.info("No verification records were produced.")



if __name__ == "__main__":
    main()


