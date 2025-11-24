# factr_app.py  (project root)

from pathlib import Path
import json
from typing import Optional  # <- add this line

import streamlit as st

from factr.config import FactrConfig
from factr.ingest import ingest_youtube
from factr.asr import transcribe_audio
from factr.claims import extract_claims
from factr.verify import verify_claims
import pandas as pd
import io  # add if not already imported
from factr.kb import kb_commentary_for_group
#Beginning of Edit – glossary support

from factr.glossary import GLOSSARY  # you already created this

def lookup_word(word: str) -> dict:
    """
    Look up a word in the local glossary and build some helpful links.

    - word: the original token from the text (can include punctuation)
    - returns a dict with:
        - 'word': original word
        - 'glossary': entry from GLOSSARY (or None)
        - 'wiktionary_url': link to Wiktionary
        - 'wikipedia_url': link to Wikipedia (best-effort)
    """
    # Normalise the token a bit: lowercase and strip punctuation/brackets
    key = word.lower().strip(".,!?;:\"'()[]")
    result = {
        "word": word,
        "glossary": GLOSSARY.get(key),
        "wiktionary_url": f"https://en.wiktionary.org/wiki/{key}",
        "wikipedia_url": f"https://en.wikipedia.org/wiki/{key.capitalize()}",
    }
    return result

#end of Edit – glossary support


#Beginning of Edit – new verdict badge mapping for per-tradition and combined verdicts
def verdict_badge_html(verdict: Optional[str]) -> str:
    """
    Render a coloured badge for a verdict.

    We support two kinds of labels:

    • Per-tradition (Islamic sources / Christian sources)
      - agrees        → AGREES      (green)
      - disagrees     → DISAGREES   (red)
      - divided       → DIVIDED     (orange)
      - insufficient  → INSUFFICIENT (grey)

    • Combined (Both sources)
      - agreement     → AGREEMENT   (green)
      - conflicted    → CONFLICTED  (red)
      - doubtful      → DOUBTFUL    (purple)
      - insufficient  → INSUFFICIENT (grey)

    For backwards-compatibility we also accept older labels like
    'supported', 'contradicted', 'both'.
    """
    if not verdict:
        label, colour = "NO DATA", "#7f8c8d"
    else:
        v = str(verdict).strip().lower()

        mapping = {
            # Per-tradition
            "agrees":       ("AGREES",      "#2ecc71"),
            "supported":    ("AGREES",      "#2ecc71"),

            "disagrees":    ("DISAGREES",   "#e74c3c"),
            "contradicted": ("DISAGREES",   "#e74c3c"),

            "divided":      ("DIVIDED",     "#f39c12"),
            "both":         ("DIVIDED",     "#f39c12"),

            "insufficient": ("INSUFFICIENT","#7f8c8d"),

            # Combined verdict (both sources)
            "agreement":    ("AGREEMENT",   "#2ecc71"),
            "conflicted":   ("CONFLICTED",  "#e74c3c"),
            "doubtful":     ("DOUBTFUL",    "#9b59b6"),
        }

        label, colour = mapping.get(v, (v.upper(), "#7f8c8d"))

    html = f"""
    <div style="
        display:inline-flex;
        padding:0.25rem 0.75rem;
        border-radius:999px;
        border:1px solid {colour};
        color:{colour};
        font-weight:600;
        font-size:0.85rem;
        letter-spacing:0.05em;
        text-transform:uppercase;
    ">{label}</div>
    """
    return html
#end of edit

#Beginning of Edit – confidence badge helper
def confidence_badge_html(confidence: Optional[float]) -> str:
    """
    Render the circular confidence badge as HTML.
    Expects confidence as 0–1 float (e.g. 0.9 -> 90%).
    """
    if confidence is None:
        # No confidence => render a grey empty circle
        return """
        <div style="display:flex;justify-content:flex-end;">
          <div style="
              width:56px;height:56px;
              border-radius:999px;
              border:2px solid #555555;
              display:flex;align-items:center;justify-content:center;
              font-size:0.8rem;
              color:#aaaaaa;
              font-weight:600;
          ">
            --
          </div>
        </div>
        """

    try:
        pct = float(confidence) * 100.0
    except Exception:
        pct = 0.0

    pct_int = round(pct)

    return f"""
    <div style="display:flex;justify-content:flex-end;">
      <div style="
          width:56px;height:56px;
          border-radius:999px;
          border:2px solid #f39c12;
          display:flex;align-items:center;justify-content:center;
          font-size:0.9rem;
          color:#f39c12;
          font-weight:700;
      ">
        {pct_int}%
      </div>
    </div>
    """
#End of Edit – confidence badge helper


#Beginning of Edit – helper to phrase evidence headings
def evidence_heading_for_trad(trad_label: str, verdict: Optional[str]) -> str:
    """
    Turn a verdict into human text for the evidence expander title.

    Examples:
      - "Islamic evidence to support this claim"
      - "Christian evidence to refute / reject this claim"
      - "Islamic evidence giving mixed or divided testimony on this claim"
    """
    v = (verdict or "").strip().lower()

    if v in {"agrees", "supported", "agreement"}:
        action = "to support this claim"
    elif v in {"disagrees", "contradicted", "conflicted"}:
        action = "to refute / reject this claim"
    elif v in {"divided", "both", "doubtful"}:
        action = "giving mixed or divided testimony on this claim"
    elif v == "insufficient":
        action = "related to this claim (insufficient evidence)"
    else:
        action = "related to this claim"

    return f"{trad_label} evidence {action}"
#end of edit



def render_source_badge(col, source_label: str, verdict: str):
    """Render 'Islamic sources' / 'Christian sources' / 'Both sources' + badge in a column."""
    col.markdown(
        f"<div style='font-size:0.75rem;margin-bottom:0.15rem;color:#bbbbbb;'>{source_label}</div>",
        unsafe_allow_html=True,
    )
    col.markdown(verdict_badge_html(verdict), unsafe_allow_html=True)


def render_confidence_circle(col, confidence):
    """Render a confidence score as a circular badge."""
    try:
        if confidence is None:
            value = 0.0
        else:
            value = float(confidence)
    except Exception:
        value = 0.0

    # assume 0–1 range; convert to %
    pct = max(0.0, min(1.0, value)) * 100.0

    html = f"""
    <div style="
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        gap:0.25rem;
    ">
      <div style="font-size:0.7rem;color:#bbbbbb;">Confidence</div>
      <div style="
          width:54px;height:54px;
          border-radius:50%;
          border:3px solid #f39c12;
          display:flex;
          align-items:center;
          justify-content:center;
          font-weight:600;
          color:#f39c12;
          font-size:0.85rem;
      ">{pct:.0f}%</div>
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)


#Beginning of Edit – new claim card layout with numbered evidence & glossary
def render_claim_card(idx: int, rec: dict) -> None:
    """
    Render a single claim as a 'card' with:

      • Claim text
      • Per-tradition verdict badges (Islamic / Christian)
      • Combined verdict badge (Both sources)
      • Confidence circle
      • Natural-language explanation (with [Islamic N] / [Christian N] refs)
      • Glossary expander for key terms
      • Separate expanders for Islamic and Christian evidence, with
        numbered quotes that match the indices used in the explanation.
    """
    claim_text = rec.get("claim_text") or ""
    claim_id = rec.get("claim_id") or f"claim_{idx+1}"

    verdict_islam = rec.get("verdict_islam")
    verdict_christian = rec.get("verdict_christian")
    verdict_overall = rec.get("verdict_overall")
    confidence = float(rec.get("confidence") or 0.0)

    explanation = rec.get("explanation") or ""

    evidence_islam = rec.get("evidence_islam") or []
    evidence_christ = rec.get("evidence_christian") or []

    # --- Claim header -------------------------------------------------------
    st.markdown(f"### Claim {idx + 1}")
    st.markdown(f"**{claim_text}**")

    cols = st.columns([2, 2, 2, 1])

    with cols[0]:
        st.caption("Islamic sources")
        st.markdown(verdict_badge_html(verdict_islam), unsafe_allow_html=True)

    with cols[1]:
        st.caption("Christian sources")
        st.markdown(verdict_badge_html(verdict_christian), unsafe_allow_html=True)

    with cols[2]:
        st.caption("Both sources (combined)")
        st.markdown(verdict_badge_html(verdict_overall), unsafe_allow_html=True)

    with cols[3]:
        st.caption("Confidence")
        # this uses your existing confidence_badge_html helper
        st.markdown(confidence_badge_html(confidence), unsafe_allow_html=True)

    # --- Legend for the labels ----------------------------------------------
    with st.expander("What do these verdict labels mean?"):
        st.markdown(
            """
**Islamic / Christian sources**

- **AGREES** – the passages in that tradition *support* the claim.
- **DISAGREES** – the passages in that tradition *refute or reject* the claim.
- **DIVIDED** – different passages in that tradition pull in different directions.
- **INSUFFICIENT** – not enough clear evidence in that tradition.

**Both sources (combined)**

- **AGREEMENT** – both traditions broadly support the claim.
- **CONFLICTED** – the two traditions clearly disagree.
- **DOUBTFUL** – at least one tradition gives mixed or weak evidence.
- **INSUFFICIENT** – not enough data from either side for a fair assessment.
"""
        )

    # --- Natural-language explanation ---------------------------------------
    if explanation:
        st.markdown("")
        st.markdown(explanation)

    # --- Glossary for key terms in this claim -------------------------------
    with st.expander("Explain key terms in this claim (glossary)"):
        render_glossary_for_claim(claim_text)

    # --- Evidence: Islamic ---------------------------------------------------
    if evidence_islam:
        heading_islam = evidence_heading_for_trad("Islamic", verdict_islam)
        with st.expander(heading_islam):
            for ev in evidence_islam:
                ev_id = ev.get("id")  # 1-based index from the KB search
                ref = ev.get("ref") or ""
                text = ev.get("text") or ""
                label = f"[Islamic {ev_id}]" if ev_id is not None else "[Islamic]"

                # main quote line – this is what [Islamic N] in the explanation refers to
                st.markdown(f"- **{label} {ref}** – {text}")

                # Optional: tafsir / commentary from the commentary KB, if available
                group_key = ev.get("group_key")
                if group_key:
                    with st.expander(f"↳ View tafsir / commentary for {label} {ref}"):
                        comments = kb_commentary_for_group(group_key, tradition="Islam")
                        if not comments:
                            st.write(
                                "No tafsir / commentary entries have been added yet for this passage."
                            )
                        else:
                            for c in comments:
                                c_ref = c.ref or ""
                                c_text = c.text or ""
                                st.markdown(f"- **{c_ref}** – {c_text}")
    else:
        st.info("No Islamic evidence was selected for this claim.")

    # --- Evidence: Christian -------------------------------------------------
    if evidence_christ:
        heading_christ = evidence_heading_for_trad("Christian", verdict_christian)
        with st.expander(heading_christ):
            for ev in evidence_christ:
                ev_id = ev.get("id")
                ref = ev.get("ref") or ""
                text = ev.get("text") or ""
                label = f"[Christian {ev_id}]" if ev_id is not None else "[Christian]"

                st.markdown(f"- **{label} {ref}** – {text}")

                group_key = ev.get("group_key")
                if group_key:
                    with st.expander(f"↳ View commentary for {label} {ref}"):
                        comments = kb_commentary_for_group(
                            group_key, tradition="Christian"
                        )
                        if not comments:
                            st.write(
                                "No commentary / exegesis entries have been added yet for this passage."
                            )
                        else:
                            for c in comments:
                                c_ref = c.ref or ""
                                c_text = c.text or ""
                                st.markdown(f"- **{c_ref}** – {c_text}")
    else:
        st.info("No Christian evidence was selected for this claim.")

    st.markdown("---")
#end of edit


#Beginning of Edit – glossary rendering helper
def render_glossary_for_claim(claim_text: str) -> None:
    """
    Very simple glossary lookup: we scan the claim text for any
    terms that appear in glossary.GLOSSARY and print them out.
    """
    if not claim_text:
        st.write("No claim text found.")
        return

    text_lower = claim_text.lower()
    found = []
    for term, info in GLOSSARY.items():
        if term.lower() in text_lower:
            found.append((term, info))

    if not found:
        st.write("No special glossary terms detected in this claim.")
        return

    for term, info in found:
        st.markdown(f"**{term}**")
        if isinstance(info, dict):
            definition = info.get("definition") or info.get("meaning") or ""
            if definition:
                st.write(definition)
            if info.get("notes"):
                st.write(info["notes"])
            if info.get("sources"):
                st.write(f"_Sources: {info['sources']}_")
        else:
            st.write(str(info))
        st.markdown("---")
#end of edit




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
            # Load rows from verification JSONL
            rows: list[dict] = []
            ver_file = ver_meta.get("verification_path", ver_path)
            with open(ver_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))

            if not rows:
                st.info("No verification records were produced.")
            else:
                # Build DataFrame for download/export
                df = pd.DataFrame(rows)

                st.subheader("Debate analysis – claims and verdicts")

                # Download all as CSV
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button(
                    "Download all as CSV",
                    data=csv_buf.getvalue(),
                    file_name="factr_verification_results.csv",
                    mime="text/csv",
                )

                st.markdown("")

                # Render each claim as a card
                for idx, rec in enumerate(rows):
                    render_claim_card(idx, rec)

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

                        #Beginning of Edit – claim text + glossary + evidence view
                        # Detailed evidence view for first few claims
                        st.subheader("Evidence details (first few claims)")
                        for _, row in df.head(5).iterrows():
                            header = (
                                f"{row.get('claim_id')} – "
                                f"{row.get('verdict_overall')} – "
                                f"{row.get('claim_text')[:80]}..."
                            )
                            with st.expander(header):
                                # Show the full claim text at the top of the panel
                                st.markdown("**Claim text (full):**")
                                st.write(row.get("claim_text", ""))

                                # Optional: glossary / key-term explanation for this claim
                                with st.expander("Explain key terms in this claim"):
                                    # This calls the helper we added earlier:
                                    # render_glossary_for_claim(claim_text: str)
                                    render_glossary_for_claim(row.get("claim_text", ""))

                                st.markdown("---")
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
                        #end of edit – claim text + glossary + evidence view
                    else:
                        st.info("No verification records were produced.")



if __name__ == "__main__":
    main()


