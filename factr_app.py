from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Optional, List
from datetime import datetime  # <- for logging timestamps

import html
import io
import re

import pandas as pd
import streamlit as st

from factr.config import FactrConfig
from factr.ingest import ingest_youtube
from factr.asr import transcribe_audio
from factr.claims import extract_claims
from factr.verify import verify_claims
from factr.kb import kb_commentary_for_group
# from factr.glossary import GLOSSARY
# from glossary import render_glossary_for_claim
from openai import OpenAI

LOG_FILE = Path("factr_glossary_errors.log")
_glossary_client = OpenAI()

# =====================================================================
# Glossary helpers
# =====================================================================

def lookup_word(word: str) -> Dict[str, Any]:
    """
    Look up a word in the local glossary and build some helpful links.

    - word: the original token from the text (can include punctuation)
    - returns a dict with:
        - 'word': original word
        - 'glossary': entry from GLOSSARY (or None)
        - 'wiktionary_url': link to Wiktionary
        - 'wikipedia_url': link to Wikipedia (best-effort)
    """
    key = word.lower().strip(".,!?;:\"'()[]")
    return {
        "word": word,
        "glossary": GLOSSARY.get(key),
        "wiktionary_url": f"https://en.wiktionary.org/wiki/{key}",
        "wikipedia_url": f"https://en.wikipedia.org/wiki/{key.capitalize()}",
    }


#Beginning of Edit – glossary helpers

# Create a dedicated OpenAI client for glossary generation
#_glossary_client = OpenAI()


def generate_glossary_for_claim(claim_text: str) -> dict:
    """
    Ask GPT to extract important technical/theological terms
    from a claim and define them clearly.

    Returns a dict: { "term": "definition", ... }
    """
    claim_text = (claim_text or "").strip()
    if not claim_text:
        return {}

    # Prompt the model
    system_msg = (
        "You are a concise theological glossary assistant. "
        "Given a short claim taken from a religious debate, "
        "identify up to 6 important terms or short phrases "
        "that an ordinary reader might not know. For each term, "
        "give a clear 1–2 sentence definition in simple English. "
        "Do not include citations or verse numbers unless they are "
        "part of the term itself."
    )

    user_msg = (
        "Extract glossary terms from the following claim.\n\n"
        "Return ONLY a JSON object mapping each term or short phrase "
        "to its definition, for example:\n\n"
        '{\n'
        '  "term 1": "definition...",\n'
        '  "term 2": "definition...",\n'
        '  "term 3": "definition..."\n'
        '}\n\n'
        "If there are no specialist terms, return an empty JSON object: {}\n\n"
        f"CLAIM:\n{claim_text}"
    )

    try:
        # Use the stable Chat Completions JSON mode
        resp = _glossary_client.chat.completions.create(
            model="gpt-4.1-mini",  # or gpt-4o-mini if you prefer
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            return {}

        # Try to parse JSON directly
        try:
            data = json.loads(raw)
        except Exception:
            # Fallback: strip any ```json ... ``` fences if the model added them
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if not m:
                raise
            data = json.loads(m.group(0))

        if isinstance(data, dict):
            # Normalise term keys a little
            return {k.strip(): v for k, v in data.items() if k and k.strip()}

        return {}

    except Exception as e:
        # Log the error but fail softly so the app keeps running
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log_f:
                log_f.write(
                    f"[{datetime.now().isoformat()}] "
                    f"GLOSSARY_ERROR: {e}\n"
                )
        except Exception:
            # If logging fails, just ignore – we still don't want to crash the app
            pass

        return {}



def render_glossary_for_claim(claim_text: str) -> None:
    """
    Streamlit helper: render a glossary for a single claim.
    """
    terms = generate_glossary_for_claim(claim_text)

    if not terms:
        st.info(
            "No glossary terms could be generated for this claim yet. "
            "This may be due to a temporary issue calling the model."
        )
        return

    for term, definition in terms.items():
        st.markdown(f"**{term}**")
        st.write(definition)
        st.markdown("---")

#end of Edit – glossary helpers



# =====================================================================
# Verdict + confidence badges
# =====================================================================

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

            "insufficient": ("INSUFFICIENT", "#7f8c8d"),

            # Combined verdict (both sources)
            "agreement":    ("AGREEMENT",   "#2ecc71"),
            "conflicted":   ("CONFLICTED",  "#e74c3c"),
            "doubtful":     ("DOUBTFUL",    "#9b59b6"),
        }

        label, colour = mapping.get(v, (v.upper(), "#7f8c8d"))

    return f"""
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


def confidence_badge_html(confidence: Optional[float]) -> str:
    """
    Render a compact circular confidence badge as HTML, aligned with the
    verdict badges.

    The value is expected in [0, 1]. We convert to a percentage and clamp.
    """
    if confidence is None:
        return ""

    try:
        pct = float(confidence) * 100.0
    except Exception:
        pct = 0.0

    pct = max(0.0, min(100.0, pct))
    pct_int = int(round(pct))

    return f"""
    <div style="
        display:flex;
        align-items:flex-start;
        justify-content:center;
        margin-top:0.6rem;
    ">
      <div style="
          width:64px;
          height:64px;
          border-radius:50%;
          border:3px solid #F7931A;
          display:flex;
          align-items:center;
          justify-content:center;
          font-weight:600;
          font-size:0.9rem;
          color:#F7931A;
          box-shadow:0 0 0 1px rgba(0,0,0,0.6);
          background:rgba(0,0,0,0.4);
      ">
        {pct_int}%
      </div>
    </div>
    """


# =====================================================================
# Evidence heading helper
# =====================================================================

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


# =====================================================================
# Claim card renderer
# =====================================================================

def render_claim_card(idx: int, rec: Dict[str, Any]) -> None:
    """
    Render one claim with:
    - verdict badges for Islamic, Christian and combined views,
    - confidence score,
    - a short natural-language explanation,
    - glossary of key terms in the claim,
    - Islamic and Christian evidence lists with optional tafsir/exegesis.
    """
    claim_text = rec.get("claim_text") or ""
    verdict_islam = rec.get("verdict_islam")
    verdict_christian = rec.get("verdict_christian")
    verdict_overall = rec.get("verdict_overall")
    confidence = rec.get("confidence")
    explanation = rec.get("explanation") or ""

    # These come from verify.py – list[dict] per tradition
    evidence_islam = rec.get("evidence_islam") or []
    evidence_christ = rec.get("evidence_christian") or []

    # --- Claim header -------------------------------------------------------
    st.markdown(f"### Claim {idx + 1}")
    st.markdown(f"**{claim_text}**")

    # --- Verdict row: Islamic / Christian / Both / Confidence ---------------
    cols = st.columns([1, 1, 1, 1])

    with cols[0]:
        st.markdown(
            "<div style='text-align:left;font-size:0.9rem;'>Islamic sources</div>",
            unsafe_allow_html=True,
        )
        st.markdown(verdict_badge_html(verdict_islam), unsafe_allow_html=True)

    with cols[1]:
        st.markdown(
            "<div style='text-align:left;font-size:0.9rem;'>Christian sources</div>",
            unsafe_allow_html=True,
        )
        st.markdown(verdict_badge_html(verdict_christian), unsafe_allow_html=True)

    with cols[2]:
        st.markdown(
            "<div style='text-align:left;font-size:0.9rem;'>Both sources (combined)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(verdict_badge_html(verdict_overall), unsafe_allow_html=True)

    with cols[3]:
        # Title + small "?" hint
        st.markdown(
            """
            <div style='text-align:center;font-size:0.9rem;'>
              Confidence
              <span style="font-size:0.8rem;opacity:0.75;"
                    title="How sure the model is about these verdicts, based on how strongly the evidence matched.">?</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        conf_html = confidence_badge_html(confidence) or ""
        st.markdown(
            f"<div style='text-align:center;'>{conf_html}</div>",
            unsafe_allow_html=True,
        )

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

**Confidence %**

- This is the model’s own estimate (0–100%) of how stable the overall verdict is,
  given the passages it has seen. Higher means *stronger, more consistent*
  evidence; lower means *weaker or more conflicted* evidence.
"""
        )

    # --- Natural-language explanation ---------------------------------------
    if explanation:
        st.markdown("**Explanation**")
        st.markdown(explanation)

    # --- Glossary for key terms in this claim -------------------------------
    with st.expander("Explain key terms in this claim (glossary)"):
        render_glossary_for_claim(claim_text)

    # --- Evidence from primary sources --------------------------------------
    if evidence_islam or evidence_christ:
        st.markdown("### Evidence from primary sources")

    # Islamic evidence
    if evidence_islam:
        heading_islam = evidence_heading_for_trad("Islamic", verdict_islam)
        with st.expander(heading_islam):
            for ev in evidence_islam:
                ev_id = ev.get("id")
                ref = ev.get("ref") or ""
                text = ev.get("text") or ""
                label = f"[Islamic {ev_id}]" if ev_id is not None else "[Islamic]"

                st.markdown(f"- **{label} {ref}** – {text}")

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

                                st.markdown(f"**{c_ref}**")
                                st.markdown(
                                    "<div style='white-space:pre-wrap;font-size:0.95rem;'>"
                                    + html.escape(c_text or "")
                                    + "</div>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown("---")
    else:
        st.info("No Islamic evidence was selected for this claim.")

    # Christian evidence
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

                                st.markdown(f"**{c_ref}**")
                                st.markdown(
                                    "<div style='white-space:pre-wrap;font-size:0.95rem;'>"
                                    + html.escape(c_text or "")
                                    + "</div>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown("---")
    else:
        st.info("No Christian evidence was selected for this claim.")

    st.markdown("---")


# =====================================================================
# Render helper for one-click results
# =====================================================================

def render_oneclick_results(rows: List[Dict[str, Any]]) -> None:
    """
    Render the 'Debate analysis – claims and verdicts' section
    (table download + per-claim cards) from a list of verification rows.
    """
    if not rows:
        st.info("No verification records were produced.")
        return

    df = pd.DataFrame(rows)

    st.subheader("Debate analysis – claims and verdicts")

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download all as CSV",
        data=csv_buf.getvalue(),
        file_name="factr_verification_results.csv",
        mime="text/csv",
    )

    st.markdown("")

    for idx, rec in enumerate(rows):
        render_claim_card(idx, rec)


# =====================================================================
# MAIN APP
# =====================================================================

def main() -> None:
    st.set_page_config(page_title="FACTR – Debate Analyzer", layout="centered")

    st.title("FACTR – Debate Analyzer")

    # Session state for cached results + run log
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
    if "run_log" not in st.session_state:
        st.session_state["run_log"] = []

    cfg = FactrConfig()
    log_file = cfg.processed_dir / "ONECLICK_RUN.log"

    # Simple helper to log messages both to session_state and to file
    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        st.session_state["run_log"].append(line)
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # Logging failure should never crash the app
            pass

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

    analyse_button = st.button("Analyse debate (end-to-end)")

    if analyse_button:
        if not url_oneclick.strip():
            st.error("Please enter a YouTube URL.")
        else:
            # fresh log for this run
            st.session_state["run_log"] = []
            log(f"Starting one-click analysis for URL: {url_oneclick.strip()}")
            log(
                f"Settings – max_chunks={max_chunks_all}, "
                f"max_claims={max_claims_all}, top_k={top_k_all}"
            )

            progress = st.progress(0.0)
            status = st.empty()
            error = False

            # ---------------------------
            # Step 1: Ingest YouTube audio
            # ---------------------------
            status.text("Step 1/4: Downloading & normalising audio…")
            log("Step 1/4 – ingest_youtube() starting.")
            try:
                result = ingest_youtube(url_oneclick.strip(), cfg=cfg)
                log("Step 1/4 – ingest_youtube() completed successfully.")
            except Exception as e:
                msg = f"Error during ingest: {e}"
                st.error(msg)
                log(msg)
                error = True

            # Normalise ingest result only if step 1 succeeded
            if not error:
                run_id = None
                audio_path = None
                ingest_meta: Dict[str, Any] = {}

                if isinstance(result, tuple):
                    if len(result) == 3:
                        run_id, audio_path, ingest_meta = result
                    elif len(result) == 2:
                        run_id, ingest_meta = result
                        if isinstance(ingest_meta, dict):
                            audio_path = ingest_meta.get("audio_path")
                    else:
                        ingest_meta = result[0]
                elif isinstance(result, dict):
                    ingest_meta = result

                if run_id is None and isinstance(ingest_meta, dict):
                    run_id = ingest_meta.get("run_id")
                if audio_path is None and isinstance(ingest_meta, dict):
                    audio_path = ingest_meta.get("audio_path")

                if not audio_path:
                    msg = "Could not determine audio_path from ingest_youtube result."
                    st.error(msg)
                    log(msg)
                    error = True
                else:
                    log(f"Using audio_path={audio_path} (run_id={run_id}).")

            progress.progress(0.15)

            # ---------------------------
            # Step 2: Transcribe audio
            # ---------------------------
            if not error:
                def asr_progress(frac: float) -> None:
                    progress.progress(0.15 + 0.45 * frac)
                    status.text(f"Step 2/4: Transcribing audio… {frac*100:.1f}%")

                status.text("Step 2/4: Transcribing audio…")
                log("Step 2/4 – transcribe_audio() starting.")
                try:
                    utterances, asr_meta = transcribe_audio(
                        audio_path,
                        cfg=cfg,
                        progress_callback=asr_progress,
                    )
                    log(
                        "Step 2/4 – transcription complete: "
                        f"{asr_meta.get('num_utterances')} utterances, "
                        f"duration={asr_meta.get('duration_sec')}s."
                    )
                except Exception as e:
                    msg = f"Error during transcription: {e}"
                    st.error(msg)
                    log(msg)
                    error = True

            progress.progress(0.60)

            # ---------------------------
            # Step 3: Extract claims
            # ---------------------------
            if not error:
                status.text("Step 3/4: Extracting claims from transcript…")
                mc_chunks = None if max_chunks_all == 0 else int(max_chunks_all)
                log(f"Step 3/4 – extract_claims() starting (max_chunks={mc_chunks}).")
                try:
                    claims_path, claims_meta = extract_claims(
                        cfg=cfg,
                        max_chunks=mc_chunks,
                    )
                    log(
                        "Step 3/4 – extract_claims() complete: "
                        f"{claims_meta.get('num_claims')} claims, "
                        f"chunks={claims_meta.get('num_chunks')}."
                    )
                except Exception as e:
                    msg = f"Error during claim extraction: {e}"
                    st.error(msg)
                    log(msg)
                    error = True

            progress.progress(0.75)

            # ---------------------------
            # Step 4: Verify claims vs KB
            # ---------------------------
            if not error:
                status.text("Step 4/4: Verifying claims against the KB…")
                mc_claims = None if max_claims_all == 0 else int(max_claims_all)
                log(
                    "Step 4/4 – verify_claims() starting "
                    f"(max_claims={mc_claims}, top_k={int(top_k_all)})."
                )
                try:
                    ver_path, ver_meta = verify_claims(
                        cfg=cfg,
                        max_claims=mc_claims,
                        top_k_evidence=int(top_k_all),
                    )
                    log(
                        "Step 4/4 – verify_claims() complete: "
                        f"{ver_meta.get('num_verifications')} verifications."
                    )
                except Exception as e:
                    msg = f"Error during verification: {e}"
                    st.error(msg)
                    log(msg)
                    error = True

            if error:
                progress.progress(0.0)
                status.text("Analysis failed – see processing log below.")
                st.session_state["last_results"] = None
                log("One-click analysis finished with ERRORS.")
            else:
                progress.progress(1.0)
                status.text("Analysis complete ✅")
                st.success("End-to-end debate analysis complete ✅")
                log("One-click analysis finished successfully.")

                # -----------------------------------------
                # Show final verification results + evidence
                # -----------------------------------------
                rows: List[Dict[str, Any]] = []
                ver_file = ver_meta.get("verification_path", ver_path)
                with open(ver_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rows.append(json.loads(line))

                if not rows:
                    st.info("No verification records were produced.")
                    st.session_state["last_results"] = None
                    log("Verification file contained no rows.")
                else:
                    st.session_state["last_results"] = rows
                    render_oneclick_results(rows)

    # If we have previous results and the button is NOT pressed on this rerun,
    # show the cached results so the page does not "forget" everything.
    if not analyse_button and st.session_state.get("last_results"):
        st.markdown("---")
        render_oneclick_results(st.session_state["last_results"])

    # Show processing log (for this session)
    with st.expander("View processing log (this session)", expanded=False):
        log_lines: List[str] = st.session_state.get("run_log") or []
        if not log_lines:
            st.write("No one-click runs logged yet.")
        else:
            # most recent first
            for line in reversed(log_lines[-200:]):
                st.text(line)

    # ======================================================
    # Step-by-step pipeline (manual control) – ADVANCED
    # ======================================================

    st.markdown("---")
    with st.expander(
        "Advanced: step-by-step pipeline (manual control)", expanded=False
    ):
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

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(frac: float) -> None:
                    progress_bar.progress(frac)
                    status_text.text(f"Transcribing… {frac * 100:.1f}%")

                with st.spinner(
                    "Transcribing audio… this may take a while for long videos."
                ):
                    try:
                        utterances, asr_meta = transcribe_audio(
                            audio_path,
                            cfg=cfg,
                            progress_callback=update_progress,
                        )
                    except Exception as e:
                        st.error(f"Error during transcription: {e}")
                    else:
                        progress_bar.progress(1.0)
                        status_text.text("Transcription 100% complete.")
                        st.success("Transcription complete ✅")

                        st.write("**Audio path:**", asr_meta["audio_path"])
                        st.write("**Language detected:**", asr_meta["language"])
                        st.write(
                            "**Duration (seconds):**", f"{asr_meta['duration_sec']:.1f}"
                        )
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
            value=2,
            step=1,
            help="Use a small number while testing to control cost.",
        )

        if st.button("Extract claims"):
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

                        rows: List[Dict[str, Any]] = []
                        with open(meta["claims_path"], "r", encoding="utf-8") as f:
                            for i, line in enumerate(f):
                                if i >= 50:
                                    break
                                rows.append(json.loads(line))
                        if rows:
                            st.subheader("Sample of extracted claims")
                            st.dataframe(pd.DataFrame(rows))
                        else:
                            st.info(
                                "No claims were extracted (model returned empty lists)."
                            )

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
            help=(
                "How many passages to retrieve from each tradition "
                "for each claim."
            ),
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
                        st.write(
                            "**Number of claims verified:**",
                            meta["num_verifications"],
                        )
                        st.write("**Model:**", meta["model"])

                        rows: List[Dict[str, Any]] = []
                        with open(
                            meta["verification_path"], "r", encoding="utf-8"
                        ) as f:
                            for i, line in enumerate(f):
                                if i >= 50:
                                    break
                                rows.append(json.loads(line))
                        if rows:
                            st.subheader("Sample of verification results")
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

                            # Detailed evidence view for first few claims
                            st.subheader("Evidence details (first few claims)")
                            for _, row in df.head(5).iterrows():
                                header = (
                                    f"{row.get('claim_id')} – "
                                    f"{row.get('verdict_overall')} – "
                                    f"{(row.get('claim_text') or '')[:80]}..."
                                )
                                with st.expander(header):
                                    st.markdown("**Claim text (full):**")
                                    st.write(row.get("claim_text", ""))

                                    with st.expander(
                                        "Explain key terms in this claim"
                                    ):
                                        render_glossary_for_claim(
                                            row.get("claim_text", "")
                                        )

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
                                        st.write(
                                            "_No Christian evidence selected._"
                                        )
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
