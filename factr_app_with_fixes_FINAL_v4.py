from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime  # <- for logging timestamps

import html
import io
import re

import pandas as pd
import streamlit as st
import uuid

from factr.config import FactrConfig
from factr.ingest import ingest_youtube
from factr.asr import transcribe_audio
from factr.claims import extract_claims
from factr.verify import verify_claims
from factr.kb import kb_commentary_for_group
# from glossary import render_glossary_for_claim
from openai import OpenAI
from ui_faq import render_faq


LOG_FILE = Path("factr_glossary_errors.log")
FEEDBACK_FILE = Path("factr_verdict_feedback.jsonl")
_glossary_client = OpenAI()

# create and store the config once
def get_cfg() -> FactrConfig:
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = FactrConfig()
    return st.session_state["cfg"]

# =====================================================================
# Glossary helpers
# =====================================================================

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

def log_verdict_feedback(**fields: Any) -> None:
    """
    Append one feedback event to FEEDBACK_FILE as JSONL.

    fields might include:
      - claim_id, claim_text
      - verdict_islam, verdict_christian, verdict_overall
      - confidence
      - reaction: "thumbs_up" / "thumbs_down"
      - comment: optional free text
    """
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **fields,
    }
    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # feedback must never crash the app
        pass


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
# Timestamp helpers
# =====================================================================

def _fmt_hms(seconds: Optional[float]) -> str:
    """Format seconds as H:MM:SS or M:SS. Returns '' if seconds is None."""
    if seconds is None:
        return ""
    try:
        s = float(seconds)
    except Exception:
        return ""
    if s < 0:
        s = 0.0
    total = int(round(s))
    h = total // 3600
    m = (total % 3600) // 60
    sec = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _coerce_time_seconds(v: Any) -> Optional[float]:
    """Coerce possible time formats into seconds."""
    if v is None:
        return None
    # already numeric
    try:
        fv = float(v)
    except Exception:
        # strings like "00:01:23" or "1:23"
        if isinstance(v, str) and ":" in v:
            parts = [p.strip() for p in v.split(":")]
            try:
                nums = [float(p) for p in parts]
            except Exception:
                return None
            if len(nums) == 3:
                h, m, s = nums
                return h * 3600 + m * 60 + s
            if len(nums) == 2:
                m, s = nums
                return m * 60 + s
        return None
    # if looks like ms
    if fv > 10_000:  # > ~2.7h in seconds, likely ms
        return fv / 1000.0
    return fv

def load_claim_time_cache(processed_dir: Path) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Build a claim_id -> (start_sec, end_sec) cache.

    FACTR claim artefacts can store time in two ways:
      1) direct numeric fields (t_start/t_end, start/end, etc.)
      2) an utterance index span (utterance_range=[i0,i1]) which must be resolved
         via UTTERANCES.parquet (utterance i -> start/end seconds).

    The app uses this cache to show timestamps in the UI and include them in CSV exports.
    Best-effort: if dependencies for reading parquet are missing, we fall back to whatever
    time fields exist in the claim artefacts.
    """
    candidates = [
        processed_dir / "CLAIMS_raw.jsonl",
        processed_dir / "CLAIMS.jsonl",
        processed_dir / "CLAIMS_extracted.jsonl",
        processed_dir / "CLAIMS_raw.json",
        processed_dir / "CLAIMS.json",
    ]

    # Optional utterance timeline (index -> start/end sec)
    utterances_fp = processed_dir / "UTTERANCES.parquet"
    utter_df = None
    start_col = end_col = None
    if utterances_fp.exists():
        try:
            import pandas as _pd  # noqa: F401
            # pandas parquet requires pyarrow or fastparquet in most environments
            utter_df = _pd.read_parquet(utterances_fp)
            cols = [c.lower() for c in utter_df.columns]
            # common column name variants
            def _pick(col_candidates):
                for cand in col_candidates:
                    if cand in cols:
                        return utter_df.columns[cols.index(cand)]
                return None
            start_col = _pick(["start", "start_time", "start_sec", "t_start", "start_s"])
            end_col   = _pick(["end", "end_time", "end_sec", "t_end", "end_s"])
            # if we can't detect, treat as unavailable
            if start_col is None or end_col is None:
                utter_df = None
        except Exception:
            utter_df = None

    cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    def _store(cid: Any, t0: Optional[float], t1: Optional[float], claim_text: Any = None) -> None:
        """Store by claim_id and (optionally) by normalised claim text."""
        if cid is not None:
            cache[str(cid)] = (t0, t1)
        # secondary key: normalised claim text (helps when claim_id formats differ across stages)
        key_txt = _norm_claim_text(claim_text)
        if key_txt:
            cache[f"txt:{key_txt}"] = (t0, t1)

    for fp in candidates:
        if not fp.exists():
            continue
        try:
            if fp.suffix.lower() == ".jsonl":
                lines = fp.read_text(encoding="utf-8").splitlines()
                items = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
            else:
                obj = json.loads(fp.read_text(encoding="utf-8"))
                items = obj if isinstance(obj, list) else (obj.get("claims") or [])

            for obj in items:
                cid = obj.get("claim_id") or obj.get("id") or obj.get("claim_idx")
                # 1) direct timestamps
                t0 = _coerce_time_seconds(
                    obj.get("t_start") or obj.get("start") or obj.get("start_time") or obj.get("start_sec") or
                    (obj.get("provenance") or {}).get("start") or (obj.get("provenance") or {}).get("start_sec")
                )
                t1 = _coerce_time_seconds(
                    obj.get("t_end") or obj.get("end") or obj.get("end_time") or obj.get("end_sec") or
                    (obj.get("provenance") or {}).get("end") or (obj.get("provenance") or {}).get("end_sec")
                )

                # 2) resolve utterance_range using UTTERANCES.parquet (if available)
                if (t0 is None or t1 is None) and utter_df is not None:
                    ur = obj.get("utterance_range") or (obj.get("provenance") or {}).get("utterance_range")
                    if isinstance(ur, (list, tuple)) and len(ur) == 2:
                        try:
                            i0 = int(ur[0])
                            i1 = int(ur[1])
                            # treat i1 as inclusive (safer for [0,24] style ranges)
                            lo = max(0, min(i0, i1))
                            hi = max(0, max(i0, i1))
                            hi = min(hi, len(utter_df) - 1)
                            span = utter_df.iloc[lo:hi+1]
                            t0 = _coerce_time_seconds(span.iloc[0][start_col])
                            t1 = _coerce_time_seconds(span.iloc[-1][end_col])
                        except Exception:
                            pass

                _store(cid, t0, t1, claim_text=obj.get('claim_text') or obj.get('text') or obj.get('claim'))

        except Exception:
            continue

        # Prefer the first artefact found that yields any timestamps.
        if cache:
            break

    return cache

def extract_claim_times(rec: Dict[str, Any], time_cache: Optional[Dict[Any, Tuple[Optional[float], Optional[float]]]] = None) -> Tuple[Optional[float], Optional[float]]:
    """Best-effort extraction of start/end time fields from a verification record."""
    # Common field names across pipelines/notebooks
    start_keys = ["t_start", "claim_start_sec", "claim_start", "start_sec", "start_seconds", "start", "start_time", "utterance_start"]
    end_keys = ["t_end", "claim_end_sec", "claim_end", "end_sec", "end_seconds", "end", "end_time", "utterance_end"]

    def _from_dict(d: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        t0 = next((_coerce_time_seconds(d.get(k)) for k in start_keys if d.get(k) is not None), None)
        t1 = next((_coerce_time_seconds(d.get(k)) for k in end_keys if d.get(k) is not None), None)
        return t0, t1

    t0, t1 = _from_dict(rec)

    # nested provenance/meta
    if t0 is None and t1 is None:
        for nk in ["provenance", "claim_provenance", "claim_meta", "meta", "metadata"]:
            nd = rec.get(nk)
            if isinstance(nd, dict):
                t0, t1 = _from_dict(nd)
                if t0 is not None or t1 is not None:
                    break

    # time cache lookup by claim_id
    if (t0 is None and t1 is None) and time_cache:
        # 1) primary key: claim_id
        cid = rec.get("claim_id") or rec.get("id") or rec.get("claim_idx")
        if cid is not None:
            key = str(cid)
            if key in time_cache:
                t0, t1 = time_cache[key]

        # 2) fallback key: normalised claim text (helps when IDs are regenerated)
        if (t0 is None and t1 is None):
            ctext = rec.get("claim_text") or rec.get("claim") or rec.get("text")
            tkey = _norm_claim_text(ctext)
            if tkey:
                k2 = f"txt:{tkey}"
                if k2 in time_cache:
                    t0, t1 = time_cache[k2]


    return t0, t1

def claim_time_label(rec: Dict[str, Any], time_cache: Optional[Dict[Any, Tuple[Optional[float], Optional[float]]]] = None) -> str:
    t0, t1 = extract_claim_times(rec, time_cache=time_cache)
    a = _fmt_hms(t0)
    b = _fmt_hms(t1)
    if a and b:
        return f"{a}–{b}"
    if a:
        return a
    return ""


def pretty_source(src: str) -> str:
    """
    Convert internal dataset/source identifiers into user-facing references.
    - strips local path prefixes (e.g. 'spa5k/')
    - removes '(local mirror)'
    - keeps the most informative tail component
    """
    if not src:
        return ""
    s = str(src)
    s = s.replace("(local mirror)", "").strip()
    s = re.sub(r"\s+", " ", s).strip()
    # drop leading path components
    if "/" in s:
        s = s.split("/")[-1]
    # common normalisations
    s = s.replace("tafsir_api", "Tafsir (source)")  # neutral, still honest
    return s.strip()

def clean_reference(ref: str) -> str:
    """Remove internal dataset/debug fragments from a reference string."""
    if not ref:
        return ""
    r = str(ref)
    # Remove internal parentheses like "(spa5k/tafsir_api (local mirror))"
    r = re.sub(r"\s*\([^)]*(local mirror|tafsir_api|spa5k|semarketi|quranjson)[^)]*\)\s*", " ", r, flags=re.IGNORECASE)
    r = re.sub(r"\s+", " ", r).strip()
    return r


def _norm_claim_text(s: Any) -> str:
    """Normalise claim text for fuzzy key matching (best-effort)."""
    if not s:
        return ""
    t = str(s).strip().lower()
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    # remove most punctuation (keep apostrophes inside words)
    t = re.sub(r"[^a-z0-9\s']", "", t)
    return t


def render_plain_text_block(text: str) -> None:
    """Render text without Markdown interpretation (avoids accidental code/highlight colours)."""
    if not text:
        return
    safe = html.escape(text)
    st.markdown(f"<div style='white-space:pre-wrap;font-family:inherit;line-height:1.45'>{safe}</div>", unsafe_allow_html=True)


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

    # Timestamp (if available)
    time_cache = st.session_state.get("time_cache")
    ts = rec.get("claim_time") or claim_time_label(rec, time_cache=time_cache)
    if ts:
        st.caption(f"🕒 {ts}")

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
    
    # --- User feedback on this verdict --------------------------------------
    feedback_key = f"feedback_claim_{rec.get('claim_id', idx)}"
    with st.expander("Feedback on this verdict (optional)"):
        st.markdown(
            "Help improve FACTR by telling us whether this verdict seems reasonable. "
            "Your feedback will be logged anonymously for research."
        )

        # Get current pilot ID (if set in sidebar)
        user_tag = st.session_state.get("user_tag")

        # Quick thumbs up / down
        fb_cols = st.columns(2)
        with fb_cols[0]:
            if st.button("👍 Verdict seems reasonable", key=f"{feedback_key}_up"):
                log_verdict_feedback(
                    claim_id=rec.get("claim_id", idx),
                    claim_text=claim_text,
                    verdict_islam=verdict_islam,
                    verdict_christian=verdict_christian,
                    verdict_overall=verdict_overall,
                    confidence=confidence,
                    reaction="thumbs_up",
                    comment=None,
                    user_tag=user_tag,
                )
                st.success("Thanks – feedback recorded.")

        with fb_cols[1]:
            if st.button("👎 Verdict seems wrong / unclear", key=f"{feedback_key}_down"):
                log_verdict_feedback(
                    claim_id=rec.get("claim_id", idx),
                    claim_text=claim_text,
                    verdict_islam=verdict_islam,
                    verdict_christian=verdict_christian,
                    verdict_overall=verdict_overall,
                    confidence=confidence,
                    reaction="thumbs_down",
                    comment=None,
                    user_tag=user_tag,
                )
                st.success("Thanks – feedback recorded. "
                           "You can also add a short comment below if you wish.")

        # Always-available comment box
        comment = st.text_area(
            "Optional: what feels wrong or confusing about this verdict?",
            key=f"{feedback_key}_comment",
        )
        if st.button("Submit comment", key=f"{feedback_key}_submit"):
            cleaned = (comment or "").strip() or None
            if cleaned:
                log_verdict_feedback(
                    claim_id=rec.get("claim_id", idx),
                    claim_text=claim_text,
                    verdict_islam=verdict_islam,
                    verdict_christian=verdict_christian,
                    verdict_overall=verdict_overall,
                    confidence=confidence,
                    reaction="comment",
                    comment=cleaned,
                    user_tag=user_tag,
                )
                st.success("Thanks – your comment was recorded.")
            else:
                st.info("Please type a comment before submitting, or just use the buttons above.")


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
                ref = clean_reference(ev.get("ref") or "")
                text = ev.get("text") or ""
                label = f"[Islamic {ev_id}]" if ev_id is not None else "[Islamic]"

                src = pretty_source(ev.get("source") or ev.get("dataset") or ev.get("collection") or "")
                page = ev.get("page") or ev.get("page_no") or ev.get("loc") or ""
                src_bits = " | ".join([b for b in [src, (f"p.{page}" if page else "")] if b])
                src_str = f" _({src_bits})_" if src_bits else ""
                st.markdown(f"- **{label} {ref}**{src_str}")
                render_plain_text_block(text)

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

                                c_src = pretty_source(getattr(c, "source", "") or getattr(c, "dataset", "") or "")
                                c_page = getattr(c, "page", "") or getattr(c, "page_no", "") or ""
                                c_bits = " | ".join([b for b in [c_src, (f"p.{c_page}" if c_page else "")] if b])
                                c_hdr = f"{c_ref}" + (f" ({c_bits})" if c_bits else "")
                                st.markdown(f"**{c_hdr}**")
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
                ref = clean_reference(ev.get("ref") or "")
                text = ev.get("text") or ""
                label = f"[Christian {ev_id}]" if ev_id is not None else "[Christian]"

                src = pretty_source(ev.get("source") or ev.get("dataset") or ev.get("collection") or "")
                page = ev.get("page") or ev.get("page_no") or ev.get("loc") or ""
                src_bits = " | ".join([b for b in [src, (f"p.{page}" if page else "")] if b])
                src_str = f" _({src_bits})_" if src_bits else ""
                st.markdown(f"- **{label} {ref}**{src_str}")
                render_plain_text_block(text)

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

                                c_src = pretty_source(getattr(c, "source", "") or getattr(c, "dataset", "") or "")
                                c_page = getattr(c, "page", "") or getattr(c, "page_no", "") or ""
                                c_bits = " | ".join([b for b in [c_src, (f"p.{c_page}" if c_page else "")] if b])
                                c_hdr = f"{c_ref}" + (f" ({c_bits})" if c_bits else "")
                                st.markdown(f"**{c_hdr}**")
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

def render_oneclick_results(rows: List[Dict[str, Any]], cfg: Optional["FactrConfig"] = None) -> None:
    """
    Render the \'Debate analysis – claims and verdicts\' section
    (table download + per-claim cards) from a list of verification rows.
    """
    if not rows:
        st.info("No verification records were produced.")
        return

    # Build a claim_id -> (start,end) cache from any available claim artefact
    time_cache = None
    try:
        if cfg is not None:
            time_cache = load_claim_time_cache(cfg.processed_dir)
    except Exception:
        time_cache = None

    # Add timestamp fields (best-effort) so they appear both on-screen and in the CSV
    rows_out: List[Dict[str, Any]] = []
    for rec in rows:
        rec2 = dict(rec)
        t0, t1 = extract_claim_times(rec, time_cache=time_cache)
        rec2["claim_start_sec"] = t0
        rec2["claim_end_sec"] = t1
        rec2["claim_time"] = claim_time_label(rec, time_cache=time_cache)
        rows_out.append(rec2)

    df = pd.DataFrame(rows_out)

    # Persist last results so mobile refresh / reconnect can still display them
    try:
        if cfg is not None:
            last_fp = cfg.processed_dir / f"STREAMLIT_LAST_RESULTS_{st.session_state.get('session_id', 'nosid')}.json"
            last_fp.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

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

    for idx, rec in enumerate(rows_out):
        render_claim_card(idx, rec)


# =====================================================================
# MAIN APP
# =====================================================================

def main() -> None:
    st.set_page_config(page_title="FACTR – Debate Analyzer", layout="centered")

    # Style: make the main Analyse button red (primary buttons only)
    st.markdown("""
    <style>
    button[data-testid="baseButton-primary"] {
        background-color: #d32f2f !important;
        border: 1px solid #d32f2f !important;
        color: white !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #b71c1c !important;
        border: 1px solid #b71c1c !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("FACTR – Debate Analyzer")

    # -----------------------------------------------------------------
    # Per-user session ID (prevents users seeing each other's last run)
    # Persisted in the URL so iOS refreshes keep the same session.
    # -----------------------------------------------------------------
    try:
        sid = st.query_params.get("sid", None)
    except Exception:
        sid = None

    if not sid:
        sid = uuid.uuid4().hex
        try:
            st.query_params["sid"] = sid
        except Exception:
            pass

    st.session_state["session_id"] = sid


    # Sidebar help / FAQ
    with st.sidebar:
        st.subheader("Help & FAQ")
        render_faq()


    # Session state for cached results + run log
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
    if "run_log" not in st.session_state:
        st.session_state["run_log"] = []

    # IMPORTANT: use the session config (do NOT recreate a new FactrConfig)
    cfg = st.session_state.get("cfg", FactrConfig())
    # Build / refresh claim timestamp cache (used for UI timestamps)
    if (st.session_state.get("time_cache") is None) or (st.session_state.get("time_cache_dir") != str(cfg.processed_dir)):
        st.session_state["time_cache"] = load_claim_time_cache(cfg.processed_dir)
        st.session_state["time_cache_dir"] = str(cfg.processed_dir)
    log_file = cfg.processed_dir / "ONECLICK_RUN.log"

    # If the browser refreshed (common on iOS), try to recover the last completed results
    if st.session_state.get("last_results") is None:
        try:
            last_fp = cfg.processed_dir / f"STREAMLIT_LAST_RESULTS_{st.session_state.get('session_id', 'nosid')}.json"
            if last_fp.exists():
                st.session_state["last_results"] = json.loads(last_fp.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Simple helper to log messages both to session_state and to file
    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        st.session_state["run_log"].append(line)
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
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

    
    # Pilot testing: capture user ID (optional)
    user_tag = st.text_input("Pilot ID (e.g. S1, S2)", value="", max_chars=10, key="pilot_id")
    st.session_state["user_tag"] = (user_tag or "").strip() or None

    # Runtime / cost controls (defaults: 2, 5, 5, 0.55)
    if "ui_defaults" not in st.session_state:
        st.session_state["ui_defaults"] = {"max_chunks": 2, "max_claims": 5, "top_k": 5, "min_similarity": 0.55}

    d = st.session_state["ui_defaults"]

    col1, col2, col3 = st.columns(3)
    with col1:
        max_chunks_all = st.number_input(
            "Max transcript chunks (0 = all)",
            min_value=0,
            max_value=200,
            value=int(d["max_chunks"]),
            step=1,
            help="Limits how much transcript is processed (runtime/cost control).",
            key="max_chunks_all",
        )
    with col2:
        max_claims_all = st.number_input(
            "Max claims to verify (0 = all)",
            min_value=0,
            max_value=300,
            value=int(d["max_claims"]),
            step=1,
            help="Caps how many extracted claims are verified (runtime/cost control).",
            key="max_claims_all",
        )
    with col3:
        top_k_all = st.number_input(
            "Top-K evidence per tradition",
            min_value=1,
            max_value=20,
            value=int(d["top_k"]),
            step=1,
            help="How many KB passages to retrieve for each claim.",
            key="top_k_all",
        )

    min_similarity_floor = st.slider(
        "Min similarity floor",
        min_value=0.0,
        max_value=1.0,
        value=float(d["min_similarity"]),
        step=0.01,
        help="Evidence below this similarity is treated as weak/ignored.",
        key="min_similarity_floor",
    )

    def _reset_ui_defaults() -> None:
        st.session_state["ui_defaults"] = {"max_chunks": 2, "max_claims": 5, "top_k": 5, "min_similarity": 0.55}
        # Reset widget-backed values (safe inside callback)
        st.session_state["max_chunks_all"] = 2
        st.session_state["max_claims_all"] = 5
        st.session_state["top_k_all"] = 5
        st.session_state["min_similarity_floor"] = 0.55

    st.button("Reset settings to defaults", key="reset_ui_defaults", on_click=_reset_ui_defaults)


    analyse_button = st.button("Analyse debate (end-to-end)", type="primary")

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
                    render_oneclick_results(rows, cfg=cfg)

    # If we have previous results and the button is NOT pressed on this rerun,
    # show the cached results so the page does not "forget" everything.
    if not analyse_button and st.session_state.get("last_results"):
        st.markdown("---")
        render_oneclick_results(st.session_state["last_results"], cfg=cfg)

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