# factr/claims.py

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

from .config import FactrConfig

client = OpenAI()


@dataclass
class ClaimRecord:
    claim_id: str
    chunk_id: str
    approx_start: Optional[float]
    approx_end: Optional[float]
    speaker: Optional[str]
    side: Optional[str]        # e.g. "Christian", "Muslim", "Neutral"
    category: Optional[str]    # e.g. "theological", "historical"
    text: str
    created_utc: str


def _load_utterances(cfg: FactrConfig) -> pd.DataFrame:
    utt_path = cfg.processed_dir / "UTTERANCES.parquet"
    if not utt_path.exists():
        raise FileNotFoundError(f"UTTERANCES.parquet not found: {utt_path}")
    df = pd.read_parquet(utt_path)
    # Ensure ordering
    if "start" in df.columns:
        df = df.sort_values("start").reset_index(drop=True)
    return df


def _chunk_utterances(
    df: pd.DataFrame,
    max_chars: int = 3500,
) -> List[Dict[str, Any]]:
    """
    Make chunks of the transcript ~max_chars long.

    Each chunk text looks like:
      [00:10–00:20] SPEAKER_00: some text.
    """
    chunks: List[Dict[str, Any]] = []
    cur_lines: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None

    def flush_chunk():
        nonlocal cur_lines, cur_start, cur_end
        if not cur_lines:
            return
        chunk_id = uuid.uuid4().hex[:8]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start": cur_start,
                "end": cur_end,
                "text": "\n".join(cur_lines),
            }
        )
        cur_lines = []
        cur_start = None
        cur_end = None

    def fmt_time(t: float) -> str:
        m = int(t // 60)
        s = int(t % 60)
        return f"{m:02d}:{s:02d}"

    cur_len = 0
    for _, row in df.iterrows():
        start = float(row.get("start", 0.0) or 0.0)
        end = float(row.get("end", start) or start)
        speaker = str(row.get("speaker", "SPEAKER_00"))
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        line = f"[{fmt_time(start)}–{fmt_time(end)}] {speaker}: {text}"
        if cur_len + len(line) > max_chars and cur_lines:
            flush_chunk()
            cur_len = 0

        if cur_start is None:
            cur_start = start
        cur_end = end
        cur_lines.append(line)
        cur_len += len(line)

    flush_chunk()
    return chunks


def _call_openai_for_chunk(
    chunk: Dict[str, Any],
    model: str = "gpt-4.1-mini",
) -> List[Dict[str, Any]]:
    """
    Ask the model to extract claims as a JSON list of objects.

    Each object should look like:
      {
        "text": "...",
        "side": "Christian" | "Muslim" | "Neutral",
        "category": "theological" | "historical" | "other",
        "approx_start": 123.4,
        "approx_end": 130.0,
        "speaker": "SPEAKER_00"
      }
    """
    #Beginning of Edit
    # Updated prompt so that claim text is **always returned in English**,
    # even when the original audio contains Arabic only.
    system_msg = (
        "You are an expert analyst extracting atomic claims from a "
        "religious debate transcript between a Christian and a Muslim. "
        "Return ONLY valid JSON, no prose."
    )

    user_msg = f"""
Transcript chunk (ID={chunk['chunk_id']}):

{chunk['text']}

Task:
- Extract the most important *distinct* claims made in this chunk.
- A claim is a specific statement that could, in principle, be checked
  against primary sources (Bible, Qur'an, early Christian writings, etc.).
- Ignore fluff, interruptions, or pure questions with no assertion.

Return a JSON array of objects, each with:

- "text": the claim text, rewritten concisely but faithfully.
- "side": "Christian", "Muslim", "Both", or "Neutral".
- "category": one of "theological", "historical", "scriptural", or "other".
- "approx_start": approximate start time in seconds (float), if known.
- "approx_end": approximate end time in seconds (float), if known.
- "speaker": speaker label if obvious (like "SPEAKER_00").

If no clear claims, return [].

Very important:
- Always write the "text" for each claim in clear **English**, even if the
  original statement was partly or fully in Arabic (or another language).
- You may keep short Arabic religious terms (for example: "Isa", "Insha'Allah")
  inside the sentence, but the overall sentence must be understandable to an
  English-speaking reader.
- Do **not** return any claim whose "text" is only Arabic with no English
  explanation. If you cannot reasonably translate or interpret a purely Arabic
  sentence, simply omit that claim instead of returning it.
"""
    #end of edit

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )

    content = resp.choices[0].message.content.strip()

    # The model should return JSON, but be defensive
    try:
        data = json.loads(content)
    except Exception:
        # Try a crude fallback: find first JSON-like block
        first_brace = content.find("[")
        last_brace = content.rfind("]")
        if first_brace != -1 and last_brace != -1:
            snippet = content[first_brace : last_brace + 1]
            data = json.loads(snippet)
        else:
            raise ValueError(f"Model output not valid JSON:\n{content}")

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got: {type(data)}")

    return data


def extract_claims(
    cfg: Optional[FactrConfig] = None,
    model: str = "gpt-4.1-mini",
    max_chunks: Optional[int] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    High-level entry point:

    - Load utterances
    - Chunk them
    - Call OpenAI on each chunk
    - Write CLAIMS.jsonl
    - Write CLAIMS_LAST.json snapshot

    Returns: (claims_path, meta_dict)
    """
    cfg = cfg or FactrConfig()
    df = _load_utterances(cfg)
    chunks = _chunk_utterances(df)

    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    claims: List[ClaimRecord] = []

    for idx, chunk in enumerate(chunks):
        print(f"[CLAIMS] Processing chunk {idx+1}/{len(chunks)} (id={chunk['chunk_id']})")
        try:
            chunk_claims = _call_openai_for_chunk(chunk, model=model)
        except Exception as e:
            print(f"  ! Error on chunk {chunk['chunk_id']}: {e}")
            continue

        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        for i, c in enumerate(chunk_claims):
            claim_id = f"{chunk['chunk_id']}_{i:03d}"
            rec = ClaimRecord(
                claim_id=claim_id,
                chunk_id=chunk["chunk_id"],
                approx_start=float(c.get("approx_start") or 0.0)
                if c.get("approx_start") is not None
                else None,
                approx_end=float(c.get("approx_end") or 0.0)
                if c.get("approx_end") is not None
                else None,
                speaker=c.get("speaker"),
                side=c.get("side"),
                category=c.get("category"),
                text=str(c.get("text", "")).strip(),
                created_utc=ts,
            )
            if rec.text:
                claims.append(rec)

    processed_dir = cfg.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    claims_path = processed_dir / "CLAIMS.jsonl"
    with claims_path.open("w", encoding="utf-8") as f:
        for rec in claims:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    meta = {
        "claims_path": str(claims_path),
        "num_claims": len(claims),
        "num_chunks": len(chunks),
        "model": model,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    snapshot_path = cfg.snapshots_dir / "CLAIMS_LAST.json"
    snapshot_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return claims_path, meta
