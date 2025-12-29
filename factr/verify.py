# factr/verify.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .config import FactrConfig
from .kb import KBHit, kb_search

client = OpenAI()


@dataclass
class VerificationRecord:
    claim_id: str
    claim_text: str
    side: Optional[str]

    # Per-tradition verdicts
    verdict_overall: str
    verdict_islam: Optional[str]
    verdict_christian: Optional[str]
    confidence: float

    explanation: str
    created_utc: str

    # For UI we also keep a short list of evidence snippets
    evidence_islam: List[Dict[str, Any]]
    evidence_christian: List[Dict[str, Any]]


def _load_claims(cfg: FactrConfig) -> List[Dict[str, Any]]:
    claims_path = cfg.processed_dir / "CLAIMS.jsonl"
    if not claims_path.exists():
        raise FileNotFoundError(f"CLAIMS.jsonl not found at {claims_path}")
    claims: List[Dict[str, Any]] = []
    with claims_path.open("r", encoding="utf-8") as f:
        for line in f:
            claims.append(json.loads(line))
    return claims


def _format_hits(label: str, hits: List[KBHit]) -> str:
    """
    Render hits as numbered list for the prompt.
    """
    lines = [f"{label} evidence:"]
    if not hits:
        lines.append("  (no strong matches found)")
        return "\n".join(lines)

    for i, h in enumerate(hits, start=1):
        ref = h.ref or ""
        src = h.source or ""
        lines.append(
            f"[{i}] score={h.score:.3f} ref={ref} source={src}\n"
            f"    text={h.text}"
        )
    return "\n".join(lines)


#Beginning of Edit
def _norm_tradition_label(label: Optional[str]) -> Optional[str]:
    """
    Normalise a per-tradition label (Islam / Christian) to one of:

        "agrees", "disagrees", "divided", "insufficient", or None
    """
    if label is None:
        return None
    l = str(label).strip().lower()
    if not l:
        return None

    if l in {
        "supported",
        "support",
        "supports",
        "agree",
        "agrees",
        "agreement",
        "affirmed",
        "true",
        "yes",
    }:
        return "agrees"

    if l in {
        "contradicted",
        "contradicts",
        "disagree",
        "disagrees",
        "refuted",
        "refutes",
        "rejects",
        "denies",
        "false",
        "no",
    }:
        return "disagrees"

    if l in {"both", "mixed", "divided", "conflicted", "split"}:
        return "divided"

    # Everything else is treated as not strong enough
    return "insufficient"


def _norm_overall_label(label: Optional[str]) -> str:
    """
    Normalise the combined verdict to one of:

        "agreement", "doubtful", "conflicted", "insufficient"
    """
    if label is None:
        return "insufficient"
    l = str(label).strip().lower()
    if not l:
        return "insufficient"

    if l in {"agreement", "agrees", "supported", "support"}:
        return "agreement"

    if l in {"doubtful", "disagrees", "contradicted", "rejects", "against"}:
        return "doubtful"

    if l in {"conflicted", "conflict", "both", "mixed", "split"}:
        return "conflicted"

    return "insufficient"
#end of edit


def _call_gpt_for_verdict(
    claim_text: str,
    side: Optional[str],
    islam_hits: List[KBHit],
    christian_hits: List[KBHit],
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Ask GPT to compare the claim against the provided evidence and
    return a JSON verdict.
    """
    #Beginning of Edit
    # Updated prompt to:
    #   * use the new verdict names (agrees / disagrees / divided / insufficient)
    #   * define how the combined verdict (agreement / doubtful / conflicted /
    #     insufficient) should be chosen
    #   * be stricter about Islamic tawhid verses contradicting claims that
    #     rely on praying to or relying on Jesus instead of Allah.
    system_msg = (
        "You are a careful fact-checker for interfaith debates. "
        "You compare a claim to Islamic and Christian primary sources. "
        "Return ONLY valid JSON, no explanation outside JSON."
    )

    user_msg = f"""
Claim from a Christian–Muslim debate:

Claim text:
{claim_text}

Speaker side (if known): {side or "unknown"}

Below are passages retrieved from a knowledge base.

{_format_hits("Islamic", islam_hits)}

{_format_hits("Christian", christian_hits)}

Task:

Using ONLY the information in these passages (do not invent new facts), do three things:

1. For the **Islamic tradition**, decide whether the claim is:
   - "agrees": the Islamic evidence clearly affirms or fits the claim.
   - "disagrees": the Islamic evidence clearly teaches the opposite of the claim
     or rejects its implications. In particular, strong verses about exclusive
     worship of Allah and prohibiting calling on anyone else should be treated
     as *disagreeing* with claims that involve praying to, worshipping or
     ultimately relying on Jesus.
   - "divided": there is meaningful evidence both for and against the claim in
     this tradition.
   - "insufficient": the passages are too weak, too generic or too unrelated to
     decide either way.

2. For the **Christian tradition**, make the same judgement with the same four
   labels ("agrees", "disagrees", "divided", "insufficient").

3. Provide an **overall verdict** that summarises both traditions together,
   using these labels:
   - "agreement": at least one tradition clearly agrees with the claim and
     neither clearly disagrees.
   - "doubtful": at least one tradition clearly disagrees with the claim and
     neither clearly agrees.
   - "conflicted": one tradition agrees while the other disagrees (explicit
     inter-faith disagreement).
   - "insufficient": both sides are too weak, ambiguous or irrelevant to judge.

Return a single JSON object with:

- "verdict_overall": overall label ("agreement", "doubtful", "conflicted",
  or "insufficient").
- "verdict_islam": label for the Islamic evidence ("agrees", "disagrees",
  "divided", or "insufficient").
- "verdict_christian": label for the Christian evidence (same options as above).
- "confidence": number between 0 and 1 for your overall verdict.
- "explanation": a short explanation (2–5 sentences) referencing evidence
  numbers like [Islamic 2], [Christian 1].
- "used_islam_ids": list of integers (evidence numbers you relied on for the
  Islamic verdict).
- "used_christian_ids": list of integers (evidence numbers you relied on for
  the Christian verdict).

If the evidence is too weak to judge for a tradition, use "insufficient" with
low confidence and explain that the passages are not strong or relevant enough.
"""
    #end of edit

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    content = resp.choices[0].message.content.strip()

    # Parse JSON defensively
    try:
        data = json.loads(content)
    except Exception:
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1:
            snippet = content[first_brace : last_brace + 1]
            data = json.loads(snippet)
        else:
            raise ValueError(f"Model output not valid JSON:\n{content}")

    return data


def verify_claims(
    cfg: Optional[FactrConfig] = None,
    max_claims: Optional[int] = None,
    top_k_evidence: int = 5,
    model: str = "gpt-4.1-mini",
) -> Tuple[Path, Dict[str, Any]]:
    """
    Main entry point:

    - Load CLAIMS.jsonl
    - For each claim (optionally limited by max_claims):
        - Retrieve Islamic + Christian evidence from KB
        - Call GPT for verdict
    - Write VERIFICATION.jsonl
    - Write VERIFY_LAST.json snapshot
    """
    cfg = cfg or FactrConfig()

    claims = _load_claims(cfg)
    if max_claims is not None:
        claims = claims[:max_claims]

    records: List[VerificationRecord] = []

    for idx, claim in enumerate(claims):
        claim_text = claim["text"]
        claim_id = claim["claim_id"]
        side = claim.get("side")

        print(f"[VERIFY] Claim {idx+1}/{len(claims)}: {claim_id}")

        islam_hits = kb_search(
            claim_text,
            cfg=cfg,
            top_k=top_k_evidence,
            tradition="Islam",
        )
        christian_hits = kb_search(
            claim_text,
            cfg=cfg,
            top_k=top_k_evidence,
            tradition="Christianity",
        )

        try:
            verdict = _call_gpt_for_verdict(
                claim_text, side, islam_hits, christian_hits, model=model
            )
        except Exception as e:
            print(f"  ! GPT error on claim {claim_id}: {e}")
            continue

        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        used_islam_ids = verdict.get("used_islam_ids") or []
        used_christian_ids = verdict.get("used_christian_ids") or []

        #Beginning of Edit
        # Normalise label strings and expose group_key so the UI can
        # fetch related commentary from the KB if needed.
        verdict_overall = _norm_overall_label(verdict.get("verdict_overall"))
        verdict_islam = _norm_tradition_label(verdict.get("verdict_islam"))
        verdict_christian = _norm_tradition_label(verdict.get("verdict_christian"))
        confidence = float(verdict.get("confidence") or 0.0)

        def select_hits(hits: List[KBHit], used_ids: List[int]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for i in used_ids:
                if 1 <= i <= len(hits):
                    h = hits[i - 1]
                    out.append(
                        {
                            "id": i,
                            "score": h.score,
                            "ref": h.ref,
                            "text": h.text,
                            "source": h.source,
                            "tradition": h.tradition,
                            "group_key": h.group_key,
                        }
                    )
            return out
        #end of edit

        rec = VerificationRecord(
            claim_id=claim_id,
            claim_text=claim_text,
            side=side,
            verdict_overall=verdict_overall,
            verdict_islam=verdict_islam,
            verdict_christian=verdict_christian,
            confidence=confidence,
            explanation=verdict.get("explanation", "").strip(),
            created_utc=ts,
            evidence_islam=select_hits(islam_hits, used_islam_ids),
            evidence_christian=select_hits(christian_hits, used_christian_ids),
        )

        records.append(rec)

    processed = cfg.processed_dir
    processed.mkdir(parents=True, exist_ok=True)

    ver_path = processed / "VERIFICATION.jsonl"
    with ver_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    meta = {
        "verification_path": str(ver_path),
        "num_verifications": len(records),
        "max_claims": max_claims,
        "top_k_evidence": top_k_evidence,
        "model": model,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    snap_path = cfg.snapshots_dir / "VERIFY_LAST.json"
    snap_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ver_path, meta
