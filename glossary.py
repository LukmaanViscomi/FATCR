# glossary.py
# =====================================================================
# AI-powered glossary for FACTR
#
# For each claim:
#   - call the OpenAI model once with the claim text
#   - get back up to N key terms + definitions as JSON
#   - cache the result in memory for this session
#
# factr_app.py imports: render_glossary_for_claim(claim_text: str)
# =====================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import json

import streamlit as st
from openai import OpenAI


@dataclass
class GlossaryTerm:
    term: str
    definition: str
    source: str  # e.g. "model"


# In-memory cache so we don't call the model repeatedly for the same claim
_GLOSSARY_CACHE: Dict[str, List[GlossaryTerm]] = {}


def _call_model_for_glossary(
    claim_text: str,
    max_terms: int = 6,
) -> List[GlossaryTerm]:
    """
    Ask the OpenAI model to identify key terms in the claim and provide
    short, neutral definitions.

    If anything goes wrong (no key, network issue, JSON parse error),
    we fail silently and return an empty list.
    """
    claim_text = (claim_text or "").strip()
    if not claim_text:
        return []

    try:
        # Uses OPENAI_API_KEY from your environment
        client = OpenAI()
    except Exception:
        return []

    prompt = f"""
You are helping with an interfaith theology debate tool.

Given this claim:

\"\"\"{claim_text}\"\"\"

1. Identify up to {max_terms} important theological or technical terms
   or short phrases that a non-expert might not understand.
2. For each term, give a brief, neutral, one-sentence definition.

Return ONLY a JSON object with this structure:

{{
  "terms": [
    {{"term": "...", "definition": "..."}},
    ...
  ]
}}
"""

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            response_format={"type": "json_object"},
            max_output_tokens=400,
        )
    except Exception:
        return []

    try:
        # Take the first text block and parse as JSON
        content = resp.output[0].content[0].text
        data = json.loads(content)
    except Exception:
        return []

    out: List[GlossaryTerm] = []
    for item in data.get("terms", []):
        term = (item.get("term") or "").strip()
        definition = (item.get("definition") or "").strip()
        if not term or not definition:
            continue
        out.append(
            GlossaryTerm(
                term=term,
                definition=definition,
                source="model",
            )
        )

    return out


def build_glossary_for_claim(
    claim_text: str,
    max_terms: int = 6,
) -> List[GlossaryTerm]:
    """
    Main builder: returns a list of GlossaryTerm for this claim.

    - Uses an in-memory cache so repeated re-runs don't re-call the API
      for the same claim text in one Streamlit session.
    """
    claim_text = (claim_text or "").strip()
    if not claim_text:
        return []

    cache_key = claim_text
    if cache_key in _GLOSSARY_CACHE:
        return _GLOSSARY_CACHE[cache_key]

    terms = _call_model_for_glossary(claim_text, max_terms=max_terms)
    _GLOSSARY_CACHE[cache_key] = terms
    return terms


def render_glossary_for_claim(claim_text: str) -> None:
    """
    Streamlit UI helper: called from factr_app.render_claim_card().

    Shows a small list of AI-suggested terms + definitions. This is what
    factr_app.py imports and calls in the claim card.
    """
    claim_text = (claim_text or "").strip()
    if not claim_text:
        st.info("No claim text available for this claim.")
        return

    terms = build_glossary_for_claim(claim_text)

    if not terms:
        st.info(
            "No glossary terms could be generated for this claim yet. "
            "Check your OPENAI_API_KEY or try again later."
        )
        return

    for t in terms:
        st.markdown(f"**{t.term}** _(AI-suggested)_")
        st.write(t.definition)
        st.markdown("---")

