# factr/kb.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import FactrConfig


@dataclass
class KBHit:
    score: float
    text: str
    ref: Optional[str]
    tradition: Optional[str]
    genre: Optional[str]
    source: Optional[str]
    group_key: Optional[str]


# Module-level cache so we only load once
_MODEL: Optional[SentenceTransformer] = None
_KB_EMB: Optional[np.ndarray] = None
_KB_META: Optional[List[Dict[str, Any]]] = None
_KB_CFG: Optional[Dict[str, Any]] = None


def _load_kb(cfg: FactrConfig) -> None:
    """
    Load KB embeddings + passages into memory (once per process).
    """
    global _MODEL, _KB_EMB, _KB_META, _KB_CFG

    if _KB_EMB is not None and _KB_META is not None and _MODEL is not None:
        return

    processed = cfg.processed_dir

    emb_path = processed / "KB_embeddings.npy"
    meta_path = processed / "KB_embeddings.meta.json"
    passages_path = processed / "KB_passages.jsonl"

    if not emb_path.exists():
        raise FileNotFoundError(f"KB_embeddings.npy not found at {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"KB_embeddings.meta.json not found at {meta_path}")
    if not passages_path.exists():
        raise FileNotFoundError(f"KB_passages.jsonl not found at {passages_path}")

    # Load embedding meta
    _KB_CFG = json.loads(meta_path.read_text(encoding="utf-8"))
    model_name = _KB_CFG["model_name"]
    normalized = bool(_KB_CFG.get("normalized", True))

    # Load embeddings
    _KB_EMB = np.load(emb_path)
    if _KB_EMB.dtype != np.float32:
        _KB_EMB = _KB_EMB.astype("float32")

    # Load passage records
    _KB_META = []
    with passages_path.open("r", encoding="utf-8") as f:
        for line in f:
            _KB_META.append(json.loads(line))

    # Load sentence-transformer model
    _MODEL = SentenceTransformer(model_name)
    _MODEL_NORMALIZED = normalized  # not used externally; just hint


def kb_search(
    query: str,
    cfg: Optional[FactrConfig] = None,
    top_k: int = 5,
    tradition: Optional[str] = None,
) -> List[KBHit]:
    """
    Search the KB for passages relevant to the query.

    Args:
        query: natural language string (e.g. a claim)
        cfg: FactrConfig instance (optional)
        top_k: number of hits
        tradition: None (all), or "Islam", "Christianity", etc.

    Returns:
        List[KBHit] sorted by descending score.
    """
    cfg = cfg or FactrConfig()
    _load_kb(cfg)

    assert _MODEL is not None
    assert _KB_EMB is not None
    assert _KB_META is not None

    # Embed query (normalized to match KB)
    q = _MODEL.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype("float32")

    # Optional filter by tradition
    if tradition:
        mask = [
            (rec.get("tradition") or "").lower().startswith(tradition.lower())
            for rec in _KB_META
        ]
        idxs = np.nonzero(mask)[0]
        if len(idxs) == 0:
            return []
        emb = _KB_EMB[idxs]
        meta = [_KB_META[i] for i in idxs]
    else:
        emb = _KB_EMB
        meta = _KB_META

    # Cosine similarity via dot product (embeddings are normalized)
    scores = emb @ q
    top_idx = np.argsort(-scores)[: top_k]

    hits: List[KBHit] = []
    for j in top_idx:
        rec = meta[int(j)]
        hits.append(
            KBHit(
                score=float(scores[int(j)]),
                text=rec.get("text", ""),
                ref=rec.get("ref"),
                tradition=rec.get("tradition"),
                genre=rec.get("genre"),
                source=rec.get("source"),
                group_key=rec.get("group_key"),
            )
        )

    return hits
