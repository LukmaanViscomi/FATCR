# factr/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class FactrConfig:
    # --- Project root & directories ---
    root: Path = Path(__file__).resolve().parents[1]

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    # --- Runtime / cost controls (Streamlit UI should override these) ---
    max_transcript_chunks: int = 12
    max_claims: int = 40
    top_k: int = 5

    # --- Retrieval quality controls ---
    min_similarity: float = 0.55  # similarity floor to accept evidence
    # If you implement “no evidence under floor”, this drives that behaviour.

    # --- Verdict strictness controls (if/when you use them) ---
    min_verifier_confidence: float = 0.60
    support_margin: float = 0.10  # delta needed between agree vs disagree

    # --- Model identifiers (optional – helps dissertation clarity) ---
    asr_model: str = "whisper"              # or "whisper-large-v2" etc.
    embedding_model: str = "text-embedding-3-small"
    verifier_model: str = "gpt-4o-mini"

