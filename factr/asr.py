# factr/asr.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
from faster_whisper import WhisperModel

from .config import FactrConfig


# ---------------------------------------------------------------------
# Device / model helpers
# ---------------------------------------------------------------------


def _detect_device() -> str:
    """
    Prefer CUDA if it is available, otherwise fall back to CPU.

    We use torch only for detection. If torch is not installed or fails,
    we silently fall back to CPU.
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------
# Core transcription function
# ---------------------------------------------------------------------


def transcribe_audio(
    audio_path: str | Path,
    cfg: Optional[FactrConfig] = None,
    language: str = "en",
    model_size: str = "large-v2",
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transcribe audio to utterances using faster-whisper, preferring the GPU.

    Parameters
    ----------
    audio_path:
        Path to an audio or video file (anything ffmpeg/faster-whisper can read).
    cfg:
        Optional FactrConfig. If None, a default FactrConfig() is created.
    language:
        Optional language code; if None or "auto", whisper will attempt detection.
    model_size:
        Name of the faster-whisper model, e.g. "small", "medium", "large-v2".
    progress_callback:
        Optional function taking a float in [0, 1]. Called as progress updates.

    Returns
    -------
    utterances:
        Pandas DataFrame with one row per utterance/segment.
    meta:
        Dictionary with basic metadata (language, duration, device, etc.).
    """
    cfg = cfg or FactrConfig()
    audio_path = Path(audio_path).resolve()

    # Where to cache / load Whisper weights
    model_dir = cfg.root / "models" / "whisper"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- Device + compute type ------------------------------------------------
    device = _detect_device()
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"[ASR] Using device={device}, compute_type={compute_type}")
    print(f"[ASR] Model size={model_size}, model_dir={model_dir}")

    # ---- Load model -----------------------------------------------------------
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        local_files_only=False,          # allow download on first run
        download_root=str(model_dir),
    )

    # ---- Transcribe -----------------------------------------------------------
    if progress_callback:
        progress_callback(0.0)

    # GPU-friendly settings, compatible with your faster-whisper version
    result, info = model.transcribe(
        str(audio_path),
        beam_size=1,          # much faster than 5 on GPU
        word_timestamps=False,
        language=language,
    )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    lang = getattr(info, "language", language)

    rows: list[Dict[str, Any]] = []

    for i, seg in enumerate(result):
        rows.append(
            {
                "segment_id": i,
                "start": float(seg.start),
                "end": float(seg.end),
                "speaker": "SPEAKER_00",   # diarisation placeholder
                "text": seg.text.strip(),
                "language": lang,
            }
        )

        # Approximate progress by segment end vs total duration
        if progress_callback and duration > 0:
            frac = min(float(seg.end) / duration, 0.99)
            progress_callback(frac)

    if progress_callback:
        progress_callback(1.0)

    utterances = pd.DataFrame(rows)

    # ---- Persist UTTERANCES.parquet for downstream steps ---------------------
    processed_dir = cfg.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    utt_path = processed_dir / "UTTERANCES.parquet"
    utterances.to_parquet(utt_path, index=False)

    # ---- Metadata + ASR_LAST snapshot ----------------------------------------
    meta: Dict[str, Any] = {
        "utterances_path": str(utt_path),
        "num_utterances": int(len(utterances)),
        "language": lang,
        "duration_sec": duration,
        "model_size": model_size,
        "device": device,
        "model_dir": str(model_dir),
        "created_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }

    snapshots_dir = cfg.snapshots_dir
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    asr_snapshot = snapshots_dir / "ASR_LAST.json"
    asr_snapshot.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return utterances, meta


