# factr/asr.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from faster_whisper import WhisperModel

from .config import FactrConfig


def _detect_device() -> str:
    try:
        from ctranslate2.devices import get_supported_devices
        if "cuda" in get_supported_devices():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def transcribe_audio(
    audio_path: str,
    cfg: Optional[FactrConfig] = None,
    language: str = "en",
    model_size: str = "large-v2",
    progress_callback: Optional[Callable[[float], None]] = None,  # <-- NEW
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transcribe audio to utterances using faster-whisper.

    `progress_callback`, if provided, will be called with a float in [0, 1]
    representing approximate progress through the audio.
    """
    cfg = cfg or FactrConfig()
    audio_path = Path(audio_path).resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = _detect_device()
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        word_timestamps=False,
    )

    total_duration = float(info.duration or 0.0)

    rows = []
    for i, seg in enumerate(segments):
        rows.append(
            {
                "segment_id": i,
                "start": float(seg.start),
                "end": float(seg.end),
                "speaker": "SPEAKER_00",
                "text": seg.text.strip(),
                "language": info.language,
            }
        )

        # ---- progress callback (approx based on segment end time) ----
        if progress_callback and total_duration > 0:
            frac = min(max(seg.end / total_duration, 0.0), 1.0)
            progress_callback(frac)

    # make sure we finish at 100%
    if progress_callback:
        progress_callback(1.0)

    utterances = pd.DataFrame(rows)

    processed_dir = cfg.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    utt_path = processed_dir / "UTTERANCES.parquet"
    utterances.to_parquet(utt_path, index=False)

    meta = {
        "audio_path": str(audio_path),
        "utterances_path": str(utt_path),
        "num_utterances": int(len(utterances)),
        "language": info.language,
        "duration_sec": float(info.duration),
        "model_size": model_size,
        "device": device,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    asr_snapshot = cfg.snapshots_dir / "ASR_LAST.json"
    asr_snapshot.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return utterances, meta

