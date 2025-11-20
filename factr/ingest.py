# factr/ingest.py

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import soundfile as sf  # <- instead of librosa
from yt_dlp import YoutubeDL

from .config import FactrConfig


def _download_audio_with_yt_dlp(
    youtube_url: str,
    out_dir: Path,
    cookies_path: Optional[Path] = None,
) -> Path:
    """
    Download best available audio from YouTube using yt-dlp and return the file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    base = uuid.uuid4().hex[:10]
    out_tmpl = str(out_dir / f"{base}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": True,
    }
    if cookies_path is not None and cookies_path.exists():
        ydl_opts["cookiefile"] = str(cookies_path)

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        downloaded = ydl.prepare_filename(info)

    return Path(downloaded)


def _ffmpeg_normalise_to_16k_mono(
    src_audio: Path,
    dst_audio: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """
    Use ffmpeg to convert arbitrary audio to 16 kHz mono WAV.
    """
    dst_audio.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src_audio),
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-vn",
        str(dst_audio),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _audio_metadata(wav_path: Path) -> Dict[str, Any]:
    """
    Compute basic metadata: duration, sample rate, channels using soundfile.
    """
    wav_path = wav_path.resolve()
    data, samplerate = sf.read(str(wav_path))

    if data.ndim == 1:  # mono
        channels = 1
        duration = len(data) / samplerate
    else:  # stereo or multi-channel: shape (n_frames, n_channels)
        channels = data.shape[1]
        duration = data.shape[0] / samplerate

    return {
        "audio_path": str(wav_path),
        "sample_rate": int(samplerate),
        "channels": int(channels),
        "duration_sec": float(duration),
    }


def ingest_youtube(
    youtube_url: str,
    cfg: Optional[FactrConfig] = None,
    cookies_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level ingest function (replacement for FACTR_02 notebook):

    - Download the YouTube audio.
    - Normalise to 16 kHz mono WAV.
    - Write LAST_INGEST.json under data/processed.
    - Return a metadata dict that downstream steps can consume.
    """
    cfg = cfg or FactrConfig()
    processed_dir = cfg.processed_dir
    snapshots_dir = cfg.snapshots_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # 1) Download audio with yt-dlp
    cookies = Path(cookies_path) if cookies_path else None
    raw_audio = _download_audio_with_yt_dlp(
        youtube_url=youtube_url,
        out_dir=processed_dir,
        cookies_path=cookies,
    )

    # 2) Normalise to 16k mono WAV
    final_wav = processed_dir / f"{run_id}_16k_mono.wav"
    _ffmpeg_normalise_to_16k_mono(raw_audio, final_wav, sample_rate=16000, channels=1)

    # 3) Gather metadata
    meta = _audio_metadata(final_wav)
    snap: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": ts,
        "youtube_url": youtube_url,
        "raw_audio_path": str(raw_audio.resolve()),
        **meta,
    }

    # 4) Write snapshot + LAST_INGEST.json
    snapshot_path = snapshots_dir / f"INGEST_SNAPSHOT_{run_id}.json"
    snapshot_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")

    last_path = processed_dir / "LAST_INGEST.json"
    last_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")

    return snap

