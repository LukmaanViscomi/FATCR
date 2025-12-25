from factr.config import FactrConfig
from factr.ingest import ingest_youtube

cfg = FactrConfig()
snap = ingest_youtube(
    "https://www.youtube.com/watch?v=speFWRuuJNs",  # put a real debate URL here
    cfg=cfg,
)

print("Run ID      :", snap["run_id"])
print("Audio path  :", snap["audio_path"])
print("Duration (s):", snap["duration_sec"])
