# factr/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class FactrConfig:
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

