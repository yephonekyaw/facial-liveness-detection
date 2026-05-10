"""Unified manifest CSV — the index that ties every dataset together.

Every preprocessed face crop produces one row. Adding CSMAD later is
literally just appending more rows with `dataset='csmad'` and an
`attack_type` like `mask_silicone`. The training pipeline never needs
to know which dataset a row came from — it filters by `split` and
trains on the union.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

# Order matters — used as CSV header.
COLUMNS = [
    "path",          # relative to PROJECT_ROOT
    "label",         # 0 = bonafide, 1 = attack
    "dataset",       # replay | 3dmad | csmad
    "source_video",  # stable video-level identifier (used for video-level eval)
    "frame_idx",     # frame number within the source video
    "split",         # train | devel | test
    "subject_id",    # e.g. "replay_client018", "3dmad_07"
    "attack_type",   # real | print | mobile | highdef | video | mask3d | mask_silicone
    "lighting",      # controlled | adverse | unknown
    "face_x",
    "face_y",
    "face_w",
    "face_h",
]


@dataclass
class ManifestRow:
    path: str
    label: int
    dataset: str
    source_video: str
    frame_idx: int
    split: str
    subject_id: str
    attack_type: str
    lighting: str
    face_x: int
    face_y: int
    face_w: int
    face_h: int

    def __post_init__(self) -> None:
        # Catch typos in fields/columns at construction time rather than at
        # CSV-write time. Dataclass field order must match COLUMNS.
        assert [f.name for f in fields(self)] == COLUMNS, "ManifestRow fields out of sync with COLUMNS"


class ManifestWriter:
    """Append-only CSV writer. Use as a context manager."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "ManifestWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        self._fh = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        if new_file:
            self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.close()

    def write(self, row: ManifestRow) -> None:
        assert self._writer is not None, "ManifestWriter must be used as a context manager"
        self._writer.writerow(asdict(row))


def load_manifest(path: Path) -> pd.DataFrame:
    """Load the CSV with sensible dtypes."""
    df = pd.read_csv(path)
    # Make filtering by split/dataset/label fast and unambiguous.
    df["label"] = df["label"].astype(int)
    df["frame_idx"] = df["frame_idx"].astype(int)
    return df
