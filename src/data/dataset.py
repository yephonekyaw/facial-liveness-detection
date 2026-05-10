"""PyTorch Dataset reading the unified manifest.

A single class handles every dataset / split combination. Cross-dataset
evaluation is just a different filter on the same manifest.

Why we keep `source_video` and `dataset` on each sample: at evaluation
time we want both frame-level and video-level metrics (the standard in
face anti-spoofing), so the eval loop needs to group predictions by
their source video. Carrying the metadata through the DataLoader is
simpler than maintaining a parallel index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from src.config import PROJECT_ROOT
from src.data.manifest import load_manifest


@dataclass
class SampleMeta:
    """Per-sample metadata for evaluation-time aggregation."""

    source_video: str
    dataset: str
    subject_id: str
    attack_type: str
    lighting: str


@dataclass
class BatchMeta:
    """Same fields as SampleMeta, but each is a list across the batch.

    Produced by `meta_collate` so the evaluation loop can index batched
    metadata the same way it indexes batched tensors.
    """

    source_video: list[str]
    dataset: list[str]
    subject_id: list[str]
    attack_type: list[str]
    lighting: list[str]


def meta_collate(batch):
    """Custom collate: stack tensors normally, gather SampleMeta into BatchMeta.

    PyTorch's default_collate doesn't know what to do with our dataclass,
    so we peel the meta out of each item, default-collate the rest, and
    rebuild meta as parallel lists.
    """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    metas = [item[2] for item in batch]

    images = default_collate(images)
    labels = default_collate(labels)
    batch_meta = BatchMeta(
        source_video=[m.source_video for m in metas],
        dataset=[m.dataset for m in metas],
        subject_id=[m.subject_id for m in metas],
        attack_type=[m.attack_type for m in metas],
        lighting=[m.lighting for m in metas],
    )
    return images, labels, batch_meta


class LivenessDataset(Dataset):
    """Reads face crops listed in `manifest.csv` and yields (image, label, meta) batches.

    Args:
        manifest_path:    Path to the unified manifest CSV.
        split:            'train' | 'devel' | 'test'.
        datasets:         Restrict to these dataset names (e.g. ['replay']).
                          None = use all.
        transform:        Albumentations Compose object. Required.
        return_meta:      If True, __getitem__ returns (image, label, SampleMeta).
                          False (default) returns (image, label) — appropriate for training.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        transform,
        datasets: list[str] | None = None,
        return_meta: bool = False,
    ):
        df = load_manifest(manifest_path)
        df = df[df["split"] == split]
        if datasets is not None:
            df = df[df["dataset"].isin(datasets)]
        if len(df) == 0:
            raise ValueError(
                f"No samples found for split={split!r}, datasets={datasets!r}. "
                f"Did you run preprocessing?"
            )
        self._df = df.reset_index(drop=True)
        self._transform = transform
        self._return_meta = return_meta

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]
        # Manifest paths are stored relative to data/. We resolve them
        # against PROJECT_ROOT/data so the absolute path follows even
        # if the working dir changes.
        abs_path = PROJECT_ROOT / "data" / row["path"]
        image_bgr = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read crop: {abs_path}")
        # Albumentations is BGR-agnostic for our pipeline (no color-space
        # transforms that care), so we leave it as BGR. The model trains
        # on whatever color order we feed it — consistency is what matters.
        image = self._transform(image=image_bgr)["image"]
        label = torch.tensor(int(row["label"]), dtype=torch.float32)

        if self._return_meta:
            meta = SampleMeta(
                source_video=row["source_video"],
                dataset=row["dataset"],
                subject_id=row["subject_id"],
                attack_type=row["attack_type"],
                lighting=row["lighting"],
            )
            return image, label, meta
        return image, label

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        transform,
        return_meta: bool = False,
    ) -> "LivenessDataset":
        """Build directly from a pre-filtered DataFrame (used by cross-validation)."""
        instance = object.__new__(cls)
        instance._df = df.reset_index(drop=True)
        instance._transform = transform
        instance._return_meta = return_meta
        return instance

    @property
    def labels(self) -> np.ndarray:
        """Useful for class-weighted sampling / scikit-learn-compatible utilities."""
        return self._df["label"].to_numpy()
