"""Albumentations pipelines for training and evaluation.

Why Albumentations: it operates on numpy arrays, which is the natural
representation coming out of cv2.imread.

We don't resize here — the preprocessing step already produced 256x256
crops. We also don't normalize with ImageNet stats: AttackNet v2.2 is
trained from scratch (no pretrained backbone), so [0, 1] scaling is
the simplest choice.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


def train_transform() -> A.Compose:
    """Augmentation pipeline applied during training only.

    The numbers (±20° rotation, p=0.5 hflip, ±20% brightness/contrast,
    σ≈0.02 gaussian noise, k=5 motion blur) introduce realistic
    capture-time variation without producing samples that are no longer
    visually plausible faces.
    """
    return A.Compose(
        [
            A.Rotate(limit=20, p=0.5, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.005, 0.03), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.3),
            # Scale to [0, 1] then to CHW float tensor.
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    )


def eval_transform() -> A.Compose:
    """No augmentation — just dtype/scale conversion to tensor."""
    return A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    )


def to_chw_float(image_bgr: np.ndarray) -> np.ndarray:
    """Convenience used by the EDA notebook to view a transform's output."""
    return eval_transform()(image=image_bgr)["image"].numpy()
