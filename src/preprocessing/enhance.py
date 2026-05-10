"""Optional 5-stage image enhancement pipeline.

Gated by `ENABLE_ENHANCEMENT` in src/config.py. When enabled, every crop
is run through this pipeline immediately before JPEG save, so the on-disk
dataset already contains the enhanced pixels and training-time code does
not need to know whether enhancement was applied.

Stages (run in order — each subsequent stage assumes the previous one ran):
    1. Bilateral filter           — edge-preserving denoise (light σ=15)
    2. Gamma correction           — lift midtones
    3. Linear contrast (αx + β)   — global scale/offset
    4. CLAHE on L channel of LAB  — local contrast without color shift
    5. Unsharp mask               — restore high-frequency detail

Order rationale: sharpen last so CLAHE/gamma don't amplify unsharp halos;
global tone (gamma, linear contrast) before CLAHE so local-contrast
equalisation operates on an already-exposed image. Bilateral is first
but kept light (σ=15) so it cleans sensor noise without flattening the
moiré/paper-grain texture that distinguishes spoofs from real faces.

Toggling the flag does not invalidate cached bbox detections in
data/face_cache/ (those are independent of pixel content), but it does
require re-running preprocessing because the enhancement is baked into
the JPEGs.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import (
    ENABLE_ENHANCEMENT,
    ENH_BILATERAL_SIGMA_COLOR,
    ENH_BILATERAL_SIGMA_SPACE,
    ENH_CLAHE_CLIP_LIMIT,
    ENH_CLAHE_TILE_SIZE,
    ENH_CONTRAST_ALPHA,
    ENH_CONTRAST_BETA,
    ENH_GAMMA,
    ENH_UNSHARP_AMOUNT,
    ENH_UNSHARP_SIGMA,
)


def _bilateral(img: np.ndarray) -> np.ndarray:
    # d=-1 lets OpenCV pick the kernel diameter from sigmaSpace.
    return cv2.bilateralFilter(
        img,
        d=-1,
        sigmaColor=ENH_BILATERAL_SIGMA_COLOR,
        sigmaSpace=ENH_BILATERAL_SIGMA_SPACE,
    )


def _clahe_lab(img_bgr: np.ndarray) -> np.ndarray:
    # Apply CLAHE to L only so chroma (a, b) is untouched — equalising
    # in BGR per-channel would shift colour balance.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=ENH_CLAHE_CLIP_LIMIT, tileGridSize=ENH_CLAHE_TILE_SIZE)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _unsharp(img: np.ndarray) -> np.ndarray:
    # Standard unsharp mask: sharp = img + amount * (img - blurred).
    # threshold=0 → apply everywhere (no low-contrast skip).
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=ENH_UNSHARP_SIGMA)
    return cv2.addWeighted(img, 1.0 + ENH_UNSHARP_AMOUNT, blurred, -ENH_UNSHARP_AMOUNT, 0)


# Build the gamma LUT once at import time — it's a function of a constant.
_GAMMA_LUT = np.clip(
    ((np.arange(256) / 255.0) ** (1.0 / ENH_GAMMA)) * 255.0, 0, 255
).astype(np.uint8)


def _gamma(img: np.ndarray) -> np.ndarray:
    return cv2.LUT(img, _GAMMA_LUT)


def _linear_contrast(img: np.ndarray) -> np.ndarray:
    # Saturating cast: out = clip(alpha*img + beta, 0, 255).
    return cv2.convertScaleAbs(img, alpha=ENH_CONTRAST_ALPHA, beta=ENH_CONTRAST_BETA)


def enhance_frame(image_bgr: np.ndarray) -> np.ndarray:
    """Run all five stages in order and return a uint8 BGR image."""
    x = _bilateral(image_bgr)
    x = _gamma(x)
    x = _linear_contrast(x)
    x = _clahe_lab(x)
    x = _unsharp(x)
    return x


def maybe_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """Apply enhancement iff ENABLE_ENHANCEMENT is set; otherwise pass through."""
    if not ENABLE_ENHANCEMENT:
        return image_bgr
    return enhance_frame(image_bgr)
