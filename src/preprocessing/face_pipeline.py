"""Unified face detection + crop pipeline.

Used by every per-dataset extractor so that 3DMAD frames, Replay-Attack
frames, and (future) CSMAD frames all go through the same RetinaFace
detector and produce identically-sized RGB crops. This is what makes the
manifest dataset-agnostic at training time.

Why a class rather than a free function: insightface's `FaceAnalysis`
holds an ONNX session on the GPU. Constructing it per-frame would be
ruinously slow, and per-process construction lets us reuse one detector
across the whole dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.config import FACE_CACHE_DIR, FACE_MARGIN, IMAGE_SIZE


@dataclass
class FaceBox:
    """Pixel-space bounding box in (x, y, w, h) form."""

    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class FaceDetector:
    """Thin wrapper around insightface RetinaFace.

    Loads `buffalo_sc` (the lightweight detection-only bundle: ~14 MB,
    no recognition stack) and runs on CUDA when available.
    """

    def __init__(self, det_size: tuple[int, int] = (640, 640)):
        self._app = FaceAnalysis(name="buffalo_sc", allowed_modules=["detection"])
        # ctx_id=0 → first GPU; ctx_id=-1 → CPU
        self._app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, image_bgr: np.ndarray) -> FaceBox | None:
        """Return the highest-confidence face box, or None if none found.

        Insightface expects BGR uint8 input (the OpenCV convention).
        """
        faces = self._app.get(image_bgr)
        if not faces:
            return None
        best = max(faces, key=lambda f: float(f.det_score))
        x1, y1, x2, y2 = best.bbox.astype(int)
        return FaceBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def expand_to_square(box: FaceBox, image_shape: tuple[int, int], margin: float = FACE_MARGIN) -> FaceBox:
    """Expand the box outward by `margin`, then square-pad it to the image bounds.

    Square crops avoid distortion when we resize to 256x256. We expand a
    little beyond the tight face box to keep some surrounding context
    (hair, jawline) which carries useful anti-spoofing signal (e.g.,
    paper edges of a print attack).
    """
    H, W = image_shape[:2]
    cx = box.x + box.w / 2
    cy = box.y + box.h / 2
    side = max(box.w, box.h) * (1 + margin)

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))

    # Clamp to image while keeping the box square. If the requested
    # crop spills past an edge, shift it inward instead of letting it
    # become rectangular.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > W:
        x1 -= x2 - W
        x2 = W
    if y2 > H:
        y1 -= y2 - H
        y2 = H
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    return FaceBox(x1, y1, x2 - x1, y2 - y1)


def crop_and_resize(image_bgr: np.ndarray, box: FaceBox, size: int = IMAGE_SIZE) -> np.ndarray:
    """Crop with the given box and resize to size×size using Lanczos.

    Lanczos preserves high-frequency texture (matters for spoofing — the
    moiré pattern of a screen replay or the paper grain of a print attack
    is a key cue). bilinear/area would smooth it out.
    """
    crop = image_bgr[box.y : box.y + box.h, box.x : box.x + box.w]
    if crop.size == 0:
        return None  # type: ignore[return-value]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)


def save_jpeg(image_bgr: np.ndarray, output_path: Path, quality: int = 95) -> None:
    """Write the crop as JPEG. Quality 95 keeps texture detail with ~30 KB/file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def cache_path_for(video_id: str) -> Path:
    """Where we keep per-video bbox detections so re-runs don't re-detect."""
    return FACE_CACHE_DIR / f"{video_id}.json"


def load_bbox_cache(video_id: str) -> dict[int, list[int]] | None:
    """Return {frame_idx: [x, y, w, h]} or None if no cache exists."""
    p = cache_path_for(video_id)
    if not p.exists():
        return None
    with p.open() as f:
        # JSON keys are strings; convert back to int frame indices.
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_bbox_cache(video_id: str, boxes: dict[int, list[int]]) -> None:
    p = cache_path_for(video_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump({str(k): v for k, v in boxes.items()}, f)
