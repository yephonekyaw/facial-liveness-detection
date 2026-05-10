"""3DMAD frame extraction.

Each HDF5 file is a single video: 300 frames @ 640x480, with both Color_Data
(RGB, channels-first) and Depth_Data (we ignore depth — the project is
RGB-only). Filename `XX_SS_NN.hdf5` decomposes as subject / session / video.

Sessions:
    01, 02 = bonafide (real client, recorded ~2 weeks apart)
    03     = 3D mask attack

Subject split (deterministic, since the dataset doesn't ship one):
    train: subjects 01..07  (7)
    devel: subjects 08..12  (5)
    test:  subjects 13..17  (5)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from src.config import FRAME_STRIDE, MANIFEST_PATH, MASK3D_OUT, MASK3D_RAW
from src.data.manifest import ManifestRow, ManifestWriter
from src.preprocessing.enhance import maybe_enhance
from src.preprocessing.face_pipeline import (
    FaceBox,
    FaceDetector,
    crop_and_resize,
    expand_to_square,
    load_bbox_cache,
    save_bbox_cache,
    save_jpeg,
)

FNAME_RE = re.compile(r"(\d{2})_(\d{2})_(\d{2})\.hdf5")

# 7 / 5 / 5 split per the 3DMAD README.
SUBJECT_SPLIT = {
    **{f"{i:02d}": "train" for i in range(1, 8)},   # 01..07
    **{f"{i:02d}": "devel" for i in range(8, 13)},  # 08..12
    **{f"{i:02d}": "test" for i in range(13, 18)},  # 13..17
}


@dataclass
class Mask3DVideo:
    path: Path
    subject: str   # "01".."17"
    session: str   # "01" | "02" | "03"
    video_no: str  # "01".."05"
    split: str
    label: int     # 0 = real (sessions 01, 02), 1 = mask attack (session 03)


def discover_videos() -> list[Mask3DVideo]:
    videos: list[Mask3DVideo] = []
    for session in ("01", "02", "03"):
        data_dir = MASK3D_RAW / f"session{session}" / "Data"
        if not data_dir.exists():
            continue
        for h5 in sorted(data_dir.glob("*.hdf5")):
            m = FNAME_RE.match(h5.name)
            if not m:
                tqdm.write(f"WARN: 3DMAD filename does not match pattern: {h5.name}")
                continue
            subject, sess, vno = m.groups()
            split = SUBJECT_SPLIT.get(subject)
            if split is None:
                tqdm.write(f"WARN: 3DMAD subject {subject} not in split table")
                continue
            videos.append(
                Mask3DVideo(
                    path=h5,
                    subject=subject,
                    session=sess,
                    video_no=vno,
                    split=split,
                    label=1 if sess == "03" else 0,
                )
            )
    return videos


def video_id(v: Mask3DVideo) -> str:
    return f"3dmad_s{v.subject}_sess{v.session}_v{v.video_no}"


def process_video(v: Mask3DVideo, detector: FaceDetector, writer: ManifestWriter) -> tuple[int, int]:
    vid = video_id(v)
    cache = load_bbox_cache(vid) or {}
    have_cache = bool(cache)
    new_cache: dict[int, list[int]] = {}

    kept = 0
    dropped = 0

    with h5py.File(v.path, "r") as f:
        # Shape: (300, 3, 480, 640), uint8, RGB.
        color = f["Color_Data"]
        n_frames = color.shape[0]

        for frame_idx in range(0, n_frames, FRAME_STRIDE):
            # (3, H, W) -> (H, W, 3) RGB -> BGR (insightface expects BGR).
            rgb = np.transpose(color[frame_idx], (1, 2, 0))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if have_cache and frame_idx in cache:
                x, y, w, h = cache[frame_idx]
                box: FaceBox | None = FaceBox(x, y, w, h)
            else:
                box = detector.detect(bgr)
                if box is not None:
                    new_cache[frame_idx] = list(box.as_tuple())

            if box is None:
                dropped += 1
                continue

            square = expand_to_square(box, bgr.shape)
            crop = crop_and_resize(bgr, square)
            if crop is None:
                dropped += 1
                continue

            crop = maybe_enhance(crop)
            out_path = MASK3D_OUT / f"{vid}_{frame_idx:04d}.jpg"
            save_jpeg(crop, out_path)
            writer.write(
                ManifestRow(
                    path=str(out_path.relative_to(out_path.parents[2])),
                    label=v.label,
                    dataset="3dmad",
                    source_video=vid,
                    frame_idx=frame_idx,
                    split=v.split,
                    subject_id=f"3dmad_{v.subject}",
                    attack_type="mask3d" if v.label == 1 else "real",
                    lighting="controlled",  # 3DMAD only has one lighting condition
                    face_x=square.x,
                    face_y=square.y,
                    face_w=square.w,
                    face_h=square.h,
                )
            )
            kept += 1

    if not have_cache and new_cache:
        save_bbox_cache(vid, new_cache)
    return kept, dropped


def run(limit: int | None = None) -> None:
    videos = discover_videos()
    if limit is not None:
        videos = videos[:limit]
    print(f"3DMAD: discovered {len(videos)} videos")

    detector = FaceDetector()
    MASK3D_OUT.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_dropped = 0
    with ManifestWriter(MANIFEST_PATH) as writer:
        for v in tqdm(videos, desc="3dmad videos"):
            k, d = process_video(v, detector, writer)
            total_kept += k
            total_dropped += d
    print(f"3DMAD: kept {total_kept} frames, dropped {total_dropped} (no face detected)")
