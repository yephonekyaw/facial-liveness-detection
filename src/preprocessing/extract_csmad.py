"""CSMAD frame extraction.

Each .h5 is a short clip from the Intel RealSense SR300 (we only use the
RGB color stream — `data/sr300/color`; depth, infrared, and the Seek
thermal channel are ignored). Inside that group every frame is its own
timestamped dataset of shape (H, W, 3) uint8, e.g.

    data/sr300/color/16_37_01_151  -> (1920, 1080, 3) uint8

so we list keys, sort them chronologically, and step through with
FRAME_STRIDE just like the other extractors.

Filename layouts:
    bonafide:  <SUBJ>_gen_i<ILLUM>_<SEQ>.h5         e.g. A_gen_i0_061.h5
    attack:    Mask_atk_<SUBJ><MASK>_i<ILLUM>_<SEQ>.h5  e.g. Mask_atk_A1_i0_001.h5

Directory layout (after extracting the .tar.gz archives):
    raw/custom-silicon-mask-attack/
        bonafide/CSMAD/bonafide/<SUBJ>/*.h5
        attack/CSMAD/attack/{WEAR,STAND}/<SUBJ>/*.h5

Bonafide and attack are extracted separately (the attack archive is huge),
so `discover_videos()` just picks up whichever subtrees currently exist —
you can run extraction once on bonafide, delete it, unpack attack, and
run again. The manifest is append-only so the two passes accumulate.

Subject split (deterministic, by wearer letter). Attack clips only exist
for subjects A-F, so the split is designed to give attacks in every
partition:

    train: A B       G H I   (attack: A-B  bonafide: A-B + G-I)
    devel: C D       J K     (attack: C-D  bonafide: C-D + J-K)
    test:  E F       L M N   (attack: E-F  bonafide: E-F + L-N)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from src.config import CSMAD_OUT, CSMAD_RAW, FRAME_STRIDE, MANIFEST_PATH
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

# Some bonafide clips have a `genglasses` token instead of `gen` — same person,
# just wearing glasses. We treat them identically (no per-row glasses flag in the
# manifest schema).
BONA_RE = re.compile(r"^([A-Z])_gen(?:glasses)?_i(\d+)_(\d+)\.h5$")
# Two naming conventions exist across WEAR/STAND subfolders:
#   WEAR:  {camera}_atk_{wearer}{mask}_i{illum}_{seq}.h5  e.g. E_atk_A1_i0_001.h5
#   STAND: Mask_atk_{wearer}{mask}_i{illum}_{seq}.h5      e.g. Mask_atk_A1_i0_001.h5
ATK_RE = re.compile(r"^(?:[A-Z]|Mask)_atk_([A-Z])(\d+)_i(\d+)_(\d+)\.h5$")

# Subject-disjoint split. Attack clips only exist for A-F, so we distribute
# those six subjects across all three partitions to ensure each has attacks.
SUBJECT_SPLIT: dict[str, str] = {
    "A": "train", "B": "train",
    "C": "devel",  "D": "devel",
    "E": "test",   "F": "test",
    "G": "train", "H": "train", "I": "train",
    "J": "devel",  "K": "devel",
    "L": "test",   "M": "test",  "N": "test",
}


@dataclass
class CsmadVideo:
    path: Path
    subject: str        # "A".."N" — wearer (for attack, the person behind the mask)
    illum: str          # "0", "1", ... from the i<N> token
    seq: str            # numeric sequence id from the filename
    pose: str           # "bonafide" | "WEAR" | "STAND"
    label: int          # 0 = bonafide, 1 = attack
    split: str


def _discover_bonafide() -> list[CsmadVideo]:
    # The tarball nests CSMAD/bonafide/ under our raw/.../bonafide/, hence the doubled segment.
    root = CSMAD_RAW / "bonafide" / "CSMAD" / "bonafide"
    if not root.exists():
        return []
    out: list[CsmadVideo] = []
    for h5 in sorted(root.rglob("*.h5")):
        m = BONA_RE.match(h5.name)
        if not m:
            tqdm.write(f"WARN: CSMAD bonafide filename does not match pattern: {h5.name}")
            continue
        subj, illum, seq = m.groups()
        out.append(CsmadVideo(h5, subj, illum, seq, "bonafide", 0, SUBJECT_SPLIT[subj]))
    return out


def _discover_attack() -> list[CsmadVideo]:
    root = CSMAD_RAW / "attack" / "CSMAD" / "attack"
    if not root.exists():
        return []
    out: list[CsmadVideo] = []
    for pose in ("WEAR", "STAND"):
        pose_dir = root / pose
        if not pose_dir.exists():
            continue
        for h5 in sorted(pose_dir.rglob("*.h5")):
            m = ATK_RE.match(h5.name)
            if not m:
                tqdm.write(f"WARN: CSMAD attack filename does not match pattern: {h5.name}")
                continue
            subj, _mask_idx, illum, seq = m.groups()
            out.append(CsmadVideo(h5, subj, illum, seq, pose, 1, SUBJECT_SPLIT[subj]))
    return out


def discover_videos() -> list[CsmadVideo]:
    return _discover_bonafide() + _discover_attack()


def video_id(v: CsmadVideo) -> str:
    return f"csmad_{v.pose.lower()}_{v.subject}_i{v.illum}_{v.seq}"


def process_video(v: CsmadVideo, detector: FaceDetector, writer: ManifestWriter) -> tuple[int, int]:
    vid = video_id(v)
    cache = load_bbox_cache(vid) or {}
    have_cache = bool(cache)
    new_cache: dict[int, list[int]] = {}

    kept = 0
    dropped = 0

    with h5py.File(v.path, "r") as f:
        color = f["data/sr300/color"]
        # Each frame is its own dataset keyed by HH_MM_SS_mmm — sort lexicographically
        # (timestamps are zero-padded, so lexical order == chronological order).
        timestamps = sorted(color.keys())

        for frame_idx, ts in enumerate(timestamps):
            if frame_idx % FRAME_STRIDE != 0:
                continue

            rgb = np.asarray(color[ts])  # (H, W, 3) uint8, RGB
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
            out_path = CSMAD_OUT / f"{vid}_{frame_idx:04d}.jpg"
            save_jpeg(crop, out_path)
            writer.write(
                ManifestRow(
                    path=str(out_path.relative_to(out_path.parents[2])),
                    label=v.label,
                    dataset="csmad",
                    source_video=vid,
                    frame_idx=frame_idx,
                    split=v.split,
                    subject_id=f"csmad_{v.subject}",
                    attack_type="real" if v.label == 0 else "mask_silicone",
                    lighting="controlled",
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
    n_bona = sum(1 for v in videos if v.label == 0)
    n_atk = len(videos) - n_bona
    print(f"CSMAD: discovered {len(videos)} videos ({n_bona} bonafide, {n_atk} attack)")
    if not videos:
        return

    detector = FaceDetector()
    CSMAD_OUT.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_dropped = 0
    with ManifestWriter(MANIFEST_PATH) as writer:
        for v in tqdm(videos, desc="csmad videos"):
            k, d = process_video(v, detector, writer)
            total_kept += k
            total_dropped += d
    print(f"CSMAD: kept {total_kept} frames, dropped {total_dropped} (no face detected)")
