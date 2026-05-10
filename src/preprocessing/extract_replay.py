"""Replay-Attack frame extraction.

Walks `raw/replay-attack/replayattack-{train,devel,test}/`, opens each
.mov, samples every `FRAME_STRIDE`-th frame, runs RetinaFace, saves a
square 256x256 RGB crop, and appends a manifest row.

Filename grammar (Replay-Attack convention):
    real:    client###_session##_webcam_authenticate_<lighting>_<idx>.mov
    attack:  attack_<print|mobile|highdef>_client###_session##_<device>_<photo|video>_<lighting>.mov

The attack subdir layer (`fixed/` vs `hand/`) tells us whether the
attacker used a stand or held the device by hand. We record this in
`subject_id` rather than as its own column to keep the schema lean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from src.config import FRAME_STRIDE, MANIFEST_PATH, REPLAY_OUT, REPLAY_RAW
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

# Match either filename convention. Lighting is the cue we can extract
# from both forms; the attack form additionally tells us the spoof type.
REAL_RE = re.compile(r"client(\d+)_session\d+_webcam_authenticate_(controlled|adverse)_\d+\.mov")
ATTACK_RE = re.compile(
    r"attack_(print|mobile|highdef)_client(\d+)_session\d+_\w+_(photo|video)_(controlled|adverse)\.mov"
)


@dataclass
class ReplayVideo:
    path: Path
    split: str            # train | devel | test
    label: int            # 0 = real, 1 = attack
    subject_id: str       # e.g. "replay_client018"
    attack_type: str      # real | print | mobile | highdef | video
    lighting: str         # controlled | adverse


def discover_videos() -> list[ReplayVideo]:
    """Walk the raw directory and produce a metadata record per .mov."""
    videos: list[ReplayVideo] = []

    for split in ("train", "devel", "test"):
        split_dir = REPLAY_RAW / f"replayattack-{split}" / split
        if not split_dir.exists():
            continue

        # Real (bonafide) accesses
        for mov in sorted((split_dir / "real").glob("*.mov")):
            m = REAL_RE.match(mov.name)
            if not m:
                # Skip anything that doesn't match the canonical pattern;
                # log it so we notice if the dataset layout changes.
                tqdm.write(f"WARN: real filename does not match pattern: {mov.name}")
                continue
            client, lighting = m.group(1), m.group(2)
            videos.append(
                ReplayVideo(
                    path=mov,
                    split=split,
                    label=0,
                    subject_id=f"replay_client{client}",
                    attack_type="real",
                    lighting=lighting,
                )
            )

        # Attacks (split into fixed/ and hand/, but the support type
        # doesn't change the spoof category, so we treat them together).
        for support_dir in ("fixed", "hand"):
            for mov in sorted((split_dir / "attack" / support_dir).glob("*.mov")):
                m = ATTACK_RE.match(mov.name)
                if not m:
                    tqdm.write(f"WARN: attack filename does not match pattern: {mov.name}")
                    continue
                spoof_kind, client, photo_or_video, lighting = m.groups()
                # Combine spoof_kind ('print'/'mobile'/'highdef') with the
                # photo/video distinction so 'video' replays are
                # separable from 'print' photo attacks during analysis.
                if spoof_kind == "print":
                    attack_type = "print"
                else:
                    # mobile|highdef paired with photo|video gives us
                    # mobile_photo, mobile_video, highdef_photo, highdef_video.
                    attack_type = f"{spoof_kind}_{photo_or_video}"
                videos.append(
                    ReplayVideo(
                        path=mov,
                        split=split,
                        label=1,
                        subject_id=f"replay_client{client}",
                        attack_type=attack_type,
                        lighting=lighting,
                    )
                )

    return videos


def video_id(v: ReplayVideo) -> str:
    """Stable identifier used for bbox caching and the `source_video` column."""
    return f"replay_{v.split}_{v.path.stem}"


def process_video(v: ReplayVideo, detector: FaceDetector, writer: ManifestWriter) -> tuple[int, int]:
    """Process one video. Returns (frames_kept, frames_dropped_no_face)."""
    vid = video_id(v)
    cache = load_bbox_cache(vid) or {}
    have_cache = bool(cache)
    new_cache: dict[int, list[int]] = {}

    cap = cv2.VideoCapture(str(v.path))
    if not cap.isOpened():
        tqdm.write(f"ERR: could not open {v.path}")
        return 0, 0

    kept = 0
    dropped = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % FRAME_STRIDE == 0:
            if have_cache and frame_idx in cache:
                x, y, w, h = cache[frame_idx]
                box: FaceBox | None = FaceBox(x, y, w, h)
            else:
                box = detector.detect(frame)
                if box is not None:
                    new_cache[frame_idx] = list(box.as_tuple())

            if box is None:
                dropped += 1
            else:
                square = expand_to_square(box, frame.shape)
                crop = crop_and_resize(frame, square)
                if crop is None:
                    dropped += 1
                else:
                    crop = maybe_enhance(crop)
                    out_path = REPLAY_OUT / f"{vid}_{frame_idx:04d}.jpg"
                    save_jpeg(crop, out_path)
                    writer.write(
                        ManifestRow(
                            path=str(out_path.relative_to(out_path.parents[2])),  # data/processed/replay_attack/...
                            label=v.label,
                            dataset="replay",
                            source_video=vid,
                            frame_idx=frame_idx,
                            split=v.split,
                            subject_id=v.subject_id,
                            attack_type=v.attack_type,
                            lighting=v.lighting,
                            face_x=square.x,
                            face_y=square.y,
                            face_w=square.w,
                            face_h=square.h,
                        )
                    )
                    kept += 1
        frame_idx += 1
    cap.release()

    if not have_cache and new_cache:
        save_bbox_cache(vid, new_cache)
    return kept, dropped


def run(limit: int | None = None) -> None:
    """Process every Replay-Attack video. `limit` caps the count for smoke tests."""
    videos = discover_videos()
    if limit is not None:
        videos = videos[:limit]
    print(f"Replay-Attack: discovered {len(videos)} videos")

    detector = FaceDetector()
    REPLAY_OUT.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_dropped = 0
    with ManifestWriter(MANIFEST_PATH) as writer:
        for v in tqdm(videos, desc="replay videos"):
            k, d = process_video(v, detector, writer)
            total_kept += k
            total_dropped += d
    print(f"Replay-Attack: kept {total_kept} frames, dropped {total_dropped} (no face detected)")
