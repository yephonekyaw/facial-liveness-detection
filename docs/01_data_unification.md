# 01 — Data Unification

## Why a unified manifest

The two datasets in scope (Replay-Attack, 3DMAD) live in completely different
formats:

- **Replay-Attack**: `.mov` videos (Motion JPEG, 320×240 @ ~25 fps), nested
  in `replayattack-{train,devel,test}/{train,devel,test}/{real,attack}/...`.
  Filenames carry attack type (`print` / `mobile` / `highdef`), photo-vs-video,
  lighting (`controlled` / `adverse`), and client ID.
- **3DMAD**: HDF5 files (`XX_SS_NN.hdf5`) containing 300 frames of 640×480
  RGB plus depth and eye annotations. Filename decomposes to subject /
  session / video number. Sessions 01–02 are bonafide; session 03 is the
  3D mask attack.

Training time we don't want to handle two different I/O paths. Instead,
preprocessing produces:

1. A directory of identically-shaped 256×256 RGB JPEG face crops
   (`data/processed/<dataset>/<video_id>_<frame_idx>.jpg`).
2. One CSV manifest (`data/processed/manifest.csv`) where every row is one
   crop, tagged with everything downstream code needs:

```
path, label, dataset, source_video, frame_idx, split,
subject_id, attack_type, lighting, face_x, face_y, face_w, face_h
```

Adding CSMAD later is just appending more rows with `dataset='csmad'` and
`attack_type='mask_silicone'`. The training loop never knows which dataset
a given row came from.

## Pipeline

```
            +-----------+    +-------------------+    +---------------+
.mov / .h5  | extract_  |    | RetinaFace        |    | square crop + |
            | replay    +--->| (insightface,     +--->| Lanczos resize|
.h5         | extract_  |    |  buffalo_sc, GPU) |    | to 256x256    |
            | 3dmad     |    +-------------------+    +-------+-------+
            +-----------+                                     |
                                                              v
                                                     +--------+--------+
                                                     | JPEG q95 +      |
                                                     | append manifest |
                                                     +-----------------+
```

Per-video bbox detections are cached to `data/face_cache/<video_id>.json`,
so re-running is cheap once detection is done.

## Frame sampling

Every 5th frame is kept. Keeping every frame would add ~3× to training
time per epoch with mostly redundant samples (adjacent frames in a
video are extremely similar). Sampling at stride 5 yields roughly
73,000 crops total — enough for AttackNet v2.2 to train without
needing curriculum tricks.

## Why RetinaFace via insightface

- Replay-Attack ships precomputed face boxes; 3DMAD ships eye positions
  (we'd need to derive a face box). Using a single detector across all
  datasets removes that asymmetry and gives us a path to CSMAD which has
  no annotations at all.
- `buffalo_sc` is the lightweight detection-only insightface bundle
  (~14 MB; det_500m.onnx). The full `buffalo_l` adds a recognition stack
  we don't need.
- Inference runs on CUDA via `onnxruntime-gpu`. This sidesteps the
  PyTorch/Pascal compatibility issue that forced us off Python 3.14
  (notes.md, Phase 1).

## Crop construction

For each detected face we:

1. Take the highest-confidence box.
2. Center-expand it by `1 + FACE_MARGIN` (= 1.3×). The margin captures
   surrounding context (hair, jawline, screen bezel for replay attacks)
   that carries useful spoofing cues.
3. Square-pad to the longer side, clamped to image bounds. Square crops
   avoid distortion when we resize.
4. Resize to 256×256 with **Lanczos** interpolation. Lanczos preserves
   high-frequency texture which is the main cue for screen replays
   (moiré) and print attacks (paper grain). Bilinear or area smooths
   that out.
5. Save as JPEG quality 95. ~20 KB/frame, ~30× smaller than uncompressed.

## 3DMAD subject split

3DMAD doesn't ship a canonical train/devel/test split — the README only
says 7/5/5. We use a deterministic, reproducible split:

| Split | Subjects                                |
|-------|-----------------------------------------|
| train | 01, 02, 03, 04, 05, 06, 07              |
| devel | 08, 09, 10, 11, 12                      |
| test  | 13, 14, 15, 16, 17                      |

This is the simplest reproducible choice. Anything more elaborate
(stratified by gender, age, etc.) requires metadata the dataset does
not publish.

## What gets dropped

Frames where RetinaFace returns no detection are dropped (logged at the
end of each extractor's `run`). On RGB-only data this is rare — Replay's
controlled-lighting frames detect at >99%, adverse lighting is mid-90s,
and 3DMAD is near-100% (all controlled-lighting Kinect captures).
Numbers from the actual run will be added to `notes.md` and §3 below.

## Modality reminder

Every crop is RGB only. 3DMAD's depth map and CSMAD's IR / depth /
thermal channels are explicitly ignored — see notes.md, Phase 1. The
unified manifest schema therefore has no modality column.
