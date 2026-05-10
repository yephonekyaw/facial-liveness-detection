# Project Journal

Running notes, decisions, and surprises encountered during implementation.
Material here gets distilled into the formal docs (`01..05_*.md`) and the final paper.

---

## Phase 1 — Environment setup

### Python 3.14 → 3.13 downgrade

The project initially pinned Python 3.14, but we hit a hard incompatibility with
the GTX 1070 (Pascal architecture, compute capability 6.1):

- The only PyTorch wheels available for Python 3.14 are the CUDA 13 builds.
- CUDA 13 PyTorch wheels were compiled with `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`
  — Pascal (`sm_61`) was dropped.
- Older PyTorch builds (e.g. `torch 2.6.0+cu124`) still ship Pascal binaries,
  but their wheels stop at Python 3.13 (`cp313`).

So the chain is: **Pascal GPU → must use cu124 PyTorch → must use Python ≤ 3.13**.

We pinned Python 3.13 in `.python-version` and tightened `requires-python`
to `>=3.13,<3.14`. Configured `[tool.uv.sources]` so `torch`/`torchvision`
resolve from `https://download.pytorch.org/whl/cu124` while every other
dependency comes from PyPI.

Verified: `torch.cuda.is_available() == True`, forward-pass on a 256x256x3
batch executes without `cudaErrorNoKernelImageForDevice`.

### Face detector choice

Picked **insightface** (RetinaFace via ONNX runtime on GPU) over `facenet-pytorch`
(MTCNN) because the user explicitly requested RetinaFace. Insightface uses
`onnxruntime-gpu` which works on Pascal without the PyTorch CUDA-arch issue.

We use the `buffalo_sc` model bundle (smaller `det_500m.onnx` — sufficient for
cropping; we don't need the full recognition stack). This downloads ~14 MB
weights to `~/.insightface/models/` on first use.

### Why `albumentations` over torchvision transforms

Our augmentation list (rotation, hflip, brightness/contrast, gaussian
noise, motion blur) maps 1:1 to albumentations primitives, and albumentations
operates on numpy arrays which is a more natural fit for our preprocessing
pipeline that already deals in OpenCV images.

---

## Phase 2 — Preprocessing run results

Full preprocessing completed in ~12 minutes on the GTX 1070 (faster than
the smoke-test estimate — OpenCV's video read cache warms up after the
first dozen videos).

| Dataset | Videos | Frames kept | Dropped (no face) |
|---------|--------|-------------|-------------------|
| Replay-Attack | 1,200 (test+devel+train; enroll skipped) | 61,996 | 4 (0.006%) |
| 3DMAD | 255 | 15,300 | 0 |
| **Total** | **1,455** | **77,296** | **4** |

Disk usage:
- `data/processed/replay_attack/`: 772 MB (61,996 × ~12 KB JPEGs)
- `data/processed/3dmad/`: 290 MB (15,300 × ~19 KB JPEGs)
- `data/face_cache/`: 3.9 MB (cached bbox JSONs)

### Class distribution

| Dataset | split | real | attack |
|---------|-------|------|--------|
| Replay  | train | 4,500 | 14,096 |
| Replay  | devel | 4,500 | 14,100 |
| Replay  | test  | 6,000 | 18,800 |
| 3DMAD   | train | 4,200 | 2,100  |
| 3DMAD   | devel | 3,000 | 1,500  |
| 3DMAD   | test  | 3,000 | 1,500  |

Replay is **5:1 attack-heavy** (by design — 300 attack vs 60 real videos
per split). 3DMAD is **2:1 real-heavy** at the frame level (sessions 01
and 02 are both bonafide vs. session 03 which is the attack).

When training on the *combined* dataset, the bonafide/attack mix becomes
~16,200 real vs. 16,196 attack at the train split — essentially balanced.
That's a happy accident: the two datasets' opposite imbalances cancel out.
We may still want class-weighted sampling within Replay-only training for
the cross-dataset experiments — to be decided after EDA.

### Attack-type breakdown

- `real`: 25,200
- `print`, `mobile_photo`, `mobile_video`, `highdef_photo`, `highdef_video`:
  ~9,400 each (Replay)
- `mask3d`: 5,100 (3DMAD)

The under-represented class is **`mask3d`** because 3DMAD is a much smaller
dataset. This is the most likely place for the model to be over-confident
on Replay-Attack vs. 3DMAD attacks. We'll watch for this in the per-attack-type
confusion matrix during evaluation.

### Lighting

Replay only: 31,000 adverse / 30,996 controlled (essentially balanced).
3DMAD: all `controlled` (single lighting condition).
