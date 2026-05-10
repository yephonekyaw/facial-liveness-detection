# 00 — Implementation Plan

The bird's-eye view of the project: goals, decisions, structure, and the
seven phases of work. The numbered docs that follow (`01_data_unification.md`
through `05_results.md`) are the per-phase write-ups produced as each
phase landed; this file is what they're elaborating on.

---

## Context

We're training a CNN-based face anti-spoofing classifier on a unified dataset built from **Replay-Attack** (1,300 `.mov` videos, 2D photo/video attacks) and **3DMAD** (HDF5 frames from 17 subjects, 3D mask attacks). Existing work (e.g. `ref/face-anti-spoofing-ref.pdf`) suggests that single-dataset models fail to generalise cross-domain while combined training closes most of that gap — dataset unification is therefore the dominant design driver.

**Goals**
1. Build a clean, *unified* dataset pipeline that already accommodates a third dataset (CSMAD) without refactoring.
2. Implement AttackNet v2.2 in PyTorch and run combined-training and zero-shot cross-dataset evaluations.
3. Run an EDA in JupyterLab to feed a 10-page write-up (`docs/`).
4. Use Optuna for hyperparameter tuning within the GTX 1070's budget (8 GB VRAM, 16 GB RAM).
5. Keep code simple and well-commented so the user can learn from it — no premature abstraction, no framework magic (plain PyTorch, no Lightning).

**Decisions already made (from clarifying questions)**
- Framework: **PyTorch**
- Frame sampling: **every 5th frame** per video
- Face detection: **RetinaFace** as a unified detector across all datasets (avoids divergent per-dataset annotation handling, and is the only viable path for CSMAD later)
- Architecture scope: **only AttackNet v2.2** (no v1/v2.1 ablation — out of scope for our deliverable)
- **Modality: RGB only.** 3DMAD's depth channel and CSMAD's IR / depth / thermal channels are explicitly ignored. Every sample is a 3-channel 8-bit RGB face crop. This keeps the model architecture single-modality and the manifest schema uniform across datasets.

---

## Project Structure

```
facial-liveness-detection/
├── docs/                        # paper material (NEW)
│   ├── 00_plan.md               # this file
│   ├── 01_data_unification.md
│   ├── 02_eda.md
│   ├── 03_model_architecture.md
│   ├── 04_training_protocol.md
│   ├── 05_results.md
│   └── notes.md                 # running journal — observations, decisions, surprises
├── src/
│   ├── __init__.py
│   ├── config.py                # paths, seeds, device, image size
│   ├── preprocessing/
│   │   ├── extract_replay.py    # .mov  → frames + bbox via RetinaFace
│   │   ├── extract_3dmad.py     # .hdf5 → frames + bbox via RetinaFace
│   │   ├── extract_csmad.py     # stub for future use, same interface
│   │   └── face_pipeline.py     # shared: detect → crop → resize → save
│   ├── data/
│   │   ├── manifest.py          # build/load the unified CSV manifest
│   │   ├── dataset.py           # PyTorch Dataset reading manifest rows
│   │   └── transforms.py        # albumentations train/eval pipelines
│   ├── models/
│   │   └── attacknet_v22.py     # the model (~150 LOC)
│   ├── training/
│   │   ├── train.py             # single-run training entry point
│   │   ├── evaluate.py          # frame- and video-level metrics
│   │   └── metrics.py           # APCER, BPCER, ACER, EER, HTER
│   └── tuning/
│       └── optuna_search.py     # bayesian HPO over LR / dropout / WD
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_results.ipynb
├── configs/
│   └── default.yaml             # hyperparams, paths, seeds
├── data/                        # gitignored
│   ├── processed/
│   │   ├── replay_attack/       # 256×256 .jpg face crops
│   │   ├── 3dmad/
│   │   └── manifest.csv         # the unified index
│   └── face_cache/              # cached RetinaFace bboxes (per source video)
├── outputs/                     # gitignored
│   ├── checkpoints/
│   ├── tensorboard/
│   └── optuna.db
├── main.py
├── pyproject.toml
└── CLAUDE.md
```

---

## Phase 1 — Dependencies (~5 min)

Add to `pyproject.toml` via `uv add`:

**Core**: `torch torchvision pytorch-cuda` (CUDA 12.1 build for GTX 1070), `numpy pandas pillow opencv-python h5py albumentations scikit-learn matplotlib seaborn pyyaml tqdm`

**Face detection**: `retina-face` (TensorFlow-free pure-PyTorch fork) OR `facenet-pytorch` (which bundles MTCNN). RetinaFace's `retinaface-pytorch` package is the cleanest. We'll also have OpenCV's Haar cascade as a fallback for the rare frame where RetinaFace fails.

**Training & tuning**: `optuna optuna-dashboard tensorboard`

**Why these specifically**:
- `albumentations` over torchvision transforms because our augmentation list (rotation, flip, brightness/contrast, gaussian noise, motion blur) maps 1:1 to its API.
- `h5py` to read 3DMAD's HDF5 files.
- `optuna-dashboard` so the user can watch trials live in a browser.

---

## Phase 2 — Unified Dataset Build (~half day)

**Manifest schema** (`data/processed/manifest.csv`):
```
path, label, dataset, source_video, frame_idx,
split, subject_id, attack_type, lighting,
face_x, face_y, face_w, face_h
```
- `label`: 0 = bonafide, 1 = attack
- `dataset`: replay / 3dmad / csmad (future)
- `split`: train / devel / test (preserves official protocol per dataset)
- `attack_type`: real / print / mobile / highdef / video / mask3d / mask_silicone (future)
- `subject_id`: stable cross-dataset (`replay_001`, `3dmad_07`, …)
- `lighting`: controlled / adverse / unknown

This schema is **the foundation** — adding CSMAD later is just appending more rows.

### `src/preprocessing/face_pipeline.py` (shared core)
1. Load RetinaFace once on GPU.
2. For each frame: detect → take highest-confidence box → expand 1.3× for context → square-pad → crop → resize **256×256** with Lanczos.
3. Save as JPEG quality 95 to `data/processed/<dataset>/<video_id>_<frame_idx>.jpg`.
4. Cache bboxes in `data/face_cache/<video_id>.json` so we never re-detect.
5. Drop frames where no face is found (log count to `docs/notes.md`).

### `src/preprocessing/extract_replay.py`
- Walk `raw/replay-attack/replayattack-{train,devel,test}/`.
- For each `.mov`: open with OpenCV → keep every 5th frame → call `face_pipeline`.
- `label`: 0 if path contains `real/`, 1 if `attack/`. `attack_type` from path (`print`, `mobile`, `highdef`, `video`).
- `lighting`: parsed from filename (`controlled` / `adverse`).
- `split`: from the directory name (Replay-Attack's official protocol).

### `src/preprocessing/extract_3dmad.py`
- Walk `raw/3d-mask-attack/session{01,02,03}/Data/*.hdf5`.
- For each file: read **only `Color_Data` (RGB)** — keep every 5th frame. `Depth_Data` is ignored.
- Filename `XX_SS_NN.hdf5` → subject `XX`, session `SS`, video `NN`.
- `label`: 0 if `session01` or `session02`, 1 if `session03` (mask attack).
- Subject split per the 3DMAD README: 7 train / 5 devel / 5 test (we hardcode subjects 01–07 / 08–12 / 13–17 — `documentation/` had no canonical split file).

### `extract_csmad.py` — stub
Implements the same interface but reads **only the SR300 `color` channel** from CSMAD HDF5s (ignoring `infrared`, `depth`, `thermal`, and the Seek thermal camera entirely). Not run now — proves the schema generalizes.

**Output of phase 2**: a single `manifest.csv` with ~58k Replay frames + ~15k 3DMAD frames ≈ **73k face crops total** (well within disk budget — ~2 GB).

---

## Phase 3 — EDA in `notebooks/01_eda.ipynb` (~2 hours)

Visuals for the writeup.

1. **Class distribution** per dataset and split (bar chart). Confirm balance/imbalance — Replay is 5:1 attack-heavy at the video level, so we'll need class weighting *or* undersampling and we should decide based on EDA.
2. **Quality scores** per dataset using four standard image-quality metrics:
   - Laplacian variance (sharpness)
   - RMS contrast
   - Mean brightness (flag if outside 40–220)
   - Tenengrad gradient magnitude (blur)
   Compute on a sample of 5k crops, show distributions, see if any dataset is systematically lower quality.
3. **Sample grid** — 4×4 panels per dataset showing bonafide vs each attack type so the user (and the professor) can eyeball the visual differences.
4. **PCA & t-SNE** on resnet18 ImageNet features (cheaper than training a probe). Color by `label`, then by `dataset`. The expected story: classes are separable within a dataset, but datasets cluster apart — this *visually motivates* the combined-dataset training in Phase 5.
5. **Lighting / attack-type breakdown** — frame counts × accuracy expectations.

Each finding gets a paragraph in `docs/02_eda.md` as we go.

---

## Phase 4 — AttackNet v2.2 Model (~2 hours)

`src/models/attacknet_v22.py` — AttackNet v2.2 architecture:

```
Input: 256×256×3
Block 1:
  Conv2d(3, 16, 3, padding=1) → ReLU → BatchNorm
  Conv2d(16, 16, 3, padding=1) → ReLU → BatchNorm
  + skip from block-1 input (1×1 conv to match channels) [ADDITION, not concat]
  MaxPool(2) → Dropout(p)
Block 2:
  Conv2d(16, 32, 3, padding=1) → ReLU → BatchNorm
  Conv2d(32, 32, 3, padding=1) → ReLU → BatchNorm
  + skip (1×1 conv 16→32)
  MaxPool(2) → Dropout(p)
Flatten
Dense(→ 64) → LeakyReLU(0.2) → BatchNorm → Dropout(p)
Dense(→ 1) → [logit]
```

**Design choices**:
- Output head: a single logit + `BCEWithLogitsLoss` with label smoothing — the standard PyTorch idiom for binary classification. Detail in `docs/03_model_architecture.md`.
- The flatten of `64×64×32 = 131,072` → `64` produces a parameter-heavy first dense layer (~8.4M of the 8.4M total). We accept that — the alternative is widening the conv stack, which we don't need for a 73k-frame dataset.

We'll add **mixed-precision (`torch.cuda.amp`) training** because the GTX 1070 supports FP16 storage which roughly halves VRAM for the activations — important since the dense layer alone is 8M params.

---

## Phase 5 — Training (`src/training/train.py`, ~half day)

Single entry point, configured via `configs/default.yaml`:

```yaml
seed: 42
image_size: 256
batch_size: 16          # fits 8GB VRAM
epochs: 30
optimizer: adam
lr: 1.0e-6              # tiny starting value (will be Optuna-tuned later)
weight_decay: 1.0e-5
dropout: 0.05
label_smoothing: 0.1
lr_scheduler:
  type: reduce_on_plateau
  factor: 0.5
  patience: 5
  min_lr: 1.0e-9
early_stopping_patience: 10
```

**Training loop** (no Lightning — explicit and educational):
- Train / val split: `split == 'train'` and `split == 'devel'` from manifest.
- Augmentation (train only): rotate ±20°, hflip 0.5, brightness/contrast ±20%, gaussian noise σ=0.02, motion blur k=5.
- Loss: `BCEWithLogitsLoss(label_smoothing=0.1)` (we implement smoothing manually since the built-in is multiclass-only).
- TensorBoard scalars: `loss/train`, `loss/val`, `acer/val`, `eer/val`, `lr`.
- Save best checkpoint by validation **ACER** (not loss — biometrics-correct).

**Evaluation** (`src/training/evaluate.py`):
- **Frame-level metrics**: APCER, BPCER, ACER, EER, ROC-AUC.
- **Video-level metrics**: aggregate frame scores per `source_video` (mean), then re-compute.
- **Cross-dataset**: train on one, evaluate on the other (zero-shot) — the headline generalisation test.
- **Combined**: train on both, evaluate on each — should match within-dataset scores.

`src/training/metrics.py` implements these from scratch (~50 LOC) so they're auditable for the writeup.

---

## Phase 6 — Optuna Tuning (`src/tuning/optuna_search.py`, ~1 day GPU time)

Search space (kept narrow — current defaults are already reasonable starting points):

| Hyperparameter   | Range / Choices                  |
|------------------|----------------------------------|
| learning rate    | 1e-7 to 1e-4 (log-uniform)       |
| dropout          | 0.01 to 0.5                      |
| weight decay     | 1e-6 to 1e-3 (log-uniform)       |
| batch size       | {8, 16, 32}                      |
| label smoothing  | {0.0, 0.05, 0.1, 0.15}           |

- ~30 trials, `MedianPruner` (kills bad trials at epoch 5).
- **Objective**: validation ACER on the combined dataset.
- Each trial: 15 epochs (not full 30 — pruner makes most trials short). On GTX 1070, a 15-epoch trial on 73k images with batch 16 ≈ 25–35 min, so ~30 trials ≈ 12–18 hours overnight.
- Storage: `optuna.db` (SQLite) so trials resume cleanly if the process dies.
- Final retrain: best params, full 30 epochs, on train+devel.

`docs/04_training_protocol.md` will record the search, justify the bounds, and include the parallel coordinates / importance plots Optuna produces.

---

## Phase 7 — Results, Writeup, Reproducibility (~1 day)

`notebooks/02_results.ipynb`:
- Confusion matrices (frame and video level).
- ROC and DET curves overlaid for: trained on Replay only, trained on 3DMAD only, trained on combined.
- Cross-dataset transfer matrix.
- Failure-case gallery — top 16 false-positives + false-negatives. Cheap and tells a story.

`docs/05_results.md` aggregates these into prose ready to lift into the 10-page paper.

---

## Critical Files to Create / Modify

| File | Reason |
|------|--------|
| `pyproject.toml` | add ML deps via `uv add` |
| `.gitignore` | exclude `data/`, `outputs/`, `raw/` (already partially) |
| `configs/default.yaml` | central knobs, single source of truth |
| `src/preprocessing/face_pipeline.py` | unified RetinaFace → crop → save |
| `src/preprocessing/extract_replay.py` | Replay-Attack reader |
| `src/preprocessing/extract_3dmad.py` | 3DMAD HDF5 reader |
| `src/preprocessing/extract_csmad.py` | stub interface for future |
| `src/data/manifest.py` | manifest CSV schema + I/O |
| `src/data/dataset.py` | PyTorch `Dataset` |
| `src/data/transforms.py` | albumentations pipelines |
| `src/models/attacknet_v22.py` | the model |
| `src/training/train.py` | training loop |
| `src/training/evaluate.py` | metrics computation |
| `src/training/metrics.py` | APCER / BPCER / ACER / EER from scratch |
| `src/tuning/optuna_search.py` | HPO entry point |
| `notebooks/01_eda.ipynb` | EDA |
| `notebooks/02_results.ipynb` | final results |
| `docs/01..05_*.md` | running paper material |
| `main.py` | thin CLI dispatcher: `python main.py preprocess|train|tune|eval` |

---

## Verification Plan

End-to-end smoke test before any long training run:

1. `uv run python main.py preprocess --dataset replay --videos 5` — process 5 videos; manifest has correct rows; crops look right (open one).
2. `uv run python main.py preprocess --dataset 3dmad --videos 5` — same.
3. `python -c "from src.models.attacknet_v22 import AttackNetV22; m = AttackNetV22(); import torch; print(m(torch.randn(2,3,256,256)).shape)"` — should print `torch.Size([2, 1])`.
4. `uv run python main.py train --epochs 1 --max-steps 50` — overfit-on-tiny smoke test; loss should drop.
5. `uv run python main.py eval --checkpoint <path>` — metrics print without crashing; ACER between 0 and 1.
6. `uv run python main.py tune --trials 2 --epochs 2` — Optuna creates `optuna.db`, trials complete.
7. Open TensorBoard, confirm scalars logged.
8. Memory check: `nvidia-smi` during training — confirm we stay under 7 GB VRAM (1 GB headroom on the 1070).

Only after all 8 pass do we kick off the full 30-trial Optuna run overnight.

---

## What we're explicitly NOT doing (and why)

- **Not** processing CSMAD now — too large, no GPU/disk headroom, but the schema and `extract_csmad.py` stub mean the future cost is minutes, not days.
- **Not** implementing v1 / v2.1 — out of scope; we focus on v2.2 only.
- **Not** using PyTorch Lightning — wraps too much for an educational project.
- **Not** writing the paper itself yet — `docs/` accumulates raw material for that step, written after the implementation lands.
