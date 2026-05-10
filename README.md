# Facial Liveness Detection — AttackNet v2.2

A face anti-spoofing system built on a small residual CNN trained across three benchmark datasets. Supports live webcam inference via a Gradio web UI and GPU-accelerated Docker deployment.

---

## Model — AttackNet v2.2

A lightweight binary classifier that takes a 256×256 RGB face crop and outputs a single logit (sigmoid > 0.5 → attack).

```
Input:  3 × 256 × 256
Block 1: Conv(3→16) × 2 + residual skip + MaxPool(2) + Dropout   → 16 × 128 × 128
Block 2: Conv(16→32) × 2 + residual skip + MaxPool(2) + Dropout  → 32 × 64 × 64
Flatten → Dense(131072→64) + LeakyReLU(0.2) + BN + Dropout
Dense(64→1) → logit
```

Skip connections use **addition** (not concatenation) to keep parameter count low. Training uses `BCEWithLogitsLoss` with label smoothing and mixed-precision (AMP) on CUDA.

---

## Datasets

Raw datasets live in `raw/` and must be downloaded separately.

| Dataset | Videos / Frames | Attack types |
|---|---|---|
| **replay-attack** | 1 300 `.mov` videos, 50 subjects | print, mobile, HD replay |
| **3d-mask-attack** | 76 500 frames, 17 subjects | 3D mask (Kinect RGB+depth) |
| **CSMAD** | 246 `.h5` clips (held-out test only) | silicon mask (WEAR + STAND) |

Unpack archives with the helper script:

```bash
./extract.sh raw/replay-attack
./extract.sh raw/3d-mask-attack
./extract.sh raw/custom-silicon-mask-attack   # bonafide first, then attack
```

---

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # install all runtime deps
uv sync --dev    # also install jupyterlab / gdown
```

GPU: PyTorch is pinned to the `cu124` wheel index (CUDA 12.4). CPU fallback works but is slow.

---

## Workflow

### 1. Preprocess

Extract and align face crops into `data/processed/` and build `data/processed/manifest.csv`:

```bash
uv run python main.py preprocess --dataset replay
uv run python main.py preprocess --dataset 3dmad
uv run python main.py preprocess --dataset csmad
# or all at once (skips CSMAD):
uv run python main.py preprocess --dataset all
```

Add `--limit N` to cap the number of videos for a quick smoke test.

### 2. Train

```bash
# combined (replay + 3dmad)
uv run python main.py train --protocol combined

# single-dataset
uv run python main.py train --protocol replay
uv run python main.py train --protocol 3dmad

# override any config key inline
uv run python main.py train --protocol combined --epochs 50 --lr 3e-4 --dropout 0.1
```

Checkpoints are saved to `outputs/checkpoints/<run_name>/best.pt` and `last.pt`.

### 3. K-Fold cross-validation

```bash
uv run python main.py train-cv --protocol combined --n-folds 5
```

### 4. Hyperparameter search (Optuna)

```bash
uv run python main.py tune --n-trials 30 --epochs-per-trial 15 --protocol combined
```

### 5. Evaluate

```bash
# standard split evaluation
uv run python main.py eval --checkpoint outputs/checkpoints/both/best.pt --split test

# cross-dataset HTER (calibrates threshold on devel, reports on test)
uv run python main.py eval-cross --checkpoint outputs/checkpoints/both/best.pt
```

---

## Live Demo (Gradio)

Run the webcam inference app locally:

```bash
uv run python app.py
# custom checkpoint:
uv run python app.py --checkpoint outputs/checkpoints/replay/best.pt
```

Opens at `http://localhost:7860`. The webcam streams automatically and runs inference every 3 seconds. The **Run once** button triggers a single inference on the current frame. Output: 256×256 face crop with a green (real) or red (attack) border, text verdict, and raw attack probability score (>0.5 = attack).

---

## Docker (GPU)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker compose up --build
```

The app binds to port `7860`. Checkpoints are mounted read-only from `./outputs/checkpoints`; the InsightFace detector cache is persisted in a named volume so it isn't re-downloaded on restart.

---

## Project structure

```
├── app.py                  # Gradio live demo
├── main.py                 # CLI dispatcher (preprocess / train / eval / tune)
├── configs/
│   └── default.yaml        # default training config
├── src/
│   ├── config.py           # paths, constants
│   ├── models/
│   │   └── attacknet_v22.py
│   ├── data/
│   │   ├── dataset.py      # PyTorch Dataset over manifest.csv
│   │   ├── manifest.py     # manifest reader/writer
│   │   └── transforms.py   # albumentations augmentation + eval transform
│   ├── preprocessing/
│   │   ├── face_pipeline.py    # InsightFace detector, crop/resize helpers
│   │   ├── extract_replay.py
│   │   ├── extract_3dmad.py
│   │   ├── extract_csmad.py
│   │   └── enhance.py          # optional 5-stage image enhancement
│   └── training/
│       ├── train.py
│       ├── cross_val.py
│       ├── eval_runner.py
│       ├── evaluate.py
│       └── metrics.py
├── Dockerfile
├── docker-compose.yml
└── ref/                    # dataset descriptions and reference papers
```
