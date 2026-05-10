# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.13, managed with `uv`. Use `uv` for all package operations:

```bash
uv run python main.py          # run the CLI entry point
uv add <package>               # add a runtime dependency
uv add --dev <package>         # add a dev dependency
uv run jupyter lab             # start JupyterLab
uv sync --frozen --no-dev      # production install (used in Docker)
```

## Project layout

```
app.py              Gradio webcam demo (loads model once, streams inference)
main.py             CLI dispatcher — preprocess / train / train-cv / eval / eval-cross / tune
configs/            YAML training configs (default.yaml + per-protocol overrides)
src/
  config.py         Single source of truth for paths, constants, DEVICE
  models/           AttackNet v2.2 (residual CNN, binary logit output)
  data/             Dataset, manifest reader/writer, albumentations transforms
  preprocessing/    Face pipeline (InsightFace), frame extractors per dataset, enhance.py
  training/         train, cross_val, eval_runner, evaluate, metrics
  tuning/           Optuna search
data/processed/     manifest.csv + face crop JPEGs (gitignored)
outputs/checkpoints/  best.pt + last.pt + history.json per run (gitignored)
raw/                Raw dataset archives (gitignored)
docs/               Write-ups and paper source
ref/                Dataset descriptions and reference PDFs
```

## CLI reference

```bash
# Preprocessing
uv run python main.py preprocess --dataset {replay,3dmad,csmad,all} [--limit N]

# Training
uv run python main.py train [--config PATH] [--protocol {combined,replay,3dmad}] [overrides]
uv run python main.py train-cv [--n-folds 5] [--protocol ...]

# Hyperparameter search
uv run python main.py tune [--n-trials 30] [--epochs-per-trial 15] [--protocol ...]

# Evaluation
uv run python main.py eval --checkpoint PATH [--split {train,devel,test}] [--datasets ...]
uv run python main.py eval-cross --checkpoint PATH [--datasets ...]

# Live demo
uv run python app.py [--checkpoint PATH]
```

## Dataset Extraction

Raw datasets live in `raw/` and ship as `.tar.gz` archives. Use `extract.sh` to unpack them:

```bash
./extract.sh raw/custom-silicon-mask-attack   # extracts all .tar.gz in that dir, deletes archives on success
./extract.sh --strict raw/3d-mask-attack      # abort on first failure
./extract.sh --log extract.log raw/replay-attack
```

## Datasets

Three benchmark datasets are stored under `raw/`:

### replay-attack
1300 `.mov` videos (320×240, ~25 fps, Motion JPEG) of 50 clients. Split into `train/`, `devel/`, and `test/`. Face bounding-box annotations are in `replayattack-face-locations-v2/`. Attack types: print, mobile (iPhone), high-def (iPad), video replay.

### 3d-mask-attack
76500 frames from 17 subjects recorded with a Kinect. Each frame is a **640×480 RGB image + 640×480 11-bit depth map**. Structure: `session01/`, `session02/` (real access), `session03/` (3D mask attack). Train/devel/test split: 7/5/5 subjects.

### custom-silicon-mask-attack (CSMAD)
HDF5 (`.h5`) files (~300 frames, ~10 s each) with four channels per file from two cameras:
- Intel RealSense SR300: `color`, `infrared` (NIR 860 nm), `depth`
- Seek Thermal Compact Pro: `thermal` (LWIR)

HDF5 path layout: `data/sr300/{color,infrared,depth,aligned_color_to_depth}` and `data/seek_compact/infrared`.

Structure: `bonafide/` (87 videos + 17 still JPGs), `attack/WEAR/` (108 videos), `attack/STAND/` (51 videos).

**CSMAD is held out as a cross-dataset test corpus.** All extracted frames are assigned `split='test'` regardless of subject, because attack files only cover subjects A–F (all train subjects), making a letter-based split impossible. The attack archive uses two filename conventions across subfolders — `extract_csmad.py` handles both.

## Model

`src/models/attacknet_v22.py` — `AttackNetV22`.

- Input: 3 × 256 × 256 RGB face crop
- Two `ResidualConvBlock` layers (3→16, 16→32 channels), each with addition-based skip + MaxPool(2) + Dropout
- Dense head: 131 072 → 64 (LeakyReLU + BN + Dropout) → 1 logit
- Output is a raw logit; `torch.sigmoid(logit) > 0.5` → attack at inference time
- Training loss: `BCEWithLogitsLoss` with label smoothing; AMP enabled by default

## Preprocessing pipeline

1. `FaceDetector` (InsightFace, ONNX, buffalo_sc model) detects the largest face box
2. `expand_to_square` adds a 30% margin and squares the box
3. `crop_and_resize` crops and resizes to 256×256
4. Optional `maybe_enhance` (off by default) runs bilateral denoise → LAB-CLAHE → unsharp mask → gamma → contrast
5. JPEG saved to `data/processed/<dataset>/`
6. Row appended to `data/processed/manifest.csv`

Bounding-box results are cached per video in `data/face_cache/` to avoid re-running the detector on re-runs.

## Training notes

- Config lives in `configs/default.yaml`; per-protocol configs can override it
- `--protocol combined` sets `train_datasets: [replay, 3dmad]`
- Checkpoints: `outputs/checkpoints/<run_name>/best.pt` (best val-ACER) + `last.pt`
- `history.json` alongside each checkpoint logs per-epoch metrics
- Optuna study name defaults to `attacknet_v22`; trials prune on plateau

## Docker

```bash
docker compose up --build
```

Requires NVIDIA Container Toolkit. The Gradio app runs on port 7860. Checkpoints are mounted read-only from `./outputs/checkpoints`. InsightFace detector weights are cached in a named Docker volume.

## Reference Material

`ref/` contains papers and extended dataset descriptions:
- `about-proj.pdf` — project overview
- `face-anti-spoofing-ref.pdf` — main reference paper
- `*-description.md` — detailed protocol notes for each dataset

`docs/` contains formal write-ups and the paper source (`paper.tex`). Keep `docs/` in sync with results as experiments complete.
