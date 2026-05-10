# CLI Reference

All commands are run through `main.py`. The general form is:

```bash
uv run python main.py <command> [options]
```

Unless noted otherwise, every command reads `configs/default.yaml` for
defaults. CLI flags override YAML values.

---

## preprocess

Extract face crops from raw datasets into `data/processed/` and write
(or append to) `data/processed/manifest.csv`.

```bash
uv run python main.py preprocess --dataset {replay,3dmad,csmad,all}
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | Which dataset to process. `all` runs replay + 3dmad. |
| `--limit N` | None | Cap number of videos to process (useful for smoke tests). |

Examples:

```bash
# Process everything
uv run python main.py preprocess --dataset all

# Smoke test: 5 videos from replay only
uv run python main.py preprocess --dataset replay --limit 5
```

---

## train

Train AttackNet v2.2 on the official train split, validate on devel.
Checkpoints, `history.json`, and best/last model weights are saved to
`outputs/checkpoints/<run_name>/`.

```bash
uv run python main.py train [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `configs/default.yaml` | Path to YAML config file. |
| `--protocol` | None | Shorthand: `combined` (replay+3dmad), `replay`, or `3dmad`. Overrides `train_datasets`/`val_datasets` in config. |
| `--epochs N` | 30 | Total training epochs. |
| `--batch-size N` | 16 | Training batch size. |
| `--lr FLOAT` | 1e-6 | Learning rate. |
| `--dropout FLOAT` | 0.05 | Dropout rate (conv blocks + dense head). |
| `--run-name STR` | auto | Custom run name (default: `YYYYMMDD-HHMMSS_datasets`). |
| `--no-amp` | False | Disable mixed-precision training. |
| `--max-steps N` | None | Cap training steps per epoch (for smoke tests). |

Examples:

```bash
# Combined training with defaults
uv run python main.py train --protocol combined

# Replay-only, fewer epochs
uv run python main.py train --protocol replay --epochs 15

# Quick overfit smoke test
uv run python main.py train --epochs 1 --max-steps 50

# Custom hyperparameters
uv run python main.py train --protocol combined --lr 5e-6 --dropout 0.1 --batch-size 32
```

Output structure:

```
outputs/checkpoints/<run_name>/
  best.pt          # checkpoint with lowest val ACER
  last.pt          # checkpoint from the final epoch
  history.json     # per-epoch metrics, config, run metadata
```

---

## train-cv

Subject-disjoint K-fold cross-validation. Merges the official train and
devel splits into one pool, partitions by `subject_id` into K folds,
and trains a fresh model per fold. The test split is never touched.

```bash
uv run python main.py train-cv [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `configs/default.yaml` | Path to YAML config file. |
| `--n-folds N` | 5 | Number of folds. |
| `--protocol` | None | `combined`, `replay`, or `3dmad`. |
| `--epochs N` | 30 | Max epochs per fold. |
| `--batch-size N` | 16 | Training batch size. |
| `--lr FLOAT` | 1e-6 | Learning rate. |
| `--dropout FLOAT` | 0.05 | Dropout rate. |
| `--run-name STR` | auto | Custom run name. |
| `--no-amp` | False | Disable mixed-precision training. |

Examples:

```bash
# Full 5-fold CV on combined dataset
uv run python main.py train-cv --protocol combined --n-folds 5

# Faster: 3 folds, 15 epochs each
uv run python main.py train-cv --protocol combined --n-folds 3 --epochs 15
```

Output structure:

```
outputs/checkpoints/cv_<K>fold_<timestamp>_<datasets>/
  fold_0/best.pt, last.pt, history.json
  fold_1/...
  ...
  summary.json     # aggregated mean +/- std across folds
```

The final console output reports per-metric mean and standard deviation:

```
   acer: 0.0065 +/- 0.0023  [0.0045  0.0078  0.0051  0.0089  0.0062]
```

---

## tune

Bayesian hyperparameter search with Optuna. Trains multiple trials with
different hyperparameter combinations, pruning unpromising ones early.
Results are stored in a SQLite database for crash recovery and
visualisation.

```bash
uv run python main.py tune [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `configs/default.yaml` | Path to YAML config file. |
| `--n-trials N` | 30 | Number of Optuna trials to run. |
| `--epochs-per-trial N` | 15 | Max epochs per trial (pruner may cut short). |
| `--study-name STR` | `attacknet_v22` | Optuna study name (for DB grouping). |
| `--protocol` | None | `combined`, `replay`, or `3dmad`. |
| `--no-amp` | False | Disable mixed-precision training. |

Search space:

| Hyperparameter | Range |
|----------------|-------|
| learning rate | 1e-7 to 1e-4 (log-uniform) |
| dropout | 0.01 to 0.5 |
| weight decay | 1e-6 to 1e-3 (log-uniform) |
| batch size | {8, 16, 32} |
| label smoothing | {0.0, 0.05, 0.1, 0.15} |

Examples:

```bash
# Full overnight run (~12-18 hours on GTX 1070)
uv run python main.py tune --n-trials 30 --epochs-per-trial 15 --protocol combined

# Quick sanity check (~15 min)
uv run python main.py tune --n-trials 3 --epochs-per-trial 3 --protocol combined

# Small sweep for paper figures (~3-4 hours)
uv run python main.py tune --n-trials 10 --epochs-per-trial 10 --protocol combined

# Resume after a crash (same command picks up where it left off)
uv run python main.py tune --n-trials 30 --epochs-per-trial 15 --protocol combined

# Visualise results in the browser
uv run optuna-dashboard sqlite:///outputs/optuna.db
```

Output:

```
outputs/optuna.db              # SQLite study (resumable, dashboard-compatible)
outputs/optuna_results.json    # best trial params + importance ranking
```

---

## eval

Evaluate a saved checkpoint on a specific split. Reports frame-level
and video-level metrics (ACER, APCER, BPCER, EER, AUC) plus a
per-attack-type breakdown.

```bash
uv run python main.py eval --checkpoint PATH [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | *(required)* | Path to a `.pt` checkpoint file. |
| `--split` | `test` | Which split to evaluate: `train`, `devel`, or `test`. |
| `--datasets` | None (all) | Filter to specific dataset names (e.g. `replay 3dmad`). |
| `--batch-size N` | 64 | Evaluation batch size. |
| `--num-workers N` | 4 | DataLoader workers. |
| `--no-amp` | False | Disable mixed-precision inference. |

Examples:

```bash
# Evaluate combined model on the full test set
uv run python main.py eval --checkpoint outputs/checkpoints/both/best.pt

# Evaluate on 3DMAD test only (cross-dataset zero-shot)
uv run python main.py eval --checkpoint outputs/checkpoints/replay/best.pt --datasets 3dmad

# Evaluate on devel split
uv run python main.py eval --checkpoint outputs/checkpoints/both/best.pt --split devel
```

---

## eval-cross

Cross-protocol evaluation: calibrate the decision threshold on the
devel split (using the EER operating point), then report HTER on the
test split at that fixed threshold. This is the standard "honest"
evaluation where the test threshold is never tuned on test data.

```bash
uv run python main.py eval-cross --checkpoint PATH [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | *(required)* | Path to a `.pt` checkpoint file. |
| `--datasets` | None (all) | Filter eval splits to these dataset names. |
| `--batch-size N` | 64 | Evaluation batch size. |
| `--num-workers N` | 4 | DataLoader workers. |
| `--no-amp` | False | Disable mixed-precision inference. |

Examples:

```bash
# Standard cross-protocol eval on all datasets
uv run python main.py eval-cross --checkpoint outputs/checkpoints/both/best.pt

# Cross-protocol on replay only
uv run python main.py eval-cross --checkpoint outputs/checkpoints/3dmad/best.pt --datasets replay
```

---

## Config file

All training-related commands read `configs/default.yaml` by default.
The full set of config keys:

```yaml
seed: 42
image_size: 256
train_datasets: [replay, 3dmad]
val_datasets:   [replay, 3dmad]
batch_size: 16
eval_batch_size: 64
epochs: 30
optimizer: adam          # adam | adamw | sgd
lr: 1.0e-6
weight_decay: 1.0e-5
momentum: 0.9           # sgd only
dropout: 0.05
label_smoothing: 0.1
lr_scheduler:
  type: reduce_on_plateau
  factor: 0.5
  patience: 5
  min_lr: 1.0e-9
early_stopping_patience: 10
amp: true
num_workers: 4
pin_memory: true
run_name: null          # null = auto-generated
```

CLI flags take precedence over YAML values. The `--protocol` shorthand
sets both `train_datasets` and `val_datasets`:

| `--protocol` | `train_datasets` | `val_datasets` |
|---|---|---|
| `combined` | `[replay, 3dmad]` | `[replay, 3dmad]` |
| `replay` | `[replay]` | `[replay]` |
| `3dmad` | `[3dmad]` | `[3dmad]` |
