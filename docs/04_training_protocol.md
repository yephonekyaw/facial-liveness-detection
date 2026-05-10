# 04 — Training Protocol

## Objective

Train AttackNet v2.2 from scratch on Replay-Attack ∪ 3DMAD frames and
evaluate on each dataset's official `test` split. The headline question
is whether single-dataset models *fail* to generalise cross-domain;
combined training is the proposed remedy. We measure both directions.

## Loss

`BCEWithLogitsLoss` with **label smoothing** toward 0.5:

```
y_smooth = y · (1 - 2s) + s,    s = label_smoothing  (default 0.1)
loss     = BCEWithLogits(logit, y_smooth)
```

PyTorch's built-in `label_smoothing` argument lives on `CrossEntropyLoss`
and only works for multiclass distributions, so we apply it manually.
Smoothing pushes the targets away from the saturating regions of the
sigmoid, which:

- discourages the network from producing extreme logits, improving
  calibration;
- helps slightly with overfitting on a relatively small dataset
  (~25k training frames after the train split).

The decision boundary is unchanged — we still predict "attack" iff
`logit ≥ 0` (equivalently `sigmoid ≥ 0.5`).

## Optimiser and schedule

| Component         | Default | Notes |
|-------------------|--------:|-------|
| Optimiser         | Adam    | |
| Learning rate     | 1e-6    | small starting point — see below |
| Weight decay      | 1e-5    | mild L2 |
| LR scheduler      | `ReduceLROnPlateau` (factor 0.5, patience 5, min_lr 1e-9) | monitors val ACER |
| Early stopping    | patience 10 epochs of no val-ACER improvement | |
| Batch size        | 16      | fits 8 GB VRAM |
| Epochs (cap)      | 30      | typically early-stops sooner |

Adam at lr=1e-6 is startlingly small but defensible:

- the dense layer has 8.4M parameters and dominates the optimisation
  landscape — large LRs blow up the BatchNorm statistics on the dense
  bottleneck within the first few steps;
- with BatchNorm everywhere, internal-covariate effects are already
  mostly tamed, so the schedule's job is just slow refinement.

Phase 6 (Optuna) will re-explore the LR; this default just gives us a
working baseline.

## Augmentation (train only)

Applied to the train split only:

- rotation ±20°
- horizontal flip (p=0.5)
- brightness/contrast ±20% (p=0.5)
- Gaussian noise σ ∈ [0.005, 0.03] (p=0.3)
- motion blur, kernel up to 5 (p=0.3)

Implemented via Albumentations, BGR-agnostic. We don't normalise with
ImageNet stats — AttackNet trains from scratch, so plain `[0, 1]`
scaling is appropriate.

## Mixed precision

`torch.amp.autocast` + `GradScaler` enabled by default. Halves activation
memory on Pascal — the dense layer's activation tensor goes from
`(B, 64) × fp32 = 4 KB` to 2 KB (negligible) but the conv activations
across both blocks save ~250 MB at batch 16. Headroom we'll need
during Optuna trials when other processes might be sharing the GPU.

If a trial diverges with NaN losses, disable AMP first with `--no-amp`
and re-run before suspecting other causes.

## Data flow

- Train split: `manifest.split == 'train'` ∧ `dataset ∈ train_datasets`.
- Val split:   `manifest.split == 'devel'` ∧ `dataset ∈ val_datasets`.
- Test split: held-out for final evaluation only.

Cross-dataset experiments use the `--protocol` flag to swap the
training/val datasets list:

| `--protocol` | `train_datasets`     | `val_datasets`       |
|--------------|----------------------|----------------------|
| `combined`   | `[replay, 3dmad]`    | `[replay, 3dmad]`    |
| `replay`     | `[replay]`           | `[replay]`           |
| `3dmad`      | `[3dmad]`            | `[3dmad]`            |

For zero-shot cross-dataset evaluation, train with one protocol and run
`eval` / `eval-cross` against the other dataset's test split.

## Metrics

Implemented from scratch in `src/training/metrics.py`. Definitions
follow ISO/IEC 30107-3 with the convention `label=1 ⇒ attack`:

- **APCER** (attack got through): `FN_attack / N_attack`
- **BPCER** (bonafide rejected): `FP_bonafide / N_bonafide`
- **ACER** = `(APCER + BPCER) / 2`  — the single number we optimise
- **EER** = error at the threshold where `APCER == BPCER`
- **HTER** = ACER computed at an externally-fixed threshold (e.g. dev-EER threshold applied to the test set)
- **ROC-AUC** for completeness

Both **frame-level** and **video-level** numbers are reported. Video-level
aggregates frame probabilities by mean per `source_video` before
thresholding. The aggregation reduces frame-wise noise and is the
standard headline metric for video-based liveness benchmarks.

The `eval-cross` subcommand calibrates the decision threshold on the
devel split and reports HTER on the test split — an honest cross-protocol
evaluation that doesn't peek at test-set labels when picking a threshold.

## Logging

- TensorBoard scalars under `outputs/tensorboard/<run_name>/`:
  `loss/train_step`, `loss/train_epoch`, `acer/val_frame`,
  `apcer/val_frame`, `bpcer/val_frame`, `eer/val_frame`, `auc/val_frame`,
  `acer/val_video`, `eer/val_video`, `lr`.
- Per-epoch console line with frame and video ACER.
- `outputs/checkpoints/<run_name>/{best.pt, last.pt}` — the model state,
  optimiser, scheduler, scaler, and the `TrainConfig` used.
- `outputs/checkpoints/<run_name>/history.json` — the per-epoch metrics
  for downstream notebook plotting.

## Reproducibility

`seed_everything(seed)` covers Python, NumPy, and Torch (including
`cudnn.deterministic = True`, which costs a small amount of throughput
but makes runs repeatable). The seed lives in `configs/default.yaml`
so it's part of the saved config — re-running a checkpoint config will
produce identical numbers.

## How to run

```bash
# Combined training (the headline experiment)
uv run python main.py train --protocol combined

# Single-dataset baselines (single-domain stress tests)
uv run python main.py train --protocol replay --run-name replay_only
uv run python main.py train --protocol 3dmad  --run-name 3dmad_only

# Smoke test before a long run (1 epoch, 50 train steps, full val)
uv run python main.py train --epochs 1 --max-steps 50 --run-name smoke

# Watch live
uv run tensorboard --logdir outputs/tensorboard

# Within-protocol test evaluation
uv run python main.py eval \
    --checkpoint outputs/checkpoints/<run_name>/best.pt \
    --split test

# Cross-dataset evaluation (the headline generalisation test)
uv run python main.py eval-cross \
    --checkpoint outputs/checkpoints/replay_only/best.pt \
    --datasets 3dmad
```

CLI overrides on top of the YAML defaults: `--epochs`, `--batch-size`,
`--lr`, `--dropout`, `--no-amp`, `--max-steps`. Phase 6 (Optuna) will
build on these same hooks.

## Verification

Smoke-test results (1 epoch × 3 train steps, full val pass on 23,100
devel frames, GTX 1070, AMP on):

- Per-step loss: ~0.7 (random init, near `ln 2 ≈ 0.693` as expected
  for an unbiased BCE on a balanced signal)
- Val pass time: ~33s
- Train throughput: ~4 it/s after warmup
- Per-attack APCER all near 1.0, BPCER ~0.04 — i.e. an untrained net
  defaults to "always bonafide". Sanity check passed.
- Checkpoints saved, TensorBoard events written, JSON history dumped.
