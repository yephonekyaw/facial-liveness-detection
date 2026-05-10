"""Subject-disjoint K-fold cross-validation for AttackNet v2.2.

Merges the official train and devel splits into one pool, generates
K folds where each fold's val subjects never appear in its train set,
and trains a fresh model per fold. The test split is never touched.

Option 1 from the plan: train+devel are folded, test is held out for
final evaluation. This gives both cross-validation robustness numbers
(mean ± std ACER across folds) *and* a clean test-set result.

Usage:
    uv run python main.py train-cv --n-folds 5 --protocol combined
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import amp
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, DEVICE, MANIFEST_PATH
from src.data.dataset import LivenessDataset, meta_collate
from src.data.manifest import load_manifest
from src.data.transforms import eval_transform, train_transform
from src.models.attacknet_v22 import AttackNetV22, count_parameters
from src.training.evaluate import (
    evaluate_loader,
    frame_report,
    metrics_summary,
    video_report,
)
from src.training.train import (
    BCEWithLabelSmoothing,
    TrainConfig,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

def generate_subject_folds(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split *df* into *n_folds* subject-disjoint (train, val) pairs.

    Subjects are shuffled then distributed round-robin so fold sizes
    stay as even as possible. Every frame belonging to a subject goes
    into the same fold — no information leaks across the split boundary.
    """
    subjects = np.array(sorted(df["subject_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    chunks = np.array_split(subjects, n_folds)

    folds = []
    for k in range(n_folds):
        val_subs = set(chunks[k])
        is_val = df["subject_id"].isin(val_subs)
        folds.append((df[~is_val].copy(), df[is_val].copy()))
    return folds


def _fold_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame) -> str:
    """One-line summary of a fold's class balance."""
    tr_atk = (train_df["label"] == 1).sum()
    tr_bon = (train_df["label"] == 0).sum()
    va_atk = (val_df["label"] == 1).sum()
    va_bon = (val_df["label"] == 0).sum()
    return (
        f"train: {len(train_df):,} ({tr_bon:,}b/{tr_atk:,}a, "
        f"{train_df['subject_id'].nunique()} subj)  "
        f"val: {len(val_df):,} ({va_bon:,}b/{va_atk:,}a, "
        f"{val_df['subject_id'].nunique()} subj)"
    )


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------

def run_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: TrainConfig,
    fold_dir: Path,
) -> dict:
    """Train one fold from scratch, return a results dict."""
    seed_everything(cfg.seed + fold_idx)
    device = DEVICE

    train_ds = LivenessDataset.from_dataframe(train_df, transform=train_transform())
    val_ds = LivenessDataset.from_dataframe(
        val_df, transform=eval_transform(), return_meta=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=meta_collate,
    )

    model = AttackNetV22(dropout=cfg.dropout, image_size=cfg.image_size).to(device)
    criterion = BCEWithLabelSmoothing(cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None

    fold_dir.mkdir(parents=True, exist_ok=True)

    best_acer = float("inf")
    epochs_since_best = 0
    history: list[dict] = []

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, epoch,
        )
        val_arrays = evaluate_loader(
            model, val_loader, device, use_amp=cfg.amp,
            desc=f"fold {fold_idx} ep {epoch:02d} val",
        )
        val_frame = frame_report(val_arrays)
        val_video = video_report(val_arrays)

        scheduler.step(val_frame.acer)
        lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        print(
            f"  fold {fold_idx} epoch {epoch:02d}/{cfg.epochs}  "
            f"loss={train_loss:.4f}  {metrics_summary(val_frame)}  "
            f"lr={lr:.2e}  t={elapsed:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_frame": val_frame.as_dict(),
            "val_video": val_video.as_dict(),
            "lr": lr,
        })

        save_checkpoint(
            fold_dir / "last.pt", model, optimizer, scheduler, scaler,
            epoch, best_acer, cfg,
        )
        if val_frame.acer < best_acer:
            best_acer = val_frame.acer
            epochs_since_best = 0
            save_checkpoint(
                fold_dir / "best.pt", model, optimizer, scheduler, scaler,
                epoch, best_acer, cfg,
            )
            print(f"   ✓ fold {fold_idx} new best val ACER: {best_acer:.4f}")
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.early_stopping_patience:
                print(f"   fold {fold_idx} early-stop at epoch {epoch}")
                break

    fold_result = {
        "fold": fold_idx,
        "best_val_acer": best_acer,
        "epochs_completed": len(history),
        "train_subjects": sorted(train_df["subject_id"].unique().tolist()),
        "val_subjects": sorted(val_df["subject_id"].unique().tolist()),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "history": history,
    }
    (fold_dir / "history.json").write_text(
        json.dumps(fold_result, indent=2, default=str),
    )
    return fold_result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_cv(cfg: TrainConfig, n_folds: int = 5) -> dict:
    """Run K-fold cross-validation and write an aggregated summary."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    proto = "+".join(sorted(cfg.train_datasets))
    run_name = cfg.run_name or f"cv_{n_folds}fold_{stamp}_{proto}"
    cv_dir = CHECKPOINT_DIR / run_name

    print(f"== {n_folds}-fold cross-validation: {run_name} ==")
    print(f"   device: {DEVICE}")
    print(f"   datasets: {cfg.train_datasets}")

    df = load_manifest(MANIFEST_PATH)
    df = df[df["split"].isin(["train", "devel"])]
    if cfg.train_datasets:
        df = df[df["dataset"].isin(cfg.train_datasets)]

    total_subjects = df["subject_id"].nunique()
    print(f"   merged train+devel: {len(df):,} frames, {total_subjects} subjects")

    folds = generate_subject_folds(df, n_folds, cfg.seed)

    # Show the fold layout before any training starts.
    for k, (tr, va) in enumerate(folds):
        print(f"   fold {k}: {_fold_split_summary(tr, va)}")

    fold_results = []
    for k, (train_df, val_df) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"  FOLD {k}/{n_folds - 1}")
        print(f"{'='*60}")
        result = run_fold(k, train_df, val_df, cfg, cv_dir / f"fold_{k}")
        fold_results.append(result)

    # --- Aggregate across folds ---
    best_metrics = []
    for r in fold_results:
        best_idx = min(
            range(len(r["history"])),
            key=lambda i: r["history"][i]["val_frame"]["acer"],
        )
        best_metrics.append(r["history"][best_idx]["val_frame"])

    metric_keys = ["acer", "apcer", "bpcer", "eer", "roc_auc"]
    agg: dict[str, dict] = {}
    for key in metric_keys:
        vals = [m[key] for m in best_metrics]
        agg[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_fold": vals,
        }

    summary = {
        "run_name": run_name,
        "n_folds": n_folds,
        "datasets": cfg.train_datasets,
        "total_subjects": total_subjects,
        "total_frames": len(df),
        "config": asdict(cfg),
        "aggregated_metrics": agg,
        "fold_results": fold_results,
    }
    cv_dir.mkdir(parents=True, exist_ok=True)
    (cv_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
    )

    print(f"\n{'='*60}")
    print(f"  {n_folds}-FOLD CV RESULTS")
    print(f"{'='*60}")
    for key in metric_keys:
        per = "  ".join(f"{v:.4f}" for v in agg[key]["per_fold"])
        print(f"   {key:>8s}: {agg[key]['mean']:.4f} ± {agg[key]['std']:.4f}  [{per}]")
    print(f"\n   saved to {cv_dir}")

    return summary
