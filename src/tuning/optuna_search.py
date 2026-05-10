"""Bayesian hyperparameter search with Optuna.

Search space (from docs/00_plan.md Phase 6):
    learning rate:    1e-7 to 1e-4 (log-uniform)
    dropout:          0.01 to 0.5
    weight decay:     1e-6 to 1e-3 (log-uniform)
    batch size:       {8, 16, 32}
    label smoothing:  {0.0, 0.05, 0.1, 0.15}

Objective: validation ACER on the devel split. MedianPruner kills
unpromising trials early (after a few warmup epochs). SQLite storage
so the study survives crashes and can be resumed with the same command.

Each trial builds a fresh AttackNetV22, trains for `epochs_per_trial`
epochs, and reports val ACER at each epoch for pruning. The best
val ACER across epochs is the trial's final value.

Usage:
    uv run python main.py tune --n-trials 30 --epochs 15 --protocol combined
    uv run python main.py tune --n-trials 5  --epochs 3               # smoke test
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import amp
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, DEVICE, MANIFEST_PATH, OUTPUTS_DIR
from src.data.dataset import LivenessDataset, meta_collate
from src.data.transforms import eval_transform, train_transform
from src.models.attacknet_v22 import AttackNetV22
from src.training.evaluate import evaluate_loader, frame_report, metrics_summary
from src.training.train import (
    BCEWithLabelSmoothing,
    TrainConfig,
    build_optimizer,
    build_scheduler,
    seed_everything,
    train_one_epoch,
)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def _build_trial_config(trial: optuna.Trial, base_cfg: TrainConfig) -> TrainConfig:
    """Sample hyperparameters and return a complete config for this trial."""
    return TrainConfig(
        seed=base_cfg.seed,
        image_size=base_cfg.image_size,
        train_datasets=base_cfg.train_datasets,
        val_datasets=base_cfg.val_datasets,
        batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
        eval_batch_size=base_cfg.eval_batch_size,
        epochs=base_cfg.epochs,
        optimizer=base_cfg.optimizer,
        lr=trial.suggest_float("lr", 1e-7, 1e-4, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        momentum=base_cfg.momentum,
        dropout=trial.suggest_float("dropout", 0.01, 0.5),
        label_smoothing=trial.suggest_categorical(
            "label_smoothing", [0.0, 0.05, 0.1, 0.15],
        ),
        lr_scheduler=base_cfg.lr_scheduler,
        early_stopping_patience=base_cfg.early_stopping_patience,
        amp=base_cfg.amp,
        num_workers=base_cfg.num_workers,
        pin_memory=base_cfg.pin_memory,
        run_name=None,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, base_cfg: TrainConfig, epochs: int) -> float:
    """Single Optuna trial: train for *epochs* epochs, return best val ACER.

    Reports val ACER after every epoch so the MedianPruner can kill
    hopeless trials early. A trial that gets pruned raises TrialPruned
    and Optuna records it without counting it as a failure.
    """
    cfg = _build_trial_config(trial, base_cfg)
    seed_everything(cfg.seed)
    device = DEVICE

    train_ds = LivenessDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
        transform=train_transform(),
        datasets=cfg.train_datasets,
    )
    val_ds = LivenessDataset(
        manifest_path=MANIFEST_PATH,
        split="devel",
        transform=eval_transform(),
        datasets=cfg.val_datasets,
        return_meta=True,
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

    best_acer = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, epoch,
        )
        val_arrays = evaluate_loader(
            model, val_loader, device, use_amp=cfg.amp,
            desc=f"trial {trial.number} ep {epoch:02d}",
        )
        val_metrics = frame_report(val_arrays)
        scheduler.step(val_metrics.acer)

        if val_metrics.acer < best_acer:
            best_acer = val_metrics.acer

        trial.report(val_metrics.acer, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        print(
            f"  trial {trial.number:03d} epoch {epoch:02d}/{epochs}  "
            f"loss={train_loss:.4f}  {metrics_summary(val_metrics)}"
        )

    return best_acer


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_search(
    base_cfg: TrainConfig,
    n_trials: int = 30,
    epochs_per_trial: int = 15,
    study_name: str = "attacknet_v22",
) -> dict:
    """Create (or resume) an Optuna study and run *n_trials* new trials.

    Results are saved in two places:
    - `outputs/optuna.db`            — full SQLite study (for dashboard / resume)
    - `outputs/optuna_results.json`  — human-readable summary of the best trial
    """
    db_path = OUTPUTS_DIR / "optuna.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path}"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )

    print(f"== Optuna HPO: {study_name} ==")
    print(f"   device: {DEVICE}")
    print(f"   datasets: {base_cfg.train_datasets}")
    print(f"   n_trials: {n_trials}, epochs_per_trial: {epochs_per_trial}")
    print(f"   storage: {db_path}")
    existing = len(study.trials)
    if existing > 0:
        print(f"   resuming: {existing} existing trials found")

    study.optimize(
        lambda trial: objective(trial, base_cfg, epochs_per_trial),
        n_trials=n_trials,
    )

    # --- Report best trial ---
    best = study.best_trial
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\n== best trial #{best.number} ==")
    print(f"   val ACER: {best.value:.4f}")
    print(f"   params:")
    for k, v in best.params.items():
        print(f"     {k}: {v}")

    importance: dict = {}
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"   param importance:")
        for k, v in importance.items():
            print(f"     {k}: {v:.3f}")
    except Exception:
        pass

    summary = {
        "study_name": study_name,
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "best_trial": best.number,
        "best_val_acer": best.value,
        "best_params": best.params,
        "param_importance": {k: float(v) for k, v in importance.items()},
        "base_config": asdict(base_cfg),
    }

    results_path = OUTPUTS_DIR / "optuna_results.json"
    results_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n   completed: {len(completed)}, pruned: {len(pruned)}")
    print(f"   results saved to {results_path}")
    print(f"   database at {db_path}")
    print(f"   run 'optuna-dashboard {storage}' to visualise")

    return summary
