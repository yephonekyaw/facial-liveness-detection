"""Training entry point for AttackNet v2.2.

Plain PyTorch — no Lightning, no Hydra, no abstraction we don't earn.
The loop is intentionally explicit: read it top-to-bottom and you can
trace every gradient, every metric, every checkpoint write.

Workflow per epoch:
    1. Train: forward → loss → backward (scaled if AMP) → optimizer step.
    2. Validate: run the model on the devel split, compute ACER.
    3. Scheduler step on val ACER, save best checkpoint, log scalars.
    4. Early-stop if ACER hasn't improved in `early_stopping_patience` epochs.

Why we monitor ACER (not loss):
    Loss is convenient for the optimiser but doesn't translate to the
    biometric figure of merit. ACER is the single number stakeholders
    actually care about, so checkpoint selection and early stopping
    track it directly.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import amp, nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, DEVICE, MANIFEST_PATH
from src.data.dataset import LivenessDataset, meta_collate
from src.data.transforms import eval_transform, train_transform
from src.models.attacknet_v22 import AttackNetV22, count_parameters
from src.training.evaluate import (
    EvalArrays,
    evaluate_loader,
    frame_report,
    metrics_summary,
    per_attack_report,
    video_report,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch.

    cudnn.deterministic=True trades a small amount of speed for repeatable
    results — a fair trade for an educational project where we'll want to
    reproduce numbers in the writeup.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Loss with manual label smoothing
# ---------------------------------------------------------------------------

class BCEWithLabelSmoothing(nn.Module):
    """BCEWithLogitsLoss + label smoothing (smoothing toward 0.5).

    PyTorch's CrossEntropyLoss has built-in label_smoothing but BCE does
    not. Smoothing replaces the target `y` with
        y_smooth = y * (1 - s) + (1 - y) * s = y(1 - 2s) + s
    which moves both 0 and 1 toward 0.5. Effect: discourages the network
    from producing extreme logits, which helps calibration and reduces
    overfitting on a relatively small dataset.
    """

    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        assert 0.0 <= smoothing < 0.5
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0:
            targets = targets * (1 - 2 * self.smoothing) + self.smoothing
        return self.bce(logits, targets)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    seed: int
    image_size: int
    train_datasets: list[str]
    val_datasets: list[str]
    batch_size: int
    eval_batch_size: int
    epochs: int
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float
    dropout: float
    label_smoothing: float
    lr_scheduler: dict[str, Any]
    early_stopping_patience: int
    amp: bool
    num_workers: int
    pin_memory: bool
    run_name: str | None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Optimiser / scheduler factories
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: TrainConfig) -> optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unknown optimizer: {cfg.optimizer!r}")


def build_scheduler(optimizer: optim.Optimizer, cfg: TrainConfig) -> ReduceLROnPlateau:
    sched = cfg.lr_scheduler
    kind = sched.get("type", "reduce_on_plateau")
    if kind != "reduce_on_plateau":
        raise ValueError(f"unsupported lr_scheduler.type: {kind!r}")
    return ReduceLROnPlateau(
        optimizer,
        mode="min",  # we pass val ACER and want it minimised
        factor=sched.get("factor", 0.5),
        patience=sched.get("patience", 5),
        min_lr=sched.get("min_lr", 1e-9),
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: amp.GradScaler | None,
    device: torch.device,
    epoch: int,
    max_steps: int | None = None,
) -> float:
    """Train for one epoch, return mean training loss.

    `scaler` is None when AMP is disabled. The branches below collapse to
    a normal fp32 loop in that case.
    """
    model.train()
    losses: list[float] = []
    pbar = tqdm(loader, desc=f"epoch {epoch:02d} train", leave=False)
    for step, (images, labels) in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        loss_val = float(loss.detach().item())
        losses.append(loss_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}")


    return float(np.mean(losses)) if losses else 0.0


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: amp.GradScaler | None,
    epoch: int,
    best_acer: float,
    cfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_acer": best_acer,
        "config": asdict(cfg),
    }
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Run name
# ---------------------------------------------------------------------------

def make_run_name(cfg: TrainConfig) -> str:
    if cfg.run_name:
        return cfg.run_name
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    proto = "+".join(sorted(cfg.train_datasets))
    return f"{stamp}_{proto}"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg: TrainConfig, max_steps: int | None = None) -> dict:
    seed_everything(cfg.seed)
    device = DEVICE

    run_name = make_run_name(cfg)
    print(f"== training run: {run_name} ==")
    print(f"   device: {device}")
    print(f"   train datasets: {cfg.train_datasets}")
    print(f"   val   datasets: {cfg.val_datasets}")

    # --- Data ---
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
        return_meta=True,  # video-level eval needs source_video
    )
    print(f"   train samples: {len(train_ds):,}    val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,  # BatchNorm hates batches of size 1
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=meta_collate,  # SampleMeta isn't auto-collatable
    )

    # --- Model ---
    model = AttackNetV22(dropout=cfg.dropout, image_size=cfg.image_size).to(device)
    trainable, total = count_parameters(model)
    print(f"   parameters: {trainable:,} trainable / {total:,} total")

    criterion = BCEWithLabelSmoothing(cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None

    ckpt_dir = CHECKPOINT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    best_acer = float("inf")
    epochs_since_best = 0
    history: list[dict] = []

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, epoch, max_steps=max_steps,
        )
        val_arrays = evaluate_loader(model, val_loader, device, use_amp=cfg.amp,
                                     desc=f"epoch {epoch:02d} val")
        val_metrics = frame_report(val_arrays)
        val_video_metrics = video_report(val_arrays)

        # ReduceLROnPlateau wants the monitored value (smaller is better
        # when mode='min'); we pass ACER directly.
        scheduler.step(val_metrics.acer)
        current_lr = optimizer.param_groups[0]["lr"]

        # Per-epoch console summary.
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:02d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val[frame]: {metrics_summary(val_metrics)}  "
            f"val[video]: acer={val_video_metrics.acer:.4f}  "
            f"lr={current_lr:.2e}  "
            f"t={elapsed:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_frame": val_metrics.as_dict(),
            "val_video": val_video_metrics.as_dict(),
            "lr": current_lr,
        })

        # Checkpoint: always last, only-on-improvement for best.
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, scaler,
                        epoch, best_acer, cfg)
        if val_metrics.acer < best_acer:
            best_acer = val_metrics.acer
            epochs_since_best = 0
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, scaler,
                            epoch, best_acer, cfg)
            print(f"   ✓ new best val ACER: {best_acer:.4f}")
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.early_stopping_patience:
                print(f"   early-stop: val ACER hasn't improved for "
                      f"{cfg.early_stopping_patience} epochs")
                break

    # Persist a JSON summary of the run for later notebooks / writeup.
    summary = {
        "run_name": run_name,
        "best_val_acer": best_acer,
        "epochs_completed": len(history),
        "config": asdict(cfg),
        "history": history,
    }
    (ckpt_dir / "history.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n== done. best val ACER: {best_acer:.4f}  ckpts at {ckpt_dir} ==")
    return summary


# ---------------------------------------------------------------------------
# CLI helpers (called from main.py)
# ---------------------------------------------------------------------------

def parse_overrides(args: argparse.Namespace, cfg: TrainConfig) -> TrainConfig:
    """Apply CLI overrides on top of the YAML-loaded config."""
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.protocol is not None:
        if args.protocol == "combined":
            cfg.train_datasets = ["replay", "3dmad"]
            cfg.val_datasets = ["replay", "3dmad"]
        elif args.protocol == "replay":
            cfg.train_datasets = ["replay"]
            cfg.val_datasets = ["replay"]
        elif args.protocol == "3dmad":
            cfg.train_datasets = ["3dmad"]
            cfg.val_datasets = ["3dmad"]
        else:
            raise ValueError(f"unknown protocol: {args.protocol!r}")
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.no_amp:
        cfg.amp = False
    return cfg
