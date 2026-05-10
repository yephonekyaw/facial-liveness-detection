"""Standalone evaluation: load a checkpoint, run it on a split, print metrics.

Used for both within-protocol evaluation (train on combined, eval on
combined-test) and cross-protocol stress tests (train on replay, eval
on 3dmad-test). Cross-protocol generalisation is the harder benchmark,
so we make it a first-class CLI flag.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import DEVICE, MANIFEST_PATH
from src.data.dataset import LivenessDataset, meta_collate
from src.data.transforms import eval_transform
from src.models.attacknet_v22 import AttackNetV22
from src.training.evaluate import (
    cross_protocol_hter,
    evaluate_loader,
    frame_report,
    metrics_summary,
    per_attack_report,
    video_report,
)


def load_model(ckpt_path: Path, device: torch.device) -> tuple[AttackNetV22, dict]:
    """Re-instantiate the model and restore weights from a checkpoint."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]
    model = AttackNetV22(dropout=cfg["dropout"], image_size=cfg["image_size"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


def evaluate_split(
    ckpt_path: Path,
    split: str,
    eval_datasets: list[str] | None,
    batch_size: int = 64,
    num_workers: int = 4,
    use_amp: bool = True,
):
    """Run the model in `ckpt_path` on the named split and print metrics."""
    device = DEVICE
    model, cfg = load_model(ckpt_path, device)

    ds = LivenessDataset(
        manifest_path=MANIFEST_PATH,
        split=split,
        transform=eval_transform(),
        datasets=eval_datasets,
        return_meta=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=meta_collate,
    )
    print(f"== eval ckpt={ckpt_path}  split={split}  datasets={eval_datasets}  n={len(ds):,} ==")

    arrays = evaluate_loader(model, loader, device, use_amp=use_amp, desc="eval")

    fr = frame_report(arrays)
    vr = video_report(arrays)
    print(f"frame: {metrics_summary(fr)}  (n_attack={fr.n_attack}, n_bonafide={fr.n_bonafide})")
    print(f"video: {metrics_summary(vr)}  (n_attack={vr.n_attack}, n_bonafide={vr.n_bonafide})")

    print("\nper-attack-type:")
    print(per_attack_report(arrays).to_string(index=False))

    return {
        "ckpt": str(ckpt_path),
        "split": split,
        "datasets": eval_datasets,
        "frame": fr.as_dict(),
        "video": vr.as_dict(),
        "trained_with": cfg.get("train_datasets"),
    }


def evaluate_cross(
    ckpt_path: Path,
    eval_datasets: list[str] | None,
    batch_size: int = 64,
    num_workers: int = 4,
    use_amp: bool = True,
):
    """Calibrate threshold on devel, report HTER on test."""
    device = DEVICE
    model, cfg = load_model(ckpt_path, device)

    def make_loader(split: str):
        ds = LivenessDataset(
            manifest_path=MANIFEST_PATH,
            split=split,
            transform=eval_transform(),
            datasets=eval_datasets,
            return_meta=True,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          collate_fn=meta_collate), ds

    devel_loader, devel_ds = make_loader("devel")
    test_loader, test_ds = make_loader("test")
    print(
        f"== cross-protocol HTER  trained_with={cfg.get('train_datasets')}  "
        f"eval_on={eval_datasets}  devel_n={len(devel_ds):,}  test_n={len(test_ds):,} =="
    )

    devel_arrays = evaluate_loader(model, devel_loader, device, use_amp=use_amp, desc="devel")
    test_arrays = evaluate_loader(model, test_loader, device, use_amp=use_amp, desc="test")

    hter, thr, test_at_thr = cross_protocol_hter(devel_arrays, test_arrays)
    print(f"calibrated threshold (devel EER): {thr:.4f}")
    print(f"test HTER at that threshold:       {hter:.4f}")
    print(f"test full metrics:                 {metrics_summary(test_at_thr)}")
    return {
        "ckpt": str(ckpt_path),
        "eval_datasets": eval_datasets,
        "trained_with": cfg.get("train_datasets"),
        "calibrated_threshold": thr,
        "hter": hter,
        "test_at_threshold": test_at_thr.as_dict(),
    }
