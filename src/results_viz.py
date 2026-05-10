"""Results visualisation — figures for the paper's results section.

Functions are called from `notebooks/03_results.ipynb`. They take
pre-loaded JSON data (CV summaries and test evaluation results) and
return matplotlib Figure objects.

Figures produced:
1. Training curves — loss and val ACER over epochs, per fold per protocol
2. ROC curves — within-dataset and cross-dataset on the test set
3. Bar charts — within-dataset ACER comparison, cross-dataset asymmetry
4. CV stability — per-fold ACER spread across protocols
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.metrics import roc_curve

PROTOCOL_LABELS = {
    "both-cv-5": "Combined",
    "replay-cv-5": "Replay",
    "3dmad-cv-5": "3DMAD",
}
PROTOCOL_COLORS = {
    "both-cv-5": "#2196F3",
    "replay-cv-5": "#FF9800",
    "3dmad-cv-5": "#4CAF50",
}


def load_cv_summaries(ckpt_dir: Path) -> dict:
    runs = {}
    for name in ["both-cv-5", "replay-cv-5", "3dmad-cv-5"]:
        path = ckpt_dir / name / "summary.json"
        with path.open() as f:
            runs[name] = json.load(f)
    return runs


def load_test_results(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(cv_runs: dict) -> plt.Figure:
    """2x3 grid: top row = training loss, bottom row = val ACER.
    One column per protocol. Thin lines per fold, thick = fold-mean."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey="row")

    for col, run_name in enumerate(["both-cv-5", "replay-cv-5", "3dmad-cv-5"]):
        run = cv_runs[run_name]
        label = PROTOCOL_LABELS[run_name]
        color = PROTOCOL_COLORS[run_name]

        folds = run["fold_results"]
        max_epochs = max(len(f["history"]) for f in folds)

        all_losses = np.full((len(folds), max_epochs), np.nan)
        all_acers = np.full((len(folds), max_epochs), np.nan)

        for i, fold in enumerate(folds):
            n = len(fold["history"])
            all_losses[i, :n] = [h["train_loss"] for h in fold["history"]]
            all_acers[i, :n] = [h["val_frame"]["acer"] for h in fold["history"]]

        epochs = np.arange(1, max_epochs + 1)

        ax_loss = axes[0, col]
        ax_acer = axes[1, col]

        for i in range(len(folds)):
            mask = ~np.isnan(all_losses[i])
            ax_loss.plot(epochs[mask], all_losses[i, mask],
                         color=color, alpha=0.25, linewidth=0.8)
            mask = ~np.isnan(all_acers[i])
            ax_acer.plot(epochs[mask], all_acers[i, mask],
                         color=color, alpha=0.25, linewidth=0.8)

        mean_loss = np.nanmean(all_losses, axis=0)
        mean_acer = np.nanmean(all_acers, axis=0)
        valid_loss = ~np.isnan(mean_loss)
        valid_acer = ~np.isnan(mean_acer)

        ax_loss.plot(epochs[valid_loss], mean_loss[valid_loss],
                     color=color, linewidth=2, label="fold mean")
        ax_acer.plot(epochs[valid_acer], mean_acer[valid_acer],
                     color=color, linewidth=2, label="fold mean")

        ax_loss.set_title(f"{label}")
        ax_loss.set_xlabel("epoch")
        ax_acer.set_xlabel("epoch")

        if col == 0:
            ax_loss.set_ylabel("training loss")
            ax_acer.set_ylabel("val ACER")

        ax_loss.grid(alpha=0.3)
        ax_acer.grid(alpha=0.3)
        ax_loss.legend(fontsize=8)

    fig.suptitle("Training Curves — 5-Fold CV", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. ROC curves
# ---------------------------------------------------------------------------

def _load_eval_arrays(ckpt_path: Path, split: str, datasets: list[str]):
    """Run inference and return (scores, labels). Imported here to avoid
    top-level torch import — keeps the module lightweight for non-GPU use."""
    import torch
    from src.config import DEVICE, MANIFEST_PATH
    from src.data.dataset import LivenessDataset, meta_collate
    from src.data.transforms import eval_transform
    from src.training.eval_runner import load_model
    from src.training.evaluate import evaluate_loader
    from torch.utils.data import DataLoader

    device = DEVICE
    model, cfg = load_model(ckpt_path, device)
    ds = LivenessDataset(
        manifest_path=MANIFEST_PATH,
        split=split,
        transform=eval_transform(),
        datasets=datasets,
        return_meta=True,
    )
    loader = DataLoader(
        ds, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=meta_collate,
    )
    arrays = evaluate_loader(model, loader, device, use_amp=True, desc=f"ROC {split}")
    return arrays.scores, arrays.labels


def plot_roc_within(ckpt_dir: Path) -> plt.Figure:
    """ROC curves for within-dataset test evaluation (3 protocols)."""
    fig, ax = plt.subplots(figsize=(6, 6))

    specs = {
        "Combined": {
            "ckpt": ckpt_dir / "both-cv-5" / "fold_1" / "best.pt",
            "datasets": ["replay", "3dmad"],
            "color": "#2196F3",
        },
        "Replay": {
            "ckpt": ckpt_dir / "replay-cv-5" / "fold_0" / "best.pt",
            "datasets": ["replay"],
            "color": "#FF9800",
        },
        "3DMAD": {
            "ckpt": ckpt_dir / "3dmad-cv-5" / "fold_0" / "best.pt",
            "datasets": ["3dmad"],
            "color": "#4CAF50",
        },
    }

    for label, spec in specs.items():
        scores, labels = _load_eval_arrays(spec["ckpt"], "test", spec["datasets"])
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        auc = float(np.trapezoid(tpr, fpr))
        ax.plot(fpr, tpr, color=spec["color"], linewidth=1.8,
                label=f"{label} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("FPR (BPCER)")
    ax.set_ylabel("TPR (1 - APCER)")
    ax.set_title("ROC — Within-Dataset Test Evaluation")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 0.15)
    ax.set_ylim(0.85, 1.005)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_roc_cross(ckpt_dir: Path) -> plt.Figure:
    """ROC curves for cross-dataset test evaluation."""
    fig, ax = plt.subplots(figsize=(6, 6))

    specs = {
        "Replay → 3DMAD": {
            "ckpt": ckpt_dir / "replay" / "best.pt",
            "datasets": ["3dmad"],
            "color": "#FF9800",
        },
        "3DMAD → Replay": {
            "ckpt": ckpt_dir / "3dmad" / "best.pt",
            "datasets": ["replay"],
            "color": "#4CAF50",
        },
    }

    for label, spec in specs.items():
        scores, labels = _load_eval_arrays(spec["ckpt"], "test", spec["datasets"])
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        auc = float(np.trapezoid(tpr, fpr))
        ax.plot(fpr, tpr, color=spec["color"], linewidth=1.8,
                label=f"{label} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("FPR (BPCER)")
    ax.set_ylabel("TPR (1 - APCER)")
    ax.set_title("ROC — Cross-Dataset Test Evaluation")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Bar charts
# ---------------------------------------------------------------------------

def plot_within_dataset_bars(test_results: dict) -> plt.Figure:
    """Grouped bar chart: ACER, APCER, BPCER for each within-dataset protocol."""
    protocols = ["combined", "replay", "3dmad"]
    labels = ["Combined", "Replay", "3DMAD"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    metrics = ["acer", "apcer", "bpcer"]
    metric_labels = ["ACER", "APCER", "BPCER"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, level, level_label in zip(axes, ["frame", "video"],
                                       ["Frame-Level", "Video-Level"]):
        x = np.arange(len(metrics))
        width = 0.25

        for i, (proto, label, color) in enumerate(zip(protocols, labels, colors)):
            r = test_results["within_dataset"][proto][level]
            vals = [r[m] for m in metrics]
            bars = ax.bar(x + i * width, vals, width, label=label, color=color)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.4f}",
                                (bar.get_x() + bar.get_width() / 2, h),
                                ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("error rate")
        ax.set_title(f"Within-Dataset Test — {level_label}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(0.015, ax.get_ylim()[1] * 1.3))

    fig.suptitle("Within-Dataset Test-Set Evaluation", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_cross_dataset_bars(test_results: dict) -> plt.Figure:
    """Side-by-side bar chart showing the cross-dataset asymmetry."""
    directions = ["replay_to_3dmad", "3dmad_to_replay"]
    dir_labels = ["Replay → 3DMAD", "3DMAD → Replay"]
    colors = ["#FF9800", "#4CAF50"]

    metrics = ["acer", "apcer", "bpcer"]
    metric_labels = ["ACER", "APCER", "BPCER"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Fixed threshold (0.5)
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.35
    for i, (d, dl, c) in enumerate(zip(directions, dir_labels, colors)):
        r = test_results["cross_dataset_raw"][d]["frame"]
        vals = [r[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=dl, color=c)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.2%}",
                            (bar.get_x() + bar.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("error rate")
    ax.set_title("Fixed Threshold (0.5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Devel-calibrated HTER
    ax = axes[1]
    hter_vals = []
    for d in directions:
        hter_vals.append(test_results["cross_dataset_calibrated"][d]["hter"])
    bars = ax.bar(dir_labels, hter_vals, color=colors, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2%}",
                    (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("HTER")
    ax.set_title("Devel-Calibrated Threshold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Cross-Dataset Generalisation", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. CV stability (per-fold spread)
# ---------------------------------------------------------------------------

def plot_cv_fold_spread(cv_runs: dict) -> plt.Figure:
    """Strip/box plot showing per-fold ACER values for each protocol."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    positions = []
    box_data = []
    tick_labels = []

    for i, run_name in enumerate(["both-cv-5", "replay-cv-5", "3dmad-cv-5"]):
        run = cv_runs[run_name]
        per_fold = run["aggregated_metrics"]["acer"]["per_fold"]
        mean = run["aggregated_metrics"]["acer"]["mean"]
        color = PROTOCOL_COLORS[run_name]
        label = PROTOCOL_LABELS[run_name]

        pos = i
        positions.append(pos)
        box_data.append(per_fold)
        tick_labels.append(label)

        ax.scatter([pos] * len(per_fold), per_fold,
                   color=color, zorder=3, s=50, alpha=0.7, edgecolors="white")
        ax.scatter([pos], [mean], color=color, zorder=4, s=120,
                   marker="D", edgecolors="black", linewidths=1.2)

    bp = ax.boxplot(box_data, positions=positions, widths=0.35,
                    patch_artist=True, showfliers=False, zorder=2)
    for patch, run_name in zip(bp["boxes"],
                                ["both-cv-5", "replay-cv-5", "3dmad-cv-5"]):
        patch.set_facecolor(PROTOCOL_COLORS[run_name])
        patch.set_alpha(0.2)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("validation ACER")
    ax.set_title("CV Fold Stability — Per-Fold ACER (diamonds = mean)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
