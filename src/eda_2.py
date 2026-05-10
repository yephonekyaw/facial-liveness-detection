"""Second-pass EDA — figures aimed at the paper / slide deck.

`src/eda.py` covered the inspection-level questions (sample grids, raw
quality histograms, feature-space scatter). This module focuses on
*summary* figures: one chart per concept that compactly tells the
reader something about the unified dataset.

Functions are called from `notebooks/02_eda.ipynb`. They take the
manifest DataFrame (and, for quality plots, a metrics DataFrame
produced by `src.eda.compute_quality_metrics`).

Figures produced:
1. Class distribution — bonafide vs attack per dataset and per split
2. Attack-type breakdown — visual of which attack types we have
3. Subject and source-video diversity — identities per cell
4. Face crop size — resolution differences between datasets
5. Quality metrics by class — bonafide vs attack separability via raw stats
6. Quality metrics by attack type — which attacks look most "real"
7. Frames per source video — sanity-check that frame yield is even
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL_NAMES = {0: "bonafide", 1: "attack"}
# A consistent palette so the same class always has the same colour
# across every figure in the notebook.
LABEL_COLORS = {"bonafide": "#2a9d8f", "attack": "#e76f51"}
DATASET_COLORS = {"replay": "#4361ee", "3dmad": "#f4a261", "csmad": "#9b5de5"}


# ---------------------------------------------------------------------------
# 1. Class distribution
# ---------------------------------------------------------------------------

def class_counts_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format bonafide/attack counts per dataset, ready for plotting."""
    counts = (
        df.assign(label_name=df["label"].map(LABEL_NAMES))
          .groupby(["dataset", "label_name"])
          .size()
          .unstack(fill_value=0)
    )
    # Re-order columns so bars always read bonafide → attack.
    return counts[["bonafide", "attack"]]


def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: bonafide vs attack frame counts per dataset.

    This is the headline distribution figure for the paper — at a glance
    it shows which class dominates each dataset before any combination.
    """
    counts = class_counts_per_dataset(df)
    datasets = counts.index.tolist()
    x = np.arange(len(datasets))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_b = ax.bar(x - width / 2, counts["bonafide"], width,
                    label="bonafide", color=LABEL_COLORS["bonafide"])
    bars_a = ax.bar(x + width / 2, counts["attack"], width,
                    label="attack",   color=LABEL_COLORS["attack"])

    # Number-on-top so the chart works in a printed paper (no hover tooltip).
    for bars in (bars_b, bars_a):
        for bar in bars:
            ax.annotate(f"{int(bar.get_height()):,}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("frame count")
    ax.set_title("Class distribution per dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_class_distribution_by_split(df: pd.DataFrame) -> plt.Figure:
    """One panel per dataset, grouped bars over splits.

    Useful to confirm the train/devel/test ratio is preserved per class —
    a sanity check before any modelling.
    """
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.2),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    splits_order = ["train", "devel", "test"]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        counts = (
            sub.assign(label_name=sub["label"].map(LABEL_NAMES))
               .groupby(["split", "label_name"])
               .size()
               .unstack(fill_value=0)
               .reindex(index=splits_order, columns=["bonafide", "attack"], fill_value=0)
        )
        x = np.arange(len(counts.index))
        width = 0.38
        ax.bar(x - width / 2, counts["bonafide"], width,
               label="bonafide", color=LABEL_COLORS["bonafide"])
        ax.bar(x + width / 2, counts["attack"], width,
               label="attack",   color=LABEL_COLORS["attack"])
        ax.set_xticks(x)
        ax.set_xticklabels(counts.index)
        ax.set_title(ds)
        ax.set_ylabel("frame count")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Per-split class distribution")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Attack types
# ---------------------------------------------------------------------------

def plot_attack_type_breakdown(df: pd.DataFrame) -> plt.Figure:
    """Stacked bar: each dataset's column shows its attack-type composition."""
    # Drop the 'real' rows — this chart is about attack composition.
    attacks = df[df["label"] == 1]
    table = (
        attacks.groupby(["dataset", "attack_type"])
               .size()
               .unstack(fill_value=0)
    )
    # Stable column order so the legend is consistent run-to-run.
    table = table.reindex(sorted(table.columns), axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bottom = np.zeros(len(table))
    cmap = plt.get_cmap("tab10")
    for i, col in enumerate(table.columns):
        vals = table[col].values
        ax.bar(table.index, vals, bottom=bottom, label=col, color=cmap(i))
        bottom += vals

    ax.set_ylabel("attack frame count")
    ax.set_title("Attack-type composition per dataset")
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Subject and video diversity
# ---------------------------------------------------------------------------

def diversity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Unique subjects and source videos per (dataset, split, label)."""
    g = df.groupby(["dataset", "split", "label"])
    out = pd.DataFrame({
        "subjects": g["subject_id"].nunique(),
        "videos":   g["source_video"].nunique(),
        "frames":   g.size(),
    })
    out.index = out.index.set_levels(
        [LABEL_NAMES[i] for i in out.index.levels[2]], level=2
    )
    return out


def plot_subject_diversity(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of unique subjects per (dataset, split, class).

    Frame counts are misleading because one subject can contribute
    thousands of frames. Subject counts tell you how many *people* the
    model has actually seen — the real generalisation budget.
    """
    g = (
        df.assign(label_name=df["label"].map(LABEL_NAMES))
          .groupby(["dataset", "split", "label_name"])["subject_id"]
          .nunique()
          .unstack("label_name", fill_value=0)
          .reindex(columns=["bonafide", "attack"], fill_value=0)
    )

    datasets = sorted({d for d, _ in g.index})
    splits_order = ["train", "devel", "test"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.0),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = g.loc[ds].reindex(splits_order, fill_value=0)
        x = np.arange(len(sub.index))
        width = 0.38
        ax.bar(x - width / 2, sub["bonafide"], width,
               color=LABEL_COLORS["bonafide"], label="bonafide")
        ax.bar(x + width / 2, sub["attack"], width,
               color=LABEL_COLORS["attack"], label="attack")
        for i, split in enumerate(sub.index):
            for off, col in [(-width / 2, "bonafide"), (width / 2, "attack")]:
                ax.text(i + off, sub[col].iloc[i] + 0.1, str(int(sub[col].iloc[i])),
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index)
        ax.set_title(ds)
        ax.set_ylabel("unique subjects")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Identity diversity per split")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Face crop size
# ---------------------------------------------------------------------------

def plot_face_size_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of face_w (in source-image pixels) per dataset.

    All crops are resampled to IMAGE_SIZE (256) before training, but the
    *original* face size is the real resolution budget — once it's small,
    upsampling can't restore detail. Datasets shot at 640×480 yield
    faces ~2× larger than Replay's 320×240 source video.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for ds, g in df.groupby("dataset"):
        ax.hist(g["face_w"], bins=50, alpha=0.55, label=ds, density=True,
                color=DATASET_COLORS.get(ds))
    ax.set_xlabel("face crop width (px in source frame)")
    ax.set_ylabel("density")
    ax.set_title("Original face-crop resolution per dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Quality metrics by class / attack type
# ---------------------------------------------------------------------------

QUALITY_COLS = ["sharpness", "contrast", "brightness", "tenengrad"]


def _boxplot_grid(metrics: pd.DataFrame, group_col: str, title: str,
                  rotate_x: bool = False) -> plt.Figure:
    """2×2 grid of boxplots, one quality metric per subplot, grouped by `group_col`."""
    cats = sorted(metrics[group_col].unique(), key=str)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.flat, QUALITY_COLS):
        data = [metrics.loc[metrics[group_col] == c, col].values for c in cats]
        bp = ax.boxplot(data, tick_labels=cats, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], cats):
            # Map class labels to the same palette used elsewhere.
            colour = LABEL_COLORS.get(str(c), "#cccccc") \
                     if group_col == "label_name" else None
            if colour:
                patch.set_facecolor(colour)
            else:
                patch.set_facecolor("#bcd0e5")
            patch.set_alpha(0.8)
        ax.set_title(col)
        ax.grid(axis="y", alpha=0.3)
        if rotate_x:
            ax.tick_params(axis="x", rotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_quality_by_class(metrics: pd.DataFrame) -> plt.Figure:
    """Boxplots of the four quality metrics, bonafide vs attack."""
    m = metrics.copy()
    m["label_name"] = m["label"].map(LABEL_NAMES)
    return _boxplot_grid(m, "label_name", "Image quality — bonafide vs attack")


def plot_quality_by_attack_type(metrics: pd.DataFrame) -> plt.Figure:
    """Boxplots of the four quality metrics, by attack type.

    Tells us which presentation attacks look most like real frames in
    raw image-statistics terms. A print attack should look very
    different from a bona-fide capture (lower sharpness, washed-out
    contrast); a high-def video replay should be much closer.
    """
    return _boxplot_grid(metrics, "attack_type",
                         "Image quality by attack type", rotate_x=True)


# ---------------------------------------------------------------------------
# 6. Frames per source video
# ---------------------------------------------------------------------------

def plot_frames_per_video(df: pd.DataFrame) -> plt.Figure:
    """How many frames each source video contributes, per dataset.

    With FRAME_STRIDE=5 we expect ~20% of original frames. Outliers (very
    short videos, or ones where face detection failed on most frames)
    show up as a tail near zero — worth knowing before training.
    """
    counts = df.groupby(["dataset", "source_video"]).size().reset_index(name="frames")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for ds, g in counts.groupby("dataset"):
        ax.hist(g["frames"], bins=40, alpha=0.55, label=ds,
                color=DATASET_COLORS.get(ds))
    ax.set_xlabel("frames retained per source video")
    ax.set_ylabel("video count")
    ax.set_title("Frame yield per source video")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
