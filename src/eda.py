"""Exploratory analysis on the unified dataset.

Functions here are called from `notebooks/01_eda.ipynb`. They operate on
the manifest CSV, sampling the actual JPEG crops only when needed (so
class-distribution analysis doesn't pay any image I/O).

Three things we want to see:
1. Class distribution (per dataset, split, attack-type, lighting).
2. Per-image quality (sharpness, contrast, brightness, blur) — to know
   if any dataset is systematically harder than others.
3. Feature-space layout via PCA / t-SNE on a frozen resnet18, to visualise
   whether bonafide vs attack are separable and whether datasets cluster
   apart (which would visually motivate combined-dataset training).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import models, transforms
from tqdm import tqdm

from src.config import DEVICE, PROJECT_ROOT, SEED


# ---------------------------------------------------------------------------
# 1. Class distribution
# ---------------------------------------------------------------------------

def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Frame counts pivoted by dataset / split / label."""
    table = df.groupby(["dataset", "split", "label"]).size().unstack(fill_value=0)
    table.columns = ["bonafide", "attack"]
    table["total"] = table.sum(axis=1)
    table["attack_ratio"] = (table["attack"] / table["total"]).round(3)
    return table


def attack_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["dataset", "attack_type"]).size().unstack(fill_value=0)


# ---------------------------------------------------------------------------
# 2. Sample grid
# ---------------------------------------------------------------------------

def _imread_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sample_grid(
    df: pd.DataFrame,
    rows: int = 4,
    cols: int = 4,
    title: str | None = None,
    seed: int = SEED,
) -> plt.Figure:
    """Plot a rows x cols grid of crops sampled from `df`."""
    sub = df.sample(n=min(rows * cols, len(df)), random_state=seed)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1)
    for ax, (_, row) in zip(axes, sub.iterrows()):
        img = _imread_rgb(PROJECT_ROOT / "data" / row["path"])
        ax.imshow(img)
        ax.set_title(f"{row['attack_type']}\n{row['lighting']}", fontsize=7)
        ax.axis("off")
    for ax in axes[len(sub):]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Image quality metrics
# ---------------------------------------------------------------------------
#
# Four standard image-quality metrics:
#   sharpness  : variance of the Laplacian (high = sharp)
#   contrast   : RMS contrast (std of grayscale)
#   brightness : mean grayscale value (40-220 = "well-exposed")
#   blur       : Tenengrad gradient magnitude (high = sharp edges)


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rms_contrast(gray: np.ndarray) -> float:
    return float(gray.astype(np.float32).std())


def mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def compute_quality_metrics(df: pd.DataFrame, n_sample: int = 5000, seed: int = SEED) -> pd.DataFrame:
    """Sample n_sample rows (stratified by dataset+label), compute the 4 metrics.

    Returns a DataFrame with columns: dataset, label, attack_type, lighting,
    sharpness, contrast, brightness, tenengrad.
    """
    rng = np.random.default_rng(seed)
    # Stratify so each (dataset, label) cell gets a fair share.
    pieces = []
    groups = df.groupby(["dataset", "label"], group_keys=False)
    per_group = max(50, n_sample // max(1, len(groups)))
    for _, g in groups:
        take = min(per_group, len(g))
        pieces.append(g.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    sub = pd.concat(pieces).reset_index(drop=True)

    rows = []
    for _, r in tqdm(sub.iterrows(), total=len(sub), desc="quality metrics"):
        img = cv2.imread(str(PROJECT_ROOT / "data" / r["path"]), cv2.IMREAD_GRAYSCALE)
        rows.append(
            {
                "dataset": r["dataset"],
                "label": r["label"],
                "attack_type": r["attack_type"],
                "lighting": r["lighting"],
                "sharpness": laplacian_var(img),
                "contrast": rms_contrast(img),
                "brightness": mean_brightness(img),
                "tenengrad": tenengrad(img),
            }
        )
    return pd.DataFrame(rows)


def quality_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    agg = metrics.groupby("dataset")[["sharpness", "contrast", "brightness", "tenengrad"]]
    return agg.agg(["mean", "std"]).round(2)


def quality_distribution_plot(metrics: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.flat, ["sharpness", "contrast", "brightness", "tenengrad"]):
        for ds, g in metrics.groupby("dataset"):
            ax.hist(g[col], bins=40, alpha=0.45, label=ds, density=True)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Feature-space visualisation (PCA + t-SNE on frozen resnet18)
# ---------------------------------------------------------------------------

def _resnet18_feature_extractor() -> torch.nn.Module:
    """Pretrained resnet18 with the final fc replaced by Identity → 512-d features.

    We use ImageNet weights only to get a reasonable embedding for
    visualisation; the actual liveness model is trained from scratch.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    return model.to(DEVICE).eval()


def _resnet_eval_transform():
    # ResNet was trained on ImageNet stats; matching them gives a more
    # meaningful embedding even though we just want clusters.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.ToTensor(),                # HWC uint8 BGR -> CHW float [0,1]
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def extract_features(
    df: pd.DataFrame,
    n_per_class: int = 1000,
    batch_size: int = 64,
    seed: int = SEED,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Sample frames stratified by (dataset, label) and embed with resnet18.

    Returns (features [N, 512], rows DataFrame [N, ...]) where rows[i]
    is the manifest row for features[i].
    """
    rng = np.random.default_rng(seed)
    sampled = []
    for _, g in df.groupby(["dataset", "label"]):
        take = min(n_per_class, len(g))
        sampled.append(g.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    sub = pd.concat(sampled).reset_index(drop=True)

    model = _resnet18_feature_extractor()
    tfm = _resnet_eval_transform()

    features = np.zeros((len(sub), 512), dtype=np.float32)
    for i in tqdm(range(0, len(sub), batch_size), desc="resnet18 features"):
        chunk = sub.iloc[i : i + batch_size]
        batch = torch.stack(
            [
                tfm(cv2.cvtColor(cv2.imread(str(PROJECT_ROOT / "data" / r["path"])), cv2.COLOR_BGR2RGB))
                for _, r in chunk.iterrows()
            ]
        ).to(DEVICE)
        out = model(batch).cpu().numpy()
        features[i : i + len(chunk)] = out
    return features, sub


def pca_2d(features: np.ndarray, seed: int = SEED) -> np.ndarray:
    return PCA(n_components=2, random_state=seed).fit_transform(features)


def tsne_2d(features: np.ndarray, perplexity: float = 30.0, seed: int = SEED) -> np.ndarray:
    # Reduce to ~50 dims with PCA first — standard t-SNE practice; speeds
    # up the t-SNE step by ~3x with no quality loss for our scale.
    pca50 = PCA(n_components=50, random_state=seed).fit_transform(features)
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(pca50)


def scatter_2d(coords: np.ndarray, labels: pd.Series, title: str, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    for cat in sorted(labels.unique(), key=str):
        mask = labels == cat
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.55, label=str(cat))
    ax.set_title(title)
    ax.legend(fontsize=7, markerscale=2, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
