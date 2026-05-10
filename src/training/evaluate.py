"""Evaluation: run the model over a split, report frame + video metrics.

`evaluate_loader` is the workhorse — it runs a forward pass over a
DataLoader and returns the raw `(scores, labels, source_videos,
attack_types, datasets)` arrays. The reporting helpers
(`frame_report`, `video_report`, `per_attack_report`) are pure-numpy
post-processing on those arrays, so they're cheap to re-run with
different thresholds or filters.

We deliberately keep the loader iteration and the metric computation
separate. That way the training loop can call `evaluate_loader` once per
epoch and slice the results different ways without re-running the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import (
    FrameMetrics,
    aggregate_video_scores,
    compute_frame_metrics,
    hter_at_threshold,
)


@dataclass
class EvalArrays:
    """Raw per-sample arrays from a single evaluation pass."""

    scores: np.ndarray         # post-sigmoid probabilities, shape (N,)
    labels: np.ndarray         # 0/1, shape (N,)
    source_videos: np.ndarray  # str, shape (N,)
    attack_types: np.ndarray   # str, shape (N,)
    datasets: np.ndarray       # str, shape (N,)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    desc: str = "eval",
) -> EvalArrays:
    """Run the model over `loader` once, collecting per-sample outputs.

    The DataLoader is expected to yield `(image, label, SampleMeta)` —
    construct the dataset with `return_meta=True`. We need `source_video`
    for video-level aggregation and `attack_type` / `dataset` for
    breakdown reports.
    """
    model.eval()
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    video_chunks: list[list[str]] = []
    attack_chunks: list[list[str]] = []
    dataset_chunks: list[list[str]] = []

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else _NullCtx()
    )

    for images, labels, meta in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(images)
        # Cast back to fp32 before sigmoid for numerical stability.
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        score_chunks.append(probs)
        label_chunks.append(labels.numpy().astype(np.int64))
        # SampleMeta fields are emitted by the default collate as lists
        # (one entry per item in the batch).
        video_chunks.append(list(meta.source_video))
        attack_chunks.append(list(meta.attack_type))
        dataset_chunks.append(list(meta.dataset))

    return EvalArrays(
        scores=np.concatenate(score_chunks),
        labels=np.concatenate(label_chunks),
        source_videos=np.array([v for chunk in video_chunks for v in chunk]),
        attack_types=np.array([a for chunk in attack_chunks for a in chunk]),
        datasets=np.array([d for chunk in dataset_chunks for d in chunk]),
    )


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def frame_report(arrays: EvalArrays, threshold: float = 0.5) -> FrameMetrics:
    """Frame-level metrics on the full evaluation set."""
    return compute_frame_metrics(arrays.scores, arrays.labels, threshold=threshold)


def video_report(arrays: EvalArrays, threshold: float = 0.5) -> FrameMetrics:
    """Video-level metrics — frame scores aggregated by source_video first."""
    v_scores, v_labels = aggregate_video_scores(arrays.scores, arrays.labels, arrays.source_videos)
    return compute_frame_metrics(v_scores, v_labels, threshold=threshold)


def per_attack_report(arrays: EvalArrays, threshold: float = 0.5) -> pd.DataFrame:
    """One row per attack-type. Useful to spot e.g. mask3d failing while
    print succeeds.

    For each attack type, we compute APCER on its samples (those of
    label=1 with that attack_type) against the full bonafide pool.
    Bonafide BPCER is reported as a single row labelled 'real'.
    """
    rows = []
    is_bonafide = arrays.labels == 0
    bonafide_scores = arrays.scores[is_bonafide]

    # Bonafide: only BPCER is meaningful.
    if len(bonafide_scores) > 0:
        bpcer = float((bonafide_scores >= threshold).sum() / len(bonafide_scores))
    else:
        bpcer = float("nan")
    rows.append({"attack_type": "real", "n": int(is_bonafide.sum()),
                 "apcer": float("nan"), "bpcer": bpcer})

    for attack in sorted(set(arrays.attack_types)):
        if attack == "real":
            continue
        mask = (arrays.attack_types == attack) & (arrays.labels == 1)
        n = int(mask.sum())
        if n == 0:
            continue
        attack_scores = arrays.scores[mask]
        # APCER for this attack family: attacks getting through.
        apcer = float((attack_scores < threshold).sum() / n)
        rows.append({"attack_type": attack, "n": n, "apcer": apcer, "bpcer": float("nan")})
    return pd.DataFrame(rows)


def cross_protocol_hter(
    devel_arrays: EvalArrays,
    test_arrays: EvalArrays,
) -> tuple[float, float, FrameMetrics]:
    """Calibrate threshold on `devel`, report HTER on `test`.

    The standard "honest" cross-protocol metric: we are not allowed to
    look at the test set when picking a threshold. Returns
    (hter_test, eer_threshold_from_dev, full_test_metrics_at_that_threshold).
    """
    dev_metrics = frame_report(devel_arrays)
    thr = dev_metrics.eer_threshold
    test_at_thr = compute_frame_metrics(test_arrays.scores, test_arrays.labels, threshold=thr)
    hter = hter_at_threshold(test_arrays.scores, test_arrays.labels, threshold=thr)
    return hter, thr, test_at_thr


def metrics_summary(metrics: FrameMetrics) -> str:
    """One-line human-readable summary, used by the training loop."""
    return (
        f"acer={metrics.acer:.4f}  apcer={metrics.apcer:.4f}  "
        f"bpcer={metrics.bpcer:.4f}  eer={metrics.eer:.4f}  "
        f"auc={metrics.roc_auc:.4f}"
    )
