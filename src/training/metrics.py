"""Biometric error metrics for face anti-spoofing.

The face-anti-spoofing community uses a specific set of metrics — the
generic ML "accuracy" or "F1" don't really tell the right story when the
two error types have very different operational costs (a falsely
accepted attack is a security breach; a falsely rejected user is an
inconvenience). The standard reference is ISO/IEC 30107-3.

Definitions in this module (label convention: 0 = bonafide, 1 = attack):

- **APCER** — Attack Presentation Classification Error Rate.
  Fraction of attack samples that the model classifies as bonafide.
  i.e. attacks that *got through* the system. APCER = FN_attack / N_attack.

- **BPCER** — Bonafide Presentation Classification Error Rate.
  Fraction of bonafide samples classified as attack — legitimate users
  that the system incorrectly rejects. BPCER = FP_bonafide / N_bonafide.

- **ACER** — Average Classification Error Rate = (APCER + BPCER) / 2.
  This is what we optimise for: a single number that weights the two
  error types equally.

- **EER** — Equal Error Rate. The threshold-independent error: vary the
  decision threshold and find the point where APCER == BPCER. The value
  at that crossover is the EER. Useful because it doesn't depend on the
  arbitrary choice of operating threshold.

- **HTER** — Half Total Error Rate. Same formula as ACER, but evaluated
  at a *fixed* threshold (typically the EER threshold from the dev set,
  applied to the test set). It's the standard cross-protocol metric.

- **ROC-AUC** — Standard ROC area-under-curve, threshold-independent.

All functions accept numpy arrays of probabilities (post-sigmoid) and
integer labels. They are intentionally short — easy to audit by reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class FrameMetrics:
    """Bundle of metrics computed at a single decision threshold.

    Threshold-independent metrics (eer, eer_threshold, roc_auc) are also
    included since they are computed from the same `(scores, labels)`
    inputs and the caller almost always wants them together.
    """

    threshold: float
    apcer: float
    bpcer: float
    acer: float
    eer: float
    eer_threshold: float
    roc_auc: float
    n_attack: int
    n_bonafide: int

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "apcer": self.apcer,
            "bpcer": self.bpcer,
            "acer": self.acer,
            "eer": self.eer,
            "eer_threshold": self.eer_threshold,
            "roc_auc": self.roc_auc,
            "n_attack": self.n_attack,
            "n_bonafide": self.n_bonafide,
        }


def apcer_bpcer(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """APCER and BPCER at a fixed threshold.

    `scores`: probability of class 1 (attack) for each sample.
    `labels`: 0 = bonafide, 1 = attack.

    Predicted attack iff score >= threshold.
    """
    pred_attack = scores >= threshold
    is_attack = labels == 1
    is_bonafide = labels == 0

    # APCER: of all attacks, how many were classified bonafide?
    n_attack = int(is_attack.sum())
    apcer = float((~pred_attack & is_attack).sum() / n_attack) if n_attack else 0.0

    # BPCER: of all bonafide, how many were classified attack?
    n_bonafide = int(is_bonafide.sum())
    bpcer = float((pred_attack & is_bonafide).sum() / n_bonafide) if n_bonafide else 0.0

    return apcer, bpcer


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find the threshold where APCER == BPCER and return (EER, threshold).

    We use sklearn's roc_curve to enumerate every operating point. roc_curve
    returns (fpr, tpr, thr) with positive class = 1 (attack):
        fpr  = bonafide-rejected rate    = BPCER
        tpr  = attack-accepted-as-attack = 1 - APCER
        ⇒ APCER = 1 - tpr,   BPCER = fpr
    The EER is where fpr == 1 - tpr, i.e. the curve crosses the
    anti-diagonal. We interpolate between the two surrounding samples
    rather than picking the closest, which reduces the error from
    discretisation when scores cluster.
    """
    if len(np.unique(labels)) < 2:
        # Degenerate: either all attacks or all bonafide. EER undefined.
        return 0.0, 0.5

    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr  # = APCER

    # Find the index where fnr crosses fpr. Sign of (fnr - fpr) flips
    # at the crossover; argmin of |fnr - fpr| is the closest sampled point.
    diff = fnr - fpr
    # Locate the first index where diff <= 0 (i.e. fnr <= fpr).
    idx = np.argmin(np.abs(diff))

    # Linear interpolate between idx-1 and idx for a smoother estimate.
    if 0 < idx < len(diff) and diff[idx - 1] * diff[idx] < 0:
        # Interpolation factor t in [0, 1] s.t. (1-t)*diff[idx-1] + t*diff[idx] == 0.
        t = diff[idx - 1] / (diff[idx - 1] - diff[idx])
        eer = float((1 - t) * fnr[idx - 1] + t * fnr[idx])
        eer_thr = float((1 - t) * thr[idx - 1] + t * thr[idx])
    else:
        eer = float(fnr[idx])
        eer_thr = float(thr[idx])
    return eer, eer_thr


def compute_frame_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> FrameMetrics:
    """Compute the full metrics bundle from per-sample scores and labels."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    apcer, bpcer = apcer_bpcer(scores, labels, threshold)
    eer, eer_thr = equal_error_rate(scores, labels)

    if len(np.unique(labels)) >= 2:
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        roc_auc = float("nan")

    return FrameMetrics(
        threshold=float(threshold),
        apcer=apcer,
        bpcer=bpcer,
        acer=(apcer + bpcer) / 2,
        eer=eer,
        eer_threshold=eer_thr,
        roc_auc=roc_auc,
        n_attack=int((labels == 1).sum()),
        n_bonafide=int((labels == 0).sum()),
    )


def hter_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Half Total Error Rate at an externally-fixed threshold.

    Used for cross-protocol evaluation: we calibrate the threshold on the
    development split and report HTER on the test split using that same
    threshold.
    """
    apcer, bpcer = apcer_bpcer(scores, labels, threshold)
    return (apcer + bpcer) / 2


def aggregate_video_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    source_videos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Group frame scores by source video, return (video_scores, video_labels).

    Aggregation is *mean of frame probabilities*. A few alternatives
    exist (median, max, voting), but mean is the most stable when
    individual frames are noisy.

    All frames within a single source video must share the same label —
    we assert this since silent label leakage would be subtle.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    source_videos = np.asarray(source_videos)

    unique_videos = np.unique(source_videos)
    v_scores = np.zeros(len(unique_videos), dtype=np.float64)
    v_labels = np.zeros(len(unique_videos), dtype=np.int64)
    for i, vid in enumerate(unique_videos):
        mask = source_videos == vid
        v_scores[i] = scores[mask].mean()
        # Sanity: every frame from a video shares its label.
        vid_labels = labels[mask]
        assert (vid_labels == vid_labels[0]).all(), (
            f"video {vid!r} has mixed labels — manifest is corrupt"
        )
        v_labels[i] = vid_labels[0]
    return v_scores, v_labels
