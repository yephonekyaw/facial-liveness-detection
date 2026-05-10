"""Run all test-set evaluations and save results to a JSON file.

Evaluations:
1. Within-dataset: best CV fold checkpoint → test split (same dataset)
   - combined (both-cv-5/fold_1) → test on [replay, 3dmad]
   - replay  (replay-cv-5/fold_0) → test on [replay]
   - 3dmad   (3dmad-cv-5/fold_0)  → test on [3dmad]

2. Cross-dataset: single-run checkpoints → test split (other dataset)
   - replay/best.pt  → test on [3dmad]
   - 3dmad/best.pt   → test on [replay]

3. Cross-dataset with devel-calibrated threshold (eval-cross):
   - replay/best.pt  → devel+test on [3dmad]
   - 3dmad/best.pt   → devel+test on [replay]
"""

from __future__ import annotations

import json
from pathlib import Path

from src.training.eval_runner import evaluate_cross, evaluate_split

CKPT_DIR = Path("outputs/checkpoints")
OUT_PATH = Path("outputs/test_eval_results.json")


def main():
    results = {}

    # --- 1. Within-dataset test evals (best CV fold) ---
    within = {
        "combined": {
            "ckpt": CKPT_DIR / "both-cv-5" / "fold_1" / "best.pt",
            "datasets": ["replay", "3dmad"],
        },
        "replay": {
            "ckpt": CKPT_DIR / "replay-cv-5" / "fold_0" / "best.pt",
            "datasets": ["replay"],
        },
        "3dmad": {
            "ckpt": CKPT_DIR / "3dmad-cv-5" / "fold_0" / "best.pt",
            "datasets": ["3dmad"],
        },
    }
    results["within_dataset"] = {}
    for name, spec in within.items():
        print(f"\n{'='*60}")
        print(f"  WITHIN-DATASET TEST: {name}")
        print(f"{'='*60}")
        r = evaluate_split(
            ckpt_path=spec["ckpt"],
            split="test",
            eval_datasets=spec["datasets"],
        )
        results["within_dataset"][name] = r

    # --- 2. Cross-dataset test evals (single-run models) ---
    # Use the single-run models (not CV folds) because they were trained
    # on the standard train split, so devel calibration is clean.
    cross_split = {
        "replay_to_3dmad": {
            "ckpt": CKPT_DIR / "replay" / "best.pt",
            "datasets": ["3dmad"],
        },
        "3dmad_to_replay": {
            "ckpt": CKPT_DIR / "3dmad" / "best.pt",
            "datasets": ["replay"],
        },
    }
    results["cross_dataset_raw"] = {}
    for name, spec in cross_split.items():
        print(f"\n{'='*60}")
        print(f"  CROSS-DATASET TEST (fixed threshold): {name}")
        print(f"{'='*60}")
        r = evaluate_split(
            ckpt_path=spec["ckpt"],
            split="test",
            eval_datasets=spec["datasets"],
        )
        results["cross_dataset_raw"][name] = r

    # --- 3. Cross-dataset with devel-calibrated threshold ---
    results["cross_dataset_calibrated"] = {}
    for name, spec in cross_split.items():
        print(f"\n{'='*60}")
        print(f"  CROSS-DATASET (devel-calibrated HTER): {name}")
        print(f"{'='*60}")
        r = evaluate_cross(
            ckpt_path=spec["ckpt"],
            eval_datasets=spec["datasets"],
        )
        results["cross_dataset_calibrated"][name] = r

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n== All results saved to {OUT_PATH} ==")


if __name__ == "__main__":
    main()
