"""Run every (model, test-set) evaluation pair and save results to JSON.

Models (single-run checkpoints under outputs/checkpoints/):
  - both    → trained on replay + 3dmad
  - replay  → trained on replay only
  - 3dmad   → trained on 3dmad only

Test sets (test split of the manifest):
  - replay
  - 3dmad
  - csmad
  - combined  (replay + 3dmad + csmad together)

For each (model, test-set) pair we record:
  1. Raw test metrics at the default 0.5 threshold (`evaluate_split`).
  2. Devel-calibrated HTER metrics (`evaluate_cross`) — threshold picked on
     the devel split of the same test set, then applied to test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.eval_runner import evaluate_cross, evaluate_split

CKPT_DIR = Path("outputs/checkpoints")
OUT_PATH = Path("outputs/test_eval_results.json")

MODELS = {
    "both":   CKPT_DIR / "both"   / "best.pt",
    "replay": CKPT_DIR / "replay" / "best.pt",
    "3dmad":  CKPT_DIR / "3dmad"  / "best.pt",
}

TEST_SETS = {
    "replay":   ["replay"],
    "3dmad":    ["3dmad"],
    "csmad":    ["csmad"],
    "combined": ["replay", "3dmad", "csmad"],
}


def main():
    results = {"raw": {}, "calibrated": {}}

    for model_name, ckpt in MODELS.items():
        results["raw"][model_name] = {}
        results["calibrated"][model_name] = {}

        for test_name, datasets in TEST_SETS.items():
            print(f"\n{'='*60}")
            print(f"  RAW  | model={model_name}  test={test_name}")
            print(f"{'='*60}")
            results["raw"][model_name][test_name] = evaluate_split(
                ckpt_path=ckpt,
                split="test",
                eval_datasets=datasets,
            )

            print(f"\n{'='*60}")
            print(f"  CALIBRATED  | model={model_name}  test={test_name}")
            print(f"{'='*60}")
            results["calibrated"][model_name][test_name] = evaluate_cross(
                ckpt_path=ckpt,
                eval_datasets=datasets,
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n== All results saved to {OUT_PATH} ==")


if __name__ == "__main__":
    main()
