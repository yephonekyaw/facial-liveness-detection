"""Apply the 5-stage enhancement pipeline to already-extracted face crops.

Reads every JPEG listed in data/processed/manifest.csv, runs enhance_frame()
on it, and saves the result under data/processed_enhanced/ mirroring the same
subdirectory layout. Then writes data/processed_enhanced/manifest.csv with
paths rewritten to point at the new location.

Original frames in data/processed/ are never modified.

Usage:
    uv run python scripts/apply_enhancement.py           # all rows
    uv run python scripts/apply_enhancement.py --limit 500  # smoke test
    uv run python scripts/apply_enhancement.py --workers 8  # override worker count

To train on enhanced data afterwards, set MANIFEST_PATH in src/config.py to:
    DATA_DIR / "processed_enhanced" / "manifest.csv"
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
from tqdm import tqdm

from src.config import DATA_DIR, PROCESSED_DIR

ENHANCED_DIR = DATA_DIR / "processed_enhanced"
ENHANCED_MANIFEST = ENHANCED_DIR / "manifest.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bake enhancement into a parallel crop directory.")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows (smoke test).")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel worker processes.")
    return p.parse_args()


def _process_one(args: tuple[str, str, str]) -> tuple[str, str]:
    """Worker function: enhance one frame and write to dst. Returns (rel_src_path, rel_dst_path).

    Returning the original rel_src_path signals skip/fail so the caller can
    preserve the manifest row pointing back at the unenhanced crop.
    """
    # Import inside worker so each subprocess initialises its own OpenCV/enhance state.
    from src.preprocessing.enhance import enhance_frame  # noqa: PLC0415

    rel_src, src_path_str, dst_path_str = args
    src_path = Path(src_path_str)
    dst_path = Path(dst_path_str)

    if not src_path.exists():
        return rel_src, rel_src  # sentinel: keep original path

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(src_path))
    if bgr is None:
        return rel_src, rel_src  # sentinel: keep original path

    enhanced = enhance_frame(bgr)
    cv2.imwrite(str(dst_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return rel_src, str(dst_path.relative_to(DATA_DIR))


def _build_tasks(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    tasks = []
    for row in df.itertuples(index=False):
        src_path = DATA_DIR / row.path
        rel_to_processed = src_path.relative_to(PROCESSED_DIR)
        dst_path = ENHANCED_DIR / rel_to_processed
        tasks.append((row.path, str(src_path), str(dst_path)))
    return tasks


def main() -> None:
    args = parse_args()

    src_manifest = PROCESSED_DIR / "manifest.csv"
    if not src_manifest.exists():
        raise FileNotFoundError(f"Source manifest not found: {src_manifest}")

    df = pd.read_csv(src_manifest)
    if args.limit is not None:
        df = df.head(args.limit)

    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(df)
    # Map original rel path → result rel path; pre-fill with originals so order is preserved.
    result_map: dict[str, str] = {rel_src: rel_src for rel_src, _, _ in tasks}

    skipped_or_failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, t): t[0] for t in tasks}
        with tqdm(total=len(tasks), desc="enhancing") as bar:
            for fut in as_completed(futures):
                rel_src, rel_dst = fut.result()
                result_map[rel_src] = rel_dst
                if rel_src == rel_dst:
                    skipped_or_failed += 1
                bar.update(1)

    # Reconstruct new_paths in original row order.
    new_paths = [result_map[row.path] for row in df.itertuples(index=False)]

    out_df = df.copy()
    out_df["path"] = new_paths
    out_df.to_csv(ENHANCED_MANIFEST, index=False)

    total = len(df)
    written = total - skipped_or_failed
    print(f"\nDone: {written} enhanced, {skipped_or_failed} skipped/failed")
    print(f"Enhanced crops → {ENHANCED_DIR}")
    print(f"Enhanced manifest → {ENHANCED_MANIFEST}")
    print(f"Workers used: {args.workers}")
    print()
    print("To train on enhanced data, change MANIFEST_PATH in src/config.py to:")
    print(f"    DATA_DIR / 'processed_enhanced' / 'manifest.csv'")


if __name__ == "__main__":
    main()
