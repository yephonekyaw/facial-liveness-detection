"""Project-wide constants and paths.

Single source of truth for filesystem layout, image size, random seed,
and device. Other modules should import from here rather than hardcoding.
"""

from pathlib import Path

import torch

# --- Filesystem ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FACE_CACHE_DIR = DATA_DIR / "face_cache"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
DOCS_DIR = PROJECT_ROOT / "docs"

# Per-dataset raw locations
REPLAY_RAW = RAW_DIR / "replay-attack"
MASK3D_RAW = RAW_DIR / "3d-mask-attack"
CSMAD_RAW = RAW_DIR / "custom-silicon-mask-attack"

# Per-dataset processed locations
REPLAY_OUT = PROCESSED_DIR / "replay_attack"
MASK3D_OUT = PROCESSED_DIR / "3dmad"
CSMAD_OUT = PROCESSED_DIR / "csmad"

# --- Model / data constants ---
IMAGE_SIZE = 256          # 256x256 RGB
FRAME_STRIDE = 5          # keep every 5th frame from each video
FACE_MARGIN = 0.3         # expand bbox by 30% before square crop
SEED = 42

# --- Optional 5-stage enhancement pipeline ---
# When True, each crop is run through bilateral denoise → LAB-CLAHE →
# unsharp mask → gamma → linear contrast before being written to disk.
# Toggling this requires re-running preprocessing (the result is baked
# into the JPEGs under data/processed/).
ENABLE_ENHANCEMENT = False
# Bilateral kept light (σ=15) so it removes sensor noise without erasing
# the moiré/paper-grain texture cues an anti-spoof model relies on.
ENH_BILATERAL_SIGMA_COLOR = 15   # range sigma (intensity)
ENH_BILATERAL_SIGMA_SPACE = 15   # spatial sigma (pixels)
ENH_CLAHE_CLIP_LIMIT = 3.0
ENH_CLAHE_TILE_SIZE = (8, 8)
ENH_UNSHARP_SIGMA = 2.0
ENH_UNSHARP_AMOUNT = 1.5
ENH_GAMMA = 1.2
ENH_CONTRAST_ALPHA = 1.1
ENH_CONTRAST_BETA = 5

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
