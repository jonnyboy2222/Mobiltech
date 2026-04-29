# ============================================================
# File: config.py
# ============================================================

import os
from pathlib import Path

# Root path for all input/output data.
# Only change this environment variable when moving environments.
CD_DATA_ROOT = Path(os.environ.get("CD_DATA_ROOT", "./cd_fake_data")).resolve()

# Subdirectories
REF_DIR = CD_DATA_ROOT / "ref"
QUERY_DIR = CD_DATA_ROOT / "query"
MAP_DIR = CD_DATA_ROOT / "maps"
RESULT_DIR = CD_DATA_ROOT / "results"

# File paths
REF_SEG_PATH = REF_DIR / "ref_segmentation.npz"
QUERY_SEG_PATH = QUERY_DIR / "query_segmentation.npz"
REF_MAP_PATH = MAP_DIR / "semantic_ref_map.pkl"
CHANGE_RESULT_PATH = RESULT_DIR / "change_result.npz"

# Data/model simulation settings
NUM_CLASSES = 4
FEAT_DIM = 16

# Voxel/map settings
VOXEL_SIZE = 0.5

# Change detection settings
RADIUS = 0.8
ALPHA_OCC = 0.35
BETA_SEM = 0.45
GAMMA_FEAT = 0.20
CHANGE_THRESHOLD = 0.55

# Fake pose noise settings
QUERY_TRANSLATION_NOISE = (0.3, -0.2, 0.0)
QUERY_YAW_NOISE_DEG = 2.0


def ensure_dirs():
    for d in [REF_DIR, QUERY_DIR, MAP_DIR, RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
