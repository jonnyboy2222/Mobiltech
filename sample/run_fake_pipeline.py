# ============================================================
# File: run_fake_pipeline.py
# ============================================================

# In actual file, use real imports:
from config import *
from fake_data_generator import save_fake_segmentation
from ref_map_aggregator import build_ref_map_from_segmentation
from ref_query_check import run_change_check


def run_pipeline():
    """
    End-to-end fake pipeline:
      1. Generate fake reference segmentation output
      2. Generate fake query segmentation output with changes
      3. Build semantic reference voxel map
      4. Compare query against ref map
      5. Save change result
    """
    ensure_dirs()

    save_fake_segmentation(
        REF_SEG_PATH,
        version="ref",
        num_classes=NUM_CLASSES,
        feat_dim=FEAT_DIM,
        noise_ratio=0.03,
        apply_pose_noise=False,
    )

    save_fake_segmentation(
        QUERY_SEG_PATH,
        version="query",
        num_classes=NUM_CLASSES,
        feat_dim=FEAT_DIM,
        noise_ratio=0.05,
        apply_pose_noise=True,
        translation=QUERY_TRANSLATION_NOISE,
        yaw_deg=QUERY_YAW_NOISE_DEG,
    )

    build_ref_map_from_segmentation(
        REF_SEG_PATH,
        REF_MAP_PATH,
        voxel_size=VOXEL_SIZE,
        num_classes=NUM_CLASSES,
        feat_dim=FEAT_DIM,
    )

    run_change_check(
        QUERY_SEG_PATH,
        REF_MAP_PATH,
        CHANGE_RESULT_PATH,
        radius=RADIUS,
        alpha=ALPHA_OCC,
        beta=BETA_SEM,
        gamma=GAMMA_FEAT,
        threshold=CHANGE_THRESHOLD,
    )


# if __name__ == "__main__":
#     run_pipeline()
