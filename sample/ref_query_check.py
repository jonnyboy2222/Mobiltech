# ============================================================
# File: ref_query_check.py
# ============================================================

import pickle
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

# In actual files, use:
# from ref_map_aggregator import load_ref_map
# from config import QUERY_SEG_PATH, REF_MAP_PATH, CHANGE_RESULT_PATH, RADIUS, ALPHA_OCC, BETA_SEM, GAMMA_FEAT, CHANGE_THRESHOLD


def cosine_similarity(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


class RefQueryChangeChecker:
    """
    Compares query segmentation output against semantic reference voxel map.

    Current version assumes query points are already roughly aligned to ref map.
    Later, replace align_query_to_ref() with GICP/FastGICP.
    """

    def __init__(self, voxel_map):
        self.voxel_map = voxel_map
        arrays = voxel_map.to_arrays()
        self.indices = arrays["indices"]
        self.centers = arrays["centers"]
        self.class_prob = arrays["class_prob"]
        self.feature_mean = arrays["feature_mean"]
        self.kdtree = cKDTree(self.centers)

    def align_query_to_ref(self, query_points):
        """
        Placeholder for registration.

        Current fake pipeline leaves residual pose noise.
        Later replacement:
          - initial pose from GNSS/INS
          - refine using GICP/FastGICP
          - return aligned query points
        """
        return query_points

    def score_point(self, q, q_cls, q_conf, q_feat, radius=0.8,
                    alpha=0.35, beta=0.45, gamma=0.20):
        neighbor_ids = self.kdtree.query_ball_point(q, r=radius)

        if len(neighbor_ids) == 0:
            return {
                "change_score": 1.0,
                "changed_by": "no_neighbor",
                "occupancy_mismatch": 1.0,
                "semantic_diff": 1.0,
                "feature_diff": 1.0,
                "num_neighbors": 0,
            }

        dists = []
        semantic_sims = []
        feature_sims = []

        for nid in neighbor_ids:
            center = self.centers[nid]
            ref_prob = self.class_prob[nid]
            ref_feat = self.feature_mean[nid]

            dists.append(np.linalg.norm(q - center))
            semantic_sims.append(float(ref_prob[int(q_cls)]))
            feature_sims.append(cosine_similarity(q_feat, ref_feat))

        min_dist = float(np.min(dists))
        semantic_sim = float(np.max(semantic_sims))
        feature_sim = float(np.max(feature_sims))

        occupancy_mismatch = 1.0 - np.exp(-min_dist / (radius + 1e-8))
        semantic_diff = 1.0 - semantic_sim
        feature_diff = 1.0 - feature_sim

        score = alpha * occupancy_mismatch + beta * semantic_diff + gamma * feature_diff

        # Low query confidence should reduce certainty, but not erase the signal entirely.
        # This line can be revised later.
        score = score * float(q_conf)

        return {
            "change_score": float(score),
            "changed_by": "score",
            "occupancy_mismatch": float(occupancy_mismatch),
            "semantic_diff": float(semantic_diff),
            "feature_diff": float(feature_diff),
            "num_neighbors": len(neighbor_ids),
        }
    
    def detect(self, query_points, query_labels, query_conf, query_features,
               radius=0.8, alpha=0.35, beta=0.45, gamma=0.20, threshold=0.55):
        query_points = self.align_query_to_ref(query_points)

        scores = np.zeros(len(query_points), dtype=np.float32)
        changed = np.zeros(len(query_points), dtype=bool)
        occ_terms = np.zeros(len(query_points), dtype=np.float32)
        sem_terms = np.zeros(len(query_points), dtype=np.float32)
        feat_terms = np.zeros(len(query_points), dtype=np.float32)
        neighbor_counts = np.zeros(len(query_points), dtype=np.int32)

        for i, (q, q_cls, q_conf, q_feat) in enumerate(zip(query_points, query_labels, query_conf, query_features)):
            out = self.score_point(q, q_cls, q_conf, q_feat, radius, alpha, beta, gamma)
            scores[i] = out["change_score"]
            changed[i] = scores[i] > threshold
            occ_terms[i] = out["occupancy_mismatch"]
            sem_terms[i] = out["semantic_diff"]
            feat_terms[i] = out["feature_diff"]
            neighbor_counts[i] = out["num_neighbors"]

        return {
            "points": query_points,
            "change_scores": scores,
            "changed": changed,
            "occupancy_mismatch": occ_terms,
            "semantic_diff": sem_terms,
            "feature_diff": feat_terms,
            "neighbor_counts": neighbor_counts,
        }
    
def run_change_check(query_seg_path, ref_map_path, result_path,
                     radius=0.8, alpha=0.35, beta=0.45, gamma=0.20, threshold=0.55):
    with open(ref_map_path, "rb") as f:
        ref_map = pickle.load(f)

    data = np.load(query_seg_path)
    query_points = data["points"]
    query_labels = data["pred_labels"]
    query_conf = data["conf"]
    query_features = data["features"]
    query_gt = data["gt_labels"]

    checker = RefQueryChangeChecker(ref_map)
    result = checker.detect(
        query_points, query_labels, query_conf, query_features,
        radius=radius, alpha=alpha, beta=beta, gamma=gamma, threshold=threshold
    )

    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        result_path,
        points=result["points"].astype(np.float32),
        query_gt=query_gt.astype(np.int64),
        change_scores=result["change_scores"].astype(np.float32),
        changed=result["changed"].astype(bool),
        occupancy_mismatch=result["occupancy_mismatch"].astype(np.float32),
        semantic_diff=result["semantic_diff"].astype(np.float32),
        feature_diff=result["feature_diff"].astype(np.float32),
        neighbor_counts=result["neighbor_counts"].astype(np.int32),
    )

    print(f"[Saved Change Result] {result_path}")
    print(f"  points={len(query_points)}")
    print(f"  changed={int(result['changed'].sum())}")
    print(f"  changed_ratio={float(result['changed'].mean()):.4f}")
    print(f"  mean_score={float(result['change_scores'].mean()):.4f}")
    return result