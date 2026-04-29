# ============================================================
# File: ref_map_aggregator.py
# ============================================================

import pickle
import numpy as np
from pathlib import Path

# In actual files, use:
# from config import VOXEL_SIZE, NUM_CLASSES, FEAT_DIM, REF_SEG_PATH, REF_MAP_PATH


class VoxelMap:
    """
    Semantic reference map.

    The map is not a single argmax label map only.
    Each voxel stores:
      - class_score distribution
      - feature_sum / feature_mean
      - count
      - occupancy
      - final_class, final_conf for summary/visualization
    """

    def __init__(self, voxel_size=0.5, num_classes=4, feat_dim=16):
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.cells = {}

    def voxel_index(self, xyz):
        return tuple(np.floor(xyz / self.voxel_size).astype(int))

    def voxel_center(self, idx):
        return (np.array(idx, dtype=np.float32) + 0.5) * self.voxel_size

    def add_points(self, points, labels, confs, features):
        for p, cls, conf, feat in zip(points, labels, confs, features):
            idx = self.voxel_index(p)

            if idx not in self.cells:
                self.cells[idx] = {
                    "center": self.voxel_center(idx),
                    "class_score": np.zeros(self.num_classes, dtype=np.float32),
                    "feature_sum": np.zeros(self.feat_dim, dtype=np.float32),
                    "feature_weight": 0.0,
                    "count": 0,
                    "occupancy": True,
                }

            cell = self.cells[idx]
            cell["class_score"][int(cls)] += float(conf)
            cell["feature_sum"] += float(conf) * feat
            cell["feature_weight"] += float(conf)
            cell["count"] += 1

    def finalize(self):
        for cell in self.cells.values():
            score = cell["class_score"]
            total = float(score.sum()) + 1e-8
            cell["class_prob"] = score / total
            cell["final_class"] = int(np.argmax(score))
            cell["final_conf"] = float(score.max() / total)
            cell["feature_mean"] = cell["feature_sum"] / (float(cell["feature_weight"]) + 1e-8)

    def to_arrays(self):
        indices = list(self.cells.keys())
        centers = np.array([self.cells[idx]["center"] for idx in indices], dtype=np.float32)
        class_prob = np.array([self.cells[idx]["class_prob"] for idx in indices], dtype=np.float32)
        feature_mean = np.array([self.cells[idx]["feature_mean"] for idx in indices], dtype=np.float32)
        final_class = np.array([self.cells[idx]["final_class"] for idx in indices], dtype=np.int64)
        final_conf = np.array([self.cells[idx]["final_conf"] for idx in indices], dtype=np.float32)
        counts = np.array([self.cells[idx]["count"] for idx in indices], dtype=np.int64)
        return {
            "indices": indices,
            "centers": centers,
            "class_prob": class_prob,
            "feature_mean": feature_mean,
            "final_class": final_class,
            "final_conf": final_conf,
            "counts": counts,
        }


def build_ref_map_from_segmentation(seg_path, map_path, voxel_size=0.5, num_classes=4, feat_dim=16):
    data = np.load(seg_path)
    points = data["points"]
    labels = data["pred_labels"]
    confs = data["conf"]
    features = data["features"]

    voxel_map = VoxelMap(voxel_size=voxel_size, num_classes=num_classes, feat_dim=feat_dim)
    voxel_map.add_points(points, labels, confs, features)
    voxel_map.finalize()

    map_path = Path(map_path)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, "wb") as f:
        pickle.dump(voxel_map, f)

    print(f"[Saved Ref Map] {map_path} | voxels={len(voxel_map.cells)}")
    return voxel_map

def load_ref_map(map_path):
    with open(map_path, "rb") as f:
        return pickle.load(f)