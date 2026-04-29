# ============================================================
# File: fake_data_generator.py
# ============================================================

import numpy as np
from pathlib import Path

# In actual files, use:
# from config import REF_SEG_PATH, QUERY_SEG_PATH, NUM_CLASSES, FEAT_DIM, ensure_dirs

CLASS_NAMES = {
    0: "ground",
    1: "building",
    2: "tree",
    3: "new_object",
}


def make_plane(n=3000, z=0.0, size=20.0, cls=0):
    x = np.random.uniform(-size, size, n)
    y = np.random.uniform(-size, size, n)
    z = np.ones(n) * z
    pts = np.stack([x, y, z], axis=1)
    labels = np.ones(n, dtype=np.int64) * cls
    return pts, labels


def make_cube(center, size=2.0, n=1000, cls=1):
    pts = np.random.uniform(-size / 2, size / 2, (n, 3)) + np.array(center)
    labels = np.ones(n, dtype=np.int64) * cls
    return pts, labels


def make_cylinder(center, radius=0.8, height=3.0, n=1000, cls=2):
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(np.random.uniform(0, 1, n))
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    z = np.random.uniform(0, height, n) + center[2]
    pts = np.stack([x, y, z], axis=1)
    labels = np.ones(n, dtype=np.int64) * cls
    return pts, labels


def make_scene(version="ref"):
    """
    Fake scene generator.

    version='ref': reference time t0.
    version='query': changed time t1.

    Change definition:
      - ref has a tree/cylinder at [4, 3, 0]
      - query removes that tree
      - query adds a new cube at [5, -4, 1]
    """

    pts_list, labels_list = [], []

    p, l = make_plane(n=3000, cls=0)
    pts_list.append(p)
    labels_list.append(l)

    p, l = make_cube(center=[-5, 0, 1], size=3.0, n=1200, cls=1)
    pts_list.append(p)
    labels_list.append(l)

    if version == "ref":
        p, l = make_cylinder(center=[4, 3, 0], radius=0.7, height=4.0, n=900, cls=2)
        pts_list.append(p)
        labels_list.append(l)

    if version == "query":
        p, l = make_cube(center=[5, -4, 1], size=2.5, n=1000, cls=3)
        pts_list.append(p)
        labels_list.append(l)

    points = np.concatenate(pts_list, axis=0)
    gt_labels = np.concatenate(labels_list, axis=0)
    return points, gt_labels

def simulate_segmentation(points, gt_labels, num_classes=4, feat_dim=16, noise_ratio=0.05):
    """
    Simulates output of a pretrained 3D semantic segmentation model.

    Output is point-wise:
      - pred_labels: predicted semantic class
      - conf: max softmax confidence-like score
      - features: final decoder/head feature embedding
    """
    n = len(points)
    pred_labels = gt_labels.copy()

    noise_mask = np.random.rand(n) < noise_ratio
    pred_labels[noise_mask] = np.random.randint(0, num_classes, noise_mask.sum())

    conf = np.random.uniform(0.75, 1.0, n)
    conf[noise_mask] = np.random.uniform(0.3, 0.7, noise_mask.sum())

    prototypes = np.random.randn(num_classes, feat_dim)
    features = prototypes[pred_labels] + 0.1 * np.random.randn(n, feat_dim)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    return pred_labels, conf, features


def apply_transform(points, translation=(0.3, -0.2, 0.0), yaw_deg=2.0):
    """
    Applies fake pose/registration noise to query points.
    Later this will be replaced by real pose + GICP/FastGICP alignment.
    """
    yaw = np.deg2rad(yaw_deg)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array(translation)
    return points @ R.T + t


def save_fake_segmentation(save_path, version, num_classes=4, feat_dim=16, noise_ratio=0.05,
                           apply_pose_noise=False, translation=(0.3, -0.2, 0.0), yaw_deg=2.0):
    points, gt_labels = make_scene(version)
    pred_labels, conf, features = simulate_segmentation(
        points, gt_labels, num_classes=num_classes, feat_dim=feat_dim, noise_ratio=noise_ratio
    )

    if apply_pose_noise:
        points = apply_transform(points, translation=translation, yaw_deg=yaw_deg)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        points=points.astype(np.float32),
        gt_labels=gt_labels.astype(np.int64),
        pred_labels=pred_labels.astype(np.int64),
        conf=conf.astype(np.float32),
        features=features.astype(np.float32),
        version=np.array(version),
    )
    print(f"[Saved] {save_path} | points={len(points)} | version={version}")