import numpy as np
from dataclasses import dataclass


@dataclass
class VoxelCell:
    key: tuple
    center: np.ndarray
    points: np.ndarray
    indices: np.ndarray
    count: int
    mean_xyz: np.ndarray
    var_xyz: np.ndarray


def point_to_voxel_key(point, voxel_size, origin):
    return tuple(np.floor((point - origin) / voxel_size).astype(np.int64))


def voxel_key_to_center(key, voxel_size, origin):
    return origin + (np.array(key, dtype=np.float32) + 0.5) * voxel_size


def build_voxel_map(points, voxel_size=0.2, origin=None, fixed_origin=True):
    points = np.asarray(points, dtype=np.float32)

    if origin is None:
        if fixed_origin:
            origin = np.zeros(3, dtype=np.float32)
        else:
            origin = points.min(axis=0).astype(np.float32)
    else:
        origin = np.asarray(origin, dtype=np.float32)

    buckets = {}

    for idx, p in enumerate(points):
        key = point_to_voxel_key(p, voxel_size, origin)
        buckets.setdefault(key, []).append(idx)

    voxel_map = {}

    for key, idxs in buckets.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        pts = points[idxs]
        center = voxel_key_to_center(key, voxel_size, origin)

        mean_xyz = pts.mean(axis=0)
        var_xyz = pts.var(axis=0) + 1e-6

        voxel_map[key] = VoxelCell(
            key=key,
            center=center,
            points=pts,
            indices=idxs,
            count=len(idxs),
            mean_xyz=mean_xyz,
            var_xyz=var_xyz,
        )

    return voxel_map, origin

