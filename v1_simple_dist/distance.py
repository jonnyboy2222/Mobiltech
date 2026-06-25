import numpy as np

from utils.voxelization import point_to_voxel_key
from utils.voxel_hash import query_neighbor_voxels


def point_to_voxel_distance(
    point,
    target_voxel_map,
    voxel_size,
    origin,
    radius=1,
    empty_score=1e6,
):
    key = point_to_voxel_key(point, voxel_size, origin)
    neighbors = query_neighbor_voxels(target_voxel_map, key, radius)

    if len(neighbors) == 0:
        return empty_score

    centers = np.stack([v.center for v in neighbors], axis=0)
    dists = np.linalg.norm(centers - point[None, :], axis=1)

    return float(dists.min())


def directional_distance_scores(
    source_points,
    target_voxel_map,
    voxel_size,
    target_origin,
    radius=1,
    empty_score=1e6,
):
    scores = np.zeros(len(source_points), dtype=np.float32)

    for i, p in enumerate(source_points):
        scores[i] = point_to_voxel_distance(
            point=p,
            target_voxel_map=target_voxel_map,
            voxel_size=voxel_size,
            origin=target_origin,
            radius=radius,
            empty_score=empty_score,
        )

    return scores


def scores_to_binary_labels(scores, threshold):
    return (scores > threshold).astype(np.int64)