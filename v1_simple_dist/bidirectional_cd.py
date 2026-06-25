from utils.voxelization import build_voxel_map
from v1_simple_dist.distance import directional_distance_scores, scores_to_binary_labels


def run_bidirectional_distance_cd(pc2016, pc2020, voxel_size=0.2, radius=1,
                                  tau_removed=0.3, tau_added=0.3):
    voxel2016, origin2016 = build_voxel_map(pc2016, voxel_size=voxel_size)
    voxel2020, origin2020 = build_voxel_map(pc2020, voxel_size=voxel_size)

    removed_scores = directional_distance_scores(
        source_points=pc2016,
        target_voxel_map=voxel2020,
        voxel_size=voxel_size,
        target_origin=origin2020,
        radius=radius,
    )

    added_scores = directional_distance_scores(
        source_points=pc2020,
        target_voxel_map=voxel2016,
        voxel_size=voxel_size,
        target_origin=origin2016,
        radius=radius,
    )

    removed_pred = scores_to_binary_labels(removed_scores, tau_removed)
    added_pred = scores_to_binary_labels(added_scores, tau_added)

    return {
        "removed_scores": removed_scores,
        "added_scores": added_scores,
        "removed_pred": removed_pred,
        "added_pred": added_pred,
    }