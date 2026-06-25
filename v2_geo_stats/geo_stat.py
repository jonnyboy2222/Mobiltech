import numpy as np

from utils.voxelization import build_voxel_map
from utils.voxel_hash import query_neighbor_voxels


def geo_stat_distance(src_voxel, tgt_voxel, eps=1e-6, min_var=None):
    diff = src_voxel.mean_xyz - tgt_voxel.mean_xyz
    var = src_voxel.var_xyz + tgt_voxel.var_xyz + eps

    if min_var is not None:
        var = np.maximum(var, min_var)

    return float(np.sum((diff ** 2) / var))


def voxel_to_voxel_geo_stat_scores(
    src_voxel_map,
    tgt_voxel_map,
    radius=1,
    eps=1e-6,
    min_var=0.01,
):
    keys = []
    scores = []
    empty_flags = []

    for key, src_voxel in src_voxel_map.items():
        neighbors = query_neighbor_voxels(tgt_voxel_map, key, radius)

        if len(neighbors) == 0:
            score = np.nan
            empty = 1
        else:
            score = min(
                geo_stat_distance(src_voxel, tgt_voxel, eps=eps, min_var=min_var)
                for tgt_voxel in neighbors
            )
            empty = 0

        keys.append(key)
        scores.append(score)
        empty_flags.append(empty)

    return keys, np.asarray(scores, dtype=np.float32), np.asarray(empty_flags, dtype=np.int64)


def voxel_scores_to_point_scores(points, voxel_map, voxel_keys, voxel_scores):
    score_dict = {k: s for k, s in zip(voxel_keys, voxel_scores)}
    point_scores = np.zeros(len(points), dtype=np.float32)

    for key, voxel in voxel_map.items():
        point_scores[voxel.indices] = score_dict[key]

    return point_scores


def summarize_scores(name, scores, pred=None):
    finite = np.isfinite(scores)
    inf_ratio = 1.0 - float(finite.mean())

    print(f"[{name}] n={len(scores)} inf_ratio={inf_ratio:.4f}")

    if finite.any():
        qs = [50, 75, 90, 95, 99, 99.5, 99.9]
        vals = np.percentile(scores[finite], qs)
        stat = ", ".join([f"p{q}={v:.4f}" for q, v in zip(qs, vals)])
        print(f"[{name}] {stat}")

    if pred is not None:
        print(f"[{name}] positive_ratio={float(pred.mean()):.4f}")


def run_bidirectional_geo_stat_cd(
    pc2016,
    pc2020,
    voxel_size=0.2,
    radius=1,
    tau_removed=1.5,
    tau_added=1.5,
    eps=1e-6,
    min_var=None,
    fixed_origin=True,
    origin=None,
    debug=False,
):
    ref_voxel_map, origin = build_voxel_map(
        pc2016,
        voxel_size=voxel_size,
        origin=origin,
        fixed_origin=fixed_origin,
    )

    query_voxel_map, _ = build_voxel_map(
        pc2020,
        voxel_size=voxel_size,
        origin=origin,
        fixed_origin=fixed_origin,
    )

    removed_keys, removed_scores, removed_empty = voxel_to_voxel_geo_stat_scores(
        src_voxel_map=ref_voxel_map,
        tgt_voxel_map=query_voxel_map,
        radius=radius,
        eps=eps,
        min_var=min_var,
    )

    added_keys, added_scores, added_empty = voxel_to_voxel_geo_stat_scores(
        src_voxel_map=query_voxel_map,
        tgt_voxel_map=ref_voxel_map,
        radius=radius,
        eps=eps,
        min_var=min_var,
    )

    removed_voxel_pred = ((removed_empty == 1) | (removed_scores > tau_removed)).astype(np.int64)
    added_voxel_pred = ((added_empty == 1) | (added_scores > tau_added)).astype(np.int64)
    

    removed_point_scores = voxel_scores_to_point_scores(
        points=pc2016,
        voxel_map=ref_voxel_map,
        voxel_keys=removed_keys,
        voxel_scores=removed_scores,
    )

    added_point_scores = voxel_scores_to_point_scores(
        points=pc2020,
        voxel_map=query_voxel_map,
        voxel_keys=added_keys,
        voxel_scores=added_scores,
    )

    removed_point_pred = voxel_scores_to_point_scores(
        points=pc2016,
        voxel_map=ref_voxel_map,
        voxel_keys=removed_keys,
        voxel_scores=removed_voxel_pred,
    ).astype(np.int64)

    added_point_pred = voxel_scores_to_point_scores(
        points=pc2020,
        voxel_map=query_voxel_map,
        voxel_keys=added_keys,
        voxel_scores=added_voxel_pred,
    ).astype(np.int64)

    if debug:
        summarize_scores("removed_voxel", removed_scores, removed_voxel_pred)
        summarize_scores("added_voxel", added_scores, added_voxel_pred)
        summarize_scores("removed_point", removed_point_scores, removed_point_pred)
        summarize_scores("added_point", added_point_scores, added_point_pred)

    return {
        "removed_pred": removed_point_pred,
        "added_pred": added_point_pred,
        "removed_scores": removed_point_scores,
        "added_scores": added_point_scores,
        "origin": origin,
    }
