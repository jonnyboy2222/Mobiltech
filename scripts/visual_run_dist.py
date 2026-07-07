"""
Clean visualization runner for V1 point-to-voxel distance baseline.

Default outputs are intentionally minimal:
    2016_rgb_render.png
    2020_rgb_render.png
    2016_gt_binary_render.png
    2020_gt_binary_render.png
    2016_pred_render.png
    2020_pred_render.png
    2016_error_render.png
    2020_error_render.png
    2016_score_render.png
    2020_score_render.png
    meta.txt

Optional outputs:
    --save_ply
    --save_merged
    --save_result_gt

Example:
python scripts/visual_run_dist.py \
  --root_dir data/SLPCCD \
  --split test \
  --idx 20 \
  --voxel_size 0.5 \
  --radius 1 \
  --tau_removed 0.5 \
  --tau_added 0.5 \
  --out_dir outputs/vis/v1_dist \
  --point_size 4.0 \
"""

import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.slpccd_loader import SLPCCDDataset
from utils.voxelization import build_voxel_map
from v1_simple_dist.distance import (
    directional_distance_scores,
    scores_to_binary_labels,
)
from utils.visualization import (
    ensure_dir,
    save_binary_task_visuals,
    save_merged_change_visuals,
    save_pair_rgb_gt_from_files,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--root_dir", type=str, default="data/SLPCCD")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--idx", type=int, default=0)

    # V1 distance baseline
    parser.add_argument("--voxel_size", type=float, default=0.5)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--tau_removed", type=float, default=0.5)
    parser.add_argument("--tau_added", type=float, default=0.5)
    parser.add_argument("--empty_score", type=float, default=1e6)

    # Output
    parser.add_argument("--out_dir", type=str, default="outputs/vis/v1_dist")

    # Clean output switches
    parser.add_argument(
        "--no_dataset_vis",
        action="store_true",
        help="Do not save 2016/2020 RGB and binary GT renders.",
    )
    parser.add_argument(
        "--no_rgb_gt",
        action="store_true",
        help="Deprecated alias of --no_dataset_vis.",
    )
    parser.add_argument(
        "--save_result_gt",
        action="store_true",
        help=(
            "Also save per-result GT renders. Usually unnecessary because "
            "binary GT renders are already saved."
        ),
    )
    parser.add_argument(
        "--save_merged",
        action="store_true",
        help="Also save merged 2016+2020 3-class prediction render.",
    )
    parser.add_argument(
        "--save_merged_gt",
        action="store_true",
        help="When --save_merged is used, also save merged GT render.",
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Also save PLY files. Disabled by default to avoid clutter.",
    )

    # Shared render options.
    # Applied to RGB, GT, prediction, error, score, and merged renders.
    parser.add_argument("--point_size", type=float, default=2.5)
    parser.add_argument(
        "--rgb_point_size",
        type=float,
        default=None,
        help="Deprecated alias. If set, overrides --point_size.",
    )
    parser.add_argument("--fig_size", type=float, default=6.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--azim", type=float, default=45.0)
    parser.add_argument("--elev", type=float, default=35.0)
    parser.add_argument("--max_points", type=int, default=None)
    parser.add_argument("--no_flip_y", action="store_true")

    args = parser.parse_args()

    if args.rgb_point_size is not None:
        args.point_size = args.rgb_point_size

    if args.no_rgb_gt:
        args.no_dataset_vis = True

    return args


def run_v1_distance_for_visualization(
    pc2016,
    pc2020,
    voxel_size,
    radius,
    tau_removed,
    tau_added,
    empty_score=1e6,
):
    """
    V1 bidirectional point-to-voxel distance baseline.

    removed:
        source = 2016
        target = 2020

    added:
        source = 2020
        target = 2016
    """
    voxel_2016, origin_2016 = build_voxel_map(
        pc2016,
        voxel_size=voxel_size,
    )

    voxel_2020, origin_2020 = build_voxel_map(
        pc2020,
        voxel_size=voxel_size,
    )

    removed_scores = directional_distance_scores(
        source_points=pc2016,
        target_voxel_map=voxel_2020,
        voxel_size=voxel_size,
        target_origin=origin_2020,
        radius=radius,
        empty_score=empty_score,
    )

    removed_pred = scores_to_binary_labels(
        scores=removed_scores,
        threshold=tau_removed,
    )

    added_scores = directional_distance_scores(
        source_points=pc2020,
        target_voxel_map=voxel_2016,
        voxel_size=voxel_size,
        target_origin=origin_2016,
        radius=radius,
        empty_score=empty_score,
    )

    added_pred = scores_to_binary_labels(
        scores=added_scores,
        threshold=tau_added,
    )

    return {
        "removed_scores": removed_scores,
        "added_scores": added_scores,
        "removed_pred": removed_pred,
        "added_pred": added_pred,
    }


def main():
    args = parse_args()

    dataset = SLPCCDDataset(
        root_dir=args.root_dir,
        split=args.split,
    )

    sample = dataset[args.idx]

    ref_xyz = sample["ref_xyz"]
    query_xyz = sample["query_xyz"]

    removed_gt = (sample["ref_label"] > 0).astype(np.int64)
    added_gt = (sample["query_label"] > 0).astype(np.int64)

    out = run_v1_distance_for_visualization(
        pc2016=ref_xyz,
        pc2020=query_xyz,
        voxel_size=args.voxel_size,
        radius=args.radius,
        tau_removed=args.tau_removed,
        tau_added=args.tau_added,
        empty_score=args.empty_score,
    )

    sample_out_dir = Path(args.out_dir) / f"sample_{args.idx:03d}"
    ensure_dir(sample_out_dir)

    render_kwargs = dict(
        point_size=args.point_size,
        azim=args.azim,
        elev=args.elev,
        dpi=args.dpi,
        figsize=(args.fig_size, args.fig_size),
        max_points=args.max_points,
        flip_y=not args.no_flip_y,
    )

    # Dataset-level paper visuals:
    # RGB + binary GT.
    # This replaces old removed_gt_render / added_gt_render in the clean pipeline.
    if not args.no_dataset_vis:
        save_pair_rgb_gt_from_files(
            ref_path=sample["ref_path"],
            query_path=sample["query_path"],
            out_dir=sample_out_dir,
            save_ply=args.save_ply,
            save_render=True,
            **render_kwargs,
        )

    # Result visuals.
    # GT render is disabled by default because it duplicates:
    # 2016_gt_binary_render.png and 2020_gt_binary_render.png.
    save_binary_task_visuals(
        points=ref_xyz,
        pred=out["removed_pred"],
        gt=removed_gt,
        scores=out["removed_scores"],
        out_dir=sample_out_dir,
        prefix="2016",
        change_color=(255, 0, 0),
        save_gt=args.save_result_gt,
        save_ply=args.save_ply,
        save_render=True,
        **render_kwargs,
    )

    save_binary_task_visuals(
        points=query_xyz,
        pred=out["added_pred"],
        gt=added_gt,
        scores=out["added_scores"],
        out_dir=sample_out_dir,
        prefix="2020",
        change_color=(255, 0, 0),
        save_gt=args.save_result_gt,
        save_ply=args.save_ply,
        save_render=True,
        **render_kwargs,
    )

    if args.save_merged:
        save_merged_change_visuals(
            ref_points=ref_xyz,
            query_points=query_xyz,
            removed_pred=out["removed_pred"],
            added_pred=out["added_pred"],
            removed_gt=removed_gt,
            added_gt=added_gt,
            out_dir=sample_out_dir,
            save_gt=args.save_merged_gt,
            save_ply=args.save_ply,
            save_render=True,
            **render_kwargs,
        )

    with open(sample_out_dir / "meta.txt", "w") as f:
        f.write("method: v1_point_to_voxel_distance\n")
        f.write(f"ref_path: {sample['ref_path']}\n")
        f.write(f"query_path: {sample['query_path']}\n")
        f.write(f"voxel_size: {args.voxel_size}\n")
        f.write(f"radius: {args.radius}\n")
        f.write(f"tau_removed: {args.tau_removed}\n")
        f.write(f"tau_added: {args.tau_added}\n")
        f.write(f"empty_score: {args.empty_score}\n")
        f.write(f"point_size: {args.point_size}\n")
        f.write(f"fig_size: {args.fig_size}\n")
        f.write(f"dpi: {args.dpi}\n")
        f.write(f"azim: {args.azim}\n")
        f.write(f"elev: {args.elev}\n")

    print(f"[Done] saved to: {sample_out_dir}")


if __name__ == "__main__":
    main()
