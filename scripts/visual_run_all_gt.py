"""
Render GT-only visualizations for the whole SLPCCD dataset.

Default output:
    outputs/vis/gt/{split}/sample_000_2016_gt_binary_render.png
    outputs/vis/gt/{split}/sample_000_2020_gt_binary_render.png

GT color:
    background / unchanged = blue
    changed = red

Example 1: all splits
python scripts/visual_run_all_gt.py \
  --root_dir data/SLPCCD \
  --split all \
  --out_dir outputs/vis/gt \
  --point_size 4.0

Example 2: val only
python scripts/visual_run_all_gt.py \
  --root_dir data/SLPCCD \
  --split val \
  --out_dir outputs/vis/gt \
  --point_size 4.0
"""

import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.slpccd_loader import SLPCCDDataset
from utils.visualization import (
    ensure_dir,
    binary_gt_to_rgb,
    save_pointcloud_render_png,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, default="data/SLPCCD")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
    )

    # 다른 시각화 파일들과 같은 루트 아래로 저장
    parser.add_argument("--out_dir", type=str, default="outputs/vis/gt")
    parser.add_argument("--overwrite", action="store_true")

    # Optional subset rendering
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)

    # Render options
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--fig_size", type=float, default=6.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--azim", type=float, default=45.0)
    parser.add_argument("--elev", type=float, default=35.0)
    parser.add_argument("--max_points", type=int, default=None)
    parser.add_argument("--no_flip_y", action="store_true")

    return parser.parse_args()


def render_one_gt(
    points,
    labels,
    save_path,
    point_size=3.0,
    fig_size=6.0,
    dpi=300,
    azim=45.0,
    elev=35.0,
    max_points=None,
    flip_y=True,
):
    """
    Save one binary GT render.

    label == 0 : blue background / unchanged
    label > 0  : red changed
    """
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels).astype(np.int64)

    gt_rgb = binary_gt_to_rgb(labels)

    save_pointcloud_render_png(
        points=points,
        colors=gt_rgb,
        save_path=save_path,
        azim=azim,
        elev=elev,
        point_size=point_size,
        dpi=dpi,
        figsize=(fig_size, fig_size),
        max_points=max_points,
        depth_shade=False,
        flip_y=flip_y,
    )


def render_split(args, split):
    dataset = SLPCCDDataset(
        root_dir=args.root_dir,
        split=split,
    )

    split_out_dir = ensure_dir(Path(args.out_dir) / split)

    n = len(dataset)
    start = max(args.start, 0)
    end = n if args.end is None else min(args.end, n)
    stride = max(args.stride, 1)

    print(f"[Info] split={split}, total={n}, rendering={start}:{end}:{stride}")
    print(f"[Info] output={split_out_dir}")

    for idx in range(start, end, stride):
        sample = dataset[idx]

        ref_xyz = sample["ref_xyz"]
        query_xyz = sample["query_xyz"]

        ref_label = sample["ref_label"]
        query_label = sample["query_label"]

        save_2016 = split_out_dir / f"sample_{idx:03d}_2016_gt_binary_render.png"
        save_2020 = split_out_dir / f"sample_{idx:03d}_2020_gt_binary_render.png"

        if not args.overwrite:
            if save_2016.exists() and save_2020.exists():
                continue

        render_one_gt(
            points=ref_xyz,
            labels=ref_label,
            save_path=save_2016,
            point_size=args.point_size,
            fig_size=args.fig_size,
            dpi=args.dpi,
            azim=args.azim,
            elev=args.elev,
            max_points=args.max_points,
            flip_y=not args.no_flip_y,
        )

        render_one_gt(
            points=query_xyz,
            labels=query_label,
            save_path=save_2020,
            point_size=args.point_size,
            fig_size=args.fig_size,
            dpi=args.dpi,
            azim=args.azim,
            elev=args.elev,
            max_points=args.max_points,
            flip_y=not args.no_flip_y,
        )

        if idx % 20 == 0:
            print(f"[{split}] rendered sample {idx:03d}")

    print(f"[Done] split={split}")


def main():
    args = parse_args()

    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    for split in splits:
        render_split(args, split)

    print(f"[Done] all GT renders saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
