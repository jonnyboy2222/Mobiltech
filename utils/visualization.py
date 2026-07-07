import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Basic I/O
# -----------------------------
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_ply_xyzrgb(points, colors, save_path):
    """
    Save xyz + rgb point cloud as ASCII PLY.

    points: (N, 3)
    colors: (N, 3), uint8 or float in [0, 1]
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    points = np.asarray(points, dtype=np.float32)
    colors = _normalize_colors(colors)

    assert points.shape[0] == colors.shape[0]
    assert points.shape[1] == 3
    assert colors.shape[1] == 3

    with open(save_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def load_slpccd_txt_xyzrgb_label(txt_path):
    """
    Load SLPCCD-style txt file.

    Expected format:
        //X Y Z Rf Gf Bf label
        N
        x y z r g b label

    RGB can be float in [0, 1] or uint8-like [0, 255].
    """
    txt_path = Path(txt_path)
    rows = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            parts = line.split()

            # skip the point-count line
            if len(parts) == 1 and len(rows) == 0:
                continue

            if len(parts) < 7:
                continue

            rows.append([float(v) for v in parts[:7]])

    if len(rows) == 0:
        raise ValueError(f"No valid xyzrgb-label rows found in: {txt_path}")

    arr = np.asarray(rows, dtype=np.float32)

    xyz = arr[:, 0:3]
    rgb = arr[:, 3:6]
    label = arr[:, 6].astype(np.int64)

    return xyz, rgb, label


# -----------------------------
# Color maps
# -----------------------------
def _normalize_colors(colors):
    colors = np.asarray(colors)

    if colors.dtype != np.uint8:
        finite = np.isfinite(colors)
        max_value = colors[finite].max() if finite.any() else 1.0

        if max_value <= 1.0:
            colors = colors * 255.0

        colors = np.clip(colors, 0, 255).astype(np.uint8)

    return colors


def binary_pred_to_rgb(
    pred,
    change_color=(255, 0, 0),
    bg_color=(160, 160, 160),
):
    """
    Binary prediction color.

    0 = background
    1 = change
    """
    pred = np.asarray(pred).astype(np.int64)

    colors = np.zeros((len(pred), 3), dtype=np.uint8)
    colors[pred == 0] = bg_color
    colors[pred == 1] = change_color

    return colors


def binary_gt_to_rgb(
    labels,
    bg_color=(0, 0, 255),
    change_color=(255, 0, 0),
):
    """
    Paper-style binary GT color.

    0       = background / unchanged = blue
    label>0 = changed = red
    """
    labels = np.asarray(labels).astype(np.int64)

    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    colors[labels == 0] = bg_color
    colors[labels > 0] = change_color

    return colors


def change_label_to_rgb(labels):
    """
    3-class change label color.

    0 = background / unchanged
    1 = removed
    2 = added
    """
    labels = np.asarray(labels).astype(np.int64)

    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    colors[labels == 0] = (160, 160, 160)
    colors[labels == 1] = (255, 0, 0)
    colors[labels == 2] = (0, 80, 255)

    return colors


def binary_error_to_rgb(pred, gt):
    """
    Binary error map.

    TN = gray
    TP = green
    FP = red
    FN = blue
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    colors = np.zeros((len(pred), 3), dtype=np.uint8)

    tn = ~pred & ~gt
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    colors[tn] = (160, 160, 160)
    colors[tp] = (0, 255, 0)
    colors[fp] = (255, 0, 0)
    colors[fn] = (0, 80, 255)

    return colors


def score_to_rgb(
    scores,
    q_low=2,
    q_high=98,
    nan_color=(160, 80, 200),
):
    """
    Score heatmap.

    Low  = dark blue
    Mid  = yellow
    High = red
    NaN  = purple
    """
    scores = np.asarray(scores, dtype=np.float32)

    colors = np.zeros((len(scores), 3), dtype=np.uint8)
    finite = np.isfinite(scores)

    if finite.sum() == 0:
        colors[:] = nan_color
        return colors

    lo = np.percentile(scores[finite], q_low)
    hi = np.percentile(scores[finite], q_high)

    if hi <= lo:
        hi = lo + 1e-6

    x = np.zeros_like(scores, dtype=np.float32)
    x[finite] = (scores[finite] - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)

    low_color = np.array([49, 54, 149], dtype=np.float32)
    mid_color = np.array([255, 255, 191], dtype=np.float32)
    high_color = np.array([165, 0, 38], dtype=np.float32)

    low_mask = x <= 0.5
    high_mask = x > 0.5

    t_low = x[low_mask] / 0.5
    colors[low_mask] = (
        low_color[None, :] * (1 - t_low[:, None])
        + mid_color[None, :] * t_low[:, None]
    ).astype(np.uint8)

    t_high = (x[high_mask] - 0.5) / 0.5
    colors[high_mask] = (
        mid_color[None, :] * (1 - t_high[:, None])
        + high_color[None, :] * t_high[:, None]
    ).astype(np.uint8)

    colors[~finite] = nan_color

    return colors


# -----------------------------
# Camera / Rendering
# -----------------------------
def _camera_project(points, azim=45.0, elev=35.0):
    """
    Lightweight orthographic camera projection.

    points: (N, 3)
    azim : horizontal rotation in degrees
    elev : vertical elevation in degrees

    returns:
        xy: projected 2D coordinates
        depth: camera depth for sorting
    """
    pts = np.asarray(points, dtype=np.float32)

    az = np.deg2rad(azim)
    el = np.deg2rad(elev)

    Rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(el), -np.sin(el)],
            [0, np.sin(el), np.cos(el)],
        ],
        dtype=np.float32,
    )

    cam = pts @ Rz.T @ Rx.T

    xy = cam[:, :2]
    depth = cam[:, 2]

    return xy, depth


def save_pointcloud_render_png(
    points,
    colors,
    save_path,
    title=None,
    azim=45.0,
    elev=35.0,
    point_size=2.5,
    dpi=300,
    figsize=(6, 6),
    background="white",
    margin=0.03,
    max_points=None,
    depth_sort=True,
    depth_shade=False,
    flip_y=True,
):
    """
    Save publication-style rendered point cloud image.

    This is not a simple axis-based scatter plot.
    It uses fixed orthographic camera projection, depth sorting,
    no axes, and high-DPI output.

    points: (N, 3)
    colors: (N, 3), uint8 or float in [0, 1]
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    points = np.asarray(points, dtype=np.float32)
    colors = _normalize_colors(colors)

    assert points.ndim == 2 and points.shape[1] == 3
    assert colors.ndim == 2 and colors.shape[1] == 3
    assert len(points) == len(colors)

    n = len(points)

    if max_points is not None and n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    xy, depth = _camera_project(points, azim=azim, elev=elev)

    if flip_y:
        xy[:, 1] = -xy[:, 1]

    rgb = colors.astype(np.float32) / 255.0

    if depth_sort:
        order = np.argsort(depth)
        xy = xy[order]
        depth = depth[order]
        rgb = rgb[order]

    if depth_shade:
        d = depth.copy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        shade = 0.65 + 0.35 * d
        rgb = np.clip(rgb * shade[:, None], 0.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if background == "black":
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
    else:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=rgb,
        s=point_size,
        marker=".",
        linewidths=0,
        alpha=1.0,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)

    x_pad = (x_max - x_min) * margin + 1e-6
    y_pad = (y_max - y_min) * margin + 1e-6

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    if title is not None:
        ax.set_title(title, fontsize=10, pad=2)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# -----------------------------
# Dataset RGB + GT visuals
# -----------------------------
def save_single_rgb_gt_visuals(
    points,
    rgb,
    labels,
    out_dir,
    prefix,
    save_ply=False,
    save_render=True,
    point_size=2.5,
    azim=45.0,
    elev=35.0,
    dpi=300,
    figsize=(6, 6),
    max_points=None,
    flip_y=True,
):
    """
    Save RGB render and paper-style binary GT render for one point cloud.

    Outputs:
        {prefix}_rgb_render.png
        {prefix}_gt_binary_render.png

    Optional:
        {prefix}_rgb.ply
        {prefix}_gt_binary.ply
    """
    out_dir = ensure_dir(out_dir)

    points = np.asarray(points, dtype=np.float32)
    rgb = _normalize_colors(rgb)
    gt_rgb = binary_gt_to_rgb(labels)

    if save_ply:
        save_ply_xyzrgb(
            points,
            rgb,
            out_dir / f"{prefix}_rgb.ply",
        )

        save_ply_xyzrgb(
            points,
            gt_rgb,
            out_dir / f"{prefix}_gt_binary.ply",
        )

    if save_render:
        save_pointcloud_render_png(
            points,
            rgb,
            out_dir / f"{prefix}_rgb_render.png",
            azim=azim,
            elev=elev,
            point_size=point_size,
            dpi=dpi,
            figsize=figsize,
            max_points=max_points,
            depth_shade=False,
            flip_y=flip_y,
        )

        save_pointcloud_render_png(
            points,
            gt_rgb,
            out_dir / f"{prefix}_gt_binary_render.png",
            azim=azim,
            elev=elev,
            point_size=point_size,
            dpi=dpi,
            figsize=figsize,
            max_points=max_points,
            depth_shade=False,
            flip_y=flip_y,
        )


def save_pair_rgb_gt_visuals(
    ref_points,
    query_points,
    ref_rgb,
    query_rgb,
    ref_label,
    query_label,
    out_dir,
    save_ply=False,
    save_render=True,
    point_size=2.5,
    azim=45.0,
    elev=35.0,
    dpi=300,
    figsize=(6, 6),
    max_points=None,
    flip_y=True,
):
    """
    Save 2016/2020 RGB and paper-style binary GT renders.
    """
    save_single_rgb_gt_visuals(
        points=ref_points,
        rgb=ref_rgb,
        labels=ref_label,
        out_dir=out_dir,
        prefix="2016",
        save_ply=save_ply,
        save_render=save_render,
        point_size=point_size,
        azim=azim,
        elev=elev,
        dpi=dpi,
        figsize=figsize,
        max_points=max_points,
        flip_y=flip_y,
    )

    save_single_rgb_gt_visuals(
        points=query_points,
        rgb=query_rgb,
        labels=query_label,
        out_dir=out_dir,
        prefix="2020",
        save_ply=save_ply,
        save_render=save_render,
        point_size=point_size,
        azim=azim,
        elev=elev,
        dpi=dpi,
        figsize=figsize,
        max_points=max_points,
        flip_y=flip_y,
    )


def save_pair_rgb_gt_from_files(
    ref_path,
    query_path,
    out_dir,
    save_ply=False,
    save_render=True,
    point_size=2.5,
    azim=45.0,
    elev=35.0,
    dpi=300,
    figsize=(6, 6),
    max_points=None,
    flip_y=True,
):
    """
    Load 2016/2020 txt files and save RGB + binary GT renders.
    """
    ref_xyz, ref_rgb, ref_label = load_slpccd_txt_xyzrgb_label(ref_path)
    query_xyz, query_rgb, query_label = load_slpccd_txt_xyzrgb_label(query_path)

    save_pair_rgb_gt_visuals(
        ref_points=ref_xyz,
        query_points=query_xyz,
        ref_rgb=ref_rgb,
        query_rgb=query_rgb,
        ref_label=ref_label,
        query_label=query_label,
        out_dir=out_dir,
        save_ply=save_ply,
        save_render=save_render,
        point_size=point_size,
        azim=azim,
        elev=elev,
        dpi=dpi,
        figsize=figsize,
        max_points=max_points,
        flip_y=flip_y,
    )


# -----------------------------
# Prediction/result visuals
# -----------------------------
def save_binary_task_visuals(
    points,
    pred,
    gt,
    scores,
    out_dir,
    prefix,
    change_color=(255, 0, 0),
    save_gt=False,
    save_ply=False,
    save_render=True,
    azim=45.0,
    elev=35.0,
    point_size=2.5,
    dpi=300,
    figsize=(6, 6),
    max_points=None,
    flip_y=True,
):
    """
    Save prediction/result visualizations for one binary task.

    Default outputs:
        {prefix}_pred_render.png
        {prefix}_error_render.png
        {prefix}_score_render.png

    Optional output:
        {prefix}_gt_render.png

    In the cleaned pipeline, save_gt=False is recommended because
    2016_gt_binary_render.png and 2020_gt_binary_render.png already exist.
    """
    out_dir = ensure_dir(out_dir)

    pred_rgb = binary_pred_to_rgb(
        pred,
        change_color=change_color,
    )

    err_rgb = binary_error_to_rgb(
        pred,
        gt,
    )

    score_rgb = score_to_rgb(scores)

    if save_gt:
        gt_rgb = binary_gt_to_rgb(gt)

    if save_ply:
        save_ply_xyzrgb(
            points,
            pred_rgb,
            out_dir / f"{prefix}_pred.ply",
        )

        save_ply_xyzrgb(
            points,
            err_rgb,
            out_dir / f"{prefix}_error.ply",
        )

        save_ply_xyzrgb(
            points,
            score_rgb,
            out_dir / f"{prefix}_score.ply",
        )

        if save_gt:
            save_ply_xyzrgb(
                points,
                gt_rgb,
                out_dir / f"{prefix}_gt.ply",
            )

    if save_render:
        common_kwargs = dict(
            azim=azim,
            elev=elev,
            point_size=point_size,
            dpi=dpi,
            figsize=figsize,
            max_points=max_points,
            depth_shade=False,
            flip_y=flip_y,
        )

        save_pointcloud_render_png(
            points,
            pred_rgb,
            out_dir / f"{prefix}_pred_render.png",
            **common_kwargs,
        )

        if save_gt:
            save_pointcloud_render_png(
                points,
                gt_rgb,
                out_dir / f"{prefix}_gt_render.png",
                **common_kwargs,
            )

        save_pointcloud_render_png(
            points,
            err_rgb,
            out_dir / f"{prefix}_error_render.png",
            **common_kwargs,
        )

        save_pointcloud_render_png(
            points,
            score_rgb,
            out_dir / f"{prefix}_score_render.png",
            **common_kwargs,
        )


def save_merged_change_visuals(
    ref_points,
    query_points,
    removed_pred,
    added_pred,
    removed_gt,
    added_gt,
    out_dir,
    save_gt=False,
    save_ply=False,
    save_render=True,
    azim=45.0,
    elev=35.0,
    point_size=2.5,
    dpi=300,
    figsize=(6, 6),
    max_points=None,
    flip_y=True,
):
    """
    Save optional merged 3-class prediction and GT renders.

    Use this only when a single merged 2016+2020 view is needed.
    """
    out_dir = ensure_dir(out_dir)

    all_points = np.concatenate(
        [ref_points, query_points],
        axis=0,
    )

    pred_labels = np.concatenate(
        [
            np.asarray(removed_pred).astype(np.int64),
            np.asarray(added_pred).astype(np.int64) * 2,
        ],
        axis=0,
    )

    pred_rgb = change_label_to_rgb(pred_labels)

    if save_gt:
        gt_labels = np.concatenate(
            [
                np.asarray(removed_gt).astype(np.int64),
                np.asarray(added_gt).astype(np.int64) * 2,
            ],
            axis=0,
        )

        gt_rgb = change_label_to_rgb(gt_labels)

    if save_ply:
        save_ply_xyzrgb(
            all_points,
            pred_rgb,
            out_dir / "merged_pred_3class.ply",
        )

        if save_gt:
            save_ply_xyzrgb(
                all_points,
                gt_rgb,
                out_dir / "merged_gt_3class.ply",
            )

    if save_render:
        common_kwargs = dict(
            azim=azim,
            elev=elev,
            point_size=point_size,
            dpi=dpi,
            figsize=figsize,
            max_points=max_points,
            depth_shade=False,
            flip_y=flip_y,
        )

        save_pointcloud_render_png(
            all_points,
            pred_rgb,
            out_dir / "merged_pred_3class_render.png",
            **common_kwargs,
        )

        if save_gt:
            save_pointcloud_render_png(
                all_points,
                gt_rgb,
                out_dir / "merged_gt_3class_render.png",
                **common_kwargs,
            )

