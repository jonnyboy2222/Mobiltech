# scripts/visual_run_v3_bayesian_lr.py

"""
python scripts/visual_run_v3_bayesian_lr.py \
  --config configs/v3_bayesian_lr.yaml \
  --mode separate_np \
  --split val \
  --idx 53

python scripts/visual_run_v3_bayesian_lr.py \
  --config configs/v3_bayesian_lr.yaml \
  --mode direct_lr \
  --split val \
  --idx 53


# point를 더 두껍게
python scripts/visual_run_v3_bayesian_lr.py \
  --config configs/v3_bayesian_lr.yaml \
  --mode separate_np \
  --split val \
  --idx 53 \
  --point_size 4.0 \
  --dpi 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
import torch
import numpy as np


# ------------------------------------------------------------
# Project root path
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from datasets.slpccd_loader import SLPCCDDataset
from v3_bayesian_3dcd_proto import V3BayesianLRModel
from utils.visualization import ensure_dir, save_pointcloud_render_png
from v3_bayesian_3dcd_proto.model import build_v3_model_from_config


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_config_with_args(cfg: dict, args: argparse.Namespace) -> dict:
    if args.mode is not None:
        cfg.setdefault("model", {})["mode"] = args.mode

    if args.k_neighbors is not None:
        cfg.setdefault("model", {})["k_neighbors"] = args.k_neighbors

    if args.feature_dim is not None:
        cfg.setdefault("model", {})["feature_dim"] = args.feature_dim

    if args.tau_removed is not None:
        cfg.setdefault("inference", {})["tau_removed"] = args.tau_removed

    if args.tau_added is not None:
        cfg.setdefault("inference", {})["tau_added"] = args.tau_added

    return cfg


# ------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------
def create_dataset(root_dir: str, split: str):
    try:
        return SLPCCDDataset(root_dir=root_dir, split=split)
    except TypeError:
        pass

    try:
        return SLPCCDDataset(root=root_dir, split=split)
    except TypeError:
        pass

    try:
        return SLPCCDDataset(root_dir, split)
    except TypeError as e:
        raise TypeError(
            "SLPCCDDataset 생성자 형태를 확인해야 합니다. "
            "SLPCCDDataset(root_dir=..., split=...) 또는 "
            "SLPCCDDataset(root=..., split=...) 형태를 우선 가정합니다."
        ) from e


def find_key(sample: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in sample:
            return sample[key]

    raise KeyError(
        f"sample에서 필요한 key를 찾지 못했습니다. "
        f"candidates={candidates}, available_keys={list(sample.keys())}"
    )


def find_optional_key(
    sample: dict[str, Any],
    candidates: list[str],
    default: Any = None,
) -> Any:
    for key in candidates:
        if key in sample:
            return sample[key]

    return default


def to_tensor(
    x: Any,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)

    return torch.as_tensor(x, device=device, dtype=dtype)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def prepare_sample(
    sample: dict[str, Any],
    device: torch.device,
    use_rgb: bool,
) -> dict[str, torch.Tensor]:
    ref_xyz = find_key(
        sample,
        ["ref_xyz", "xyz_ref", "points_ref", "point2016", "src_xyz"],
    )
    query_xyz = find_key(
        sample,
        ["query_xyz", "xyz_query", "points_query", "point2020", "tgt_xyz"],
    )

    ref_label = find_key(
        sample,
        ["ref_label", "label_ref", "labels_ref", "removed_label"],
    )
    query_label = find_key(
        sample,
        ["query_label", "label_query", "labels_query", "added_label"],
    )

    ref_xyz = to_tensor(ref_xyz, device=device)
    query_xyz = to_tensor(query_xyz, device=device)

    ref_label = to_tensor(ref_label, device=device)
    query_label = to_tensor(query_label, device=device)

    removed_label = (ref_label > 0).long()
    added_label = (query_label > 0).long()

    ref_rgb = None
    query_rgb = None

    if use_rgb:
        ref_rgb = find_optional_key(
            sample,
            ["ref_rgb", "rgb_ref", "color_ref", "colors_ref", "ref_color"],
            default=None,
        )
        query_rgb = find_optional_key(
            sample,
            ["query_rgb", "rgb_query", "color_query", "colors_query", "query_color"],
            default=None,
        )

        if ref_rgb is None:
            ref_rgb = torch.zeros_like(ref_xyz)
        else:
            ref_rgb = to_tensor(ref_rgb, device=device)

        if query_rgb is None:
            query_rgb = torch.zeros_like(query_xyz)
        else:
            query_rgb = to_tensor(query_rgb, device=device)

    return {
        "ref_xyz": ref_xyz,
        "query_xyz": query_xyz,
        "ref_rgb": ref_rgb,
        "query_rgb": query_rgb,
        "removed_label": removed_label,
        "added_label": added_label,
    }


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def build_model_from_cfg(cfg: dict) -> V3BayesianLRModel:
    model_cfg = cfg.get("model", {})

    return V3BayesianLRModel(
        mode=model_cfg.get("mode", "separate_np"),
        use_rgb=model_cfg.get("use_rgb", True),
        normalize_xyz=model_cfg.get("normalize_xyz", True),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        feature_dim=model_cfg.get("feature_dim", 128),
        head_hidden_dim=model_cfg.get("head_hidden_dim", 128),
        dropout=model_cfg.get("dropout", 0.1),
        k_neighbors=model_cfg.get("k_neighbors", 16),
        prior_logit=model_cfg.get("prior_logit", 0.0),
    )


def resolve_checkpoint_path(cfg: dict, args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint)

    mode = cfg.get("model", {}).get("mode", "separate_np")
    checkpoint_dir = cfg.get("output", {}).get(
        "checkpoint_dir",
        "outputs/checkpoints/v3_bayesian_lr",
    )

    return Path(checkpoint_dir) / mode / "best.pt"


def load_checkpoint(
    model: V3BayesianLRModel,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    return ckpt


def inference_amp_enabled(cfg: dict, device: torch.device) -> bool:
    infer_cfg = cfg.get("inference", {})
    return bool(infer_cfg.get("use_amp", False)) and device.type == "cuda"


def inference_autocast_context(cfg: dict, device: torch.device):
    return torch.amp.autocast(
        device_type=device.type,
        enabled=inference_amp_enabled(cfg, device),
    )


# ------------------------------------------------------------
# Visualization color helpers
# ------------------------------------------------------------
def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)

    if rgb.size == 0:
        return rgb

    if rgb.max() <= 1.5:
        rgb = rgb * 255.0

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def binary_gt_to_rgb(
    label: np.ndarray,
    bg_color: tuple[int, int, int] = (0, 0, 255),
    change_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    label = label.reshape(-1)
    colors = np.zeros((label.shape[0], 3), dtype=np.uint8)
    colors[:] = np.asarray(bg_color, dtype=np.uint8)
    colors[label > 0] = np.asarray(change_color, dtype=np.uint8)
    return colors


def binary_pred_to_rgb(
    pred: np.ndarray,
    bg_color: tuple[int, int, int] = (160, 160, 160),
    change_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    pred = pred.reshape(-1)
    colors = np.zeros((pred.shape[0], 3), dtype=np.uint8)
    colors[:] = np.asarray(bg_color, dtype=np.uint8)
    colors[pred > 0] = np.asarray(change_color, dtype=np.uint8)
    return colors


def binary_error_to_rgb(
    pred: np.ndarray,
    label: np.ndarray,
) -> np.ndarray:
    """
    background correct: blue
    change correct    : red
    false positive    : yellow
    false negative    : cyan
    """
    pred = pred.reshape(-1).astype(bool)
    label = label.reshape(-1).astype(bool)

    colors = np.zeros((pred.shape[0], 3), dtype=np.uint8)

    tn = (~pred) & (~label)
    tp = pred & label
    fp = pred & (~label)
    fn = (~pred) & label

    colors[tn] = np.array([0, 0, 255], dtype=np.uint8)
    colors[tp] = np.array([255, 0, 0], dtype=np.uint8)
    colors[fp] = np.array([255, 255, 0], dtype=np.uint8)
    colors[fn] = np.array([0, 255, 255], dtype=np.uint8)

    return colors


def score_to_rgb(score: np.ndarray) -> np.ndarray:
    """
    Simple heatmap:
      low score  -> blue
      high score -> red
    """
    score = score.reshape(-1).astype(np.float32)
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    score = np.clip(score, 0.0, 1.0)

    colors = np.zeros((score.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = (score * 255).astype(np.uint8)
    colors[:, 2] = ((1.0 - score) * 255).astype(np.uint8)

    return colors


def save_render(
    path: str | Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    point_size: float,
    fig_size: float,
    dpi: int,
    azim: float,
    elev: float,
    max_points: int | None,
    flip_y: bool,
) -> None:
    save_pointcloud_render_png(
        xyz,
        rgb,
        str(path),
        point_size=point_size,
        figsize=(fig_size, fig_size),
        dpi=dpi,
        azim=azim,
        elev=elev,
        max_points=max_points,
        flip_y=flip_y,
    )


# ------------------------------------------------------------
# Metrics for meta
# ------------------------------------------------------------
def binary_metrics(
    pred: np.ndarray,
    label: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, float]:
    pred = pred.reshape(-1).astype(bool)
    label = label.reshape(-1).astype(bool)

    tp = np.logical_and(pred, label).sum()
    fp = np.logical_and(pred, ~label).sum()
    fn = np.logical_and(~pred, label).sum()
    tn = np.logical_and(~pred, ~label).sum()

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "acc": float(acc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def write_meta(
    path: str | Path,
    info: dict[str, Any],
) -> None:
    path = Path(path)

    with open(path, "w", encoding="utf-8") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize v3 Bayesian LR result for one SLPCCD sample."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/v3_bayesian_lr.yaml",
        help="Path to config yaml.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, use output.checkpoint_dir/mode/best.pt.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split.",
    )

    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Sample index.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["direct_lr", "separate_np"],
        help="Override model mode.",
    )

    parser.add_argument(
        "--tau_removed",
        type=float,
        default=None,
        help="Override removed threshold.",
    )

    parser.add_argument(
        "--tau_added",
        type=float,
        default=None,
        help="Override added threshold.",
    )

    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=None,
        help="Override KNN size.",
    )

    parser.add_argument(
        "--feature_dim",
        type=int,
        default=None,
        help="Override feature dim.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. If omitted, use output.vis_dir/mode/split/sample_xxx.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cuda:0, or cpu. Default: cuda if available.",
    )

    # render options
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--fig_size", type=float, default=6.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--azim", type=float, default=45.0)
    parser.add_argument("--elev", type=float, default=35.0)
    parser.add_argument("--max_points", type=int, default=None)
    parser.add_argument("--no_flip_y", action="store_true")

    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
@torch.no_grad()
def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    cfg = override_config_with_args(cfg, args)

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})
    output_cfg = cfg.get("output", {})

    root_dir = dataset_cfg.get("root_dir", "data/SLPCCD")
    mode = model_cfg.get("mode", "separate_np")
    use_rgb = model_cfg.get("use_rgb", True)

    tau_removed = float(infer_cfg.get("tau_removed", 0.5))
    tau_added = float(infer_cfg.get("tau_added", 0.5))

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = resolve_checkpoint_path(cfg, args)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        vis_dir = Path(output_cfg.get("vis_dir", "outputs/vis/v3_bayesian_lr"))
        out_dir = vis_dir / mode / args.split / f"sample_{args.idx:04d}"

    ensure_dir(str(out_dir))

    print("=" * 80)
    print("V3 Bayesian LR Visualization")
    print(f"mode        : {mode}")
    print(f"device      : {device}")
    print(f"root_dir    : {root_dir}")
    print(f"split       : {args.split}")
    print(f"idx         : {args.idx}")
    print(f"checkpoint  : {checkpoint_path}")
    print(f"out_dir     : {out_dir}")
    print(f"tau_removed : {tau_removed}")
    print(f"tau_added   : {tau_added}")
    print("=" * 80)

    dataset = create_dataset(root_dir=root_dir, split=args.split)

    if args.idx < 0 or args.idx >= len(dataset):
        raise IndexError(f"idx out of range: {args.idx}, dataset length={len(dataset)}")

    sample = dataset[args.idx]

    batch = prepare_sample(
        sample=sample,
        device=device,
        use_rgb=use_rgb,
    )

    model = build_v3_model_from_config(cfg).to(device)
    load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model.eval()


    with inference_autocast_context(cfg, device):
        output = model(
            ref_xyz=batch["ref_xyz"],
            query_xyz=batch["query_xyz"],
            ref_rgb=batch["ref_rgb"],
            query_rgb=batch["query_rgb"],
        )

    # tensor -> numpy
    ref_xyz = to_numpy(batch["ref_xyz"])
    query_xyz = to_numpy(batch["query_xyz"])

    ref_label = to_numpy(batch["removed_label"]).astype(np.int64)
    query_label = to_numpy(batch["added_label"]).astype(np.int64)

    removed_score = to_numpy(output["removed_score"]).reshape(-1)
    added_score = to_numpy(output["added_score"]).reshape(-1)

    removed_pred = (removed_score >= tau_removed).astype(np.int64)
    added_pred = (added_score >= tau_added).astype(np.int64)

    if use_rgb:
        ref_rgb = to_numpy(batch["ref_rgb"])
        query_rgb = to_numpy(batch["query_rgb"])
        ref_rgb = normalize_rgb(ref_rgb)
        query_rgb = normalize_rgb(query_rgb)
    else:
        ref_rgb = np.zeros_like(ref_xyz, dtype=np.uint8)
        query_rgb = np.zeros_like(query_xyz, dtype=np.uint8)

    ref_gt_rgb = binary_gt_to_rgb(ref_label)
    query_gt_rgb = binary_gt_to_rgb(query_label)

    ref_pred_rgb = binary_pred_to_rgb(removed_pred)
    query_pred_rgb = binary_pred_to_rgb(added_pred)

    ref_error_rgb = binary_error_to_rgb(removed_pred, ref_label)
    query_error_rgb = binary_error_to_rgb(added_pred, query_label)

    ref_score_rgb = score_to_rgb(removed_score)
    query_score_rgb = score_to_rgb(added_score)

    render_kwargs = {
        "point_size": args.point_size,
        "fig_size": args.fig_size,
        "dpi": args.dpi,
        "azim": args.azim,
        "elev": args.elev,
        "max_points": args.max_points,
        "flip_y": not args.no_flip_y,
    }

    # RGB
    save_render(
        out_dir / "2016_rgb_render.png",
        ref_xyz,
        ref_rgb,
        **render_kwargs,
    )
    save_render(
        out_dir / "2020_rgb_render.png",
        query_xyz,
        query_rgb,
        **render_kwargs,
    )

    # GT
    save_render(
        out_dir / "2016_gt_binary_render.png",
        ref_xyz,
        ref_gt_rgb,
        **render_kwargs,
    )
    save_render(
        out_dir / "2020_gt_binary_render.png",
        query_xyz,
        query_gt_rgb,
        **render_kwargs,
    )

    # Prediction
    save_render(
        out_dir / "2016_pred_render.png",
        ref_xyz,
        ref_pred_rgb,
        **render_kwargs,
    )
    save_render(
        out_dir / "2020_pred_render.png",
        query_xyz,
        query_pred_rgb,
        **render_kwargs,
    )

    # Error
    save_render(
        out_dir / "2016_error_render.png",
        ref_xyz,
        ref_error_rgb,
        **render_kwargs,
    )
    save_render(
        out_dir / "2020_error_render.png",
        query_xyz,
        query_error_rgb,
        **render_kwargs,
    )

    # Score
    save_render(
        out_dir / "2016_score_render.png",
        ref_xyz,
        ref_score_rgb,
        **render_kwargs,
    )
    save_render(
        out_dir / "2020_score_render.png",
        query_xyz,
        query_score_rgb,
        **render_kwargs,
    )

    removed_metrics = binary_metrics(removed_pred, ref_label)
    added_metrics = binary_metrics(added_pred, query_label)

    meta = {
        "mode": mode,
        "split": args.split,
        "idx": args.idx,
        "checkpoint": checkpoint_path,
        "tau_removed": tau_removed,
        "tau_added": tau_added,
        "num_2016_points": int(ref_xyz.shape[0]),
        "num_2020_points": int(query_xyz.shape[0]),
        "num_removed_gt": int(ref_label.sum()),
        "num_added_gt": int(query_label.sum()),
        "num_removed_pred": int(removed_pred.sum()),
        "num_added_pred": int(added_pred.sum()),
        "removed_iou": removed_metrics["iou"],
        "removed_precision": removed_metrics["precision"],
        "removed_recall": removed_metrics["recall"],
        "removed_acc": removed_metrics["acc"],
        "added_iou": added_metrics["iou"],
        "added_precision": added_metrics["precision"],
        "added_recall": added_metrics["recall"],
        "added_acc": added_metrics["acc"],
        "point_size": args.point_size,
        "fig_size": args.fig_size,
        "dpi": args.dpi,
        "azim": args.azim,
        "elev": args.elev,
        "max_points": args.max_points,
        "flip_y": not args.no_flip_y,
    }

    if mode == "separate_np":
        removed_h0_energy = to_numpy(output["removed_h0_energy"]).reshape(-1)
        removed_h1_energy = to_numpy(output["removed_h1_energy"]).reshape(-1)
        added_h0_energy = to_numpy(output["added_h0_energy"]).reshape(-1)
        added_h1_energy = to_numpy(output["added_h1_energy"]).reshape(-1)

        meta.update(
            {
                "removed_h0_energy_mean": float(removed_h0_energy.mean()),
                "removed_h1_energy_mean": float(removed_h1_energy.mean()),
                "added_h0_energy_mean": float(added_h0_energy.mean()),
                "added_h1_energy_mean": float(added_h1_energy.mean()),
            }
        )

    write_meta(out_dir / "meta.txt", meta)

    # score 저장
    torch.save(
        {
            "removed_score": torch.from_numpy(removed_score),
            "added_score": torch.from_numpy(added_score),
            "removed_pred": torch.from_numpy(removed_pred),
            "added_pred": torch.from_numpy(added_pred),
            "removed_gt": torch.from_numpy(ref_label),
            "added_gt": torch.from_numpy(query_label),
        },
        out_dir / "scores.pt",
    )

    print("-" * 80)
    print(f"Saved visualization to: {out_dir}")
    print(f"removed IoU: {removed_metrics['iou']:.4f}")
    print(f"added IoU  : {added_metrics['iou']:.4f}")
    print("-" * 80)


if __name__ == "__main__":
    main()