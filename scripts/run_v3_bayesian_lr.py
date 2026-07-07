# scripts/run_v3_bayesian_lr.py

"""
python scripts/run_v3_bayesian_lr.py \
  --config configs/v3_bayesian_lr.yaml \
  --mode direct_lr \
  --split test

python scripts/run_v3_bayesian_lr.py \
  --config configs/v3_bayesian_lr.yaml \
  --mode separate_np \
  --split val \
  --checkpoint outputs/checkpoints/v3_bayesian_lr/separate_np/best.pt \
  --tau_removed 0.5 \
  --tau_added 0.5 \
  --save_scores
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ------------------------------------------------------------
# Project root path
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from datasets.slpccd_loader import SLPCCDDataset
from v3_bayesian_3dcd_proto import V3BayesianLRModel
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
            "현재 run script는 SLPCCDDataset(root_dir=..., split=...) "
            "또는 SLPCCDDataset(root=..., split=...) 형태를 우선 가정합니다."
        ) from e


def collate_batch_size_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("현재 v3 평가 스크립트는 batch_size=1 기준입니다.")

    return batch[0]


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
# Metrics
# ------------------------------------------------------------
class ConfusionMatrix3:
    """
    class 0: background
    class 1: removed
    class 2: added
    """

    def __init__(self) -> None:
        self.matrix = torch.zeros((3, 3), dtype=torch.long)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().cpu().long().view(-1)
        target = target.detach().cpu().long().view(-1)

        valid = (target >= 0) & (target < 3)
        pred = pred[valid]
        target = target[valid]

        idx = target * 3 + pred
        bincount = torch.bincount(idx, minlength=9)
        self.matrix += bincount.view(3, 3)

    def compute(self, eps: float = 1e-8) -> dict[str, float]:
        cm = self.matrix.float()

        ious = []
        accs = []

        for c in range(3):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom_iou = tp + fp + fn

            iou = tp / (denom_iou + eps)

            class_total = cm[c, :].sum()
            acc = tp / (class_total + eps)

            ious.append(float(iou))
            accs.append(float(acc))

        oa = float(torch.diag(cm).sum() / (cm.sum() + eps))

        miou = sum(ious) / 3.0

        return {
            "background_iou": ious[0],
            "removed_iou": ious[1],
            "added_iou": ious[2],
            "background_acc": accs[0],
            "removed_acc": accs[1],
            "added_acc": accs[2],
            "miou": miou,
            "oa": oa,
        }


@torch.no_grad()
def evaluate(
    model: V3BayesianLRModel,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    save_scores: bool = False,
    score_save_dir: str | Path | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()

    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})

    use_rgb = model_cfg.get("use_rgb", True)

    tau_removed = float(infer_cfg.get("tau_removed", 0.5))
    tau_added = float(infer_cfg.get("tau_added", 0.5))

    cm = ConfusionMatrix3()
    per_sample_rows: list[dict[str, Any]] = []

    if save_scores:
        if score_save_dir is None:
            raise ValueError("save_scores=True이면 score_save_dir가 필요합니다.")
        score_save_dir = Path(score_save_dir)
        score_save_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(loader, desc="Evaluate", leave=True)

    for idx, sample in enumerate(pbar):
        batch = prepare_sample(
            sample=sample,
            device=device,
            use_rgb=use_rgb,
        )

        with inference_autocast_context(cfg, device):
            output = model(
                ref_xyz=batch["ref_xyz"],
                query_xyz=batch["query_xyz"],
                ref_rgb=batch["ref_rgb"],
                query_rgb=batch["query_rgb"],
            )

        removed_score = output["removed_score"].squeeze(0)
        added_score = output["added_score"].squeeze(0)

        removed_pred_binary = (removed_score >= tau_removed).long()
        added_pred_binary = (added_score >= tau_added).long()

        removed_gt_binary = batch["removed_label"].squeeze(0).long()
        added_gt_binary = batch["added_label"].squeeze(0).long()

        # 3-class 통합 metric
        # ref/2016 side: background=0, removed=1
        ref_pred_3 = torch.where(
            removed_pred_binary > 0,
            torch.ones_like(removed_pred_binary),
            torch.zeros_like(removed_pred_binary),
        )
        ref_gt_3 = torch.where(
            removed_gt_binary > 0,
            torch.ones_like(removed_gt_binary),
            torch.zeros_like(removed_gt_binary),
        )

        # query/2020 side: background=0, added=2
        query_pred_3 = torch.where(
            added_pred_binary > 0,
            torch.full_like(added_pred_binary, 2),
            torch.zeros_like(added_pred_binary),
        )
        query_gt_3 = torch.where(
            added_gt_binary > 0,
            torch.full_like(added_gt_binary, 2),
            torch.zeros_like(added_gt_binary),
        )

        pred_3 = torch.cat([ref_pred_3.view(-1), query_pred_3.view(-1)], dim=0)
        gt_3 = torch.cat([ref_gt_3.view(-1), query_gt_3.view(-1)], dim=0)

        sample_cm = ConfusionMatrix3()
        sample_cm.update(pred_3, gt_3)
        sample_metrics = sample_cm.compute()

        cm.update(pred_3, gt_3)

        row = {
            "sample_idx": idx,
            "background_iou": sample_metrics["background_iou"],
            "removed_iou": sample_metrics["removed_iou"],
            "added_iou": sample_metrics["added_iou"],
            "background_acc": sample_metrics["background_acc"],
            "removed_acc": sample_metrics["removed_acc"],
            "added_acc": sample_metrics["added_acc"],
            "miou": sample_metrics["miou"],
            "oa": sample_metrics["oa"],
            "num_ref_points": int(ref_pred_3.numel()),
            "num_query_points": int(query_pred_3.numel()),
            "num_removed_gt": int((ref_gt_3 == 1).sum().item()),
            "num_added_gt": int((query_gt_3 == 2).sum().item()),
            "num_removed_pred": int((ref_pred_3 == 1).sum().item()),
            "num_added_pred": int((query_pred_3 == 2).sum().item()),
        }

        per_sample_rows.append(row)

        total_metrics = cm.compute()
        pbar.set_postfix(
            {
                "mIoU": f"{total_metrics['miou']:.4f}",
                "R-IoU": f"{total_metrics['removed_iou']:.4f}",
                "A-IoU": f"{total_metrics['added_iou']:.4f}",
            }
        )

        if save_scores:
            save_path = Path(score_save_dir) / f"sample_{idx:04d}_scores.pt"

            torch.save(
                {
                    "removed_score": removed_score.detach().cpu(),
                    "added_score": added_score.detach().cpu(),
                    "removed_pred": removed_pred_binary.detach().cpu(),
                    "added_pred": added_pred_binary.detach().cpu(),
                    "removed_gt": removed_gt_binary.detach().cpu(),
                    "added_gt": added_gt_binary.detach().cpu(),
                },
                save_path,
            )

    metrics = cm.compute()

    return metrics, per_sample_rows


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
def save_summary_csv(
    path: str | Path,
    row: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(row.keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def save_per_sample_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v3 Bayesian likelihood-ratio model for SLPCCD."
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
        default=None,
        choices=["train", "val", "test"],
        help="Dataset split. If omitted, use dataset.test_split from config.",
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
        "--device",
        type=str,
        default=None,
        help="cuda, cuda:0, or cpu. Default: cuda if available.",
    )

    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="Save per-sample scores as .pt files.",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    cfg = override_config_with_args(cfg, args)

    dataset_cfg = cfg.get("dataset", {})
    output_cfg = cfg.get("output", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})

    root_dir = dataset_cfg.get("root_dir", "data/SLPCCD")

    if args.split is not None:
        split = args.split
    else:
        split = dataset_cfg.get("test_split", "test")

    mode = model_cfg.get("mode", "separate_np")

    tau_removed = float(infer_cfg.get("tau_removed", 0.5))
    tau_added = float(infer_cfg.get("tau_added", 0.5))

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = resolve_checkpoint_path(cfg, args)

    print("=" * 80)
    print("V3 Bayesian LR Evaluation")
    print(f"mode          : {mode}")
    print(f"device        : {device}")
    print(f"root_dir      : {root_dir}")
    print(f"split         : {split}")
    print(f"checkpoint    : {checkpoint_path}")
    print(f"tau_removed   : {tau_removed}")
    print(f"tau_added     : {tau_added}")
    print("=" * 80)

    dataset = create_dataset(root_dir=root_dir, split=split)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch_size_one,
    )

    model = build_v3_model_from_config(cfg).to(device)

    ckpt = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.eval()

    result_dir = Path(output_cfg.get("result_dir", "outputs/results/v3_bayesian_lr"))
    result_dir = result_dir / mode / split
    result_dir.mkdir(parents=True, exist_ok=True)

    score_save_dir = result_dir / "scores" if args.save_scores else None

    metrics, per_sample_rows = evaluate(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        save_scores=args.save_scores,
        score_save_dir=score_save_dir,
    )

    summary_row = {
        "split": split,
        "mode": mode,
        "checkpoint": str(checkpoint_path),
        "tau_removed": tau_removed,
        "tau_added": tau_added,
        "background_iou": metrics["background_iou"],
        "removed_iou": metrics["removed_iou"],
        "added_iou": metrics["added_iou"],
        "background_acc": metrics["background_acc"],
        "removed_acc": metrics["removed_acc"],
        "added_acc": metrics["added_acc"],
        "miou": metrics["miou"],
        "oa": metrics["oa"],
    }

    summary_csv = result_dir / "summary_metrics.csv"
    per_sample_csv = result_dir / "per_sample_metrics.csv"

    save_summary_csv(summary_csv, summary_row)
    save_per_sample_csv(per_sample_csv, per_sample_rows)

    print("-" * 80)
    print("Evaluation results")
    print(f"background_iou : {metrics['background_iou']:.4f}")
    print(f"removed_iou    : {metrics['removed_iou']:.4f}")
    print(f"added_iou      : {metrics['added_iou']:.4f}")
    print(f"background_acc : {metrics['background_acc']:.4f}")
    print(f"removed_acc    : {metrics['removed_acc']:.4f}")
    print(f"added_acc      : {metrics['added_acc']:.4f}")
    print(f"mIoU           : {metrics['miou']:.4f}")
    print(f"OA             : {metrics['oa']:.4f}")
    print("-" * 80)
    print(f"Saved summary     : {summary_csv}")
    print(f"Saved per-sample  : {per_sample_csv}")

    if args.save_scores:
        print(f"Saved scores      : {score_save_dir}")

    print("=" * 80)


if __name__ == "__main__":
    main()