# scripts/train_v3_bayesian_lr.py

"""
python scripts/train_v3_bayesian_lr.py \
  --config configs/v3_bayesian_lr.yaml \
  --mode direct_lr

python scripts/train_v3_bayesian_lr.py \
  --config configs/v3_bayesian_lr.yaml \
  --mode separate_np
"""


from __future__ import annotations

import argparse
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
from v3_bayesian_3dcd_proto import V3BayesianLRModel, V3BayesianLoss
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

    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs

    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = args.lr

    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size

    if args.k_neighbors is not None:
        cfg.setdefault("model", {})["k_neighbors"] = args.k_neighbors

    if args.feature_dim is not None:
        cfg.setdefault("model", {})["feature_dim"] = args.feature_dim

    return cfg


# ------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------
def create_dataset(root_dir: str, split: str):
    """
    기존 SLPCCDDataset 생성자 형태가 다를 수 있어서
    가장 흔한 형태부터 순서대로 시도한다.
    """
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
            "현재 train script는 SLPCCDDataset(root_dir=..., split=...) "
            "또는 SLPCCDDataset(root=..., split=...) 형태를 우선 가정합니다."
        ) from e


def collate_batch_size_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Point cloud는 sample마다 point 수가 다를 수 있으므로
    처음에는 batch_size=1을 안전하게 사용한다.
    """
    if len(batch) != 1:
        raise ValueError(
            "현재 v3 prototype train script는 batch_size=1을 권장합니다. "
            "variable-size point cloud batching은 이후 padding/cropping으로 확장하세요."
        )

    return batch[0]


def find_key(sample: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in sample:
            return sample[key]

    raise KeyError(
        f"sample에서 필요한 key를 찾지 못했습니다. candidates={candidates}, "
        f"available_keys={list(sample.keys())}"
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


def to_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)

    return torch.as_tensor(x, device=device, dtype=dtype)


def prepare_sample(
    sample: dict[str, Any],
    device: torch.device,
    use_rgb: bool,
) -> dict[str, torch.Tensor]:
    """
    SLPCCD sample을 model input 형태로 변환.

    예상 key:
        ref_xyz, query_xyz
        ref_rgb, query_rgb
        ref_label, query_label

    fallback key도 일부 지원.
    """
    ref_xyz = find_key(sample, ["ref_xyz", "xyz_ref", "points_ref", "point2016", "src_xyz"])
    query_xyz = find_key(sample, ["query_xyz", "xyz_query", "points_query", "point2020", "tgt_xyz"])

    ref_label = find_key(sample, ["ref_label", "label_ref", "labels_ref", "removed_label"])
    query_label = find_key(sample, ["query_label", "label_query", "labels_query", "added_label"])

    ref_xyz = to_tensor(ref_xyz, device=device)
    query_xyz = to_tensor(query_xyz, device=device)

    ref_label = to_tensor(ref_label, device=device)
    query_label = to_tensor(query_label, device=device)

    # binary label
    # SLPCCD에서 label > 0이면 changed object point로 처리
    removed_label = (ref_label > 0).float()
    added_label = (query_label > 0).float()

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
            # loader가 RGB를 안 주는 경우 임시로 0 RGB 사용
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
# Model / loss builders
# ------------------------------------------------------------
def build_model_from_cfg(cfg: dict) -> V3BayesianLRModel:
    model_cfg = cfg.get("model", {})

    return V3BayesianLRModel(
        mode=model_cfg.get("mode", "separate_np"),
        use_rgb=model_cfg.get("use_rgb", True),
        normalize_xyz=model_cfg.get("normalize_xyz", True),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        feature_dim=model_cfg.get("feature_dim", 64),
        head_hidden_dim=model_cfg.get("head_hidden_dim", 64),
        dropout=model_cfg.get("dropout", 0.1),
        k_neighbors=model_cfg.get("k_neighbors", 8),
        prior_logit=model_cfg.get("prior_logit", 0.0),
        nfd_chunk_size=model_cfg.get("nfd_chunk_size", 512),
    )


def build_loss_from_cfg(cfg: dict) -> V3BayesianLoss:
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    return V3BayesianLoss(
        mode=model_cfg.get("mode", "separate_np"),
        pos_weight=train_cfg.get("pos_weight", None),
        lambda_energy=train_cfg.get("lambda_energy", 0.01),
        lambda_margin=train_cfg.get("lambda_margin", 0.1),
        margin=train_cfg.get("margin", 1.0),
    )

# amp
def amp_enabled(cfg: dict, device: torch.device) -> bool:
    train_cfg = cfg.get("train", {})
    return bool(train_cfg.get("use_amp", False)) and device.type == "cuda"


def autocast_context(cfg: dict, device: torch.device):
    enabled = amp_enabled(cfg, device)

    # CUDA 아니면 autocast 비활성
    return torch.amp.autocast(
        device_type=device.type,
        enabled=enabled,
    )

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
@torch.no_grad()
def binary_iou_from_logits(
    logit: torch.Tensor,
    label: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, float]:
    prob = torch.sigmoid(logit).reshape(-1)
    label = label.float().reshape(-1)

    pred = (prob >= threshold).float()

    tp = ((pred == 1) & (label == 1)).sum().item()
    fp = ((pred == 1) & (label == 0)).sum().item()
    fn = ((pred == 0) & (label == 1)).sum().item()
    tn = ((pred == 0) & (label == 0)).sum().item()

    pos_iou = tp / (tp + fp + fn + eps)
    bg_iou = tn / (tn + fp + fn + eps)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    total = tp + fp + fn + tn + eps
    pred_pos_ratio = (tp + fp) / total
    gt_pos_ratio = (tp + fn) / total

    mean_score = prob.mean().item()

    pos_mask = label == 1
    neg_mask = label == 0

    pos_score = prob[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_score = prob[neg_mask].mean().item() if neg_mask.any() else 0.0

    return {
        "iou": float(pos_iou),
        "bg_iou": float(bg_iou),
        "precision": float(precision),
        "recall": float(recall),
        "acc": float(acc),

        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),

        "pred_pos_ratio": float(pred_pos_ratio),
        "gt_pos_ratio": float(gt_pos_ratio),

        "mean_score": float(mean_score),
        "pos_score": float(pos_score),
        "neg_score": float(neg_score),
    }


# ------------------------------------------------------------
# Train / validation
# ------------------------------------------------------------
def train_one_epoch(
    model: V3BayesianLRModel,
    criterion: V3BayesianLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    epoch: int,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    model.train()

    model_cfg = cfg.get("model", {})
    use_rgb = model_cfg.get("use_rgb", True)

    total_loss = 0.0
    total_bce = 0.0
    total_energy = 0.0
    total_margin = 0.0
    count = 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)

    for sample in pbar:
        batch = prepare_sample(
            sample=sample,
            device=device,
            use_rgb=use_rgb,
        )

        output = model(
            ref_xyz=batch["ref_xyz"],
            query_xyz=batch["query_xyz"],
            ref_rgb=batch["ref_rgb"],
            query_rgb=batch["query_rgb"],
        )

        loss_dict = criterion(
            output=output,
            removed_label=batch["removed_label"],
            added_label=batch["added_label"],
        )

        loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(cfg, device):
            output = model(
                ref_xyz=batch["ref_xyz"],
                query_xyz=batch["query_xyz"],
                ref_rgb=batch["ref_rgb"],
                query_rgb=batch["query_rgb"],
            )

            loss_dict = criterion(
                output=output,
                removed_label=batch["removed_label"],
                added_label=batch["added_label"],
            )

            loss = loss_dict["loss"]

        if amp_enabled(cfg, device):
            scaler.scale(loss).backward()

            # optional gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_bce += float(loss_dict["bce_loss"].detach().cpu())

        if "energy_loss" in loss_dict:
            total_energy += float(loss_dict["energy_loss"].detach().cpu())

        if "margin_loss" in loss_dict:
            total_margin += float(loss_dict["margin_loss"].detach().cpu())

        count += 1

        pbar.set_postfix(
            {
                "loss": total_loss / count,
                "bce": total_bce / count,
            }
        )

    return {
        "loss": total_loss / max(count, 1),
        "bce_loss": total_bce / max(count, 1),
        "energy_loss": total_energy / max(count, 1),
        "margin_loss": total_margin / max(count, 1),
    }


@torch.no_grad()
def validate(
    model: V3BayesianLRModel,
    criterion: V3BayesianLoss,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
) -> dict[str, float]:
    model.eval()

    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})

    use_rgb = model_cfg.get("use_rgb", True)

    tau_removed = float(infer_cfg.get("tau_removed", 0.5))
    tau_added = float(infer_cfg.get("tau_added", 0.5))

    total_loss = 0.0
    total_removed_iou = 0.0
    total_added_iou = 0.0
    total_removed_acc = 0.0
    total_added_acc = 0.0
    count = 0

    total_removed_tp = 0.0
    total_removed_fp = 0.0
    total_removed_fn = 0.0
    total_removed_tn = 0.0

    total_added_tp = 0.0
    total_added_fp = 0.0
    total_added_fn = 0.0
    total_added_tn = 0.0

    total_removed_pred_pos_ratio = 0.0
    total_added_pred_pos_ratio = 0.0
    total_removed_gt_pos_ratio = 0.0
    total_added_gt_pos_ratio = 0.0

    total_removed_pos_score = 0.0
    total_removed_neg_score = 0.0
    total_added_pos_score = 0.0
    total_added_neg_score = 0.0

    pbar = tqdm(loader, desc="Validation", leave=False)

    for sample in pbar:
        batch = prepare_sample(
            sample=sample,
            device=device,
            use_rgb=use_rgb,
        )

        with autocast_context(cfg, device):
            output = model(
                ref_xyz=batch["ref_xyz"],
                query_xyz=batch["query_xyz"],
                ref_rgb=batch["ref_rgb"],
                query_rgb=batch["query_rgb"],
            )

            loss_dict = criterion(
                output=output,
                removed_label=batch["removed_label"],
                added_label=batch["added_label"],
            )

        removed_metric = binary_iou_from_logits(
            logit=output["removed_logit"],
            label=batch["removed_label"],
            threshold=tau_removed,
        )

        added_metric = binary_iou_from_logits(
            logit=output["added_logit"],
            label=batch["added_label"],
            threshold=tau_added,
        )

        total_loss += float(loss_dict["loss"].detach().cpu())
        total_removed_iou += removed_metric["iou"]
        total_added_iou += added_metric["iou"]
        total_removed_acc += removed_metric["acc"]
        total_added_acc += added_metric["acc"]

        total_removed_tp += removed_metric["tp"]
        total_removed_fp += removed_metric["fp"]
        total_removed_fn += removed_metric["fn"]
        total_removed_tn += removed_metric["tn"]

        total_added_tp += added_metric["tp"]
        total_added_fp += added_metric["fp"]
        total_added_fn += added_metric["fn"]
        total_added_tn += added_metric["tn"]

        total_removed_pred_pos_ratio += removed_metric["pred_pos_ratio"]
        total_removed_gt_pos_ratio += removed_metric["gt_pos_ratio"]
        total_added_pred_pos_ratio += added_metric["pred_pos_ratio"]
        total_added_gt_pos_ratio += added_metric["gt_pos_ratio"]

        total_removed_pos_score += removed_metric["pos_score"]
        total_removed_neg_score += removed_metric["neg_score"]
        total_added_pos_score += added_metric["pos_score"]
        total_added_neg_score += added_metric["neg_score"]
        count += 1

        mean_iou = (total_removed_iou + total_added_iou) / (2 * count)

        pbar.set_postfix(
            {
                "loss": total_loss / count,
                "mIoU": mean_iou,
            }
        )

    removed_iou = total_removed_tp / (
    total_removed_tp + total_removed_fp + total_removed_fn + 1e-8
    )

    added_iou = total_added_tp / (
        total_added_tp + total_added_fp + total_added_fn + 1e-8
    )

    background_iou = (total_removed_tn + total_added_tn) / (
        total_removed_tn + total_added_tn
        + total_removed_fp + total_removed_fn
        + total_added_fp + total_added_fn
        + 1e-8
    )

    miou = (background_iou + removed_iou + added_iou) / 3.0
    mean_change_iou = (removed_iou + added_iou) / 2.0

    removed_total = (
    total_removed_tp
    + total_removed_fp
    + total_removed_fn
    + total_removed_tn
    + 1e-8
    )

    added_total = (
        total_added_tp
        + total_added_fp
        + total_added_fn
        + total_added_tn
        + 1e-8
    )

    removed_pred_pos_ratio = (
        total_removed_tp + total_removed_fp
    ) / removed_total

    removed_gt_pos_ratio = (
        total_removed_tp + total_removed_fn
    ) / removed_total

    added_pred_pos_ratio = (
        total_added_tp + total_added_fp
    ) / added_total

    added_gt_pos_ratio = (
        total_added_tp + total_added_fn
    ) / added_total

    return {
        "loss": total_loss / max(count, 1),

        "background_iou": background_iou,
        "removed_iou": removed_iou,
        "added_iou": added_iou,
        "miou": miou,
        "mean_change_iou": mean_change_iou,

        "removed_acc": total_removed_acc / max(count, 1),
        "added_acc": total_added_acc / max(count, 1),

        "removed_pred_pos_ratio": removed_pred_pos_ratio,
        "removed_gt_pos_ratio": removed_gt_pos_ratio,
        "added_pred_pos_ratio": added_pred_pos_ratio,
        "added_gt_pos_ratio": added_gt_pos_ratio,

        "removed_pos_score": total_removed_pos_score / max(count, 1),
        "removed_neg_score": total_removed_neg_score / max(count, 1),
        "added_pos_score": total_added_pos_score / max(count, 1),
        "added_neg_score": total_added_neg_score / max(count, 1),
    }


# ------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: V3BayesianLRModel,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    epoch: int,
    best_metric: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train v3 Bayesian likelihood-ratio model for SLPCCD."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/v3_bayesian_lr.yaml",
        help="Path to config yaml.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["direct_lr", "separate_np"],
        help="Override model mode.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size. Current prototype recommends batch_size=1.",
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
        help="Override point feature dim.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cuda:0, or cpu. Default: cuda if available.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    cfg = override_config_with_args(cfg, args)

    dataset_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    output_cfg = cfg.get("output", {})

    root_dir = dataset_cfg.get("root_dir", "data/SLPCCD")
    train_split = dataset_cfg.get("train_split", "train")
    val_split = dataset_cfg.get("val_split", "val")

    batch_size = int(train_cfg.get("batch_size", 1))
    epochs = int(train_cfg.get("epochs", 100))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    if batch_size != 1:
        print(
            "[Warning] 현재 prototype collate는 batch_size=1을 권장합니다. "
            "batch_size>1은 variable-size point cloud에서 실패할 수 있습니다."
        )

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = cfg.get("model", {}).get("mode", "separate_np")

    print("=" * 80)
    print("V3 Bayesian LR Training")
    print(f"mode        : {mode}")
    print(f"device      : {device}")
    print(f"root_dir    : {root_dir}")
    print(f"train_split : {train_split}")
    print(f"val_split   : {val_split}")
    print(f"epochs      : {epochs}")
    print(f"lr          : {lr}")
    print("=" * 80)

    train_dataset = create_dataset(root_dir=root_dir, split=train_split)
    val_dataset = create_dataset(root_dir=root_dir, split=val_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch_size_one,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch_size_one,
    )

    model = build_v3_model_from_config(cfg).to(device)
    criterion = build_loss_from_cfg(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    use_amp = amp_enabled(cfg, device)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "outputs/checkpoints/v3_bayesian_lr"))
    checkpoint_dir = checkpoint_dir / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    best_metric = -1.0

    for epoch in range(1, epochs + 1):
        train_log = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            epoch=epoch,
            scaler=scaler,
        )

        val_log = validate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            cfg=cfg,
        )

        current_metric = val_log["mean_change_iou"]

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss={train_log['loss']:.4f} "
            f"val_loss={val_log['loss']:.4f} "
            f"energy={train_log['energy_loss']:.4f} "
            f"margin={train_log['margin_loss']:.4f} "
            f"bce={train_log['bce_loss']:.4f} "
            f"bg_iou={val_log['background_iou']:.4f} "
            f"removed_iou={val_log['removed_iou']:.4f} "
            f"added_iou={val_log['added_iou']:.4f} "
            f"miou={val_log['miou']:.4f} "
            f"mean_change_iou={val_log['mean_change_iou']:.4f} "
            f"r_pred={val_log['removed_pred_pos_ratio']:.4f} "
            f"r_gt={val_log['removed_gt_pos_ratio']:.4f} "
            f"a_pred={val_log['added_pred_pos_ratio']:.4f} "
            f"a_gt={val_log['added_gt_pos_ratio']:.4f} "
            f"r_pos_s={val_log['removed_pos_score']:.4f} "
            f"r_neg_s={val_log['removed_neg_score']:.4f} "
            f"a_pos_s={val_log['added_pos_score']:.4f} "
            f"a_neg_s={val_log['added_neg_score']:.4f}"
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
            best_metric=best_metric,
        )

        if current_metric > best_metric:
            best_metric = current_metric

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                best_metric=best_metric,
            )

            print(f"  -> Saved best checkpoint: {best_path} | best={best_metric:.4f}")

    print("=" * 80)
    print(f"Training finished. Best mean_change_iou = {best_metric:.4f}")
    print(f"Best checkpoint: {best_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()