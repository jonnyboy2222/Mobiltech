"""
python scripts/sweep_added_threshold.py \
  --scores_dir outputs/results/v3_bayesian_lr/separate_np/val/scores \
  --tau_removed 0.5 \
  --out_csv outputs/sweeps/added_threshold_sweep_val.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def binary_metrics_from_score(score: torch.Tensor, gt: torch.Tensor, tau: float, eps: float = 1e-8):
    score = score.float().view(-1)
    gt = gt.long().view(-1)

    pred = (score >= tau).long()

    tp = ((pred == 1) & (gt == 1)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()
    tn = ((pred == 0) & (gt == 0)).sum().item()

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)

    pred_pos_ratio = (tp + fp) / (tp + fp + fn + tn + eps)
    gt_pos_ratio = (tp + fn) / (tp + fp + fn + tn + eps)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "acc": acc,
        "pred_pos_ratio": pred_pos_ratio,
        "gt_pos_ratio": gt_pos_ratio,
    }


def update_cm3(cm: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor):
    pred = pred.long().view(-1)
    gt = gt.long().view(-1)

    valid = (gt >= 0) & (gt < 3)
    idx = gt[valid] * 3 + pred[valid]

    cm += torch.bincount(idx, minlength=9).view(3, 3)


def compute_cm3(cm: torch.Tensor, eps: float = 1e-8):
    cm = cm.float()

    ious = []
    accs = []

    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        iou = tp / (tp + fp + fn + eps)
        acc = tp / (cm[c, :].sum() + eps)

        ious.append(float(iou))
        accs.append(float(acc))

    oa = float(torch.diag(cm).sum() / (cm.sum() + eps))

    return {
        "background_iou": ious[0],
        "removed_iou": ious[1],
        "added_iou": ious[2],
        "background_acc": accs[0],
        "removed_acc": accs[1],
        "added_acc": accs[2],
        "miou": sum(ious) / 3.0,
        "oa": oa,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="outputs/sweeps/added_threshold_sweep.csv")
    parser.add_argument("--tau_removed", type=float, default=0.5)
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--end", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    files = sorted(scores_dir.glob("sample_*_scores.pt"))

    if len(files) == 0:
        raise FileNotFoundError(f"No score files found in {scores_dir}")

    taus = []
    t = args.start
    while t <= args.end + 1e-9:
        taus.append(round(t, 4))
        t += args.step

    rows = []

    for tau_added in taus:
        added_score_all = []
        added_gt_all = []

        cm3 = torch.zeros((3, 3), dtype=torch.long)

        for path in files:
            data = torch.load(path, map_location="cpu")

            added_score = data["added_score"].view(-1).float()
            added_gt = data["added_gt"].view(-1).long()

            removed_score = data["removed_score"].view(-1).float()
            removed_gt = data["removed_gt"].view(-1).long()

            added_score_all.append(added_score)
            added_gt_all.append(added_gt)

            removed_pred = (removed_score >= args.tau_removed).long()
            added_pred = (added_score >= tau_added).long()

            ref_pred_3 = torch.where(
                removed_pred > 0,
                torch.ones_like(removed_pred),
                torch.zeros_like(removed_pred),
            )
            ref_gt_3 = torch.where(
                removed_gt > 0,
                torch.ones_like(removed_gt),
                torch.zeros_like(removed_gt),
            )

            query_pred_3 = torch.where(
                added_pred > 0,
                torch.full_like(added_pred, 2),
                torch.zeros_like(added_pred),
            )
            query_gt_3 = torch.where(
                added_gt > 0,
                torch.full_like(added_gt, 2),
                torch.zeros_like(added_gt),
            )

            pred_3 = torch.cat([ref_pred_3, query_pred_3], dim=0)
            gt_3 = torch.cat([ref_gt_3, query_gt_3], dim=0)

            update_cm3(cm3, pred_3, gt_3)

        added_score_all = torch.cat(added_score_all, dim=0)
        added_gt_all = torch.cat(added_gt_all, dim=0)

        added_bin = binary_metrics_from_score(
            score=added_score_all,
            gt=added_gt_all,
            tau=tau_added,
        )

        cm_metric = compute_cm3(cm3)

        row = {
            "tau_added": tau_added,

            "added_iou_binary": added_bin["iou"],
            "added_precision": added_bin["precision"],
            "added_recall": added_bin["recall"],
            "added_acc_binary": added_bin["acc"],
            "added_tp": added_bin["tp"],
            "added_fp": added_bin["fp"],
            "added_fn": added_bin["fn"],
            "added_tn": added_bin["tn"],
            "added_pred_pos_ratio": added_bin["pred_pos_ratio"],
            "added_gt_pos_ratio": added_bin["gt_pos_ratio"],

            "background_iou_3class": cm_metric["background_iou"],
            "removed_iou_3class": cm_metric["removed_iou"],
            "added_iou_3class": cm_metric["added_iou"],
            "miou_3class": cm_metric["miou"],
            "oa_3class": cm_metric["oa"],
        }

        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda x: x["added_iou_binary"])

    print("=" * 80)
    print(f"Saved sweep CSV: {out_csv}")
    print("-" * 80)
    print("Best by added_iou_binary")
    print(f"tau_added      : {best['tau_added']}")
    print(f"added_iou      : {best['added_iou_binary']:.4f}")
    print(f"precision      : {best['added_precision']:.4f}")
    print(f"recall         : {best['added_recall']:.4f}")
    print(f"pred_pos_ratio : {best['added_pred_pos_ratio']:.4f}")
    print(f"gt_pos_ratio   : {best['added_gt_pos_ratio']:.4f}")
    print(f"mIoU 3-class   : {best['miou_3class']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()