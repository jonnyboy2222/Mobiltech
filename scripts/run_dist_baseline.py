import argparse
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.slpccd_loader import SLPCCDDataset
from v1_simple_dist.bidirectional_cd import run_bidirectional_distance_cd
from utils.metrics import compute_slpccd_metrics

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def compute_metrics_from_cm(cm):
    class_names = ["background", "removed", "added"]
    results = {}

    ious = []
    accs = []

    for c, name in enumerate(class_names):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        gt = cm[c, :].sum()

        iou = tp / (tp + fp + fn + 1e-8)
        acc = tp / (gt + 1e-8)

        results[f"{name}_iou"] = iou
        results[f"{name}_acc"] = acc

        ious.append(iou)
        accs.append(acc)

    results["miou"] = float(np.mean(ious))
    results["oa"] = float(np.trace(cm) / (cm.sum() + 1e-8))
    results["confusion_matrix"] = cm

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dist_baseline.yaml")

    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--tau_removed", type=float, default=None)
    parser.add_argument("--tau_added", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    root_dir = args.root_dir or cfg["data"]["root_dir"]
    split = args.split or cfg["data"]["split"]

    voxel_size = args.voxel_size if args.voxel_size is not None else cfg["method"]["voxel_size"]
    radius = args.radius if args.radius is not None else cfg["method"]["radius"]
    tau_removed = args.tau_removed if args.tau_removed is not None else cfg["method"]["tau_removed"]
    tau_added = args.tau_added if args.tau_added is not None else cfg["method"]["tau_added"]

    print("\n========== Config ==========")
    print(f"root_dir      : {root_dir}")
    print(f"split         : {split}")
    print(f"voxel_size    : {voxel_size}")
    print(f"radius        : {radius}")
    print(f"tau_removed   : {tau_removed}")
    print(f"tau_added     : {tau_added}")
    print("================================\n")

    dataset = SLPCCDDataset(
        root_dir=root_dir,
        split=split
    )

    total_cm = np.zeros((3, 3), dtype=np.int64)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        out = run_bidirectional_distance_cd(
            pc2016=sample["ref_xyz"],
            pc2020=sample["query_xyz"],
            voxel_size=voxel_size,
            radius=radius,
            tau_removed=tau_removed,
            tau_added=tau_added,
        )

        sample_metrics = compute_slpccd_metrics(
            removed_pred=out["removed_pred"],
            added_pred=out["added_pred"],
            removed_gt=sample["ref_label"],
            added_gt=sample["query_label"],
        )

        total_cm += sample_metrics["confusion_matrix"]

        if i % 10 == 0:
            print(
                f"[{i}/{len(dataset)}] "
                f"mIoU={sample_metrics['miou']:.4f} "
                f"OA={sample_metrics['oa']:.4f} "
                f"BG IoU={sample_metrics['background_iou']:.4f} "
                f"REM IoU={sample_metrics['removed_iou']:.4f} "
                f"ADD IoU={sample_metrics['added_iou']:.4f}"
            )

    final_metrics = compute_metrics_from_cm(total_cm)

    print("\n================ Final Results ================\n")
    print(f"{'Metric':<20} {'Score':>10}")
    print("-" * 32)
    print(f"{'Background IoU':<20} {final_metrics['background_iou']:.4f}")
    print(f"{'Removed IoU':<20} {final_metrics['removed_iou']:.4f}")
    print(f"{'Added IoU':<20} {final_metrics['added_iou']:.4f}")
    print("-" * 32)
    print(f"{'Background Acc':<20} {final_metrics['background_acc']:.4f}")
    print(f"{'Removed Acc':<20} {final_metrics['removed_acc']:.4f}")
    print(f"{'Added Acc':<20} {final_metrics['added_acc']:.4f}")
    print("-" * 32)
    print(f"{'mIoU':<20} {final_metrics['miou']:.4f}")
    print(f"{'OA':<20} {final_metrics['oa']:.4f}")

    print("\nConfusion Matrix:")
    print(final_metrics["confusion_matrix"])

    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "split": split,
        "voxel_size": voxel_size,
        "radius": radius,
        "tau_removed": tau_removed,
        "tau_added": tau_added,
        "background_iou": final_metrics["background_iou"],
        "removed_iou": final_metrics["removed_iou"],
        "added_iou": final_metrics["added_iou"],
        "background_acc": final_metrics["background_acc"],
        "removed_acc": final_metrics["removed_acc"],
        "added_acc": final_metrics["added_acc"],
        "miou": final_metrics["miou"],
        "oa": final_metrics["oa"],
    }

    df = pd.DataFrame([result_dict])
    exp_name = cfg["experiment"]["name"]

    csv_path = Path(f"outputs/metrics/{exp_name}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    save_cfg_path = csv_path.parent / "last_config.yaml"

    with open(save_cfg_path, "w") as f:
        yaml.dump(cfg, f)


if __name__ == "__main__":
    main()