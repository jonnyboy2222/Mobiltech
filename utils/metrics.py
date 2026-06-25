import numpy as np
from sklearn.metrics import confusion_matrix


def compute_slpccd_metrics(removed_pred, added_pred,
                           removed_gt, added_gt):

    pred_valid = np.hstack([
        removed_pred,
        added_pred * 2
    ])

    gt_valid = np.hstack([
        removed_gt,
        added_gt * 2
    ])

    cm = confusion_matrix(
        gt_valid,
        pred_valid,
        labels=[0, 1, 2]
    )

    class_names = [
        "background",
        "removed",
        "added"
    ]

    ious = {}
    accs = {}

    for c, name in enumerate(class_names):

        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        # IoU
        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0.0

        # Accuracy
        gt_count = cm[c, :].sum()
        acc = tp / gt_count if gt_count > 0 else 0.0

        ious[name] = iou
        accs[name] = acc

    oa = np.trace(cm) / (cm.sum() + 1e-8)

    miou = np.mean(list(ious.values()))

    return {
        "background_iou": ious["background"],
        "removed_iou": ious["removed"],
        "added_iou": ious["added"],

        "background_acc": accs["background"],
        "removed_acc": accs["removed"],
        "added_acc": accs["added"],

        "miou": miou,
        "oa": oa,

        "confusion_matrix": cm
    }