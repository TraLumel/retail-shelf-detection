import torch


def make_map_metric():
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    return MeanAveragePrecision(
        iou_type="bbox", max_detection_thresholds=[10, 100, 300]
    )


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def precision_recall_f1(preds, targets, iou_thr=0.5, conf_thr=0.25):

    tp = fp = fn = 0
    for p, t in zip(preds, targets):
        keep = p["scores"] >= conf_thr
        boxes = p["boxes"][keep]
        order = p["scores"][keep].argsort(descending=True)
        boxes = boxes[order]
        gt = t["boxes"]
        if len(gt) == 0:
            fp += len(boxes)
            continue
        if len(boxes) == 0:
            fn += len(gt)
            continue
        iou = box_iou(boxes, gt)
        matched = torch.zeros(len(gt), dtype=torch.bool)
        for i in range(len(boxes)):
            iou_row = iou[i].clone()
            iou_row[matched] = 0.0
            j = iou_row.argmax()
            if iou_row[j] >= iou_thr:
                matched[j] = True
                tp += 1
            else:
                fp += 1
        fn += int((~matched).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}
