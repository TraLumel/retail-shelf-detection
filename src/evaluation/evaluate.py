import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset.dataset import SKUDataset, collate_fn
from src.evaluation.metrics import make_map_metric, precision_recall_f1
from src.models.factory import build_model
from src.utils.utils import (
    PROJECT_ROOT, fix_data_yaml, get_device, load_json, update_metrics_json,
)
from src.utils.visualize import save_prediction_images

N_VIS = 4


def eval_ultralytics(name: str, cfg: dict, smoke: bool = False):
    from ultralytics import RTDETR, YOLO
    run_name = f"{name}_smoke" if smoke else name
    run_dir = PROJECT_ROOT / "results" / "logs" / run_name
    best = run_dir / "weights" / "best.pt"
    Model = RTDETR if "rtdetr" in name else YOLO
    model = Model(str(best))
    data_yaml = fix_data_yaml(cfg)

    r = model.val(data=str(data_yaml), split="test", max_det=300, workers=0,
                  project=str(PROJECT_ROOT / "results" / "logs"),
                  name=f"{run_name}_test", exist_ok=True)

    infer_ms = float(r.speed.get("inference", 0.0))

    summary = load_json(run_dir / "summary.json", {})
    metrics = {
        "framework": "ultralytics",
        "mAP50": round(float(r.box.map50), 4),
        "mAP50_95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "f1": round(2 * float(r.box.mp) * float(r.box.mr)
                    / (float(r.box.mp) + float(r.box.mr) + 1e-9), 4),
        "infer_ms_per_img": round(infer_ms, 1),
        **{k: summary.get(k) for k in ("params_M", "train_time_min", "epochs", "imgsz", "batch")},
    }
    update_metrics_json(name, metrics)


    test_imgs = sorted((PROJECT_ROOT / cfg["data"]["root"] / "images" / "test").glob("*.jpg"))[:N_VIS]
    out_dir = PROJECT_ROOT / "results" / "plots" / f"preds_{name}"
    model.predict([str(p) for p in test_imgs], max_det=300, save=True,
                  project=str(out_dir.parent), name=out_dir.name, exist_ok=True,
                  line_width=1, show_labels=False, conf=0.25)
    print(f"[{name}] test mAP50={metrics['mAP50']} mAP={metrics['mAP50_95']}")
    return metrics


def eval_torchvision(name: str, cfg: dict, smoke: bool = False,
                     overrides: dict | None = None, run_name: str | None = None):
    device = get_device()
    mcfg = {**cfg["models"][name], **(overrides or {})}
    root = PROJECT_ROOT / cfg["data"]["root"]
    run_name = run_name or (f"{name}_smoke" if smoke else name)
    run_dir = PROJECT_ROOT / "results" / "logs" / run_name

    model = build_model(name, cfg["data"]["num_classes"], mcfg["imgsz"]).to(device)
    ckpt_path = run_dir / ("best.pt" if (run_dir / "best.pt").exists() else "last.pt")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    test_ds = SKUDataset(root, "test", train=False, limit=24 if smoke else None)
    test_dl = DataLoader(test_ds, batch_size=2, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    metric = make_map_metric()
    all_preds, all_targets = [], []
    t_infer, n_img = 0.0, 0
    with torch.no_grad():
        for images, targets in test_dl:
            images_d = [im.to(device) for im in images]
            t0 = time.time()
            preds = model(images_d)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_infer += time.time() - t0
            n_img += len(images)
            preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
            metric.update(preds, targets)
            all_preds.extend(preds)
            all_targets.extend(targets)

    res = metric.compute()
    prf = precision_recall_f1(all_preds, all_targets)
    summary = load_json(run_dir / "summary.json", {})
    metrics = {
        "framework": "torchvision",
        "mAP50": round(float(res["map_50"]), 4),
        "mAP50_95": round(float(res["map"]), 4),
        "precision": round(prf["precision"], 4),
        "recall": round(prf["recall"], 4),
        "f1": round(prf["f1"], 4),
        "infer_ms_per_img": round(1000 * t_infer / max(1, n_img), 1),
        **{k: summary.get(k) for k in ("params_M", "train_time_min", "epochs", "imgsz", "batch")},
    }
    update_metrics_json(run_name if not smoke else name, metrics)

    out_dir = PROJECT_ROOT / "results" / "plots" / f"preds_{run_name}"
    save_prediction_images(model, test_ds, device, out_dir, n=N_VIS)
    print(f"[{run_name}] test mAP50={metrics['mAP50']} mAP={metrics['mAP50_95']}")
    return metrics


def evaluate_model(name: str, cfg: dict, smoke: bool = False):
    fw = (cfg["models"].get(name) or {}).get("framework", "ultralytics")
    if fw == "ultralytics" or name in cfg.get("experiments", {}):
        return eval_ultralytics(name, cfg, smoke=smoke)
    return eval_torchvision(name, cfg, smoke=smoke)
