import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset.dataset import SKUDataset, collate_fn
from src.evaluation.metrics import make_map_metric
from src.models.factory import build_model
from src.utils.utils import (
    PROJECT_ROOT, CSVLogger, count_params_m, get_device, save_json, set_seed,
)


@torch.no_grad()
def evaluate_map(model, loader, device):
    model.eval()
    metric = make_map_metric()
    for images, targets in loader:
        images = [im.to(device) for im in images]
        preds = model(images)
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        metric.update(preds, targets)
    res = metric.compute()
    return float(res["map_50"]), float(res["map"])


def train_torchvision(name: str, cfg: dict, smoke: bool = False,
                      overrides: dict | None = None, run_name: str | None = None):
    set_seed(cfg["seed"])
    device = get_device()
    mcfg = {**cfg["models"][name], **(overrides or {})}
    root = PROJECT_ROOT / cfg["data"]["root"]
    run_name = run_name or (f"{name}_smoke" if smoke else name)
    run_dir = PROJECT_ROOT / "results" / "logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)


    if not smoke and (run_dir / "summary.json").exists():
        print(f"[{name}] уже обучена (есть summary.json), пропускаю")
        return

    epochs = 2 if smoke else mcfg["epochs"]
    limit = 48 if smoke else None
    batch = 2 if smoke else mcfg["batch"]

    train_ds = SKUDataset(root, "train", train=True, limit=limit)
    val_ds = SKUDataset(root, "val", train=False, limit=24 if smoke else None)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=0, collate_fn=collate_fn, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=max(1, batch // 2), shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    model = build_model(name, cfg["data"]["num_classes"], mcfg["imgsz"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=mcfg["lr"], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best_map50, train_time = 0, 0.0, 0.0
    last_ckpt = run_dir / "last.pt"
    if last_ckpt.exists() and not smoke:
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_map50 = ck.get("best_map50", 0.0)
        train_time = ck.get("train_time", 0.0)
        print(f"[{name}] возобновление с эпохи {start_epoch}")

    logger = CSVLogger(run_dir / "train_log.csv",
                       ["epoch", "lr", "train_loss", "val_map50", "val_map", "epoch_time_s"])

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for images, targets in train_dl:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not math.isfinite(loss.item()):
                print(f"[{name}] нечисловой loss, батч пропущен")
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        epoch_time = time.time() - t0
        train_time += epoch_time

        map50, mp = evaluate_map(model, val_dl, device)
        avg_loss = total_loss / max(1, n_batches)
        print(f"[{name}] epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} "
              f"mAP50={map50:.4f} mAP={mp:.4f} ({epoch_time:.0f}s)")
        logger.log({"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": round(avg_loss, 5), "val_map50": round(map50, 5),
                    "val_map": round(mp, 5), "epoch_time_s": round(epoch_time, 1)})

        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "scheduler": scheduler.state_dict(), "epoch": epoch,
              "best_map50": best_map50, "train_time": train_time}
        torch.save(ck, last_ckpt)
        if map50 > best_map50:
            best_map50 = map50
            ck["best_map50"] = best_map50
            torch.save(ck, run_dir / "best.pt")

    save_json({"model": run_name, "base_model": name,
               "params_M": round(count_params_m(model), 2),
               "epochs": epochs, "batch": batch, "imgsz": mcfg["imgsz"],
               "lr": mcfg["lr"], "optimizer": "SGD",
               "train_time_min": round(train_time / 60, 1),
               "best_val_map50": round(best_map50, 4)},
              run_dir / "summary.json")
    print(f"[{name}] обучение завершено, best val mAP50={best_map50:.4f}")
