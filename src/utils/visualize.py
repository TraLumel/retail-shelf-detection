import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.utils import PROJECT_ROOT, load_json

PLOTS = PROJECT_ROOT / "results" / "plots"


@torch.no_grad()
def save_prediction_images(model, dataset, device, out_dir, n=4, conf=0.25):
    from torchvision.transforms import functional as TF
    from torchvision.utils import draw_bounding_boxes
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(dataset))):
        img, _ = dataset[i]
        pred = model([img.to(device)])[0]
        keep = pred["scores"] >= conf
        boxes = pred["boxes"][keep].cpu()
        img_u8 = (img * 255).to(torch.uint8)
        drawn = draw_bounding_boxes(img_u8, boxes, colors="red", width=2)
        TF.to_pil_image(drawn).save(out_dir / f"pred_{i}.jpg")


def _load_curves(name: str):

    run_dir = PROJECT_ROOT / "results" / "logs" / name
    ucsv = run_dir / "results.csv"
    tcsv = run_dir / "train_log.csv"
    if ucsv.exists():
        df = pd.read_csv(ucsv)
        df.columns = [c.strip() for c in df.columns]
        loss_cols = [c for c in df.columns if c.startswith("train/") and c.endswith("loss")]
        loss = df[loss_cols].sum(axis=1)
        map_col = next((c for c in df.columns if "mAP50(B)" in c), None)
        return df["epoch"], loss, (df[map_col] if map_col else None)
    if tcsv.exists():
        df = pd.read_csv(tcsv)
        return df["epoch"], df["train_loss"], df["val_map50"]
    return None, None, None


def plot_training_curves(model_names):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in model_names:
        ep, loss, map50 = _load_curves(name)
        if ep is None:
            continue
        axes[0].plot(ep, loss, label=name)
        if map50 is not None:
            axes[1].plot(ep, map50, label=name)
    axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("Суммарный train loss")
    axes[0].set_title("Кривые функции потерь"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Эпоха"); axes[1].set_ylabel("mAP@50 (val)")
    axes[1].set_title("Динамика mAP@50 на валидации"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "training_curves.png", dpi=150)
    plt.close(fig)
    print("Сохранено:", PLOTS / "training_curves.png")


def plot_comparison():
    metrics = load_json(PROJECT_ROOT / "results" / "metrics.json", {})
    base = {k: v for k, v in metrics.items() if not k.startswith("exp_")}
    if not base:
        print("Нет метрик для сравнения")
        return
    names = list(base.keys())
    keys = ["mAP50", "mAP50_95", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(names))
    width = 0.16
    for i, k in enumerate(keys):
        vals = [base[n].get(k) or 0 for n in names]
        bars = ax.bar([xi + i * width for xi in x], vals, width, label=k)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks([xi + 2 * width for xi in x])
    ax.set_xticklabels(names)
    ax.set_title("Сравнение моделей на тестовой выборке")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS / "model_comparison.png", dpi=150)
    plt.close(fig)


    rows = ["| Модель | mAP@50 | mAP@50-95 | Precision | Recall | F1 | Параметры, млн | Обучение, мин | Инференс, мс |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        m = base[n]
        rows.append(f"| {n} | {m.get('mAP50')} | {m.get('mAP50_95')} | {m.get('precision')} "
                    f"| {m.get('recall')} | {m.get('f1')} | {m.get('params_M')} "
                    f"| {m.get('train_time_min')} | {m.get('infer_ms_per_img')} |")
    (PLOTS / "comparison_table.md").write_text("\n".join(rows), encoding="utf-8")
    print("Сохранено:", PLOTS / "model_comparison.png")


def plot_experiments():
    metrics = load_json(PROJECT_ROOT / "results" / "metrics.json", {})
    exps = {k: v for k, v in metrics.items()
            if k.startswith("exp_") or k in ("yolov8n", "fasterrcnn")}
    if len(exps) < 2:
        return
    names = list(exps.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, k in enumerate(["mAP50", "mAP50_95"]):
        vals = [exps[n].get(k) or 0 for n in names]
        bars = ax.bar([xi + i * 0.35 for xi in range(len(names))], vals, 0.35, label=k)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks([xi + 0.175 for xi in range(len(names))])
    ax.set_xticklabels(names, rotation=10)
    ax.set_title("Влияние гиперпараметров (базовая модель YOLOv8n)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS / "experiments.png", dpi=150)
    plt.close(fig)
    print("Сохранено:", PLOTS / "experiments.png")
