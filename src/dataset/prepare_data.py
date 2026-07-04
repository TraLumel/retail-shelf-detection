import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.utils import PROJECT_ROOT, save_json

HF_DATASET = "benjamintli/sku110k"
SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}


def process_split(split: str, n: int, out_root: Path, max_side: int, stats: dict):
    from datasets import load_dataset
    from tqdm import tqdm

    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(HF_DATASET, split=SPLIT_MAP[split], streaming=True)
    n_boxes, box_areas, saved = [], [], 0

    for i, sample in enumerate(tqdm(ds, total=n, desc=split)):
        if saved >= n:
            break
        img = sample["image"]
        bboxes = sample["objects"]["bbox"]
        if img is None or not bboxes:
            continue
        img = img.convert("RGB")
        w0, h0 = img.size
        scale = min(1.0, max_side / max(w0, h0))
        w, h = int(round(w0 * scale)), int(round(h0 * scale))
        if scale < 1.0:
            img = img.resize((w, h))

        lines = []
        for (bx, by, bw, bh) in bboxes:
            bx, by, bw, bh = bx * scale, by * scale, bw * scale, bh * scale

            x1, y1 = max(0.0, bx), max(0.0, by)
            x2, y2 = min(float(w), bx + bw), min(float(h), by + bh)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            nw, nh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            box_areas.append((x2 - x1) * (y2 - y1))
        if not lines:
            continue

        stem = f"{split}_{saved:05d}"
        img.save(img_dir / f"{stem}.jpg", quality=90)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_boxes.append(len(lines))
        saved += 1

    stats[split] = {
        "images": saved,
        "boxes_total": int(np.sum(n_boxes)),
        "boxes_per_image_mean": float(np.mean(n_boxes)),
        "boxes_per_image_min": int(np.min(n_boxes)),
        "boxes_per_image_max": int(np.max(n_boxes)),
        "box_area_mean_px": float(np.mean(box_areas)),
        "box_area_median_px": float(np.median(box_areas)),
    }
    print(f"[{split}] сохранено {saved} изображений, {int(np.sum(n_boxes))} боксов")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=1500)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--out", type=str, default="data/processed")
    args = p.parse_args()

    out_root = (PROJECT_ROOT / args.out).resolve()
    stats = {}
    for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        process_split(split, n, out_root, args.max_side, stats)


    data_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "product"},
    }
    with open(out_root / "sku110k.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f)

    save_json(stats, out_root / "stats.json")
    print("Готово. Статистика:", out_root / "stats.json")


if __name__ == "__main__":
    main()
