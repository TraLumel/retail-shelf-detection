import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class SKUDataset(Dataset):

    def __init__(self, root, split="train", train=False, limit=None):
        self.img_dir = Path(root) / "images" / split
        self.lbl_dir = Path(root) / "labels" / split
        self.files = sorted(self.img_dir.glob("*.jpg"))
        if limit:
            self.files = self.files[:limit]
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        for line in lbl_path.read_text().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
        boxes = torch.tensor(boxes, dtype=torch.float32)

        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < 0.5:
                img = TF.adjust_brightness(img, 0.7 + 0.6 * random.random())
                img = TF.adjust_saturation(img, 0.7 + 0.6 * random.random())

        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return TF.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))
