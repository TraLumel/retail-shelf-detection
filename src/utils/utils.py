import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(PROJECT_ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fix_data_yaml(cfg: dict) -> Path:

    yaml_path = PROJECT_ROOT / cfg["data"]["yaml"]
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = str((PROJECT_ROOT / cfg["data"]["root"]).resolve())
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return yaml_path


def count_params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_metrics_json(name: str, metrics: dict, path=None):

    path = Path(path or PROJECT_ROOT / "results" / "metrics.json")
    all_m = load_json(path, {})
    all_m[name] = metrics
    save_json(all_m, path)


class CSVLogger:


    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, self.fieldnames).writerow(row)
