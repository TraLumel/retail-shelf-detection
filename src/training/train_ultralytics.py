import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.utils import PROJECT_ROOT, fix_data_yaml, save_json


def train_ultralytics(name: str, cfg: dict, smoke: bool = False, overrides: dict | None = None):
    from ultralytics import RTDETR, YOLO

    mcfg = cfg["models"][cfg["experiments"][name]["base"]] if name in cfg.get("experiments", {})        else cfg["models"][name]
    data_yaml = fix_data_yaml(cfg)
    run_name = f"{name}_smoke" if smoke else name
    run_dir = PROJECT_ROOT / "results" / "logs" / run_name
    weights = mcfg["weights"]


    if not smoke and (run_dir / "summary.json").exists():
        print(f"[{name}] уже обучена (есть summary.json), пропускаю")
        return


    local_w = PROJECT_ROOT / "weights" / weights
    weights_src = str(local_w) if local_w.exists() else weights

    last = run_dir / "weights" / "last.pt"
    resume = last.exists() and not smoke
    Model = RTDETR if "rtdetr" in weights else YOLO
    model = Model(str(last) if resume else weights_src)

    args = dict(
        data=str(data_yaml),
        epochs=2 if smoke else mcfg["epochs"],
        imgsz=mcfg["imgsz"],
        batch=2 if smoke else mcfg["batch"],
        seed=cfg["seed"],
        deterministic=True,
        workers=0,
        project=str(PROJECT_ROOT / "results" / "logs"),
        name=run_name,
        exist_ok=True,
        fraction=0.05 if smoke else 1.0,
        plots=True,
    )
    if name in cfg.get("experiments", {}):
        args.update(cfg["experiments"][name]["overrides"])
    if overrides:
        args.update(overrides)
    if resume:
        args = {"resume": True}
        print(f"[{name}] возобновление обучения с чекпоинта")

    t0 = time.time()
    model.train(**args)
    train_time = time.time() - t0

    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    save_json({"model": name, "params_M": round(n_params, 2),
               "epochs": args.get("epochs", mcfg["epochs"]),
               "batch": mcfg["batch"], "imgsz": args.get("imgsz", mcfg["imgsz"]),
               "train_time_min": round(train_time / 60, 1)},
              run_dir / "summary.json")
    print(f"[{name}] обучение завершено за {train_time / 60:.1f} мин")
