import argparse
import traceback

from src.utils.utils import load_config


def is_oom(exc: BaseException) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("out of memory", "cuda", "paging file", "1455"))


def train_one(name: str, cfg: dict, smoke: bool) -> str:

    from src.utils.utils import PROJECT_ROOT
    fw = cfg["models"][name]["framework"] if name in cfg["models"] else "ultralytics"


    if fw == "ultralytics" and name in cfg["models"]:
        w = cfg["models"][name].get("weights")
        fb = cfg.get("fallbacks", {}).get(name)
        if w and fb and not (PROJECT_ROOT / "weights" / w).exists():
            print(f"[!] {name}: веса {w} не скачаны, обучаю {fb} вместо неё")
            return train_one(fb, cfg, smoke)
    try:
        if fw == "torchvision":
            from src.training.train_torchvision import train_torchvision
            train_torchvision(name, cfg, smoke=smoke)
        else:
            from src.training.train_ultralytics import train_ultralytics
            train_ultralytics(name, cfg, smoke=smoke)
        return name
    except Exception as e:
        traceback.print_exc()
        fb = cfg.get("fallbacks", {}).get(name)
        if fb and is_oom(e):
            print(f"\n[!] {name}: не хватило видеопамяти, переключаюсь на {fb}\n")
            return train_one(fb, cfg, smoke)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all",
                   choices=["prepare", "train", "eval", "experiments", "plots", "all"])
    p.add_argument("--model", default=None, help="имя модели из configs/default.yaml")
    p.add_argument("--smoke", action="store_true", help="быстрый тест: 2 эпохи, часть данных")
    args = p.parse_args()
    cfg = load_config()

    if args.stage == "prepare":
        from src.dataset.prepare_data import main as prep
        import sys
        sys.argv = [sys.argv[0]]
        prep()
        return

    trained = []
    if args.stage in ("train", "all"):
        names = [args.model] if args.model else cfg["train_order"]
        for name in names:
            print(f"\n{'=' * 60}\n  Обучение: {name}\n{'=' * 60}")
            trained.append(train_one(name, cfg, args.smoke))

    if args.stage in ("eval", "all"):
        from src.evaluation.evaluate import evaluate_model
        names = trained or ([args.model] if args.model else cfg["train_order"])
        for name in names:
            print(f"\n--- Оценка: {name} ---")
            try:
                evaluate_model(name, cfg, smoke=args.smoke)
            except Exception:
                traceback.print_exc()

    if args.stage == "experiments":
        from src.training.train_ultralytics import train_ultralytics
        from src.training.train_torchvision import train_torchvision
        from src.evaluation.evaluate import eval_ultralytics, eval_torchvision
        for exp, ecfg in cfg["experiments"].items():
            base = ecfg["base"]
            fw = cfg["models"][base].get("framework", "ultralytics")
            print(f"\n{'=' * 60}\n  Эксперимент: {exp} (база: {base})\n{'=' * 60}")
            try:
                if fw == "torchvision":
                    train_torchvision(base, cfg, smoke=args.smoke,
                                      overrides=ecfg["overrides"], run_name=exp)
                    eval_torchvision(base, cfg, overrides=ecfg["overrides"], run_name=exp)
                else:
                    train_ultralytics(exp, cfg, smoke=args.smoke)
                    eval_ultralytics(exp, cfg)
            except Exception:
                traceback.print_exc()

    if args.stage in ("plots", "all"):
        from src.utils.visualize import plot_comparison, plot_experiments, plot_training_curves
        from src.utils.utils import PROJECT_ROOT, load_json
        metrics = load_json(PROJECT_ROOT / "results" / "metrics.json", {})
        plot_training_curves([m for m in metrics if not m.startswith("exp_")])
        plot_comparison()
        plot_experiments()


if __name__ == "__main__":
    main()
