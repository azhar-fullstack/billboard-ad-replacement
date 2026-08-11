#!/usr/bin/env python3
"""
Train a YOLO billboard detector.

Dataset layout (YOLO):
  datasets/billboard/
    train/images/ ...
    train/labels/ ...
    valid/images/ ...
    valid/labels/ ...
    test/images/  ...   (optional)
    test/labels/  ...

Or pass an existing data.yaml with --data.

Usage:
  python src/train.py --data datasets/billboard --epochs 100 --device 0
  python src/train.py --data path/to/data.yaml --model yolo11n.pt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def resolve_data_yaml(data_arg: str, root: Path) -> Path:
    """Accept a data.yaml path or a dataset folder and return a usable yaml path."""
    p = Path(data_arg)
    if not p.is_absolute():
        p = (root / p).resolve()

    if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}:
        return p

    if not p.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {p}\n"
            "Pass a data.yaml or a folder with train/valid (and optional test) images."
        )

    train_img = p / "train" / "images"
    # support both valid/ and val/
    val_img = p / "valid" / "images"
    if not val_img.exists():
        val_img = p / "val" / "images"
    test_img = p / "test" / "images"

    if not train_img.exists() or not val_img.exists():
        raise FileNotFoundError(
            f"Expected YOLO folders under {p}:\n"
            "  train/images, valid/images (or val/images)\n"
            "  (+ matching labels/ next to images/)"
        )

    data = {
        "path": str(p.resolve()),
        "train": "train/images",
        "val": str(val_img.relative_to(p)).replace("\\", "/"),
        "nc": 1,
        "names": ["billboard"],
    }
    if test_img.exists():
        data["test"] = "test/images"

    out = p / "data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out}")
    return out


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Train billboard YOLO detector")
    p.add_argument(
        "--data",
        default=str(root / "datasets" / "billboard"),
        help="Path to data.yaml OR dataset root folder",
    )
    p.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Base checkpoint (yolo11n.pt, yolo11n-seg.pt, or a local .pt)",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0", help="CUDA id or cpu")
    p.add_argument("--project", default=str(root / "runs"))
    p.add_argument("--name", default="billboard_det")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--cos-lr", action="store_true", help="Cosine LR schedule")
    p.add_argument(
        "--copy-best",
        action="store_true",
        default=True,
        help="Copy best.pt into weights/best.pt (default: on)",
    )
    p.add_argument("--no-copy-best", action="store_false", dest="copy_best")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_yaml = resolve_data_yaml(args.data, root)

    print(f"Data  : {data_yaml}")
    print(f"Model : {args.model}")
    print(f"Device: {args.device}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        cos_lr=args.cos_lr,
        patience=args.patience,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        print("Training finished but best weights were not found.")
        return 1

    print(f"Best weights: {best}")

    # optional test split eval
    try:
        with open(data_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if cfg.get("test"):
            metrics = model.val(data=str(data_yaml), split="test", device=args.device)
            print("Test metrics:", getattr(metrics, "results_dict", metrics))
    except Exception as exc:
        print(f"Test eval skipped: {exc}")

    if args.copy_best:
        dest = root / "weights" / "best.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"Copied to {dest}")

    print("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
