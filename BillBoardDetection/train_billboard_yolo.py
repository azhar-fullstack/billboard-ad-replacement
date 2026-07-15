from pathlib import Path
import argparse
import yaml

from ultralytics import YOLO


def make_data_yaml(project_dir: Path) -> Path:
    data = {
        "train": str((project_dir / "train" / "images").resolve()),
        "val": str((project_dir / "valid" / "images").resolve()),
        "test": str((project_dir / "test" / "images").resolve()),
        "nc": 1,
        "names": ["billboard"],
    }
    out = project_dir / "data_abs.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Train billboard detector (YOLO).")
    p.add_argument(
        "--model",
        default="weights/yolo11n.pt",
        help="Base checkpoint path (defaults to your YOLO11 billboard model)",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (use 4 on ~6GB GPUs; increase if you have headroom)",
    )
    p.add_argument("--device", default="0", help="CUDA device id or cpu")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="billboard_det")
    p.add_argument(
        "--cos_lr",
        action="store_true",
        help="Cosine LR schedule (often helps late-epoch mAP)",
    )
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs)")
    return p.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_yaml = make_data_yaml(project_dir)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir / args.project),
        name=args.name,
        exist_ok=True,
        cos_lr=args.cos_lr,
        patience=args.patience,
    )

    best = project_dir / args.project / args.name / "weights" / "best.pt"
    if best.exists():
        metrics = model.val(data=str(data_yaml), split="test", device=args.device)
        print("Training complete.")
        print(f"Best weights: {best}")
        print("Test metrics:", metrics.results_dict)
    else:
        print("Training finished but best weights were not found.")


if __name__ == "__main__":
    main()
