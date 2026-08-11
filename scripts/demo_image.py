#!/usr/bin/env python3
"""
Portfolio demo — detect a billboard on the sample image and warp a new ad onto it.

Usage (from repo root):
  python scripts/demo_image.py
  python scripts/demo_image.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replace_image import YOLO, run_image  # noqa: E402


def side_by_side(before: Path, after: Path, out: Path) -> None:
    a = cv2.imread(str(before))
    b = cv2.imread(str(after))
    if a is None or b is None:
        raise FileNotFoundError("Could not read before/after images for collage")
    h = min(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    gap = np.full((h, 12, 3), 30, dtype=np.uint8)
    collage = np.hstack([a, gap, b])
    cv2.putText(collage, "BEFORE", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        collage,
        "AFTER",
        (a.shape[1] + 28, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), collage)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Billboard ad-replacement image demo")
    p.add_argument("--model", default=str(ROOT / "weights" / "best.pt"))
    p.add_argument("--input", default=str(ROOT / "assets" / "sample_billboard.webp"))
    p.add_argument("--ad-image", default=str(ROOT / "assets" / "sample_ad.webp"))
    p.add_argument("--output", default=str(ROOT / "demos" / "replaced.jpg"))
    p.add_argument("--collage", default=str(ROOT / "demos" / "before_after.jpg"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu", help="cpu or CUDA device id")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            "ERROR: weights not found:\n"
            f"  {model_path}\n\n"
            "Place your trained billboard model at:\n"
            "  weights/best.pt\n"
            "Weights are gitignored (too large for GitHub)."
        )
        return 1

    input_path = Path(args.input)
    ad_path = Path(args.ad_image)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model : {model_path}")
    print(f"Input : {input_path}")
    print(f"Ad    : {ad_path}")
    print(f"Device: {args.device}")

    model = YOLO(str(model_path))
    run_image(
        model=model,
        input_path=input_path,
        output_path=output_path,
        ad_path=ad_path,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    side_by_side(input_path, output_path, Path(args.collage))
    # also refresh portfolio asset
    side_by_side(input_path, output_path, ROOT / "assets" / "before_after.jpg")
    print(f"Collage: {args.collage}")
    print("Demo OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
