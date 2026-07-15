from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment billboard on all images in a folder.")
    p.add_argument("--weights", required=True, help="Path to YOLO segmentation model (.pt)")
    p.add_argument("--input-dir", required=True, help="Folder containing input images")
    p.add_argument("--output-dir", required=True, help="Folder to save segmented images")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="cpu", help="Inference device: cpu or cuda id")
    p.add_argument(
        "--keep-largest-only",
        action="store_true",
        help="Keep only the largest billboard mask per image",
    )
    p.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay segmentation mask on original image instead of black background",
    )
    return p.parse_args()


def resolve_path(base_dir: Path, path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def collect_images(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def build_mask(result, h: int, w: int, keep_largest_only: bool) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None or len(result.masks.data) == 0:
        return mask

    data = result.masks.data.detach().cpu().numpy()  # [N, Hm, Wm]
    if keep_largest_only:
        areas = np.sum(data > 0.5, axis=(1, 2))
        data = data[int(np.argmax(areas)) : int(np.argmax(areas)) + 1]

    merged = (np.max(data, axis=0) > 0.5).astype(np.uint8) * 255
    if merged.shape != (h, w):
        merged = cv2.resize(merged, (w, h), interpolation=cv2.INTER_NEAREST)
    return merged


def apply_output(img: np.ndarray, mask: np.ndarray, overlay: bool) -> np.ndarray:
    if not overlay:
        return cv2.bitwise_and(img, img, mask=mask)

    out = img.copy()
    tint = np.zeros_like(img)
    tint[:, :] = (0, 255, 0)
    alpha = (mask.astype(np.float32) / 255.0) * 0.45
    out = (out.astype(np.float32) * (1.0 - alpha[..., None]) + tint.astype(np.float32) * alpha[..., None]).astype(
        np.uint8
    )
    return out


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    weights = resolve_path(base_dir, args.weights)
    input_dir = resolve_path(base_dir, args.input_dir)
    output_dir = resolve_path(base_dir, args.output_dir)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    model = YOLO(str(weights))

    total = len(images)
    with_masks = 0
    for idx, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        h, w = img.shape[:2]
        result = model.predict(img, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        mask = build_mask(result, h, w, args.keep_largest_only)
        if np.count_nonzero(mask) > 0:
            with_masks += 1

        segmented = apply_output(img, mask, args.overlay)
        out_path = output_dir / f"{img_path.stem}_segmented.png"
        cv2.imwrite(str(out_path), segmented)

        if idx % 25 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    print(f"Done. Input images: {total}")
    print(f"Images with detected masks: {with_masks}")
    print(f"Saved outputs in: {output_dir}")


if __name__ == "__main__":
    main()
