import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def mask_to_yolo_polygons(mask: np.ndarray, class_id: int, area_thresh: float = 20.0) -> list[str]:
    """
    Converts a binary mask into YOLO-seg polygon lines with a given `class_id`.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 0).astype(np.uint8) * 255
    h, w = binary.shape[:2]

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < area_thresh:
            continue
        eps = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        coords: list[str] = []
        for x, y in pts:
            x = min(max(int(x), 0), w - 1)
            y = min(max(int(y), 0), h - 1)
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")
        if len(coords) >= 6:
            lines.append(f"{class_id} " + " ".join(coords))
    return lines


def copy_existing_yolo_dataset(src_root: Path, dst_root: Path) -> tuple[int, int]:
    train_count = 0
    val_count = 0
    for split in ("train", "val"):
        src_img = src_root / "images" / split
        src_lbl = src_root / "labels" / split
        dst_img = dst_root / "images" / split
        dst_lbl = dst_root / "labels" / split
        if not src_img.exists() or not src_lbl.exists():
            continue
        for img_path in sorted(src_img.iterdir()):
            if not img_path.is_file():
                continue
            lbl_path = src_lbl / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            shutil.copy2(img_path, dst_img / img_path.name)
            shutil.copy2(lbl_path, dst_lbl / lbl_path.name)
            if split == "train":
                train_count += 1
            else:
                val_count += 1
    return train_count, val_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine existing YOLO-seg dataset with image+mask dataset.")
    parser.add_argument("--existing-yolo-root", required=True, help="Existing YOLO-seg dataset root (contains images/labels).")
    parser.add_argument("--new-images-dir", required=True, help="New dataset image folder.")
    parser.add_argument("--new-masks-dir", required=True, help="New dataset mask folder.")
    parser.add_argument("--out-root", required=True, help="Output combined dataset root.")
    parser.add_argument(
        "--new-val-ratio",
        type=float,
        default=0.1,
        help="Fraction of NEW (image+mask) pairs placed in val; e.g. 0.1 = 10%% in val, rest in train.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--exclude-stems",
        default="",
        help="Comma-separated image stems to exclude from NEW dataset (example: frame_000000,frame_000051).",
    )
    args = parser.parse_args()

    existing_root = Path(args.existing_yolo_root)
    new_images_dir = Path(args.new_images_dir)
    new_masks_dir = Path(args.new_masks_dir)
    out_root = Path(args.out_root)

    if not existing_root.exists():
        raise FileNotFoundError(f"Existing YOLO dataset not found: {existing_root}")
    if not new_images_dir.exists():
        raise FileNotFoundError(f"New images directory not found: {new_images_dir}")
    if not new_masks_dir.exists():
        raise FileNotFoundError(f"New masks directory not found: {new_masks_dir}")
    if not (0.0 <= args.new_val_ratio < 0.5):
        raise ValueError("--new-val-ratio must be in [0.0, 0.5).")
    exclude_stems = {s.strip() for s in args.exclude_stems.split(",") if s.strip()}

    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    base_train, base_val = copy_existing_yolo_dataset(existing_root, out_root)

    new_image_paths = sorted(
        [p for p in new_images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    )

    # Pair images and masks by "stem" so extensions can differ (e.g. image.jpg + mask.png).
    mask_by_stem: dict[str, Path] = {}
    for mask_path in new_masks_dir.iterdir():
        if not mask_path.is_file():
            continue
        if mask_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        mask_by_stem[mask_path.stem] = mask_path

    pairs: list[tuple[Path, Path]] = []
    for img_path in new_image_paths:
        if img_path.stem in exclude_stems:
            continue
        mask_path = new_masks_dir / img_path.name
        if not mask_path.exists():
            mask_path = mask_by_stem.get(img_path.stem)  # fall back to stem match
        if mask_path is not None and mask_path.exists():
            pairs.append((img_path, mask_path))

    random.seed(args.seed)
    random.shuffle(pairs)

    # If new_val_ratio=0, we put all new images into train (no new images in val).
    n_val = int(len(pairs) * args.new_val_ratio) if pairs else 0
    if args.new_val_ratio > 0 and n_val == 0 and pairs:
        n_val = 1
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    added_train = 0
    added_val = 0
    skipped = 0

    def write_split(split: str, items: list[tuple[Path, Path]]) -> int:
        nonlocal skipped
        written = 0
        for img_path, mask_path in items:
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if img is None or mask is None:
                skipped += 1
                continue

            # Treat provided mask as billboard (class 0) only.
            if mask.ndim == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask
            board_bin = (mask_gray > 0).astype(np.uint8)

            lines = mask_to_yolo_polygons(board_bin, class_id=0)
            if not lines:
                skipped += 1
                continue

            # Prefix new files to avoid collisions with existing dataset names.
            base_name = f"new_{img_path.stem}"
            dst_img = out_root / "images" / split / f"{base_name}{img_path.suffix.lower()}"
            dst_lbl = out_root / "labels" / split / f"{base_name}.txt"
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written += 1
        return written

    added_train = write_split("train", train_pairs)
    added_val = write_split("val", val_pairs)

    data_yaml = f"""path: {out_root.as_posix()}
train: images/train
val: images/val
names:
  0: billboard
"""
    (out_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"Combined dataset created: {out_root}")
    print(f"Base dataset -> train: {base_train}, val: {base_val}")
    print(f"New dataset -> train added: {added_train}, val added: {added_val}, skipped: {skipped}")
    print(f"Data config: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
