import argparse
import base64
import json
import random
import shutil
import zlib
from pathlib import Path

import cv2
import numpy as np


def decode_bitmap(bitmap_data: str) -> np.ndarray | None:
    z = zlib.decompress(base64.b64decode(bitmap_data))
    n = np.frombuffer(z, np.uint8)
    decoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None
    if len(decoded.shape) == 3 and decoded.shape[2] == 4:
        return decoded[:, :, 3]
    if len(decoded.shape) == 2:
        return decoded
    return None


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert bitmap JSON annotations to YOLO segmentation labels.")
    parser.add_argument("--img-dir", required=True, help="Input image directory.")
    parser.add_argument("--ann-dir", required=True, help="Input annotation JSON directory.")
    parser.add_argument("--out-dir", required=True, help="Output dataset root directory.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    if not (0.5 <= args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in [0.5, 1.0).")

    # Fresh output for reproducible dataset generation
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.json"))
    records: list[tuple[Path, list[str]]] = []
    skipped = 0

    for ann_path in ann_files:
        img_path = img_dir / ann_path.stem
        if not img_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        full_board_mask_bin = np.zeros((h, w), dtype=np.uint8)
        for obj in data.get("objects", []):
            bitmap = obj.get("bitmap")
            if not bitmap or "data" not in bitmap or "origin" not in bitmap:
                continue

            mask_small = decode_bitmap(bitmap["data"])
            if mask_small is None:
                continue

            x, y = bitmap["origin"]
            x, y = int(x), int(y)
            mh, mw = mask_small.shape
            if x < 0 or y < 0 or x >= w or y >= h:
                continue

            x2 = min(x + mw, w)
            y2 = min(y + mh, h)

            patch_bin = (mask_small[: y2 - y, : x2 - x] > 0).astype(np.uint8)
            if patch_bin.size == 0:
                continue
            full_board_mask_bin[y:y2, x:x2] = np.maximum(full_board_mask_bin[y:y2, x:x2], patch_bin)

        if full_board_mask_bin.sum() == 0:
            skipped += 1
            continue

        billboard_lines = mask_to_yolo_polygons(full_board_mask_bin, class_id=0)
        label_lines = billboard_lines

        if not label_lines:
            skipped += 1
            continue

        records.append((img_path, label_lines))

    random.seed(args.seed)
    random.shuffle(records)

    n_train = int(len(records) * args.train_ratio)
    train_set = records[:n_train]
    val_set = records[n_train:]

    def write_split(split_name: str, split_records: list[tuple[Path, list[str]]]) -> None:
        for src_img, lines in split_records:
            dst_img = out_dir / "images" / split_name / src_img.name
            dst_lbl = out_dir / "labels" / split_name / f"{src_img.stem}.txt"
            shutil.copy2(src_img, dst_img)
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_split("train", train_set)
    write_split("val", val_set)

    yaml_text = f"""path: {out_dir.as_posix()}
train: images/train
val: images/val
names:
  0: billboard
"""
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Prepared dataset at: {out_dir}")
    print(f"Total usable images: {len(records)}")
    print(f"Train images: {len(train_set)}")
    print(f"Val images: {len(val_set)}")
    print(f"Skipped images: {skipped}")
    print(f"Data config: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
