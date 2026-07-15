# ============================================================
# FIRST 200 IMAGES
# ORIGINAL + DIRECTION-AWARE SMART FILLED SEGMENTATION
# ============================================================

import os
import json
import cv2
import numpy as np
import base64
import zlib
import argparse
from pathlib import Path
from tqdm import tqdm

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMG_DIR = BASE_DIR / "dataset" / "football" / "images"
DEFAULT_ANN_DIR = BASE_DIR / "dataset" / "football" / "annotations"
DEFAULT_OUT_DIR = BASE_DIR / "segmentation_directional_preview"

# -------------------------------
# DECODE FUNCTION (KAGGLE STYLE)
# -------------------------------
def decode_bitmap(bitmap_data):
    z = zlib.decompress(base64.b64decode(bitmap_data))
    n = np.frombuffer(z, np.uint8)
    decoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)

    if decoded is None:
        return None

    if len(decoded.shape) == 3 and decoded.shape[2] == 4:
        return decoded[:, :, 3]
    elif len(decoded.shape) == 2:
        return decoded
    return None

# -------------------------------
# ESTIMATE DOMINANT BOARD ANGLE
# -------------------------------
def get_dominant_direction(mask):
    """
    Returns unit direction vector (vx, vy) along the main board direction.
    Falls back to horizontal if no line is found.
    """
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=20
    )

    if lines is None:
        return 1.0, 0.0  # horizontal fallback

    best_len = 0
    best_vec = (1.0, 0.0)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5

        if length > best_len and length > 1e-6:
            best_len = length
            best_vec = (dx / length, dy / length)

    return best_vec

# -------------------------------
# SMART DIRECTIONAL FILL
# - connects missing board pieces
# - tries to preserve player gaps
# -------------------------------
def directional_fill(mask, search_len=35, local_band=5, min_support=3):
    """
    mask: uint8 mask (0 or 255)
    search_len: how far to search in both directions
    local_band: vertical/horizontal neighborhood support
    min_support: support threshold to avoid filling isolated holes
    """
    mask_bin = (mask > 0).astype(np.uint8)
    filled = mask_bin.copy()

    h, w = mask_bin.shape
    vx, vy = get_dominant_direction(mask_bin * 255)

    # Normal direction helps reject player-like thin holes
    nx, ny = -vy, vx

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return mask_bin * 255

    # Only inspect region around current board to speed up a bit
    x_min, x_max = max(0, xs.min() - 20), min(w - 1, xs.max() + 20)
    y_min, y_max = max(0, ys.min() - 20), min(h - 1, ys.max() + 20)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if mask_bin[y, x] == 1:
                continue

            forward_hit = False
            backward_hit = False

            # Search forward/backward along board direction
            for step in range(1, search_len + 1):
                xf = int(round(x + vx * step))
                yf = int(round(y + vy * step))
                xb = int(round(x - vx * step))
                yb = int(round(y - vy * step))

                if 0 <= xf < w and 0 <= yf < h:
                    if mask_bin[yf, xf] == 1:
                        forward_hit = True

                if 0 <= xb < w and 0 <= yb < h:
                    if mask_bin[yb, xb] == 1:
                        backward_hit = True

                if forward_hit and backward_hit:
                    break

            if not (forward_hit and backward_hit):
                continue

            # Local support check across the normal direction.
            # Missing board piece usually has surrounding board structure.
            # A player gap is often thin/irregular and fails this more often.
            support = 0
            for step in range(-local_band, local_band + 1):
                xn = int(round(x + nx * step))
                yn = int(round(y + ny * step))
                if 0 <= xn < w and 0 <= yn < h:
                    support += int(mask_bin[yn, xn])

            if support >= min_support:
                filled[y, x] = 1

    # Small cleanup only
    kernel = np.ones((3, 3), np.uint8)
    filled = cv2.morphologyEx(filled * 255, cv2.MORPH_CLOSE, kernel)

    return filled

# -------------------------------
# PROCESS FIRST 200 FILES
# -------------------------------
def run(img_dir, ann_dir, out_dir, limit):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")])[:limit]
    saved = 0

    for file in tqdm(files):
        ann_path = os.path.join(ann_dir, file)
        img_name = file.replace(".json", "")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for obj in data.get("objects", []):
            if "bitmap" not in obj:
                continue

            bitmap = obj["bitmap"]
            if "data" not in bitmap or "origin" not in bitmap:
                continue

            mask_small = decode_bitmap(bitmap["data"])
            if mask_small is None:
                continue

            x, y = bitmap["origin"]  # this dataset behaves as [x, y]
            x, y = int(x), int(y)
            mh, mw = mask_small.shape

            if x < 0 or y < 0 or x >= w or y >= h:
                continue

            x2 = min(x + mw, w)
            y2 = min(y + mh, h)

            patch = mask_small[: y2 - y, : x2 - x]
            full_mask[y:y2, x:x2] = np.maximum(full_mask[y:y2, x:x2], patch)

        full_mask = (full_mask > 0).astype(np.uint8) * 255

        # Direction-aware refinement
        fixed_mask = directional_fill(
            full_mask,
            search_len=35,
            local_band=5,
            min_support=3
        )

        # Overlay
        overlay = img.copy()
        overlay[fixed_mask > 0] = [0, 0, 255]

        # Side-by-side
        combined = np.hstack((img, overlay))

        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, combined)
        saved += 1

    print(f"DONE: saved {saved} images in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate directional segmentation preview images from bitmap annotations."
    )
    parser.add_argument(
        "--img-dir",
        default=str(DEFAULT_IMG_DIR),
        help=f"Folder containing source images (default: {DEFAULT_IMG_DIR})"
    )
    parser.add_argument(
        "--ann-dir",
        default=str(DEFAULT_ANN_DIR),
        help=f"Folder containing annotation JSON files (default: {DEFAULT_ANN_DIR})"
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Folder to save output previews (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of annotation files to process (default: 200)"
    )

    args = parser.parse_args()
    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")
    if not os.path.isdir(args.ann_dir):
        raise FileNotFoundError(f"Annotation directory not found: {args.ann_dir}")
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    run(args.img_dir, args.ann_dir, args.out_dir, args.limit)