from pathlib import Path
import argparse

import cv2
import numpy as np
from ultralytics import YOLO


def fit_to_box(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    target_ratio = width / height
    src_ratio = w / h
    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        crop = image[:, x0 : x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        crop = image[y0 : y0 + new_h, :]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LANCZOS4)


def segment_grabcut(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    pad = 4
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w - 1, x2 + pad)
    ry2 = min(h - 1, y2 + pad)
    rect = (rx1, ry1, max(1, rx2 - rx1), max(1, ry2 - ry1))
    cv2.grabCut(frame, mask, rect, bg, fg, 4, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)


def overlay(frame: np.ndarray, ad: np.ndarray, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    ad_fit = fit_to_box(ad, w, h)
    local = mask[y1:y2, x1:x2].copy()
    if np.count_nonzero(local) == 0:
        local[:] = 255
    local = cv2.GaussianBlur(local, (9, 9), 0)
    local = np.where(local > 8, 255, 0).astype(np.uint8)

    clone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    clone_mask[y1:y2, x1:x2] = local
    src = np.zeros_like(frame)
    src[y1:y2, x1:x2] = ad_fit
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return cv2.seamlessClone(src, frame, clone_mask, center, cv2.NORMAL_CLONE)


def parse_args():
    p = argparse.ArgumentParser(description="Detect + replace billboard in video.")
    p.add_argument("--weights", required=True)
    p.add_argument("--video_in", default="video.mp4")
    p.add_argument("--ad_image", default="ad.webp")
    p.add_argument("--video_out", default="video_replaced_model.mp4")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--detect_width", type=int, default=1280)
    return p.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    model = YOLO(str((project_dir / args.weights).resolve()))

    ad = cv2.imread(str((project_dir / args.ad_image).resolve()), cv2.IMREAD_UNCHANGED)
    if ad is None:
        raise FileNotFoundError(f"ad image missing: {args.ad_image}")
    if ad.ndim == 3 and ad.shape[2] == 4:
        ad = ad[:, :, :3]

    cap = cv2.VideoCapture(str((project_dir / args.video_in).resolve()))
    if not cap.isOpened():
        raise FileNotFoundError(f"video missing/open failed: {args.video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str((project_dir / args.video_out).resolve()),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("failed to open video writer")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        if w > args.detect_width:
            scale = args.detect_width / float(w)
            small = cv2.resize(frame, (args.detect_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = frame

        pred = model.predict(small, conf=args.conf, verbose=False)[0]
        out = frame.copy()
        for b in pred.boxes:
            sx1, sy1, sx2, sy2 = b.xyxy[0].tolist()
            x1 = max(0, min(int(sx1 / scale), w - 1))
            y1 = max(0, min(int(sy1 / scale), h - 1))
            x2 = max(x1 + 1, min(int(sx2 / scale), w))
            y2 = max(y1 + 1, min(int(sy2 / scale), h))
            seg_mask = segment_grabcut(frame, x1, y1, x2, y2)
            out = overlay(out, ad, seg_mask, x1, y1, x2, y2)

        writer.write(out)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"processed {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"saved: {project_dir / args.video_out}")


if __name__ == "__main__":
    main()
