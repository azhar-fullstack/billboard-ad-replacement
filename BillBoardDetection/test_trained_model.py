from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test trained billboard model on image or video.")
    p.add_argument(
        "--model",
        default="runs/yolo11_finetune/weights/best.pt",
        help="Path to trained weights",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="0", help="CUDA device id or cpu")
    p.add_argument("--ad-image", default="ad.webp", help="Ad image used for replacement")

    sub = p.add_subparsers(dest="mode", required=True)

    p_img = sub.add_parser("image", help="Run inference on one image")
    p_img.add_argument("--input", default="billBoard.webp", help="Input image path")
    p_img.add_argument("--output", default="test_image_replaced.jpg", help="Output replaced image")

    p_vid = sub.add_parser("video", help="Run inference on one video")
    p_vid.add_argument("--input", default="video.mp4", help="Input video path")
    p_vid.add_argument("--output", default="test_video_replaced.mp4", help="Output replaced video")
    p_vid.add_argument(
        "--detect-width",
        type=int,
        default=1280,
        help="Resize width for detection speed (annotations written at original resolution)",
    )
    p_vid.add_argument(
        "--replace-all",
        action="store_true",
        help="Replace all detected billboards (default: only the largest for temporal stability)",
    )
    p_vid.add_argument(
        "--ema-alpha",
        type=float,
        default=0.75,
        help="EMA smoothing for the replacement box corners (0=no smoothing). Higher=more stable.",
    )
    p_vid.add_argument(
        "--hold-frames",
        type=int,
        default=2,
        help="If detection drops for a few frames, keep replacing using the last smoothed box.",
    )
    p_vid.add_argument(
        "--use-grabcut",
        action="store_true",
        help="Use GrabCut per frame for tighter masks (slower). Default uses stable box mask (faster).",
    )
    p_vid.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="Run YOLO detection every N frames (higher = faster, but can miss sudden changes).",
    )
    p_vid.add_argument(
        "--blend-mode",
        default="alpha",
        choices=["alpha", "clone"],
        help="alpha = fast stable blend (recommended for speed). clone = cv2.seamlessClone (slower).",
    )
    return p.parse_args()


def resolve_path(base_dir: Path, path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def segment_with_grabcut(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    pad = 4
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w - 1, x2 + pad)
    ry2 = min(h - 1, y2 + pad)
    rect = (rx1, ry1, max(1, rx2 - rx1), max(1, ry2 - ry1))
    cv2.grabCut(img, mask, rect, bg, fg, 4, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)


def fit_ad_to_box(ad_img: np.ndarray, w: int, h: int) -> np.ndarray:
    if ad_img.ndim == 3 and ad_img.shape[2] == 4:
        ad_rgb = ad_img[:, :, :3]
    else:
        ad_rgb = ad_img
    ah, aw = ad_rgb.shape[:2]
    target_ratio = w / h
    src_ratio = aw / ah
    if src_ratio > target_ratio:
        new_w = int(ah * target_ratio)
        x0 = (aw - new_w) // 2
        crop = ad_rgb[:, x0 : x0 + new_w]
    else:
        new_h = int(aw / target_ratio)
        y0 = (ah - new_h) // 2
        crop = ad_rgb[y0 : y0 + new_h, :]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)


def overlay_ad(base_img: np.ndarray, ad_img: np.ndarray, seg_mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    ad_fit = fit_ad_to_box(ad_img, w, h)
    local_mask = seg_mask[y1:y2, x1:x2].copy()
    if np.count_nonzero(local_mask) == 0:
        local_mask[:] = 255
    local_mask = cv2.GaussianBlur(local_mask, (9, 9), 0)
    local_mask = np.where(local_mask > 8, 255, 0).astype(np.uint8)
    # Default in this helper remains seamlessClone for compatibility, but the
    # video runner can bypass this by using alpha blending directly.
    clone_mask = np.zeros(base_img.shape[:2], dtype=np.uint8)
    clone_mask[y1:y2, x1:x2] = local_mask
    src = np.zeros_like(base_img)
    src[y1:y2, x1:x2] = ad_fit
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return cv2.seamlessClone(src, base_img, clone_mask, center, cv2.NORMAL_CLONE)


def overlay_ad_alpha(
    base_img: np.ndarray,
    ad_img: np.ndarray,
    seg_mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Fast replacement using alpha blending over the detected mask ROI."""
    out = base_img.copy()
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    ad_fit = fit_ad_to_box(ad_img, w, h)

    local = seg_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
    # Soft edge for stability.
    local = cv2.GaussianBlur(local, (9, 9), 0)
    local = np.clip(local, 0.0, 1.0)

    roi = out[y1:y2, x1:x2].astype(np.float32)
    roi = roi * (1.0 - local[..., None]) + ad_fit.astype(np.float32) * local[..., None]
    out[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)
    return out


def rectangular_mask(shape_hw: tuple[int, int], x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


def run_image(
    model: YOLO,
    input_path: Path,
    output_path: Path,
    ad_path: Path,
    conf: float,
    imgsz: int,
    device: str,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    ad_img = cv2.imread(str(ad_path), cv2.IMREAD_UNCHANGED)
    if ad_img is None:
        raise FileNotFoundError(f"Ad image not found: {ad_path}")

    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    result = model.predict(str(input_path), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    replaced = img.copy()
    print(f"Detections: {len(result.boxes)}")
    for i, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0].item())
        seg_mask = segment_with_grabcut(img, x1, y1, x2, y2)
        replaced = overlay_ad(replaced, ad_img, seg_mask, x1, y1, x2, y2)
        print(f"  {i}. xyxy=({x1}, {y1}, {x2}, {y2}) conf={score:.4f}")
    cv2.imwrite(str(output_path), replaced)
    print(f"Saved replaced image: {output_path}")


def run_video(
    model: YOLO,
    input_path: Path,
    output_path: Path,
    ad_path: Path,
    conf: float,
    imgsz: int,
    device: str,
    detect_width: int,
    replace_all: bool,
    ema_alpha: float,
    hold_frames: int,
    use_grabcut: bool,
    detect_every: int,
    blend_mode: str,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    ad_img = cv2.imread(str(ad_path), cv2.IMREAD_UNCHANGED)
    if ad_img is None:
        raise FileNotFoundError(f"Ad image not found: {ad_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    frame_idx = 0
    frames_with_det = 0
    total_dets = 0
    last_box: tuple[int, int, int, int] | None = None
    hold_left = 0
    last_dets: list[tuple[int, int, int, int]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        if w > detect_width:
            scale = detect_width / float(w)
            small = cv2.resize(frame, (detect_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = frame

        run_det = (frame_idx % max(1, detect_every) == 0) or (last_box is None)
        boxes = None
        if run_det:
            pred = model.predict(small, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            boxes = pred.boxes

        replaced = frame.copy()
        cur_boxes: list[tuple[int, int, int, int]] = []
        if boxes is not None and len(boxes) > 0:
            n = len(boxes)
            total_dets += n
            frames_with_det += 1
            for box in boxes:
                sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
                x1 = max(0, min(int(sx1 / scale), w - 1))
                y1 = max(0, min(int(sy1 / scale), h - 1))
                x2 = max(x1 + 1, min(int(sx2 / scale), w - 1))
                y2 = max(y1 + 1, min(int(sy2 / scale), h - 1))
                cur_boxes.append((x1, y1, x2, y2))
            last_dets = cur_boxes.copy()
        elif last_dets:
            # Reuse detections from the last YOLO step for speed.
            cur_boxes = last_dets.copy()

        # Default (paper-ish stability): only replace the largest billboard.
        if not replace_all and cur_boxes:
            cur_boxes = [max(cur_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))]

        if not cur_boxes:
            if last_box is not None and hold_left > 0:
                cur_boxes = [last_box]
                hold_left -= 1
            else:
                # No detection and no hold budget: write original frame.
                writer.write(frame)
                frame_idx += 1
                if frame_idx % 50 == 0:
                    print(f"Processed {frame_idx}/{total} frames")
                continue
        else:
            hold_left = hold_frames

        for (x1, y1, x2, y2) in cur_boxes:
            if last_box is None:
                smoothed = (x1, y1, x2, y2)
            else:
                px1, py1, px2, py2 = last_box
                smoothed = (
                    int(ema_alpha * px1 + (1.0 - ema_alpha) * x1),
                    int(ema_alpha * py1 + (1.0 - ema_alpha) * y1),
                    int(ema_alpha * px2 + (1.0 - ema_alpha) * x2),
                    int(ema_alpha * py2 + (1.0 - ema_alpha) * y2),
                )
            last_box = smoothed
            if use_grabcut:
                seg_mask = segment_with_grabcut(frame, smoothed[0], smoothed[1], smoothed[2], smoothed[3])
            else:
                seg_mask = rectangular_mask((h, w), smoothed[0], smoothed[1], smoothed[2], smoothed[3])

            if blend_mode == "clone":
                replaced = overlay_ad(replaced, ad_img, seg_mask, smoothed[0], smoothed[1], smoothed[2], smoothed[3])
            else:
                replaced = overlay_ad_alpha(replaced, ad_img, seg_mask, smoothed[0], smoothed[1], smoothed[2], smoothed[3])

        writer.write(replaced)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total} frames")

    cap.release()
    writer.release()

    print(f"Saved replaced video: {output_path}")
    print(f"Frames processed: {frame_idx}")
    print(f"Frames with detections: {frames_with_det}")
    print(f"Total detections: {total_dets}")


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    model_path = resolve_path(base_dir, args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    if args.mode == "image":
        input_path = resolve_path(base_dir, args.input)
        output_path = resolve_path(base_dir, args.output)
        ad_path = resolve_path(base_dir, args.ad_image)
        run_image(model, input_path, output_path, ad_path, args.conf, args.imgsz, args.device)
    else:
        input_path = resolve_path(base_dir, args.input)
        output_path = resolve_path(base_dir, args.output)
        ad_path = resolve_path(base_dir, args.ad_image)
        run_video(
            model,
            input_path,
            output_path,
            ad_path,
            args.conf,
            args.imgsz,
            args.device,
            args.detect_width,
            replace_all=args.replace_all,
            ema_alpha=args.ema_alpha,
            hold_frames=args.hold_frames,
            use_grabcut=args.use_grabcut,
            detect_every=args.detect_every,
            blend_mode=args.blend_mode,
        )


if __name__ == "__main__":
    main()
