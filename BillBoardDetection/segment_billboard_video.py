from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment billboard from a video using a YOLO segmentation model.")
    p.add_argument("--weights", required=True, help="Path to YOLO segmentation weights (.pt)")
    p.add_argument("--video-in", required=True, help="Input video path")
    p.add_argument("--video-out", required=True, help="Output segmented video path")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="0", help="Device for inference (e.g. 0 or cpu)")
    p.add_argument(
        "--keep-largest-only",
        action="store_true",
        help="If set, keep only the largest segmented billboard each frame",
    )
    p.add_argument(
        "--overlay",
        action="store_true",
        help="If set, draw transparent mask over original frame instead of blacking background",
    )
    return p.parse_args()


def resolve_path(base_dir: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base_dir / path)


def masks_to_binary(mask_data: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    """Convert model mask tensor [N, Hm, Wm] into one binary frame mask [H, W]."""
    if mask_data.size == 0:
        return np.zeros((frame_h, frame_w), dtype=np.uint8)

    merged = np.max(mask_data, axis=0)  # [Hm, Wm]
    merged = (merged > 0.5).astype(np.uint8) * 255
    if merged.shape[0] != frame_h or merged.shape[1] != frame_w:
        merged = cv2.resize(merged, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
    return merged


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    weights = resolve_path(base_dir, args.weights)
    video_in = resolve_path(base_dir, args.video_in)
    video_out = resolve_path(base_dir, args.video_out)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not video_in.exists():
        raise FileNotFoundError(f"Input video not found: {video_in}")

    video_out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {video_out}")

    frame_idx = 0
    frames_with_masks = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]

        frame_mask = np.zeros((height, width), dtype=np.uint8)
        if result.masks is not None and len(result.masks.data) > 0:
            frames_with_masks += 1
            mask_data = result.masks.data.detach().cpu().numpy()  # [N, Hm, Wm]

            if args.keep_largest_only:
                areas = np.sum(mask_data > 0.5, axis=(1, 2))
                largest_idx = int(np.argmax(areas))
                mask_data = mask_data[largest_idx : largest_idx + 1]

            frame_mask = masks_to_binary(mask_data, height, width)

        if args.overlay:
            output_frame = frame.copy()
            green = np.zeros_like(frame)
            green[:, :] = (0, 255, 0)
            alpha = (frame_mask.astype(np.float32) / 255.0) * 0.45
            output_frame = (
                output_frame.astype(np.float32) * (1.0 - alpha[..., None])
                + green.astype(np.float32) * alpha[..., None]
            ).astype(np.uint8)
        else:
            output_frame = cv2.bitwise_and(frame, frame, mask=frame_mask)

        writer.write(output_frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total} frames")

    cap.release()
    writer.release()

    print(f"Saved segmented video: {video_out}")
    print(f"Frames processed: {frame_idx}")
    print(f"Frames with segmentation masks: {frames_with_masks}")


if __name__ == "__main__":
    main()
