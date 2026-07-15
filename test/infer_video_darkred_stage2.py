import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer video with dark-red opaque segmentation overlay.")
    parser.add_argument("--model", required=True, help="Path to YOLO segmentation weights (best.pt).")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--out", required=True, help="Output video path (.mp4).")
    parser.add_argument(
        "--mode",
        choices=["overlay", "mask"],
        default="overlay",
        help="overlay = original + dark red segments. mask = black background + dark red segments.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Mask opacity for overlay mode (0..1).")
    args = parser.parse_args()

    model = YOLO(args.model)
    in_video = Path(args.input)
    out_video = Path(args.out)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {in_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video writer: {out_video}")

    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            retina_masks=True,
            verbose=False,
        )[0]

        # BGR: billboard segmentation (single-class training uses class 0 only).
        red = np.array([0, 0, 139], dtype=np.uint8)

        if args.mode == "mask":
            out_frame = np.zeros_like(frame)
        else:
            out_frame = frame.copy()

        if res.masks is not None and len(res.masks.data) > 0:
            masks_bin = res.masks.data.cpu().numpy() > 0.5  # (n, Hm, Wm)
            n = masks_bin.shape[0]

            for j in range(n):
                m = masks_bin[j].astype(np.uint8)
                if m.shape[0] != h or m.shape[1] != w:
                    m = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
                else:
                    m = (m > 0).astype(np.uint8)
                idx = m == 1
                if not np.any(idx):
                    continue
                if args.mode == "mask" or args.alpha >= 1.0:
                    out_frame[idx] = red
                else:
                    out_frame[idx] = (
                        args.alpha * red + (1.0 - args.alpha) * out_frame[idx]
                    ).round().astype(np.uint8)

        writer.write(out_frame)
        total += 1

    cap.release()
    writer.release()
    print(f"Saved: {out_video}")
    print(f"Frames: {total} | FPS: {fps:.2f} | Size: {w}x{h}")


if __name__ == "__main__":
    main()

