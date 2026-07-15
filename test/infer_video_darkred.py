from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main() -> None:
    model = YOLO(r"d:/Editors/Vs code/Code/03_FiverrProjects/test/runs/billboard_seg_train/weights/best.pt")
    in_video = Path(r"d:/Editors/Vs code/Code/03_FiverrProjects/test/t/video3.MP4")
    out_dir = Path(r"d:/Editors/Vs code/Code/03_FiverrProjects/test/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / "video3_darkred_seg.mp4"

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

        res = model.predict(source=frame, imgsz=640, conf=0.25, retina_masks=True, verbose=False)[0]
        out_frame = frame.copy()

        if res.masks is not None and len(res.masks.data) > 0:
            masks = res.masks.data.cpu().numpy() > 0.5
            union = np.any(masks, axis=0).astype(np.uint8)

            if union.shape[0] != h or union.shape[1] != w:
                union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)

            # Dark red and fully opaque in segmented area.
            out_frame[union == 1] = (0, 0, 139)

        writer.write(out_frame)
        total += 1

    cap.release()
    writer.release()
    print(f"Saved: {out_video}")
    print(f"Frames: {total} | FPS: {fps:.2f} | Size: {w}x{h}")


if __name__ == "__main__":
    main()
