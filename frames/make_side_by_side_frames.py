"""
Read three aligned MP4s and save each timestep as one horizontal strip:
  left: YOLO11n  |  middle: YOLO11l #1  |  right: YOLO11l #2
Writes PNGs incrementally (each frame saved as soon as it is built).
"""
from pathlib import Path

import cv2
import numpy as np

FRAMES_DIR = Path(__file__).resolve().parent
# Order: yolo11n first, then yolo11l "1", then yolo11l "2"
VIDEOS = (
    FRAMES_DIR / "video_freeze_yolo11n_overlay.mp4",
    FRAMES_DIR / "video_yolo11l_best1_overlay.mp4",
    FRAMES_DIR / "video_yolo11l_best1_overlay2.mp4",
)
OUT_DIR = FRAMES_DIR / "side_by_side_frames"
LABELS = ("yolo11n", "yolo11l_1", "yolo11l_2")


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def main() -> None:
    for p in VIDEOS:
        if not p.is_file():
            raise FileNotFoundError(f"Missing video: {p}")

    caps = [cv2.VideoCapture(str(p)) for p in VIDEOS]
    if not all(c.isOpened() for c in caps):
        for c in caps:
            c.release()
        raise RuntimeError("Could not open one or more video files.")

    try:
        counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
        n = min(counts)
        print(f"Frame counts: {dict(zip(LABELS, counts))} -> using {n} frames")

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            frames = []
            for c in caps:
                ok, frame = c.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to read frame {i}")
                frames.append(frame)

            target_h = min(f.shape[0] for f in frames)
            resized = [resize_to_height(f, target_h) for f in frames]
            strip = np.hstack(resized)

            out_path = OUT_DIR / f"frame_{i:06d}.png"
            if not cv2.imwrite(str(out_path), strip):
                raise RuntimeError(f"Failed to write {out_path}")
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Saved {out_path.name} ({i + 1}/{n})")
    finally:
        for c in caps:
            c.release()

    print(f"Done. {n} images in {OUT_DIR}")


if __name__ == "__main__":
    main()
