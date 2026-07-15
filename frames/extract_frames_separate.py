"""Extract each video to its own folder of PNG frames (no combining)."""
from pathlib import Path

import cv2

FRAMES_DIR = Path(__file__).resolve().parent
VIDEOS = [
    FRAMES_DIR / "video.mp4",
    FRAMES_DIR / "video3.MP4",
]


def extract_one(video_path: Path) -> int:
    stem = video_path.stem
    out_dir = FRAMES_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            out_path = out_dir / f"frame_{n:06d}.png"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Failed to write {out_path}")
            n += 1
            if n % 200 == 0:
                print(f"  {stem}: {n} frames...")
    finally:
        cap.release()
    print(f"{video_path.name} -> {out_dir} ({n} frames)")
    return n


def main() -> None:
    for vp in VIDEOS:
        if not vp.is_file():
            raise FileNotFoundError(f"Missing: {vp}")
    total = 0
    for vp in VIDEOS:
        total += extract_one(vp)
    print(f"Done. {total} frames total.")


if __name__ == "__main__":
    main()
