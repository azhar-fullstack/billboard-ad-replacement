"""Rename PNGs in video/ then video3/ to 1,2,3,... with video3 continuing after video/."""
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
FOLDERS = [BASE / "video", BASE / "video3"]


def frame_sort_key(path: Path) -> tuple:
    m = re.search(r"frame_(\d+)", path.name, re.I)
    if m:
        return (0, int(m.group(1)))
    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, path.name.lower())


def png_files(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]


def renumber_folder(folder: Path, start_num: int) -> int:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    files = png_files(folder)
    files.sort(key=frame_sort_key)
    if not files:
        return 0
    # Two-phase rename to avoid clashes with target names
    temps: list[tuple[Path, int]] = []
    for i, p in enumerate(files):
        tmp = folder / f"_renumber_tmp_{i:05d}.png"
        p.rename(tmp)
        temps.append((tmp, start_num + i))
    for tmp, num in temps:
        tmp.rename(folder / f"{num}.png")
    return len(files)


def main() -> None:
    n = 1
    for folder in FOLDERS:
        k = renumber_folder(folder, n)
        print(f"{folder.name}: {k} files -> {n} .. {n + k - 1 if k else n - 1}")
        n += k
    print(f"Done. Last index: {n - 1}")


if __name__ == "__main__":
    main()
