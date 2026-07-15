"""
Advanced billboard replacement (paper-style pipeline):
  1) YOLO detection (your fine-tuned weights)
  2) SAM — box-prompted segmentation (Grounded-SAM style: detector + SAM)
  3) OpenCV perspective warp of the ad onto the mask quad (paper inpainting step)
  4) Optional YOLO ByteTrack — stabler boxes across frames
  5) Temporal EMA on quad corners — less flicker

Requires: ultralytics, opencv-python, torch (CUDA recommended).
First run downloads SAM weights (e.g. sam_b.pt) automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import SAM, YOLO


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order corners: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def mask_to_quad(mask: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """4x2 float32 quad from binary mask; falls back to axis-aligned box."""
    x1, y1, x2, y2 = xyxy
    mx = float(mask.max())
    bin_mask = (mask > 127).astype(np.uint8) if mx > 1.0 else (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return order_points(
            np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        )

    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        return order_points(pts)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_points(box)


def ema_quad(prev: np.ndarray | None, cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None or alpha <= 0:
        return cur.copy()
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def rasterize_quad(shape_hw: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 255)
    return m


def perspective_replace(
    frame: np.ndarray,
    ad_bgr: np.ndarray,
    quad_dst: np.ndarray,
    feather: int,
) -> np.ndarray:
    """Warp ad to quad and blend over frame."""
    ah, aw = ad_bgr.shape[:2]
    src = np.array([[0, 0], [aw - 1, 0], [aw - 1, ah - 1], [0, ah - 1]], dtype=np.float32)
    dst = quad_dst.astype(np.float32)
    h_mat = cv2.getPerspectiveTransform(src, dst)
    fh, fw = frame.shape[:2]
    warped = cv2.warpPerspective(
        ad_bgr,
        h_mat,
        (fw, fh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    mask = rasterize_quad((fh, fw), dst)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather | 1, feather | 1), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out = frame.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _mask_tensor_to_u8(m: np.ndarray, fh: int, fw: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.max() <= 1.0:
        m = (m > 0.5).astype(np.uint8) * 255
    else:
        m = (m > 127).astype(np.uint8) * 255
    if m.shape[:2] != (fh, fw):
        m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.uint8)


def sam_masks_from_boxes(
    sam: SAM,
    frame: np.ndarray,
    boxes_xyxy: list[tuple[int, int, int, int]],
    device: str,
) -> list[np.ndarray | None]:
    """One SAM forward with all boxes; returns one full-frame mask per box (or None)."""
    if not boxes_xyxy:
        return []
    bboxes = [[float(a), float(b), float(c), float(d)] for a, b, c, d in boxes_xyxy]
    r = sam.predict(
        source=frame,
        bboxes=bboxes,
        device=device,
        verbose=False,
        imgsz=1024,
    )[0]
    fh, fw = frame.shape[:2]
    out: list[np.ndarray | None] = []
    if r.masks is None or len(r.masks) == 0:
        return [None] * len(boxes_xyxy)
    data = r.masks.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    n = data.shape[0] if data.ndim >= 3 else 1
    for i in range(min(n, len(boxes_xyxy))):
        mi = data[i] if data.ndim >= 3 else data
        out.append(_mask_tensor_to_u8(mi, fh, fw))
    while len(out) < len(boxes_xyxy):
        out.append(None)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Advanced SAM + perspective billboard replacement.")
    p.add_argument("--weights", default="runs/paper_like_det/weights/best.pt", help="YOLO detection weights")
    p.add_argument("--sam-model", default="sam_b.pt", help="SAM weights (sam_b.pt, mobile_sam.pt, sam2_t.pt, …)")
    p.add_argument("--video_in", default="video.mp4")
    p.add_argument("--ad_image", default="ad.webp")
    p.add_argument("--video_out", default="video_replaced_advanced.mp4")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--detect_width", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--track", action="store_true", help="Use YOLO ByteTrack for stabler boxes")
    p.add_argument("--ema", type=float, default=0.35, help="Temporal smoothing 0=no smooth, 0.5=heavy")
    p.add_argument("--feather", type=int, default=15, help="Edge feather (odd kernel size recommended)")
    p.add_argument(
        "--replace-all",
        action="store_true",
        help="Replace every detection each frame (default: largest billboard only, paper-style)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    device = args.device

    det = YOLO(str((project_dir / args.weights).resolve()))
    sam = SAM(args.sam_model)  # downloads on first use

    ad = cv2.imread(str((project_dir / args.ad_image).resolve()), cv2.IMREAD_UNCHANGED)
    if ad is None:
        raise FileNotFoundError(f"ad image missing: {args.ad_image}")
    if ad.ndim == 3 and ad.shape[2] == 4:
        ad = ad[:, :, :3]

    cap = cv2.VideoCapture(str((project_dir / args.video_in).resolve()))
    if not cap.isOpened():
        raise FileNotFoundError(f"video missing: {args.video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = project_dir / args.video_out
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter")

    prev_quads: dict[int, np.ndarray] = {}
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

        if args.track:
            pred = det.track(
                small,
                conf=args.conf,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml",
                device=device,
            )[0]
        else:
            pred = det.predict(small, conf=args.conf, verbose=False, device=device)[0]

        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            writer.write(frame)
            frame_idx += 1
            prev_quads.clear()
            continue

        entries: list[tuple[int, int, int, int, int, int]] = []
        for bi, b in enumerate(boxes):
            sx1, sy1, sx2, sy2 = b.xyxy[0].tolist()
            x1 = max(0, min(int(sx1 / scale), w - 1))
            y1 = max(0, min(int(sy1 / scale), h - 1))
            x2 = max(x1 + 1, min(int(sx2 / scale), w))
            y2 = max(y1 + 1, min(int(sy2 / scale), h))
            area = (x2 - x1) * (y2 - y1)
            tid = bi
            if args.track and b.id is not None:
                tid = int(b.id[0].item())
            entries.append((tid, x1, y1, x2, y2, area))

        if not args.replace_all:
            entries = [max(entries, key=lambda e: e[5])]

        active_ids = {e[0] for e in entries}
        for k in list(prev_quads.keys()):
            if k not in active_ids:
                del prev_quads[k]

        box_list = [(e[1], e[2], e[3], e[4]) for e in entries]
        masks = sam_masks_from_boxes(sam, frame, box_list, device)

        out = frame.copy()
        for (tid, x1, y1, x2, y2, _), m in zip(entries, masks):
            if m is None:
                m = np.zeros((h, w), np.uint8)
                m[y1:y2, x1:x2] = 255

            quad = mask_to_quad(m, (x1, y1, x2, y2))
            quad = ema_quad(prev_quads.get(tid), quad, args.ema)
            prev_quads[tid] = quad.copy()
            out = perspective_replace(out, ad, quad, args.feather)

        writer.write(out)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"frame {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
