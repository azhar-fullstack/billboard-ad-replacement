from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order quad corners as:
      [top-left, top-right, bottom-right, bottom-left]
    pts: (4,2) float32
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def quad_from_mask(mask_u8: np.ndarray) -> np.ndarray | None:
    """Compute a best-effort 4-corner quad from a binary mask."""
    m = (mask_u8 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        return order_points(pts)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_points(box)


def segment_with_grabcut(frame_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """GrabCut mask (0/255) in frame coordinates from a bbox."""
    x1, y1, x2, y2 = bbox_xyxy
    h, w = frame_bgr.shape[:2]
    pad = 4
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w - 1, x2 + pad)
    ry2 = min(h - 1, y2 + pad)
    rect = (rx1, ry1, max(1, rx2 - rx1), max(1, ry2 - ry1))

    mask = np.zeros((h, w), np.uint8)
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    cv2.grabCut(frame_bgr, mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)


def pick_largest_box(pred) -> tuple[int, int, int, int] | None:
    """Pick the largest bbox from YOLO prediction results."""
    boxes = pred.boxes
    if boxes is None or len(boxes) == 0:
        return None
    # YOLO box format: xyxy, conf, class. We'll use area as heuristic.
    best = None
    best_area = -1
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        area = max(0, x2i - x1i) * max(0, y2i - y1i)
        if area > best_area:
            best_area = area
            best = (x1i, y1i, x2i, y2i)
    return best


def warp_ad_to_quad(
    frame_bgr: np.ndarray,
    ad_bgr: np.ndarray,
    quad_dst: np.ndarray,
    feather: int,
) -> np.ndarray:
    """Warp ad image onto quad_dst and alpha-blend over frame_bgr.

    Important: this uses an ROI warp (not full-frame warp) for speed.
    """
    fh, fw = frame_bgr.shape[:2]
    ah, aw = ad_bgr.shape[:2]

    quad = quad_dst.astype(np.float32)
    x, y, w, h = cv2.boundingRect(np.round(quad).astype(np.int32))
    if w <= 1 or h <= 1:
        return frame_bgr

    # Clamp ROI to frame.
    pad = max(10, feather)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(fw, x + w + pad)
    y2 = min(fh, y + h + pad)

    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)

    quad_local = quad.copy()
    quad_local[:, 0] -= float(x1)
    quad_local[:, 1] -= float(y1)

    src = np.array([[0, 0], [aw - 1, 0], [aw - 1, ah - 1], [0, ah - 1]], dtype=np.float32)
    dst = quad_local.astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        ad_bgr,
        H,
        (roi_w, roi_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst).astype(np.int32), 255)
    if feather > 0:
        k = feather | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    alpha = mask.astype(np.float32) / 255.0
    out = frame_bgr.copy()
    roi = out[y1:y2, x1:x2].astype(np.float32)
    warped_f = warped.astype(np.float32)
    roi = roi * (1.0 - alpha[..., None]) + warped_f * alpha[..., None]
    out[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent billboard replacement (detect once, track quad).")
    p.add_argument("--weights", default="runs/yolo11_finetune/weights/best.pt", help="Trained YOLO weights")
    p.add_argument("--video_in", default="video.mp4", help="Input video")
    p.add_argument("--ad_image", default="ad.webp", help="Ad image to overlay")
    p.add_argument("--video_out", default="video_replaced_persistent.mp4", help="Output video")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    p.add_argument("--detect_width", type=int, default=1280, help="Resize width for faster detection")
    p.add_argument("--device", default="0", help="CUDA device id or cpu")
    p.add_argument("--feather", type=int, default=25, help="Feather width for mask blending")
    p.add_argument("--max-boards", type=int, default=3, help="Replace up to N billboards (tracked persistently)")
    p.add_argument("--min-conf", type=float, default=0.20, help="Min confidence for initial board selection")
    p.add_argument(
        "--flow-width",
        type=int,
        default=720,
        help="Optical-flow tracking width (smaller = faster). 0 = full resolution.",
    )

    p.add_argument("--ema-alpha", type=float, default=0.8, help="Smoothing for tracked quad corners (0..1)")
    p.add_argument("--flow-window", type=int, default=21, help="Optical flow window size (odd recommended)")
    p.add_argument("--flow-levels", type=int, default=3, help="Optical flow pyramid levels")
    p.add_argument("--flow-max-fails", type=int, default=6, help="Re-detect after N bad tracking frames")
    p.add_argument("--reinit-every", type=int, default=0, help="Optional periodic re-detection (0=off)")

    p.add_argument(
        "--use-opencl",
        action="store_true",
        help="Try OpenCV OpenCL acceleration (only if OpenCL is available).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Set OpenCV thread count (0 = leave default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    # Speed tweaks for CPU path (OpenCV CUDA isn't available on this setup).
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    if args.threads and args.threads > 0:
        try:
            cv2.setNumThreads(int(args.threads))
        except Exception:
            pass
    if args.use_opencl:
        try:
            if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "haveOpenCL") and cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                print("OpenCL: enabled", flush=True)
            else:
                print("OpenCL: not available", flush=True)
        except Exception:
            print("OpenCL: enable failed", flush=True)

    model_path = Path(args.weights)
    if not model_path.is_absolute():
        model_path = base_dir / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"weights not found: {model_path}")

    video_in = Path(args.video_in)
    if not video_in.is_absolute():
        video_in = base_dir / video_in
    ad_path = Path(args.ad_image)
    if not ad_path.is_absolute():
        ad_path = base_dir / ad_path
    video_out = Path(args.video_out)
    if not video_out.is_absolute():
        video_out = base_dir / video_out

    ad_bgr = cv2.imread(str(ad_path), cv2.IMREAD_UNCHANGED)
    if ad_bgr is None:
        raise FileNotFoundError(f"Ad image not readable: {ad_path}")
    if ad_bgr.ndim == 3 and ad_bgr.shape[2] == 4:
        ad_bgr = ad_bgr[:, :, :3]

    det = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prefer AVI on Windows for reliable container finalization.
    if video_out.suffix.lower() == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (fw, fh))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {video_out}")

    # Optical flow tracking resolution (CPU speed-up).
    if args.flow_width and args.flow_width > 0 and fw > args.flow_width:
        flow_w = int(args.flow_width)
        flow_h = max(1, int(fh * (flow_w / float(fw))))
    else:
        flow_w, flow_h = fw, fh
    sx = flow_w / float(fw)
    sy = flow_h / float(fh)

    def to_flow_quad(quad_orig: np.ndarray) -> np.ndarray:
        q = quad_orig.astype(np.float32).copy()
        q[:, 0] *= float(sx)
        q[:, 1] *= float(sy)
        return q

    def to_orig_quad(quad_flow: np.ndarray) -> np.ndarray:
        q = quad_flow.astype(np.float32).copy()
        q[:, 0] /= float(sx)
        q[:, 1] /= float(sy)
        return q

    def gray_flow(frame_bgr: np.ndarray) -> np.ndarray:
        if flow_w != fw or flow_h != fh:
            small = cv2.resize(frame_bgr, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def detect_quads(frame_bgr: np.ndarray) -> list[np.ndarray]:
        """Detect up to max-boards in the given frame and return ordered quads."""
        h, w = frame_bgr.shape[:2]
        if w > args.detect_width:
            scale = args.detect_width / float(w)
            small = cv2.resize(frame_bgr, (args.detect_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = frame_bgr

        pred = det.predict(small, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            return []

        candidates: list[tuple[float, int, int, int, int]] = []
        for bi, b in enumerate(boxes):
            if float(b.conf[0].item()) < float(args.min_conf):
                continue
            sx1, sy1, sx2, sy2 = b.xyxy[0].tolist()
            x1 = int(sx1 / scale)
            y1 = int(sy1 / scale)
            x2 = int(sx2 / scale)
            y2 = int(sy2 / scale)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            area = float((x2 - x1) * (y2 - y1))
            candidates.append((area, x1, y1, x2, y2))

        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[: max(1, int(args.max_boards))]

        quads: list[np.ndarray] = []
        for _area, x1, y1, x2, y2 in candidates:
            mask = segment_with_grabcut(frame_bgr, (x1, y1, x2, y2))
            quad = quad_from_mask(mask)
            if quad is None:
                quad = order_points(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
            quads.append(quad)
        return quads

    def stacked_pts_from_quads(quads_orig: list[np.ndarray]) -> np.ndarray:
        pts_flow = [to_flow_quad(q).reshape(-1, 1, 2) for q in quads_orig]
        return np.concatenate(pts_flow, axis=0).astype(np.float32)

    try:
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError("Video has no frames.")

        quads = detect_quads(frame0)
        if not quads:
            # No detection ever: just copy.
            while True:
                writer.write(frame0)
                ok, frame0 = cap.read()
                if not ok:
                    break
            print("No billboard detected; video copied.", flush=True)
            return

        last_good_quads = [q.copy() for q in quads]
        prev_gray = gray_flow(frame0)
        prev_pts = stacked_pts_from_quads(last_good_quads)

        bad_count = 0
        frame_idx = 0
        while True:
            if frame_idx == 0:
                frame = frame0
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            gray = gray_flow(frame)

            # Decide whether to re-detect all boards.
            need_reinit = False
            if args.reinit_every and frame_idx % args.reinit_every == 0:
                need_reinit = True
            if bad_count >= args.flow_max_fails:
                need_reinit = True

            if need_reinit:
                quads_new = detect_quads(frame)
                if quads_new:
                    last_good_quads = [q.copy() for q in quads_new]
                    prev_pts = stacked_pts_from_quads(last_good_quads)
                    # Write with fresh quads.
                    out = frame.copy()
                    for q in last_good_quads:
                        out = warp_ad_to_quad(out, ad_bgr, q, args.feather)
                    writer.write(out)
                    prev_gray = gray
                    bad_count = 0
                    frame_idx += 1
                    continue
                else:
                    # Could not detect; write original frame.
                    writer.write(frame)
                    prev_gray = gray
                    frame_idx += 1
                    continue

            next_pts, st, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_pts,
                None,
                winSize=(args.flow_window, args.flow_window),
                maxLevel=args.flow_levels,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            if next_pts is None or st is None:
                bad_count += 1
                out = frame.copy()
                for q in last_good_quads:
                    out = warp_ad_to_quad(out, ad_bgr, q, args.feather)
                writer.write(out)
            else:
                st_1 = st.reshape(-1).astype(bool) if st is not None else None
                if st_1 is None or len(st_1) != (len(last_good_quads) * 4):
                    bad_count += 1
                    out = frame.copy()
                    for q in last_good_quads:
                        out = warp_ad_to_quad(out, ad_bgr, q, args.feather)
                    writer.write(out)
                else:
                    # Update each board independently. If some corners fail,
                    # keep those corners from last frame instead of freezing everything.
                    any_valid = False
                    quads_updated: list[np.ndarray] = []
                    for i in range(len(last_good_quads)):
                        quad_i_flow = next_pts[i * 4 : (i + 1) * 4].reshape(4, 2).astype(np.float32)
                        quad_i_orig = to_orig_quad(quad_i_flow)

                        st_i = st_1[i * 4 : (i + 1) * 4]  # 4 booleans
                        if int(np.sum(st_i)) > 0:
                            any_valid = True

                        mixed = last_good_quads[i].astype(np.float32).copy()
                        for j in range(4):
                            if bool(st_i[j]):
                                mixed[j] = quad_i_orig[j]

                        quad_i_s = ema_quad_points(last_good_quads[i], mixed, args.ema_alpha)
                        quads_updated.append(quad_i_s)

                    if not any_valid:
                        bad_count += 1
                    else:
                        bad_count = 0

                    last_good_quads = quads_updated
                    out = frame.copy()
                    for q in last_good_quads:
                        out = warp_ad_to_quad(out, ad_bgr, q, args.feather)
                    writer.write(out)

                    # Keep prev_pts in updated (smoothed) location.
                    prev_pts = stacked_pts_from_quads(last_good_quads)

            prev_gray = gray
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"frame {frame_idx}/{total} bad={bad_count}", flush=True)

        print(f"Saved: {video_out}", flush=True)
    finally:
        cap.release()
        writer.release()


def ema_quad_points(prev_quad: np.ndarray, cur_quad: np.ndarray, ema_alpha: float) -> np.ndarray:
    """EMA smoothing: higher ema_alpha => smoother (more history)."""
    prev_quad = prev_quad.astype(np.float32)
    cur_quad = cur_quad.astype(np.float32)
    a = float(ema_alpha)
    return (a * prev_quad + (1.0 - a) * cur_quad).astype(np.float32)


if __name__ == "__main__":
    main()

