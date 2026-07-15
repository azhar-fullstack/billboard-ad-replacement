from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order quad corners: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def quad_from_mask(mask_u8: np.ndarray) -> np.ndarray | None:
    """Compute best-effort 4-corner quad from binary mask."""
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


def pick_top_boxes(pred, min_conf: float, max_boards: int) -> list[tuple[int, int, int, int]]:
    """Pick top-N boxes by area for single-class detections."""
    boxes = pred.boxes
    if boxes is None or len(boxes) == 0:
        return []
    candidates = []
    for b in boxes:
        conf = float(b.conf[0].item())
        if conf < float(min_conf):
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
        area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        candidates.append((area, xi1, yi1, xi2, yi2))
    candidates.sort(key=lambda t: t[0], reverse=True)
    candidates = candidates[: max(1, max_boards)]
    return [(c[1], c[2], c[3], c[4]) for c in candidates]


def warp_ad_to_quad_roi(frame_bgr: np.ndarray, ad_bgr: np.ndarray, quad_dst: np.ndarray, feather: int) -> np.ndarray:
    """Warp ad into the billboard quad ROI and alpha-blend (ROI warp for speed)."""
    fh, fw = frame_bgr.shape[:2]
    ah, aw = ad_bgr.shape[:2]
    quad = quad_dst.astype(np.float32)

    x, y, w, h = cv2.boundingRect(np.round(quad).astype(np.int32))
    if w <= 1 or h <= 1:
        return frame_bgr

    pad = max(10, feather)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(fw, x + w + pad)
    y2 = min(fh, y + h + pad)

    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)
    if roi_w <= 1 or roi_h <= 1:
        return frame_bgr

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
    p = argparse.ArgumentParser(description="Persistent billboard replacement via homography tracking.")
    p.add_argument("--weights", default="runs/yolo11_finetune/weights/best.pt")
    p.add_argument("--video_in", default="video.mp4")
    p.add_argument("--ad_image", default="ad.webp")
    p.add_argument("--video_out", default="video_replaced_persistent_homography.mp4")
    p.add_argument("--codec", default="mp4v", help="FourCC codec (recommended: mp4v for .mp4, XVID for .avi)")

    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    p.add_argument("--detect_width", type=int, default=1280, help="Resize width for faster initial detection")
    p.add_argument("--device", default="0", help="CUDA device id or cpu")

    p.add_argument("--min-conf", type=float, default=0.20, help="Min confidence for initial multi-board selection")
    p.add_argument("--max-boards", type=int, default=3, help="Replace up to N billboards")

    p.add_argument("--flow-width", type=int, default=720, help="Optical-flow tracking resolution width")
    p.add_argument("--flow-window", type=int, default=21, help="LK optical flow window size")
    p.add_argument("--flow-levels", type=int, default=3, help="LK pyramid levels")
    p.add_argument("--flow-max-fails", type=int, default=6, help="Re-detect after N bad tracking frames")

    p.add_argument("--features-per-board", type=int, default=120, help="Corner features per board for homography")
    p.add_argument("--ransac-thresh", type=float, default=3.0, help="RANSAC reprojection threshold")
    p.add_argument("--min-inliers", type=int, default=10, help="Minimum inliers to accept homography")
    p.add_argument(
        "--motion-model",
        choices=["affine", "homography"],
        default="affine",
        help="Tracking transform model. 'affine' is more stable and prevents skew drift.",
    )
    p.add_argument("--redetect-interval", type=int, default=20, help="Run detector every N frames to correct drift")
    p.add_argument("--redetect-iou", type=float, default=0.15, help="Min IoU to snap tracked board to detected board")
    p.add_argument("--redetect-alpha", type=float, default=0.65, help="Blend weight when snapping to detection")
    p.add_argument("--min-area-scale", type=float, default=0.45, help="Reject updates shrinking area too much")
    p.add_argument("--max-area-scale", type=float, default=2.2, help="Reject updates growing area too much")
    p.add_argument("--max-corner-step", type=float, default=90.0, help="Reject updates with excessive corner jump")
    p.add_argument("--min-feature-pool", type=int, default=50, help="Replenish points if tracked features are too low")

    p.add_argument("--feather", type=int, default=25)
    p.add_argument("--ema-alpha", type=float, default=0.8, help="EMA smoothing for quad corners")
    return p.parse_args()


def ema_quad_points(prev_quad: np.ndarray, cur_quad: np.ndarray, ema_alpha: float) -> np.ndarray:
    prev = prev_quad.astype(np.float32)
    cur = cur_quad.astype(np.float32)
    a = float(ema_alpha)
    return (a * prev + (1.0 - a) * cur).astype(np.float32)


def quad_area(quad: np.ndarray) -> float:
    return float(abs(cv2.contourArea(quad.astype(np.float32))))


def bbox_from_quad(quad: np.ndarray) -> tuple[float, float, float, float]:
    x1 = float(np.min(quad[:, 0]))
    y1 = float(np.min(quad[:, 1]))
    x2 = float(np.max(quad[:, 0]))
    y2 = float(np.max(quad[:, 1]))
    return x1, y1, x2, y2


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def is_reasonable_quad_update(
    prev_quad: np.ndarray,
    new_quad: np.ndarray,
    fw: int,
    fh: int,
    min_area_scale: float,
    max_area_scale: float,
    max_corner_step: float,
) -> bool:
    prev_area = max(1.0, quad_area(prev_quad))
    new_area = quad_area(new_quad)
    scale = new_area / prev_area
    if scale < float(min_area_scale) or scale > float(max_area_scale):
        return False

    step = np.linalg.norm(new_quad.astype(np.float32) - prev_quad.astype(np.float32), axis=1)
    if float(np.max(step)) > float(max_corner_step):
        return False

    if np.any(new_quad[:, 0] < -20) or np.any(new_quad[:, 0] > fw + 20):
        return False
    if np.any(new_quad[:, 1] < -20) or np.any(new_quad[:, 1] > fh + 20):
        return False
    return True


def replenish_features(gray: np.ndarray, quad_flow: np.ndarray, existing_pts: np.ndarray, target_count: int) -> np.ndarray:
    pts = existing_pts
    cur_n = 0 if pts is None else int(len(pts))
    if cur_n >= target_count:
        return pts.astype(np.float32)

    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad_flow).astype(np.int32), 255)
    need = max(0, target_count - cur_n)
    new_pts = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=max(need * 2, 25),
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
    )
    if new_pts is None:
        return pts.astype(np.float32)
    if pts is None or len(pts) == 0:
        return new_pts[:target_count].astype(np.float32)

    old = pts.reshape(-1, 2).astype(np.float32)
    candidates = new_pts.reshape(-1, 2).astype(np.float32)
    keep = []
    for c in candidates:
        d = np.linalg.norm(old - c[None, :], axis=1)
        if float(np.min(d)) >= 4.0:
            keep.append(c)
        if len(keep) >= need:
            break
    if not keep:
        return pts.astype(np.float32)
    merged = np.vstack([old, np.array(keep, dtype=np.float32)]).reshape(-1, 1, 2)
    if len(merged) > target_count:
        merged = merged[:target_count]
    return merged.astype(np.float32)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = base_dir / weights
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    video_in = Path(args.video_in)
    if not video_in.is_absolute():
        video_in = base_dir / video_in
    if not video_in.exists():
        raise FileNotFoundError(f"video not found: {video_in}")

    ad_path = Path(args.ad_image)
    if not ad_path.is_absolute():
        ad_path = base_dir / ad_path
    if not ad_path.exists():
        raise FileNotFoundError(f"ad image not found: {ad_path}")

    video_out = Path(args.video_out)
    if not video_out.is_absolute():
        video_out = base_dir / video_out

    det = YOLO(str(weights))

    ad_bgr = cv2.imread(str(ad_path), cv2.IMREAD_UNCHANGED)
    if ad_bgr is None:
        raise FileNotFoundError(f"Ad image not readable: {ad_path}")
    if ad_bgr.ndim == 3 and ad_bgr.shape[2] == 4:
        ad_bgr = ad_bgr[:, :, :3]

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    codec = (args.codec or "mp4v").strip()
    if len(codec) != 4:
        raise ValueError(f"Invalid codec '{codec}'. Use a 4-character FourCC such as mp4v or XVID.")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (fw, fh))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {video_out}")

    flow_w = int(args.flow_width)
    flow_h = max(1, int(fh * (flow_w / float(fw))))
    sx = flow_w / float(fw)
    sy = flow_h / float(fh)

    def to_flow_quad(q_orig: np.ndarray) -> np.ndarray:
        q = q_orig.astype(np.float32).copy()
        q[:, 0] *= sx
        q[:, 1] *= sy
        return q

    def to_orig_quad(q_flow: np.ndarray) -> np.ndarray:
        q = q_flow.astype(np.float32).copy()
        q[:, 0] /= sx
        q[:, 1] /= sy
        return q

    def gray_flow(frame_bgr: np.ndarray) -> np.ndarray:
        if flow_w == fw and flow_h == fh:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(frame_bgr, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def detect_initial_quads(frame0_bgr: np.ndarray) -> list[np.ndarray]:
        """Detect up to N billboards, return one ordered quad per board."""
        h, w = frame0_bgr.shape[:2]
        if w > args.detect_width:
            scale = args.detect_width / float(w)
            small = cv2.resize(frame0_bgr, (args.detect_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = frame0_bgr

        pred = det.predict(small, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        boxes = pick_top_boxes(pred, min_conf=args.min_conf, max_boards=args.max_boards)
        quads: list[np.ndarray] = []
        for (x1, y1, x2, y2) in boxes:
            # boxes were computed from pred coords already in original image coords? We rescaled.
            if scale != 1.0:
                # Undo scale from the prediction space -> original
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            mask = segment_with_grabcut(frame0_bgr, (x1, y1, x2, y2))
            quad = quad_from_mask(mask)
            if quad is None:
                quad = order_points(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
            quads.append(quad)
        return quads[: max(1, args.max_boards)]

    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        writer.release()
        raise RuntimeError("Video has no frames.")

    quads = detect_initial_quads(frame0)
    if not quads:
        writer.write(frame0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        return

    # Init tracking features for each quad (in flow coordinates).
    gray_prev = gray_flow(frame0)
    board_state = []
    corners_flow = []
    for q_orig in quads:
        q_flow = to_flow_quad(q_orig)
        corners_flow.append(q_flow)
        mask_poly = np.zeros((flow_h, flow_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_poly, np.round(q_flow).astype(np.int32), 255)
        feats = cv2.goodFeaturesToTrack(
            gray_prev,
            mask=mask_poly,
            maxCorners=int(args.features_per_board),
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7,
        )
        if feats is None:
            feats = np.zeros((0, 1, 2), dtype=np.float32)
        board_state.append(
            {
                "q_orig": q_orig.astype(np.float32),
                "q_flow": q_flow.astype(np.float32),
                "prev_pts": feats.astype(np.float32),
                "fail": 0,
            }
        )

    # Always write frame 0 replacement.
    out = frame0.copy()
    for st in board_state:
        out = warp_ad_to_quad_roi(out, ad_bgr, st["q_orig"], args.feather)
    writer.write(out)

    frame_idx = 1
    bad_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = gray_flow(frame)
        out = frame.copy()

        any_homography_ok = False
        for bi, st in enumerate(board_state):
            prev_pts = st["prev_pts"]
            if prev_pts is None or len(prev_pts) < 4:
                st["fail"] += 1
                continue

            next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                gray_prev,
                gray,
                prev_pts,
                None,
                winSize=(args.flow_window, args.flow_window),
                maxLevel=args.flow_levels,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            if next_pts is None or status is None:
                st["fail"] += 1
                continue

            status_1 = status.reshape(-1).astype(bool)
            prev_good = prev_pts[status_1].reshape(-1, 2)
            next_good = next_pts[status_1].reshape(-1, 2)

            if len(prev_good) < args.min_inliers:
                st["fail"] += 1
                # Still update prev_pts to keep tracking alive if we got some points.
                st["prev_pts"] = next_pts[status_1].reshape(-1, 1, 2).astype(np.float32)
                continue

            # Use affine by default to keep billboard motion rigid (avoid shear/keystone drift).
            q_flow_prev = st["q_flow"].astype(np.float32).reshape(-1, 1, 2)
            if args.motion_model == "affine":
                A, _inliers = cv2.estimateAffinePartial2D(
                    prev_good,
                    next_good,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=float(args.ransac_thresh),
                )
                if A is None:
                    st["fail"] += 1
                    continue
                q_flow_new = cv2.transform(q_flow_prev, A).reshape(-1, 2).astype(np.float32)
            else:
                H, _mask = cv2.findHomography(
                    prev_good,
                    next_good,
                    cv2.RANSAC,
                    ransacReprojThreshold=float(args.ransac_thresh),
                )
                if H is None:
                    st["fail"] += 1
                    continue
                q_flow_new = cv2.perspectiveTransform(q_flow_prev, H).reshape(-1, 2).astype(np.float32)

            q_orig_new = to_orig_quad(q_flow_new)
            q_orig_new = order_points(q_orig_new)
            if not is_reasonable_quad_update(
                st["q_orig"],
                q_orig_new,
                fw=fw,
                fh=fh,
                min_area_scale=args.min_area_scale,
                max_area_scale=args.max_area_scale,
                max_corner_step=args.max_corner_step,
            ):
                st["fail"] += 1
                continue
            q_orig_sm = ema_quad_points(st["q_orig"], q_orig_new, args.ema_alpha)

            st["q_orig"] = q_orig_sm
            st["q_flow"] = q_flow_new

            # Advance feature points for next iteration.
            st["prev_pts"] = next_good.reshape(-1, 1, 2).astype(np.float32)
            st["prev_pts"] = replenish_features(
                gray,
                st["q_flow"],
                st["prev_pts"],
                target_count=max(int(args.min_feature_pool), int(args.features_per_board)),
            )
            st["fail"] = 0
            any_homography_ok = True

            out = warp_ad_to_quad_roi(out, ad_bgr, st["q_orig"], args.feather)

        if not any_homography_ok:
            bad_count += 1
        else:
            bad_count = 0

        # Periodic detector snap keeps long sequences stable under zoom/rotation and suppresses drift.
        if args.redetect_interval > 0 and frame_idx % int(args.redetect_interval) == 0 and board_state:
            quads_det = detect_initial_quads(frame)
            if quads_det:
                used_det = set()
                for st in board_state:
                    prev_bb = bbox_from_quad(st["q_orig"])
                    best_i = -1
                    best_iou = 0.0
                    for i, qd in enumerate(quads_det):
                        if i in used_det:
                            continue
                        iou = bbox_iou(prev_bb, bbox_from_quad(qd))
                        if iou > best_iou:
                            best_iou = iou
                            best_i = i
                    if best_i >= 0 and best_iou >= float(args.redetect_iou):
                        used_det.add(best_i)
                        q_det = order_points(quads_det[best_i].astype(np.float32))
                        st["q_orig"] = ema_quad_points(st["q_orig"], q_det, float(args.redetect_alpha))
                        st["q_flow"] = to_flow_quad(st["q_orig"])
                        st["prev_pts"] = replenish_features(
                            gray,
                            st["q_flow"],
                            np.zeros((0, 1, 2), dtype=np.float32),
                            target_count=max(int(args.min_feature_pool), int(args.features_per_board)),
                        )
                        st["fail"] = 0

        # If tracking is consistently bad, re-detect all quads once.
        if bad_count >= args.flow_max_fails:
            quads_new = detect_initial_quads(frame)
            if not quads_new:
                writer.write(frame)
                gray_prev = gray
                frame_idx += 1
                continue

            board_state = []
            for q_orig in quads_new[: max(1, args.max_boards)]:
                q_flow = to_flow_quad(q_orig)
                mask_poly = np.zeros((flow_h, flow_w), dtype=np.uint8)
                cv2.fillConvexPoly(mask_poly, np.round(q_flow).astype(np.int32), 255)
                feats = cv2.goodFeaturesToTrack(
                    gray,
                    mask=mask_poly,
                    maxCorners=int(args.features_per_board),
                    qualityLevel=0.01,
                    minDistance=5,
                    blockSize=7,
                )
                if feats is None:
                    feats = np.zeros((0, 1, 2), dtype=np.float32)
                board_state.append(
                    {
                        "q_orig": q_orig.astype(np.float32),
                        "q_flow": q_flow.astype(np.float32),
                        "prev_pts": feats.astype(np.float32),
                        "fail": 0,
                    }
                )
            out = frame.copy()
            for st in board_state:
                out = warp_ad_to_quad_roi(out, ad_bgr, st["q_orig"], args.feather)
            bad_count = 0

        writer.write(out)
        gray_prev = gray
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"frame {frame_idx}/{total} bad={bad_count}", flush=True)

    cap.release()
    writer.release()
    print(f"Saved: {video_out}")


if __name__ == "__main__":
    main()

