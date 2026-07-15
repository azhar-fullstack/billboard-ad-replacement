"""
Billboard replacement — single YOLO-seg model.

  1) Perspective warp; blend with seg mask alpha (feathered).
  2) Optional *adaptive mosaic*: number of tiles along the billboard depends on how large the
     mask is in the frame (zoom) and how elongated the region is (distant “line” of boards).
     Large mask / zoom-in → few tiles (often 1). Small + thin strip / zoom-out → more tiles.
     Slot count is EMA-smoothed per track ID for temporal stability.
  3) ByteTrack + quad smoothing: Kalman filter on corners (default) or EMA; optional color
     histogram re-ID to restore slot/gap/mask state when a new track id matches a lost billboard.

Usage:
  python replace_video_seg_stable.py --video-in data/video3.MP4 --ad data/ad_a.jpg data/ad_b.jpg --out out/out.mp4

  Multiple --ad files cycle over panels; each cell warps a full image (unless --slice-single-ad).
  Use --mosaic-layout grid for a balanced grid instead of one horizontal strip; --mosaic-panel-compose
  overwrite keeps panels independent (no seam averaging). --texture-temporal blends the warped ad
  toward the previous frame where the mask overlaps to reduce jitter.

  Occlusion: pass --occluder-weights yolo11n-seg.pt (or similar) so people/objects can cover the ad
  where they overlap the billboard (second forward pass; optional).

  Panel gaps: --panel-gap leaves black UV gutters between tiles (same composite layer as ads, so
  occluders affect gutters too). --ema-gap smooths gap size per track. --alpha-edge-power / sharpen
  options tune boundary vs interior look.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def mask_to_quad(mask_u8: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    bin_mask = (mask_u8 > 127).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return order_points(
            np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        )
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return order_points(approx.reshape(4, 2).astype(np.float32))
    rect = cv2.minAreaRect(c)
    return order_points(cv2.boxPoints(rect).astype(np.float32))


def ema_quad(prev: np.ndarray | None, cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None or alpha <= 0:
        return cur.copy()
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def ema_float(prev: float | None, cur: float, alpha: float) -> float:
    if prev is None or alpha <= 0:
        return cur
    return float(alpha * prev + (1.0 - alpha) * cur)


class QuadKalman:
    """Constant-position Kalman on 8D quad (4 corners × xy); smooths jittery detections."""

    def __init__(self, process_noise: float, meas_noise: float) -> None:
        self.kf = cv2.KalmanFilter(8, 8)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(8, dtype=np.float32) * float(meas_noise)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 100.0
        self._ok = False

    def reset_with_quad(self, quad: np.ndarray) -> None:
        z = quad.reshape(8, 1).astype(np.float32)
        self.kf.statePre = z.copy()
        self.kf.statePost = z.copy()
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        self._ok = True

    def update(self, quad_meas: np.ndarray) -> np.ndarray:
        z = quad_meas.reshape(8, 1).astype(np.float32)
        if not self._ok:
            self.reset_with_quad(quad_meas)
            return quad_meas.copy()
        self.kf.predict()
        self.kf.correct(z)
        return self.kf.statePost.reshape(4, 2).astype(np.float32)


def billboard_color_fingerprint(frame: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Normalized H-S histogram on masked billboard pixels (for re-ID when track id resets)."""
    m = mask_u8 > 127
    if not np.any(m):
        return np.zeros(256, dtype=np.float32)
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = frame[y0:y1, x0:x1]
    mc = m[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros(256, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mc.astype(np.uint8), [16, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    n = float(np.linalg.norm(hist) + 1e-8)
    return hist / n


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or not np.any(a) or not np.any(b):
        return 0.0
    return float(np.dot(a, b))


@dataclass
class ReIdEntry:
    hist: np.ndarray
    slot_ema: float | None
    gap_ema: float | None
    quad: np.ndarray
    mask_f: np.ndarray | None


@dataclass
class FlowState:
    pts: np.ndarray


def try_reid_match(
    fp: np.ndarray,
    gallery: list[ReIdEntry],
    threshold: float,
) -> tuple[ReIdEntry | None, int | None]:
    best_s = -1.0
    best_i: int | None = None
    for i, e in enumerate(gallery):
        s = hist_similarity(fp, e.hist)
        if s > best_s:
            best_s = s
            best_i = i
    if best_i is not None and best_s >= threshold:
        return gallery[best_i], best_i
    return None, None


def apply_reid_restore(
    tid: int,
    entry: ReIdEntry,
    prev_slot_ema: dict[int, float],
    prev_gap_ema: dict[int, float],
    prev_mask_f: dict[int, np.ndarray],
    prev_quads: dict[int, np.ndarray],
    kalman_by_tid: dict[int, QuadKalman],
    use_kalman: bool,
    kf_pn: float,
    kf_mn: float,
) -> None:
    if entry.slot_ema is not None:
        prev_slot_ema[tid] = entry.slot_ema
    if entry.gap_ema is not None:
        prev_gap_ema[tid] = entry.gap_ema
    if entry.mask_f is not None:
        prev_mask_f[tid] = entry.mask_f.copy()
    if use_kalman:
        kalman_by_tid[tid] = QuadKalman(kf_pn, kf_mn)
        kalman_by_tid[tid].reset_with_quad(entry.quad)
    prev_quads[tid] = entry.quad.copy()


def smooth_quad_measured(
    tid: int,
    quad_meas: np.ndarray,
    use_kalman: bool,
    ema_alpha: float,
    prev_quads: dict[int, np.ndarray],
    kalman_by_tid: dict[int, QuadKalman],
    kf_pn: float,
    kf_mn: float,
) -> np.ndarray:
    if use_kalman:
        if tid not in kalman_by_tid:
            kalman_by_tid[tid] = QuadKalman(kf_pn, kf_mn)
        q = kalman_by_tid[tid].update(quad_meas)
    else:
        q = ema_quad(prev_quads.get(tid), quad_meas, ema_alpha)
    prev_quads[tid] = q.copy()
    return q


def quad_after_reid_smooth(
    tid: int,
    quad_meas: np.ndarray,
    mask_u8: np.ndarray,
    frame: np.ndarray,
    last_fp: dict[int, np.ndarray],
    args,
    prev_quads: dict[int, np.ndarray],
    prev_slot_ema: dict[int, float],
    prev_gap_ema: dict[int, float],
    prev_mask_f: dict[int, np.ndarray],
    kalman_by_tid: dict[int, QuadKalman],
    reid_gallery: list[ReIdEntry],
) -> np.ndarray:
    fp = billboard_color_fingerprint(frame, mask_u8)
    last_fp[tid] = fp
    use_kalman = args.quad_smooth == "kalman"
    if not args.no_reid and tid not in prev_quads:
        ent, ix = try_reid_match(fp, reid_gallery, args.reid_threshold)
        if ent is not None and ix is not None:
            reid_gallery.pop(ix)
            apply_reid_restore(
                tid,
                ent,
                prev_slot_ema,
                prev_gap_ema,
                prev_mask_f,
                prev_quads,
                kalman_by_tid,
                use_kalman,
                args.kalman_process_noise,
                args.kalman_meas_noise,
            )
    return smooth_quad_measured(
        tid,
        quad_meas,
        use_kalman,
        args.ema_quad,
        prev_quads,
        kalman_by_tid,
        args.kalman_process_noise,
        args.kalman_meas_noise,
    )


def rasterize_quad(shape_hw: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 255)
    return m


def quad_bilinear(quad: np.ndarray, u: float, v: float) -> np.ndarray:
    """quad ordered TL, TR, BR, BL; u,v in [0,1]."""
    tl, tr, br, bl = quad[0], quad[1], quad[2], quad[3]
    return (1 - u) * (1 - v) * tl + u * (1 - v) * tr + u * v * br + (1 - u) * v * bl


def uv_cell_with_gap(
    i: int, j: int, rows: int, cols: int, gap_frac: float
) -> tuple[float, float, float, float]:
    """UV bounds for cell (i,j); gap_frac in [0,0.45] shrinks each cell so gutters stay black on canvas."""
    cu, cv = 1.0 / float(cols), 1.0 / float(rows)
    if gap_frac <= 0 or (rows == 1 and cols == 1):
        return j * cu, (j + 1) * cu, i * cv, (i + 1) * cv
    mu = gap_frac * cu * 0.5
    mv = gap_frac * cv * 0.5
    u0 = j * cu + mu
    u1 = (j + 1) * cu - mu
    v0 = i * cv + mv
    v1 = (i + 1) * cv - mv
    if u1 <= u0 + 1e-4 or v1 <= v0 + 1e-4:
        return j * cu, (j + 1) * cu, i * cv, (i + 1) * cv
    return u0, u1, v0, v1


def sub_quad_from_grid(
    quad: np.ndarray,
    i: int,
    j: int,
    rows: int,
    cols: int,
    gap_frac: float = 0.0,
) -> np.ndarray:
    u0, u1, v0, v1 = uv_cell_with_gap(i, j, rows, cols, gap_frac)
    pts = np.array(
        [
            quad_bilinear(quad, u0, v0),
            quad_bilinear(quad, u1, v0),
            quad_bilinear(quad, u1, v1),
            quad_bilinear(quad, u0, v1),
        ],
        dtype=np.float32,
    )
    return order_points(pts)


def inset_quad_uv(
    quad: np.ndarray,
    inset_left: float,
    inset_right: float,
    inset_top: float,
    inset_bottom: float,
) -> np.ndarray:
    u0 = float(np.clip(inset_left, 0.0, 0.45))
    u1 = float(np.clip(1.0 - inset_right, u0 + 1e-4, 1.0))
    v0 = float(np.clip(inset_top, 0.0, 0.45))
    v1 = float(np.clip(1.0 - inset_bottom, v0 + 1e-4, 1.0))
    pts = np.array(
        [
            quad_bilinear(quad, u0, v0),
            quad_bilinear(quad, u1, v0),
            quad_bilinear(quad, u1, v1),
            quad_bilinear(quad, u0, v1),
        ],
        dtype=np.float32,
    )
    return order_points(pts)


def quad_panel_metrics(quad: np.ndarray) -> tuple[float, float, float]:
    top = float(np.linalg.norm(quad[1] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    width = 0.5 * (top + bottom)
    height = 0.5 * (left + right)
    area = float(cv2.contourArea(quad.astype(np.float32)))
    return width, height, area


def sharpen_bgr(img_bgr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img_bgr
    base = img_bgr.astype(np.float32)
    low = cv2.GaussianBlur(img_bgr, (0, 0), 1.2).astype(np.float32)
    sharp = np.clip(base + float(amount) * (base - low), 0, 255)
    return sharp.astype(np.uint8)


def prepare_panel_source(
    ad_bgr: np.ndarray,
    target_w: float,
    target_h: float,
    oversample: float,
    sharpen_amount: float,
) -> np.ndarray:
    tw = max(2, int(round(float(target_w) * max(1.0, oversample))))
    th = max(2, int(round(float(target_h) * max(1.0, oversample))))
    aw, ah = ad_bgr.shape[1], ad_bgr.shape[0]
    out_w = max(aw, tw)
    out_h = max(ah, th)
    if out_w != aw or out_h != ah:
        panel = cv2.resize(ad_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        panel = ad_bgr
    return sharpen_bgr(panel, sharpen_amount)


def build_single_warp(
    ad_bgr: np.ndarray,
    quad: np.ndarray,
    fh: int,
    fw: int,
    warp_interp: int,
    panel_oversample: float,
    panel_pre_sharpen: float,
) -> tuple[np.ndarray, np.ndarray]:
    pw, ph, _ = quad_panel_metrics(quad)
    ad_src = prepare_panel_source(ad_bgr, pw, ph, panel_oversample, panel_pre_sharpen)
    warped = warp_ad_to_quad(ad_src, quad, fh, fw, warp_interp).astype(np.uint8)
    cover = rasterize_quad((fh, fw), quad).astype(np.float32) / 255.0
    return warped, cover


def mask_geometry(mask_bin: np.ndarray, frame_hw: tuple[int, int]) -> tuple[float, float, bool]:
    """
    Returns (area_fraction, elongation, horizontal_dominant).
    Uses axis-aligned bbox of the largest contour (stable for strips vs minAreaRect axis swap).
    """
    h, w = frame_hw
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 1.0, True
    c = max(contours, key=cv2.contourArea)
    area_frac = float(cv2.contourArea(c)) / float(max(1, h * w))
    _x, _y, bw, bh = cv2.boundingRect(c)
    bw, bh = float(max(1, bw)), float(max(1, bh))
    elong = max(bw, bh) / min(bw, bh)
    horizontal = bw >= bh
    return area_frac, elong, horizontal


def adaptive_slot_float(
    area_frac: float,
    elong: float,
    max_slots: int,
    area_close: float,
    area_far: float,
    elong_gain: float,
) -> float:
    """
    Continuous target slot count (before EMA rounding).
    Large area_frac (zoom-in) -> ~1; small area + high elongation (distant strip) -> more.
    """
    af = float(np.clip(area_frac, 1e-6, 1.0))
    # 0 = close / big mask, 1 = far / small mask
    zoom_out = float(np.clip((area_close - af) / max(area_close - area_far, 1e-6), 0.0, 1.0))
    elong = float(max(1.0, elong))
    el_term = 1.0 + elong_gain * np.log1p(elong - 1.0)
    n = 1.0 + zoom_out * (float(max_slots) - 1.0) * el_term
    return float(np.clip(n, 1.0, float(max_slots)))


def layout_rows_cols(
    n_slots: int, horizontal: bool, layout_mode: str = "strip"
) -> tuple[int, int]:
    """
    strip: legacy — one row of N panels (horizontal strip) or N×1 (vertical).
    grid: balanced rows×cols so each panel is a separate full-ad warp (less “one long strip”).
    """
    n_slots = max(1, int(n_slots))
    if layout_mode == "grid":
        if n_slots <= 1:
            return 1, 1
        cols = int(np.ceil(np.sqrt(n_slots)))
        rows = int(np.ceil(n_slots / float(cols)))
        return rows, cols
    if horizontal:
        return 1, n_slots
    return n_slots, 1


def choose_rule_based_slots(
    area_frac: float,
    elong: float,
    horizontal: bool,
    max_slots: int,
    zoom_single_area: float,
    thin_strip_area_max: float,
    thin_strip_elong: float,
    thin_strip_slot_step: float,
) -> int:
    """
    Hard rules:
    - Large close-up billboard -> 1 ad.
    - Long thin zoomed-out strip -> multiple ads.
    - Otherwise fall back to moderate adaptive behavior elsewhere.
    """
    if area_frac >= float(zoom_single_area):
        return 1
    if horizontal and area_frac <= float(thin_strip_area_max) and elong >= float(thin_strip_elong):
        step = max(1.0, float(thin_strip_slot_step))
        n = int(round(elong / step))
        return max(2, min(int(max_slots), n))
    return 0


def fit_slots_to_quad(
    quad: np.ndarray,
    desired_slots: int,
    horizontal: bool,
    layout_mode: str,
    min_panel_short_side: float,
    min_panel_area: float,
    panel_gap: float,
) -> tuple[int, int, int]:
    """
    Reduce slot count until every chosen panel is drawable, so we don't leave empty black cells.
    Returns (n_slots, rows, cols).
    """
    desired_slots = max(1, int(desired_slots))
    for n_slots in range(desired_slots, 0, -1):
        rows, cols = layout_rows_cols(n_slots, horizontal, layout_mode)
        ok = True
        for idx in range(n_slots):
            i, j = idx // cols, idx % cols
            sq = sub_quad_from_grid(quad, i, j, rows, cols, panel_gap)
            pw, ph, pa = quad_panel_metrics(sq)
            if min(pw, ph) < float(min_panel_short_side) or pa < float(min_panel_area):
                ok = False
                break
        if ok:
            return n_slots, rows, cols
    return 1, 1, 1


def strip_band_quad(
    quad: np.ndarray,
    band_ratio: float,
    anchor: str,
) -> np.ndarray:
    band_ratio = float(np.clip(band_ratio, 0.05, 1.0))
    if band_ratio >= 0.999:
        return quad.copy()
    if anchor == "top":
        return inset_quad_uv(quad, 0.0, 0.0, 0.0, 1.0 - band_ratio)
    if anchor == "center":
        pad = 0.5 * (1.0 - band_ratio)
        return inset_quad_uv(quad, 0.0, 0.0, pad, pad)
    return inset_quad_uv(quad, 0.0, 0.0, 1.0 - band_ratio, 0.0)


def height_based_strip_layout(
    quad: np.ndarray,
    max_slots: int,
    panel_aspect: float,
    gap_ratio: float,
    min_gap_px: float,
    min_panel_short_side: float,
    min_panel_area: float,
) -> tuple[int, int, int, float]:
    band_w, band_h, _ = quad_panel_metrics(quad)
    panel_h = max(1.0, float(band_h))
    panel_w = max(1.0, panel_h * max(0.5, float(panel_aspect)))
    gap_px = max(float(min_gap_px), round(panel_h * max(0.0, float(gap_ratio))))
    slots = int(max(1, np.floor((band_w + gap_px) / max(panel_w + gap_px, 1e-6))))
    slots = max(1, min(int(max_slots), slots))
    slots, rows, cols = fit_slots_to_quad(
        quad,
        slots,
        True,
        "strip",
        min_panel_short_side,
        min_panel_area,
        0.0,
    )
    cell_w = max(1.0, band_w / float(max(1, cols)))
    gap_frac = float(np.clip(gap_px / cell_w, 0.0, 0.45))
    return slots, rows, cols, gap_frac


def build_mosaic_warp(
    ads: list[np.ndarray],
    quad: np.ndarray,
    rows: int,
    cols: int,
    fh: int,
    fw: int,
    slice_single_sheet: bool,
    panel_gap: float = 0.0,
    panel_compose: str = "average",
    max_cells: int | None = None,
    min_panel_short_side: float = 0.0,
    min_panel_area: float = 0.0,
    warp_interp: int = cv2.INTER_LINEAR,
    panel_oversample: float = 1.0,
    panel_pre_sharpen: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill a rows×cols grid on the quad. Each cell gets a *full* ad image warped to that sub-quad,
    cycling through ads[0], ads[1], … (real multi-panel billboards).

    If slice_single_sheet is True and len(ads)==1, use the old behaviour: crop one big image
    into a grid of slices (sprite-sheet style).

    panel_compose:
      average — blend overlaps at sub-quad edges (legacy).
      overwrite — later cells win; no cross-panel averaging (each panel looks independent).

    max_cells — cap how many cells are drawn (adaptive mosaic may use a grid with spare slots).
    """
    rows, cols = max(1, rows), max(1, cols)
    total = rows * cols
    n_fill = total if max_cells is None else min(total, max(1, int(max_cells)))
    use_avg = panel_compose == "average"

    canvas = np.zeros((fh, fw, 3), dtype=np.float32)
    wsum = np.zeros((fh, fw), dtype=np.float32) if use_avg else None
    cover = np.zeros((fh, fw), dtype=np.float32)

    def accumulate(warped: np.ndarray, m: np.ndarray) -> None:
        nonlocal canvas, wsum, cover
        if use_avg:
            assert wsum is not None
            canvas = canvas + warped * m[..., None]
            wsum = wsum + m
        else:
            mk = m[..., None]
            canvas = np.where(mk > 1e-6, warped, canvas)
        cover = np.maximum(cover, m)

    if slice_single_sheet and len(ads) == 1:
        ad_bgr = ads[0]
        ah, aw = ad_bgr.shape[:2]
        cell_h = max(1, ah // rows)
        cell_w = max(1, aw // cols)
        for idx in range(n_fill):
            i, j = idx // cols, idx % cols
            y0, x0 = i * cell_h, j * cell_w
            crop = ad_bgr[y0 : min(ah, y0 + cell_h), x0 : min(aw, x0 + cell_w)].copy()
            if crop.size == 0:
                continue
            sq = sub_quad_from_grid(quad, i, j, rows, cols, panel_gap)
            pw, ph, pa = quad_panel_metrics(sq)
            if min(pw, ph) < float(min_panel_short_side) or pa < float(min_panel_area):
                continue
            crop_src = prepare_panel_source(crop, pw, ph, panel_oversample, panel_pre_sharpen)
            warped = warp_ad_to_quad(crop_src, sq, fh, fw, warp_interp).astype(np.float32)
            m = rasterize_quad((fh, fw), sq).astype(np.float32) / 255.0
            accumulate(warped, m)
    else:
        if not ads:
            raise ValueError("ads list is empty")
        for idx in range(n_fill):
            i, j = idx // cols, idx % cols
            ad_bgr = ads[idx % len(ads)]
            if ad_bgr.size == 0:
                continue
            sq = sub_quad_from_grid(quad, i, j, rows, cols, panel_gap)
            pw, ph, pa = quad_panel_metrics(sq)
            if min(pw, ph) < float(min_panel_short_side) or pa < float(min_panel_area):
                continue
            ad_src = prepare_panel_source(ad_bgr, pw, ph, panel_oversample, panel_pre_sharpen)
            warped = warp_ad_to_quad(ad_src, sq, fh, fw, warp_interp).astype(np.float32)
            m = rasterize_quad((fh, fw), sq).astype(np.float32) / 255.0
            accumulate(warped, m)

    if use_avg:
        assert wsum is not None
        wsum = np.maximum(wsum, 1e-6)
        canvas /= wsum[..., None]
    return canvas, np.clip(cover, 0.0, 1.0)


def compute_warped_texture(
    ads: list[np.ndarray],
    mask_u8: np.ndarray,
    quad: np.ndarray,
    fh: int,
    fw: int,
    tid: int,
    fixed_grid: bool,
    grid_rows: int,
    grid_cols: int,
    max_slots: int,
    ema_slots: float,
    area_close: float,
    area_far: float,
    elong_gain: float,
    prev_slot_ema: dict[int, float],
    slice_single_sheet: bool,
    panel_gap: float,
    prev_gap_ema: dict[int, float],
    ema_gap: float,
    mosaic_layout: str,
    panel_compose: str,
    prev_layout_by_tid: dict[int, tuple[int, int, int]],
    lock_layout: bool,
    min_panel_short_side: float,
    min_panel_area: float,
    warp_interp: int,
    quad_inset_left: float,
    quad_inset_right: float,
    quad_inset_top: float,
    quad_inset_bottom: float,
    panel_oversample: float,
    panel_pre_sharpen: float,
    use_rule_based_slots: bool,
    zoom_single_area: float,
    thin_strip_area_max: float,
    thin_strip_elong: float,
    thin_strip_slot_step: float,
    use_height_based_strip: bool,
    strip_band_ratio: float,
    strip_band_anchor: str,
    strip_panel_aspect: float,
    strip_gap_ratio: float,
    strip_min_gap_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    quad_use = inset_quad_uv(
        quad, quad_inset_left, quad_inset_right, quad_inset_top, quad_inset_bottom
    )
    mask_pixels = max(1, int(np.count_nonzero(mask_u8 > 127)))
    gap_use = float(np.clip(panel_gap, 0.0, 0.45))
    if ema_gap > 0 and tid is not None:
        gap_use = ema_float(prev_gap_ema.get(tid), gap_use, ema_gap)
        prev_gap_ema[tid] = gap_use

    if fixed_grid:
        rows, cols = max(1, grid_rows), max(1, grid_cols)
        if rows == 1 and cols == 1:
            pw, ph, _ = quad_panel_metrics(quad_use)
            ad_src = prepare_panel_source(ads[0], pw, ph, panel_oversample, panel_pre_sharpen)
            warped = warp_ad_to_quad(ad_src, quad_use, fh, fw, warp_interp)
            cover = rasterize_quad((fh, fw), quad_use).astype(np.float32) / 255.0
            return warped, cover
        warped, cover = build_mosaic_warp(
            ads,
            quad_use,
            rows,
            cols,
            fh,
            fw,
            slice_single_sheet,
            gap_use,
            panel_compose,
            None,
            min_panel_short_side,
            min_panel_area,
            warp_interp,
            panel_oversample,
            panel_pre_sharpen,
        )
        if int(np.count_nonzero(cover > 0.01)) < max(4, int(0.08 * mask_pixels)):
            return build_single_warp(
                ads[0], quad_use, fh, fw, warp_interp, panel_oversample, panel_pre_sharpen
            )
        return warped.clip(0, 255).astype(np.uint8), cover

    bin_mask = (mask_u8 > 127).astype(np.uint8)
    area_frac, elong, horizontal = mask_geometry(bin_mask, (fh, fw))
    if use_height_based_strip and horizontal and area_frac < float(zoom_single_area):
        quad_strip = strip_band_quad(quad_use, strip_band_ratio, strip_band_anchor)
        n_int, rows, cols, gap_use = height_based_strip_layout(
            quad_strip,
            max_slots,
            strip_panel_aspect,
            strip_gap_ratio,
            strip_min_gap_px,
            min_panel_short_side,
            min_panel_area,
        )
        if tid is not None:
            prev_slot_ema[tid] = float(n_int)
        if tid is not None and lock_layout:
            if tid in prev_layout_by_tid:
                n_int, rows, cols = prev_layout_by_tid[tid]
            else:
                prev_layout_by_tid[tid] = (n_int, rows, cols)
        warped, cover = build_mosaic_warp(
            ads,
            quad_strip,
            rows,
            cols,
            fh,
            fw,
            slice_single_sheet,
            gap_use,
            panel_compose,
            n_int,
            min_panel_short_side,
            min_panel_area,
            warp_interp,
            panel_oversample,
            panel_pre_sharpen,
        )
        if int(np.count_nonzero(cover > 0.01)) < max(4, int(0.08 * mask_pixels)):
            return build_single_warp(
                ads[0], quad_use, fh, fw, warp_interp, panel_oversample, panel_pre_sharpen
            )
        return warped.clip(0, 255).astype(np.uint8), cover

    n_rule = 0
    if use_rule_based_slots:
        n_rule = choose_rule_based_slots(
            area_frac,
            elong,
            horizontal,
            max_slots,
            zoom_single_area,
            thin_strip_area_max,
            thin_strip_elong,
            thin_strip_slot_step,
        )
    if n_rule > 0:
        n_int = int(n_rule)
        prev_slot_ema[tid] = float(n_int)
    else:
        n_raw = adaptive_slot_float(
            area_frac, elong, max_slots, area_close, area_far, elong_gain
        )
        n_sm = ema_float(prev_slot_ema.get(tid), n_raw, ema_slots)
        prev_slot_ema[tid] = n_sm
        n_int = int(round(n_sm))
        n_int = max(1, min(max_slots, n_int))
    n_int, rows, cols = fit_slots_to_quad(
        quad_use,
        n_int,
        horizontal,
        mosaic_layout,
        min_panel_short_side,
        min_panel_area,
        gap_use,
    )
    if tid is not None and lock_layout:
        if tid in prev_layout_by_tid:
            n_int, rows, cols = prev_layout_by_tid[tid]
        else:
            prev_layout_by_tid[tid] = (n_int, rows, cols)
    warped, cover = build_mosaic_warp(
        ads,
        quad_use,
        rows,
        cols,
        fh,
        fw,
        slice_single_sheet,
        gap_use,
        panel_compose,
        n_int,
        min_panel_short_side,
        min_panel_area,
        warp_interp,
        panel_oversample,
        panel_pre_sharpen,
    )
    if int(np.count_nonzero(cover > 0.01)) < max(4, int(0.08 * mask_pixels)):
        return build_single_warp(
            ads[0], quad_use, fh, fw, warp_interp, panel_oversample, panel_pre_sharpen
        )
    return warped.clip(0, 255).astype(np.uint8), cover


def blend_warped_temporal(
    warped: np.ndarray,
    alpha: np.ndarray,
    prev_w: np.ndarray | None,
    prev_a: np.ndarray | None,
    beta: float,
) -> np.ndarray:
    """
    Blend current warped ad with previous frame’s warped texture where both frames cover
    the same pixel (alpha overlap). Reduces per-frame re-warp jitter; new mask regions stay
    mostly current-frame.
    """
    if beta <= 0 or prev_w is None or prev_a is None:
        return warped
    if prev_w.shape != warped.shape or prev_a.shape != alpha.shape:
        return warped
    pw = prev_w.astype(np.float32)
    cw = warped.astype(np.float32)
    pa = np.clip(prev_a.astype(np.float32), 0.0, 1.0)
    ca = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    w = np.clip(pa * ca, 0.0, 1.0)[..., None]
    out = cw * (1.0 - beta * w) + pw * (beta * w)
    return np.clip(out, 0, 255).astype(np.uint8)


def texture_quad_smoothed(
    tid: int,
    quad: np.ndarray,
    alpha_tex: float,
    prev_tex: dict[int, np.ndarray],
) -> np.ndarray:
    """Separate EMA state for homography used only to warp the ad texture."""
    if alpha_tex <= 0:
        return quad
    q = ema_quad(prev_tex.get(tid), quad, alpha_tex)
    prev_tex[tid] = q.copy()
    return q


def flow_pick_points(
    gray: np.ndarray,
    mask_u8: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_distance: float,
) -> np.ndarray | None:
    mm = (mask_u8 > 127).astype(np.uint8) * 255
    if not np.any(mm):
        return None
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(8, int(max_corners)),
        qualityLevel=max(1e-4, float(quality_level)),
        minDistance=max(2.0, float(min_distance)),
        mask=mm,
        blockSize=7,
    )
    if pts is None or len(pts) == 0:
        return None
    return pts.astype(np.float32)


def warp_mask01(mask01: np.ndarray, h_mat: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    out = cv2.warpPerspective(
        mask01.astype(np.float32),
        h_mat.astype(np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.clip(out, 0.0, 1.0)


def flow_homography(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    prev_pts: np.ndarray,
    win_size: int,
    max_level: int,
    ransac_thresh: float,
    min_points: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if prev_pts is None or len(prev_pts) < max(4, min_points):
        return None, None
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts.astype(np.float32),
        None,
        winSize=(max(5, int(win_size)), max(5, int(win_size))),
        maxLevel=max(0, int(max_level)),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if next_pts is None or status is None:
        return None, None
    ok = status.reshape(-1) > 0
    if int(np.count_nonzero(ok)) < max(4, min_points):
        return None, None
    src = prev_pts.reshape(-1, 2)[ok]
    dst = next_pts.reshape(-1, 2)[ok]
    h_mat, inliers = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_thresh))
    if h_mat is None or inliers is None:
        return None, None
    good = inliers.reshape(-1) > 0
    if int(np.count_nonzero(good)) < max(4, min_points):
        return None, None
    return h_mat.astype(np.float32), dst[good].reshape(-1, 1, 2).astype(np.float32)


def warp_bgr_with_h(
    img_bgr: np.ndarray, h_mat: np.ndarray, shape_hw: tuple[int, int]
) -> np.ndarray:
    h, w = shape_hw
    return cv2.warpPerspective(
        img_bgr,
        h_mat.astype(np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def refresh_new_regions_only(
    propagated_warped: np.ndarray,
    propagated_alpha: np.ndarray,
    current_warped: np.ndarray,
    current_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep the propagated previous ad where it is still visible, and only paint fresh
    current-frame ad content where the current visible billboard extends beyond the
    propagated region.
    """
    pa = np.clip(propagated_alpha.astype(np.float32), 0.0, 1.0)
    ca = np.clip(current_alpha.astype(np.float32), 0.0, 1.0)
    new_region = np.clip(ca - pa, 0.0, 1.0)
    keep_region = np.clip(ca - new_region, 0.0, 1.0)
    out = (
        propagated_warped.astype(np.float32) * keep_region[..., None]
        + current_warped.astype(np.float32) * new_region[..., None]
    )
    denom = np.maximum((keep_region + new_region)[..., None], 1e-6)
    return np.clip(out / denom, 0, 255).astype(np.uint8), ca


def mask_tensor_to_u8(m: np.ndarray, fh: int, fw: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.max() <= 1.0:
        m = (m > 0.5).astype(np.uint8) * 255
    else:
        m = (m > 127).astype(np.uint8) * 255
    if m.shape[:2] != (fh, fw):
        m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_LINEAR)
    return m


def feather_alpha_float(m01: np.ndarray, k: int) -> np.ndarray:
    """Temporal-smoothed mask in [0,1]; blur for soft edges."""
    if k <= 1:
        return np.clip(m01, 0.0, 1.0)
    k = k | 1
    g = cv2.GaussianBlur(m01.astype(np.float32), (k, k), 0)
    return np.clip(g, 0.0, 1.0)


def feather_alpha_with_soft_edge(m01: np.ndarray, feather_px: int, edge_power: float) -> np.ndarray:
    """Gaussian feather then gamma: edge_power < 1 softens transition into background (less pasted-on)."""
    a = feather_alpha_float(m01, feather_px)
    if abs(edge_power - 1.0) < 1e-6:
        return a
    p = float(edge_power)
    return np.clip(np.power(np.clip(a, 0.0, 1.0), p), 0.0, 1.0)


def enhance_warped_interior_edge(
    warped_bgr: np.ndarray,
    mask_u8: np.ndarray,
    sharpen: float,
    edge_band_px: int,
    edge_blur_sigma: float,
    thin_center_boost: float,
) -> np.ndarray:
    """Sharper detail in the interior of the board; softer toward the inner edge of the mask."""
    if sharpen <= 0 and edge_blur_sigma <= 0:
        return warped_bgr
    h, w = warped_bgr.shape[:2]
    bin_u8 = (mask_u8 > 127).astype(np.uint8)
    if bin_u8.shape[:2] != (h, w):
        bin_u8 = cv2.resize(bin_u8, (w, h), interpolation=cv2.INTER_NEAREST)
    if not np.any(bin_u8):
        return warped_bgr
    dist = cv2.distanceTransform(bin_u8, cv2.DIST_L2, 5)
    band = max(1, int(edge_band_px))
    max_dist = float(max(1.0, dist.max()))
    t_edge = np.clip(dist.astype(np.float32) / float(band), 0.0, 1.0)
    t_thin = np.power(np.clip(dist.astype(np.float32) / max_dist, 0.0, 1.0), 0.8)
    t = np.maximum(t_edge, float(np.clip(thin_center_boost, 0.0, 1.0)) * t_thin)
    wf = warped_bgr.astype(np.float32)
    sigma_soft = float(edge_blur_sigma) if edge_blur_sigma > 0 else 2.0
    soft = cv2.GaussianBlur(warped_bgr, (0, 0), sigma_soft).astype(np.float32)
    if sharpen > 0:
        sigma_b = max(1.0, sigma_soft * 0.75)
        low = cv2.GaussianBlur(warped_bgr, (0, 0), sigma_b).astype(np.float32)
        sharp = np.clip(wf + float(sharpen) * (wf - low), 0, 255)
    else:
        sharp = wf
    out = t[..., None] * sharp + (1.0 - t)[..., None] * soft
    return np.clip(out, 0, 255).astype(np.uint8)


def warp_ad_to_quad(
    ad_bgr: np.ndarray,
    quad_dst: np.ndarray,
    fh: int,
    fw: int,
    interp: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    ah, aw = ad_bgr.shape[:2]
    src = np.array([[0, 0], [aw - 1, 0], [aw - 1, ah - 1], [0, ah - 1]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(src, quad_dst.astype(np.float32))
    return cv2.warpPerspective(
        ad_bgr,
        h_mat,
        (fw, fh),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def composite_texture_with_seg(
    frame: np.ndarray,
    warped_ad: np.ndarray,
    seg_alpha: np.ndarray,
) -> np.ndarray:
    """
    seg_alpha: float32 [H,W] in [0,1] — from segmentation (feathered), not the quad polygon.
    """
    a = seg_alpha[..., None]
    out = frame.astype(np.float32) * (1.0 - a) + warped_ad.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def dilate_mask01(m01: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return m01
    u = (np.clip(m01, 0, 1) * 255).astype(np.uint8)
    k = 2 * radius + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    u = cv2.dilate(u, ker)
    return (u.astype(np.float32) / 255.0).clip(0.0, 1.0)


def occluder_union_mask(
    result,
    fh: int,
    fw: int,
    class_ids: set[int],
) -> np.ndarray:
    """Union of instance masks for selected COCO (or dataset) class ids; [0,1] float32."""
    out = np.zeros((fh, fw), dtype=np.float32)
    if result.boxes is None or len(result.boxes) == 0:
        return out
    if result.masks is None or len(result.masks.data) == 0:
        return out
    n = len(result.boxes)
    for i in range(n):
        cid = int(result.boxes.cls[i].item())
        if class_ids and cid not in class_ids:
            continue
        m = result.masks.data[i].detach().cpu().numpy()
        m = mask_tensor_to_u8(m, fh, fw).astype(np.float32) / 255.0
        out = np.maximum(out, m)
    return out


def apply_occluder_to_alpha(
    billboard_alpha: np.ndarray,
    occluder01: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Reduce ad visibility where occluder (people, etc.) is present."""
    if strength <= 0:
        return billboard_alpha
    occ = np.clip(occluder01 * strength, 0.0, 1.0)
    return np.clip(billboard_alpha * (1.0 - occ), 0.0, 1.0)


def iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def match_prev_boxes(
    prev: list[tuple[int, tuple[int, int, int, int]]],
    curr_boxes: list[tuple[int, int, int, int]],
    next_id: int,
    iou_thr: float,
) -> tuple[list[tuple[int, tuple[int, int, int, int]]], int]:
    if not curr_boxes:
        return [], next_id
    used: set[int] = set()
    out: list[tuple[int, tuple[int, int, int, int]]] = []
    for cb in curr_boxes:
        best_iou, best_pid = 0.0, None
        for pid, pb in prev:
            if pid in used:
                continue
            v = iou_xyxy(pb, cb)
            if v > best_iou:
                best_iou, best_pid = v, pid
        if best_pid is not None and best_iou >= iou_thr:
            used.add(best_pid)
            out.append((best_pid, cb))
        else:
            out.append((next_id, cb))
            next_id += 1
    return out, next_id


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Billboard replace: seg mask alpha + perspective + temporal smooth.")
    p.add_argument("--weights", type=str, default=str(root / "model" / "best.pt"))
    p.add_argument("--video-in", type=str, default=str(root / "data" / "video.mp4"))
    p.add_argument(
        "--ad",
        nargs="+",
        type=str,
        default=[str(root / "data" / "ad.jpg")],
        help="One or more ad images. With multiple panels, they cycle: panel0→ad[0], panel1→ad[1], …",
    )
    p.add_argument(
        "--slice-single-ad",
        action="store_true",
        help="If only ONE image is passed, slice it into a grid (sprite sheet). Default OFF: one image repeats on every panel.",
    )
    p.add_argument("--out", type=str, default=str(root / "out" / "replaced.mp4"))
    p.add_argument("--conf", type=float, default=0.28, help="Min detection confidence; slightly higher reduces jittery masks.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda id or cpu; default auto.",
    )
    p.add_argument("--detect-width", type=int, default=1280)
    p.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="Run billboard segmentation every N frames; in-between, propagate previous billboard with optical flow.",
    )
    p.add_argument(
        "--refresh-mode",
        choices=["full", "new_only"],
        default="full",
        help="full = rebuild ad on refresh frames; new_only = keep propagated ad and only paint newly visible billboard area.",
    )
    p.add_argument(
        "--quad-smooth",
        choices=["kalman", "ema"],
        default="kalman",
        help="kalman = Kalman filter on quad corners (less jitter); ema = exponential moving average.",
    )
    p.add_argument("--ema-quad", type=float, default=0.78, help="Used when --quad-smooth ema; higher = calmer.")
    p.add_argument(
        "--kalman-process-noise",
        type=float,
        default=1.5e-4,
        help="Kalman process noise (lower = slower state evolution, calmer quads).",
    )
    p.add_argument(
        "--kalman-meas-noise",
        type=float,
        default=16.0,
        help="Kalman measurement noise (higher = trust single-frame quads less, less jitter).",
    )
    p.add_argument(
        "--no-reid",
        action="store_true",
        help="Disable appearance gallery: when a new track id appears, do not restore slot/gap from lost tracks.",
    )
    p.add_argument("--reid-threshold", type=float, default=0.78, help="Min cosine similarity of color hist to re-link.")
    p.add_argument("--reid-gallery-max", type=int, default=24, help="Max stored lost-track fingerprints (FIFO).")
    p.add_argument(
        "--ema-mask",
        type=float,
        default=0.55,
        help="Temporal smooth on seg mask [0–1]; 0 = off. Higher = calmer contour before quad fit.",
    )
    p.add_argument("--feather", type=int, default=9, help="Edge feather (odd-ish kernel for mask blur).")
    p.add_argument(
        "--alpha-edge-power",
        type=float,
        default=0.88,
        help="<1 = softer blend into scene at billboard boundary (less sticker look). 1 = off.",
    )
    p.add_argument(
        "--panel-gap",
        type=float,
        default=0.0,
        help="Black gutters between mosaic panels, as fraction of each cell (0=no gap, ~0.15=max).",
    )
    p.add_argument(
        "--ema-gap",
        type=float,
        default=0.82,
        help="Temporal smooth on panel gap per track (0 = no EMA).",
    )
    p.add_argument(
        "--texture-temporal",
        type=float,
        default=0.5,
        help="0=off. Blend warped ad toward previous frame where alpha overlaps (less per-frame jitter).",
    )
    p.add_argument(
        "--texture-quad-ema",
        type=float,
        default=0.0,
        help="Extra EMA on quad used only for warping (0=off). Higher (e.g. 0.88) = stickier homography.",
    )
    p.add_argument(
        "--mosaic-layout",
        choices=["strip", "grid"],
        default="strip",
        help="strip = 1×N horizontal strip; grid = balanced rows×cols (each cell = full ad, not one long row).",
    )
    p.add_argument(
        "--mosaic-panel-compose",
        choices=["average", "overwrite"],
        default="overwrite",
        help="average = blend overlaps at seams; overwrite = z-order (independent panels, no cross-blend).",
    )
    p.add_argument(
        "--lock-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the chosen panel count/layout fixed for a track so one physical block does not split into multiple ads later.",
    )
    p.add_argument("--quad-inset-left", type=float, default=0.0, help="Crop usable billboard quad from the left in UV space [0..1].")
    p.add_argument("--quad-inset-right", type=float, default=0.0, help="Crop usable billboard quad from the right in UV space [0..1].")
    p.add_argument("--quad-inset-top", type=float, default=0.0, help="Crop usable billboard quad from the top in UV space [0..1].")
    p.add_argument("--quad-inset-bottom", type=float, default=0.0, help="Crop usable billboard quad from the bottom in UV space [0..1].")
    p.add_argument(
        "--min-panel-short-side",
        type=float,
        default=1.0,
        help="Do not place an ad in sub-panels thinner than this many pixels on the short side.",
    )
    p.add_argument(
        "--min-panel-area",
        type=float,
        default=4.0,
        help="Do not place an ad in tiny sub-panels below this area in pixels.",
    )
    p.add_argument(
        "--warp-interp",
        choices=["linear", "cubic", "lanczos"],
        default="lanczos",
        help="Interpolation used for ad warping. lanczos is sharpest.",
    )
    p.add_argument(
        "--ad-upscale",
        type=float,
        default=2.0,
        help="Upscale input ad images before perspective warp for more detail on skewed boards.",
    )
    p.add_argument(
        "--ad-pre-sharpen",
        type=float,
        default=0.9,
        help="Sharpen input ad images before warping to fight blur on oblique panels.",
    )
    p.add_argument(
        "--panel-oversample",
        type=float,
        default=3.5,
        help="Extra per-panel source oversampling based on the target panel size; improves skewed panel detail.",
    )
    p.add_argument(
        "--panel-pre-sharpen",
        type=float,
        default=0.35,
        help="Extra sharpening after panel oversampling and before warp.",
    )
    p.add_argument(
        "--sharpen-interior",
        type=float,
        default=0.95,
        help="Unsharp strength in board interior; higher gives a cleaner, less blurry ad.",
    )
    p.add_argument("--sharpen-edge-band", type=int, default=14, help="Px from mask edge treated as soft band.")
    p.add_argument(
        "--warp-edge-blur",
        type=float,
        default=2.4,
        help="Gaussian sigma for soft band near inner mask edge (ad texture only).",
    )
    p.add_argument(
        "--thin-center-boost",
        type=float,
        default=0.75,
        help="Extra center sharpening weight so very thin boards still keep detail while edges stay soft.",
    )
    p.add_argument("--retina-masks", action="store_true")
    p.add_argument("--close-kernel", type=int, default=0)
    p.add_argument("--flow-max-corners", type=int, default=120)
    p.add_argument("--flow-quality", type=float, default=0.01)
    p.add_argument("--flow-min-distance", type=float, default=8.0)
    p.add_argument("--flow-win-size", type=int, default=21)
    p.add_argument("--flow-max-level", type=int, default=3)
    p.add_argument("--flow-ransac-thresh", type=float, default=3.0)
    p.add_argument("--flow-min-points", type=int, default=10)
    p.add_argument(
        "--flow-mask-thresh",
        type=float,
        default=0.22,
        help="Threshold for remembered mask when propagating billboard pixels across skipped frames.",
    )
    p.add_argument(
        "--fixed-grid",
        action="store_true",
        help="Use --grid-rows / --grid-cols for a static mosaic; default is adaptive slot count.",
    )
    p.add_argument("--grid-rows", type=int, default=1)
    p.add_argument("--grid-cols", type=int, default=1)
    p.add_argument("--max-ad-slots", type=int, default=16, help="Upper bound for adaptive number of ad tiles.")
    p.add_argument(
        "--use-rule-based-slots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use hardcoded slot rules: large close-up board = 1 ad, long thin zoomed-out strip = multiple ads.",
    )
    p.add_argument(
        "--zoom-single-area",
        type=float,
        default=0.30,
        help="If billboard mask covers at least this fraction of the frame, force a single ad.",
    )
    p.add_argument(
        "--thin-strip-area-max",
        type=float,
        default=0.12,
        help="Only treat a board as a long thin zoomed-out strip below this mask area fraction.",
    )
    p.add_argument(
        "--thin-strip-elong",
        type=float,
        default=5.5,
        help="Minimum elongation to treat the board as a long thin strip.",
    )
    p.add_argument(
        "--thin-strip-slot-step",
        type=float,
        default=2.8,
        help="Lower = more ads on a long thin strip. Roughly one ad per this many elongation units.",
    )
    p.add_argument(
        "--use-height-based-strip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For horizontal strips, derive ad count from strip height and width instead of a fixed panel count.",
    )
    p.add_argument(
        "--strip-band-ratio",
        type=float,
        default=1.0,
        help="Use only this vertical fraction of a horizontal strip quad for ad placement.",
    )
    p.add_argument(
        "--strip-band-anchor",
        choices=["bottom", "center", "top"],
        default="bottom",
        help="Where to place the usable ad band inside a horizontal strip quad.",
    )
    p.add_argument(
        "--strip-panel-aspect",
        type=float,
        default=1.0,
        help="Width/height ratio for each repeated ad in a thin strip.",
    )
    p.add_argument(
        "--strip-gap-ratio",
        type=float,
        default=0.25,
        help="Gap between strip ads as a fraction of strip height.",
    )
    p.add_argument(
        "--strip-min-gap-px",
        type=float,
        default=1.0,
        help="Minimum pixel gap between strip ads.",
    )
    p.add_argument(
        "--ema-slots",
        type=float,
        default=0.78,
        help="EMA on adaptive slot count per track (higher = smoother, slower to react).",
    )
    p.add_argument(
        "--area-close",
        type=float,
        default=0.20,
        help="Mask area / frame above this → few tiles (zoomed-in billboard).",
    )
    p.add_argument(
        "--area-far",
        type=float,
        default=0.028,
        help="Mask area / frame at full zoom-out side of ramp (used with --area-close).",
    )
    p.add_argument(
        "--elong-gain",
        type=float,
        default=0.42,
        help="Extra tiles when the mask is elongated (distant horizontal strip).",
    )
    p.add_argument("--replace-all", action="store_true")
    p.add_argument("--no-tracker", action="store_true")
    p.add_argument("--iou-match", type=float, default=0.15)
    p.add_argument(
        "--occluder-weights",
        type=str,
        default=None,
        help="Optional YOLO-seg weights (e.g. yolo11n-seg.pt) to segment people/objects in front of the board. "
        "Downloads on first use. Combines with billboard alpha so occluders hide the ad.",
    )
    p.add_argument(
        "--occluder-classes",
        type=str,
        default="0",
        help="Comma-separated class ids for occluder model (COCO default 0=person). Example: 0,2,7",
    )
    p.add_argument("--occluder-conf", type=float, default=0.35)
    p.add_argument(
        "--occluder-dilate",
        type=int,
        default=4,
        help="Grow occluder mask (px) so edges are not jaggy.",
    )
    p.add_argument(
        "--occluder-strength",
        type=float,
        default=1.0,
        help="1.0 = full cutout where occluder overlaps the billboard.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
    warp_interp = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }[args.warp_interp]

    root = Path(__file__).resolve().parent
    weights = Path(args.weights)
    video_in = Path(args.video_in)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise FileNotFoundError(weights)
    if not video_in.exists():
        raise FileNotFoundError(video_in)

    ad_paths = [Path(p) for p in args.ad]
    for ap in ad_paths:
        if not ap.exists():
            raise FileNotFoundError(ap)

    ads: list[np.ndarray] = []
    for ap in ad_paths:
        im = cv2.imread(str(ap), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(ap)
        if im.ndim == 3 and im.shape[2] == 4:
            im = im[:, :, :3]
        if args.ad_upscale > 1.0:
            ah, aw = im.shape[:2]
            up_w = max(1, int(round(aw * float(args.ad_upscale))))
            up_h = max(1, int(round(ah * float(args.ad_upscale))))
            im = cv2.resize(im, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
        im = sharpen_bgr(im, args.ad_pre_sharpen)
        ads.append(im)

    model = YOLO(str(weights))

    occ_class_ids = {int(x.strip()) for x in args.occluder_classes.split(",") if x.strip() != ""}
    occ_model: YOLO | None = None
    if args.occluder_weights:
        occ_model = YOLO(args.occluder_weights)

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(video_in)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("VideoWriter failed")

    use_tracker = not args.no_tracker
    prev_quads: dict[int, np.ndarray] = {}
    prev_quad_tex: dict[int, np.ndarray] = {}
    prev_mask_f: dict[int, np.ndarray] = {}
    prev_slot_ema: dict[int, float] = {}
    prev_gap_ema: dict[int, float] = {}
    prev_layout_by_tid: dict[int, tuple[int, int, int]] = {}
    prev_warped_tex: dict[int, np.ndarray] = {}
    prev_alpha_tex: dict[int, np.ndarray] = {}
    flow_by_tid: dict[int, FlowState] = {}
    prev_boxes_state: list[tuple[int, tuple[int, int, int, int]]] = []
    kalman_by_tid: dict[int, QuadKalman] = {}
    reid_gallery: list[ReIdEntry] = []
    last_fp: dict[int, np.ndarray] = {}
    last_seen_tids: set[int] = set()
    next_fake_id = 0
    frame_idx = 0
    prev_gray: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        if w > args.detect_width > 0:
            scale = args.detect_width / float(w)
            small = cv2.resize(frame, (args.detect_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        occ01 = np.zeros((h, w), dtype=np.float32)
        if occ_model is not None:
            okw: dict = {
                "conf": args.occluder_conf,
                "imgsz": args.imgsz,
                "device": args.device,
                "verbose": False,
            }
            if args.retina_masks:
                okw["retina_masks"] = True
            orr = occ_model.predict(small, **okw)[0]
            sh, sw = small.shape[:2]
            occ_s = occluder_union_mask(orr, sh, sw, occ_class_ids)
            if occ_s.shape[0] != h or occ_s.shape[1] != w:
                occ01 = cv2.resize(occ_s, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                occ01 = occ_s
            occ01 = dilate_mask01(occ01, args.occluder_dilate)

        instances: list[dict] = []
        propagated_by_tid: dict[int, dict] = {}
        use_flow_memory = (
            args.detect_every > 1
            and prev_gray is not None
            and flow_by_tid
            and (frame_idx % max(1, args.detect_every)) != 0
        )
        if prev_gray is not None and flow_by_tid:
            for tid in sorted(flow_by_tid.keys()):
                state = flow_by_tid.get(tid)
                prev_quad = prev_quads.get(tid)
                prev_mf = prev_mask_f.get(tid)
                if state is None or prev_quad is None or prev_mf is None:
                    continue
                h_mat, tracked_pts = flow_homography(
                    prev_gray,
                    gray,
                    state.pts,
                    args.flow_win_size,
                    args.flow_max_level,
                    args.flow_ransac_thresh,
                    args.flow_min_points,
                )
                if h_mat is None or tracked_pts is None:
                    continue
                quad = cv2.perspectiveTransform(
                    prev_quad.reshape(1, 4, 2).astype(np.float32), h_mat
                ).reshape(4, 2)
                m_f = warp_mask01(prev_mf, h_mat, (h, w))
                mask_u8 = (m_f >= args.flow_mask_thresh).astype(np.uint8) * 255
                if args.close_kernel > 0:
                    kk = args.close_kernel | 1
                    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
                    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ker)
                    m_f = np.clip(mask_u8.astype(np.float32) / 255.0, 0.0, 1.0)
                if int(np.count_nonzero(mask_u8)) < 32 or len(tracked_pts) < args.flow_min_points:
                    continue
                refreshed_pts = flow_pick_points(
                    gray,
                    mask_u8,
                    args.flow_max_corners,
                    args.flow_quality,
                    args.flow_min_distance,
                )
                flow_by_tid[tid] = FlowState(
                    pts=tracked_pts if refreshed_pts is None else refreshed_pts
                )
                prev_quads[tid] = quad.copy()
                prev_mask_f[tid] = m_f.copy()
                prop_alpha = None
                prop_warped = None
                if tid in prev_alpha_tex:
                    prop_alpha = warp_mask01(prev_alpha_tex[tid], h_mat, (h, w))
                if tid in prev_warped_tex:
                    prop_warped = warp_bgr_with_h(prev_warped_tex[tid], h_mat, (h, w))
                propagated_by_tid[tid] = {
                    "quad": quad.copy(),
                    "mask_u8": mask_u8.copy(),
                    "mask_f": m_f.copy(),
                    "alpha": prop_alpha,
                    "warped": prop_warped,
                }
                if use_flow_memory:
                    if prop_alpha is not None and prop_warped is not None:
                        prev_warped_tex[tid] = prop_warped.copy()
                        prev_alpha_tex[tid] = prop_alpha.copy()
                        instances.append({"warped": prop_warped, "alpha": prop_alpha, "tid": tid})
                    else:
                        quad_warp = texture_quad_smoothed(
                            tid, quad, args.texture_quad_ema, prev_quad_tex
                        )
                        alpha = feather_alpha_with_soft_edge(
                            m_f, args.feather, args.alpha_edge_power
                        )
                        warped, cover01 = compute_warped_texture(
                            ads,
                            mask_u8,
                            quad_warp,
                            h,
                            w,
                            tid,
                            args.fixed_grid,
                            args.grid_rows,
                            args.grid_cols,
                            args.max_ad_slots,
                            args.ema_slots,
                            args.area_close,
                            args.area_far,
                            args.elong_gain,
                            prev_slot_ema,
                            args.slice_single_ad,
                            args.panel_gap,
                            prev_gap_ema,
                            args.ema_gap,
                            args.mosaic_layout,
                            args.mosaic_panel_compose,
                            prev_layout_by_tid,
                            args.lock_layout,
                            args.min_panel_short_side,
                            args.min_panel_area,
                            warp_interp,
                            args.quad_inset_left,
                            args.quad_inset_right,
                            args.quad_inset_top,
                            args.quad_inset_bottom,
                            args.panel_oversample,
                            args.panel_pre_sharpen,
                            args.use_rule_based_slots,
                            args.zoom_single_area,
                            args.thin_strip_area_max,
                            args.thin_strip_elong,
                            args.thin_strip_slot_step,
                            args.use_height_based_strip,
                            args.strip_band_ratio,
                            args.strip_band_anchor,
                            args.strip_panel_aspect,
                            args.strip_gap_ratio,
                            args.strip_min_gap_px,
                        )
                        alpha = np.clip(alpha * cover01, 0.0, 1.0)
                        warped = enhance_warped_interior_edge(
                            warped,
                            mask_u8,
                            args.sharpen_interior,
                            args.sharpen_edge_band,
                            args.warp_edge_blur,
                            args.thin_center_boost,
                        )
                        warped = blend_warped_temporal(
                            warped,
                            alpha,
                            prev_warped_tex.get(tid),
                            prev_alpha_tex.get(tid),
                            args.texture_temporal,
                        )
                        prev_warped_tex[tid] = warped.copy()
                        prev_alpha_tex[tid] = alpha.copy()
                        instances.append({"warped": warped, "alpha": alpha, "tid": tid})

        if not instances:
            kw: dict = {
                "conf": args.conf,
                "imgsz": args.imgsz,
                "device": args.device,
                "verbose": False,
            }
            if args.retina_masks:
                kw["retina_masks"] = True

            if use_tracker:
                pred = model.track(small, persist=True, tracker="bytetrack.yaml", **kw)[0]
            else:
                pred = model.predict(small, **kw)[0]

            boxes = pred.boxes
            if boxes is None or len(boxes) == 0:
                if not args.no_reid:
                    for otid in last_seen_tids:
                        if otid in last_fp and otid in prev_quads:
                            reid_gallery.append(
                                ReIdEntry(
                                    hist=last_fp[otid].copy(),
                                    slot_ema=prev_slot_ema.get(otid),
                                    gap_ema=prev_gap_ema.get(otid),
                                    quad=prev_quads[otid].copy(),
                                    mask_f=prev_mask_f.get(otid),
                                )
                            )
                            while len(reid_gallery) > args.reid_gallery_max:
                                reid_gallery.pop(0)
                prev_quads.clear()
                prev_quad_tex.clear()
                prev_mask_f.clear()
                prev_slot_ema.clear()
                prev_gap_ema.clear()
                prev_layout_by_tid.clear()
                prev_warped_tex.clear()
                prev_alpha_tex.clear()
                flow_by_tid.clear()
                kalman_by_tid.clear()
                prev_boxes_state.clear()
                last_fp.clear()
                last_seen_tids.clear()
                prev_gray = gray.copy()
                writer.write(frame)
                frame_idx += 1
                continue

            n = len(boxes)
            cur_boxes: list[tuple[int, int, int, int]] = []
            raw: list[tuple[np.ndarray | None, tuple[int, int, int, int], int | None]] = []

            for i in range(n):
                b = boxes[i]
                sx1, sy1, sx2, sy2 = b.xyxy[0].tolist()
                x1 = max(0, min(int(sx1 / scale), w - 1))
                y1 = max(0, min(int(sy1 / scale), h - 1))
                x2 = max(x1 + 1, min(int(sx2 / scale), w))
                y2 = max(y1 + 1, min(int(sy2 / scale), h))
                tid = int(b.id[0].item()) if b.id is not None else None
                cur_boxes.append((x1, y1, x2, y2))
                m_t = None
                if pred.masks is not None and len(pred.masks.data) > i:
                    m_t = pred.masks.data[i].detach().cpu().numpy()
                raw.append((m_t, (x1, y1, x2, y2), tid))

            if not args.replace_all and cur_boxes:
                k = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in cur_boxes]))
                raw = [raw[k]]
                cur_boxes = [cur_boxes[k]]

            if use_tracker:
                for idx, (m_t, xyxy, tid) in enumerate(raw):
                    if tid is None:
                        tid = 10000 + idx
                    x1, y1, x2, y2 = xyxy
                    if m_t is not None:
                        mask_u8 = mask_tensor_to_u8(m_t, h, w)
                    else:
                        mask_u8 = np.zeros((h, w), np.uint8)
                        mask_u8[y1:y2, x1:x2] = 255

                    if args.close_kernel > 0:
                        kk = args.close_kernel | 1
                        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ker)

                    quad_meas = mask_to_quad(mask_u8, xyxy)
                    quad = quad_after_reid_smooth(
                        tid,
                        quad_meas,
                        mask_u8,
                        frame,
                        last_fp,
                        args,
                        prev_quads,
                        prev_slot_ema,
                        prev_gap_ema,
                        prev_mask_f,
                        kalman_by_tid,
                        reid_gallery,
                    )
                    quad_warp = texture_quad_smoothed(
                        tid, quad, args.texture_quad_ema, prev_quad_tex
                    )

                    m_f = mask_u8.astype(np.float32) / 255.0
                    if args.ema_mask > 0 and tid in prev_mask_f:
                        m_f = args.ema_mask * prev_mask_f[tid] + (1.0 - args.ema_mask) * m_f
                    prev_mask_f[tid] = m_f.copy()

                    flow_mask_u8 = (m_f >= args.flow_mask_thresh).astype(np.uint8) * 255
                    pts = flow_pick_points(
                        gray,
                        flow_mask_u8,
                        args.flow_max_corners,
                        args.flow_quality,
                        args.flow_min_distance,
                    )
                    if pts is not None and len(pts) >= args.flow_min_points:
                        flow_by_tid[tid] = FlowState(pts=pts)
                    else:
                        flow_by_tid.pop(tid, None)

                    alpha = feather_alpha_with_soft_edge(m_f, args.feather, args.alpha_edge_power)
                    current_warped, cover01 = compute_warped_texture(
                        ads,
                        mask_u8,
                        quad_warp,
                        h,
                        w,
                        tid,
                        args.fixed_grid,
                        args.grid_rows,
                        args.grid_cols,
                        args.max_ad_slots,
                        args.ema_slots,
                        args.area_close,
                        args.area_far,
                        args.elong_gain,
                        prev_slot_ema,
                        args.slice_single_ad,
                        args.panel_gap,
                        prev_gap_ema,
                        args.ema_gap,
                        args.mosaic_layout,
                        args.mosaic_panel_compose,
                        prev_layout_by_tid,
                        args.lock_layout,
                        args.min_panel_short_side,
                        args.min_panel_area,
                        warp_interp,
                        args.quad_inset_left,
                        args.quad_inset_right,
                        args.quad_inset_top,
                        args.quad_inset_bottom,
                        args.panel_oversample,
                        args.panel_pre_sharpen,
                        args.use_rule_based_slots,
                        args.zoom_single_area,
                        args.thin_strip_area_max,
                        args.thin_strip_elong,
                        args.thin_strip_slot_step,
                        args.use_height_based_strip,
                        args.strip_band_ratio,
                        args.strip_band_anchor,
                        args.strip_panel_aspect,
                        args.strip_gap_ratio,
                        args.strip_min_gap_px,
                    )
                    alpha = np.clip(alpha * cover01, 0.0, 1.0)
                    current_warped = enhance_warped_interior_edge(
                        current_warped,
                        mask_u8,
                        args.sharpen_interior,
                        args.sharpen_edge_band,
                        args.warp_edge_blur,
                        args.thin_center_boost,
                    )
                    prop = propagated_by_tid.get(tid)
                    if (
                        args.refresh_mode == "new_only"
                        and prop is not None
                        and prop.get("warped") is not None
                        and prop.get("alpha") is not None
                    ):
                        warped, alpha = refresh_new_regions_only(
                            prop["warped"], prop["alpha"], current_warped, alpha
                        )
                    else:
                        warped = blend_warped_temporal(
                            current_warped,
                            alpha,
                            prev_warped_tex.get(tid),
                            prev_alpha_tex.get(tid),
                            args.texture_temporal,
                        )
                    prev_warped_tex[tid] = warped.copy()
                    prev_alpha_tex[tid] = alpha.copy()
                    instances.append({"warped": warped, "alpha": alpha, "tid": tid})
            else:
                matched, next_fake_id = match_prev_boxes(prev_boxes_state, cur_boxes, next_fake_id, args.iou_match)
                prev_boxes_state = matched
                for (tid, _), (m_t, xyxy, _) in zip(matched, raw):
                    x1, y1, x2, y2 = xyxy
                    if m_t is not None:
                        mask_u8 = mask_tensor_to_u8(m_t, h, w)
                    else:
                        mask_u8 = np.zeros((h, w), np.uint8)
                        mask_u8[y1:y2, x1:x2] = 255
                    if args.close_kernel > 0:
                        kk = args.close_kernel | 1
                        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ker)

                    quad_meas = mask_to_quad(mask_u8, xyxy)
                    quad = quad_after_reid_smooth(
                        tid,
                        quad_meas,
                        mask_u8,
                        frame,
                        last_fp,
                        args,
                        prev_quads,
                        prev_slot_ema,
                        prev_gap_ema,
                        prev_mask_f,
                        kalman_by_tid,
                        reid_gallery,
                    )
                    quad_warp = texture_quad_smoothed(
                        tid, quad, args.texture_quad_ema, prev_quad_tex
                    )

                    m_f = mask_u8.astype(np.float32) / 255.0
                    if args.ema_mask > 0 and tid in prev_mask_f:
                        m_f = args.ema_mask * prev_mask_f[tid] + (1.0 - args.ema_mask) * m_f
                    prev_mask_f[tid] = m_f.copy()

                    flow_mask_u8 = (m_f >= args.flow_mask_thresh).astype(np.uint8) * 255
                    pts = flow_pick_points(
                        gray,
                        flow_mask_u8,
                        args.flow_max_corners,
                        args.flow_quality,
                        args.flow_min_distance,
                    )
                    if pts is not None and len(pts) >= args.flow_min_points:
                        flow_by_tid[tid] = FlowState(pts=pts)
                    else:
                        flow_by_tid.pop(tid, None)

                    alpha = feather_alpha_with_soft_edge(m_f, args.feather, args.alpha_edge_power)
                    current_warped, cover01 = compute_warped_texture(
                        ads,
                        mask_u8,
                        quad_warp,
                        h,
                        w,
                        tid,
                        args.fixed_grid,
                        args.grid_rows,
                        args.grid_cols,
                        args.max_ad_slots,
                        args.ema_slots,
                        args.area_close,
                        args.area_far,
                        args.elong_gain,
                        prev_slot_ema,
                        args.slice_single_ad,
                        args.panel_gap,
                        prev_gap_ema,
                        args.ema_gap,
                        args.mosaic_layout,
                        args.mosaic_panel_compose,
                        prev_layout_by_tid,
                        args.lock_layout,
                        args.min_panel_short_side,
                        args.min_panel_area,
                        warp_interp,
                        args.quad_inset_left,
                        args.quad_inset_right,
                        args.quad_inset_top,
                        args.quad_inset_bottom,
                        args.panel_oversample,
                        args.panel_pre_sharpen,
                        args.use_rule_based_slots,
                        args.zoom_single_area,
                        args.thin_strip_area_max,
                        args.thin_strip_elong,
                        args.thin_strip_slot_step,
                        args.use_height_based_strip,
                        args.strip_band_ratio,
                        args.strip_band_anchor,
                        args.strip_panel_aspect,
                        args.strip_gap_ratio,
                        args.strip_min_gap_px,
                    )
                    alpha = np.clip(alpha * cover01, 0.0, 1.0)
                    current_warped = enhance_warped_interior_edge(
                        current_warped,
                        mask_u8,
                        args.sharpen_interior,
                        args.sharpen_edge_band,
                        args.warp_edge_blur,
                        args.thin_center_boost,
                    )
                    prop = propagated_by_tid.get(tid)
                    if (
                        args.refresh_mode == "new_only"
                        and prop is not None
                        and prop.get("warped") is not None
                        and prop.get("alpha") is not None
                    ):
                        warped, alpha = refresh_new_regions_only(
                            prop["warped"], prop["alpha"], current_warped, alpha
                        )
                    else:
                        warped = blend_warped_temporal(
                            current_warped,
                            alpha,
                            prev_warped_tex.get(tid),
                            prev_alpha_tex.get(tid),
                            args.texture_temporal,
                        )
                    prev_warped_tex[tid] = warped.copy()
                    prev_alpha_tex[tid] = alpha.copy()
                    instances.append({"warped": warped, "alpha": alpha, "tid": tid})

        ids_now = {inst["tid"] for inst in instances}
        for k in list(prev_quads.keys()):
            if k not in ids_now:
                if not args.no_reid and k in last_fp and k in prev_quads:
                    reid_gallery.append(
                        ReIdEntry(
                            hist=last_fp[k].copy(),
                            slot_ema=prev_slot_ema.get(k),
                            gap_ema=prev_gap_ema.get(k),
                            quad=prev_quads[k].copy(),
                            mask_f=prev_mask_f.get(k),
                        )
                    )
                    while len(reid_gallery) > args.reid_gallery_max:
                        reid_gallery.pop(0)
                del prev_quads[k]
                prev_quad_tex.pop(k, None)
                prev_mask_f.pop(k, None)
                prev_slot_ema.pop(k, None)
                prev_gap_ema.pop(k, None)
                prev_layout_by_tid.pop(k, None)
                prev_warped_tex.pop(k, None)
                prev_alpha_tex.pop(k, None)
                flow_by_tid.pop(k, None)
                kalman_by_tid.pop(k, None)
                last_fp.pop(k, None)

        last_seen_tids = ids_now.copy()

        out = frame.copy()
        for inst in instances:
            alpha_use = apply_occluder_to_alpha(inst["alpha"], occ01, args.occluder_strength)
            out = composite_texture_with_seg(out, inst["warped"], alpha_use)

        writer.write(out)
        prev_gray = gray.copy()
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"frame {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
