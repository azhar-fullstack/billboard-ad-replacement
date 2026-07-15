"""
Football Ad Board Replacement Pipeline
=======================================
Solves:
  1. Motion blur matching     - ad blurs naturally with camera motion
  2. Camera cut detection     - fade in/out on scene cuts, no snapping
  3. Color/light matching     - ad matches scene lighting
  4. Temporal smoothing       - corners smoothed over frames, no jitter
  5. Occlusion handling       - obstacles stay on top of ad

Requirements:
    pip install ultralytics opencv-python numpy torch torchvision

Usage:
    Set paths at bottom of file and run:
        python ad_replacement.py
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  HELPER: Scene Cut Detector
# ─────────────────────────────────────────────

class SceneCutDetector:
    """
    Detects hard cuts by comparing histogram difference between frames.
    When a cut is detected, signals the pipeline to reset and fade in.
    """
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.prev_hist = None

    def is_cut(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if self.prev_hist is None:
            self.prev_hist = hist
            return False

        diff = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        self.prev_hist = hist
        return diff > self.threshold


# ─────────────────────────────────────────────
#  HELPER: Corner Temporal Smoother
# ─────────────────────────────────────────────

class CornerSmoother:
    """
    Smooths the 4 corner points of detected board over a sliding window.
    Weighted by detection confidence — uncertain frames contribute less.
    Prevents jitter from frame-to-frame YOLO noise.
    """
    def __init__(self, window=6):
        self.window = window
        self.corner_history = deque(maxlen=window)
        self.conf_history   = deque(maxlen=window)

    def reset(self):
        self.corner_history.clear()
        self.conf_history.clear()

    def update(self, corners, confidence=1.0):
        self.corner_history.append(corners)
        self.conf_history.append(confidence)

    def get_smooth_corners(self):
        if not self.corner_history:
            return None
        weights = np.array(self.conf_history, dtype=np.float32)
        weights /= weights.sum()
        stacked = np.array(self.corner_history, dtype=np.float32)  # (N, 4, 2)
        smoothed = np.sum(stacked * weights[:, None, None], axis=0)
        return smoothed.astype(np.float32)


# ─────────────────────────────────────────────
#  HELPER: Motion Blur Estimator
# ─────────────────────────────────────────────

class MotionBlurEstimator:
    """
    Estimates camera motion between consecutive frames using optical flow
    on a sparse grid of points. Returns blur magnitude and angle so we
    can apply the same blur to the replacement ad.
    """
    def __init__(self):
        self.prev_gray = None
        self.prev_pts  = None

    def estimate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur_magnitude = 0.0
        blur_angle     = 0.0

        if self.prev_gray is not None and self.prev_pts is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None,
                winSize=(15, 15), maxLevel=2
            )
            good_new = new_pts[status.flatten() == 1]
            good_old = self.prev_pts[status.flatten() == 1]

            if len(good_new) > 5:
                flow          = (good_new - good_old).reshape(-1, 2)
                median_flow   = np.median(flow, axis=0)   # shape (2,)
                blur_magnitude = float(np.linalg.norm(median_flow))
                blur_angle     = float(np.degrees(np.arctan2(median_flow[1], median_flow[0])))

        # Sample grid points for next frame
        h, w = gray.shape
        grid_y = np.linspace(10, h - 10, 8).astype(np.float32)
        grid_x = np.linspace(10, w - 10, 8).astype(np.float32)
        xx, yy = np.meshgrid(grid_x, grid_y)
        self.prev_pts  = np.stack([xx.flatten(), yy.flatten()], axis=1)[:, None, :]
        self.prev_gray = gray

        return blur_magnitude, blur_angle


# ─────────────────────────────────────────────
#  HELPER: Apply Motion Blur to Ad Image
# ─────────────────────────────────────────────

def apply_motion_blur(image, magnitude, angle):
    """
    Applies directional motion blur to the ad image matching camera motion.
    If magnitude is low, no blur is applied (camera is static).
    """
    size = int(min(magnitude * 1.5, 30))
    if size < 2:
        return image
    if size % 2 == 0:
        size += 1

    kernel = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    angle_rad = np.radians(angle)

    for i in range(size):
        offset = i - cx
        x = int(cx + offset * np.cos(angle_rad))
        y = int(cy + offset * np.sin(angle_rad))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    else:
        return image

    return cv2.filter2D(image, -1, kernel)


# ─────────────────────────────────────────────
#  HELPER: Color / Lighting Match
# ─────────────────────────────────────────────

def match_color_lighting(ad_img, frame, board_corners):
    """
    Samples color temperature and brightness from the board region
    in the original frame and applies subtle adjustment to the ad.
    Makes the ad look like it belongs in that lighting environment.
    """
    h, w = frame.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    pts  = board_corners.astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)

    board_pixels = frame[mask > 0]
    if len(board_pixels) < 10:
        return ad_img

    frame_mean = board_pixels.mean(axis=0).astype(np.float32)
    frame_std  = board_pixels.std(axis=0).astype(np.float32) + 1e-6

    ad_float   = ad_img.astype(np.float32)
    ad_mean    = ad_float.mean(axis=(0, 1))
    ad_std     = ad_float.std(axis=(0, 1)) + 1e-6

    blend    = 0.4   # 0 = no adjustment, 1 = full scene color transfer
    adjusted = (ad_float - ad_mean) / ad_std \
               * (frame_std * blend + ad_std * (1 - blend)) \
               + (frame_mean * blend + ad_mean * (1 - blend))

    return np.clip(adjusted, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  HELPER: Extract Corners from YOLO Mask
# ─────────────────────────────────────────────

def extract_corners_from_mask(mask, frame_shape):
    """
    Given a binary segmentation mask from YOLO, extracts the 4 corner
    points of the board using contour approximation.
    Returns corners ordered: top-left, top-right, bottom-right, bottom-left
    or None if extraction fails.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx  = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        x, y, w, h = cv2.boundingRect(contour)
        corners = np.array([
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h]
        ], dtype=np.float32)

    # Sort to: TL, TR, BR, BL
    center  = corners.mean(axis=0)
    def angle_key(pt):
        return np.arctan2(pt[1] - center[1], pt[0] - center[0])
    corners = sorted(corners.tolist(), key=angle_key)
    corners = np.array(corners, dtype=np.float32)

    # After angle sort: index 0=right, going counter-clockwise
    # Reorder to TL, TR, BR, BL
    corners = np.array([
        corners[3],
        corners[0],
        corners[1],
        corners[2],
    ], dtype=np.float32)

    return corners


# ─────────────────────────────────────────────
#  HELPER: Warp and Blend Ad into Frame
# ─────────────────────────────────────────────

def warp_and_blend_ad(frame, ad_img, dst_corners, obstacle_mask=None, alpha=1.0):
    """
    Warps the ad image into the board region using homography,
    then blends it into the frame respecting obstacle mask and alpha.

    alpha:          0.0 = invisible, 1.0 = fully visible (for fade in/out)
    obstacle_mask:  binary mask where values > 0 = obstacle pixel (player etc.)
                    These pixels are NOT replaced by the ad.
    """
    h_ad, w_ad = ad_img.shape[:2]
    h_fr, w_fr = frame.shape[:2]

    src_corners = np.array([
        [0,     0    ],
        [w_ad,  0    ],
        [w_ad,  h_ad ],
        [0,     h_ad ]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_corners, dst_corners)
    if H is None:
        return frame

    warped_ad = cv2.warpPerspective(ad_img, H, (w_fr, h_fr))

    # Board region mask
    board_mask = np.zeros((h_fr, w_fr), dtype=np.uint8)
    pts = dst_corners.astype(np.int32)
    cv2.fillConvexPoly(board_mask, pts, 255)

    # Cut out obstacle pixels
    if obstacle_mask is not None:
        obs = (obstacle_mask > 0).astype(np.uint8) * 255
        obs = cv2.resize(obs, (w_fr, h_fr), interpolation=cv2.INTER_NEAREST)
        board_mask = cv2.bitwise_and(board_mask, cv2.bitwise_not(obs))

    # Blend with alpha
    blend_mask = (board_mask / 255.0) * alpha
    result  = frame.copy().astype(np.float32)
    warped_f = warped_ad.astype(np.float32)
    bm      = blend_mask[:, :, None]
    result  = result * (1 - bm) + warped_f * bm

    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def process_video(
    video_path:     str,
    ad_path:        str,
    model_path:     str,
    output_path:    str,
    smooth_window:  int   = 6,
    cut_threshold:  float = 0.4,
    fade_frames:    int   = 4,
    conf_threshold: float = 0.4,
    board_class:    int   = 0,
    obstacle_class: int   = 1,
):
    """
    Main pipeline. Processes input video frame by frame and writes output.

    Args:
        video_path:      Path to input football match video
        ad_path:         Path to your replacement ad image (PNG)
        model_path:      Path to your trained YOLO segmentation model (.pt)
        output_path:     Path to write output video
        smooth_window:   Frames to smooth corners over (higher = smoother, more lag)
        cut_threshold:   Scene cut sensitivity 0.3–0.6
        fade_frames:     Frames to fade in/out on scene cuts
        conf_threshold:  Minimum YOLO confidence to accept a detection
        board_class:     YOLO class index for ad board (default 0)
        obstacle_class:  YOLO class index for obstacles/players (default 1)
    """

    # ── Load assets ──────────────────────────────
    model  = YOLO(model_path)
    ad_img = cv2.imread(ad_path, cv2.IMREAD_COLOR)
    cap    = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    if ad_img is None:
        raise RuntimeError(f"Cannot load ad image: {ad_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ── Initialize helpers ───────────────────────
    cut_detector   = SceneCutDetector(threshold=cut_threshold)
    smoother       = CornerSmoother(window=smooth_window)
    blur_estimator = MotionBlurEstimator()

    fade_counter = 0
    frame_idx    = 0
    last_corners = None

    print(f"Processing {total} frames at {fps:.1f} fps ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── 1. Estimate camera motion for blur ───
        blur_mag, blur_angle = blur_estimator.estimate(frame)

        # ── 2. Detect scene cut ──────────────────
        is_cut = cut_detector.is_cut(frame)
        if is_cut:
            smoother.reset()
            last_corners = None
            fade_counter = fade_frames
            print(f"  [frame {frame_idx}] Scene cut detected")

        # ── 3. Run YOLO segmentation ─────────────
        results = model(frame, verbose=False)[0]

        detected_corners = None
        best_conf        = 0.0
        obstacle_mask    = None

        if results.masks is not None:
            for mask_data, cls, conf in zip(
                results.masks.data,
                results.boxes.cls,
                results.boxes.conf
            ):
                conf_val = float(conf)
                cls_val  = int(cls)

                mask_np = mask_data.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_np, (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

                # Ad board detection
                if cls_val == board_class and conf_val >= conf_threshold:
                    if conf_val > best_conf:
                        corners = extract_corners_from_mask(mask_resized, frame.shape)
                        if corners is not None:
                            detected_corners = corners
                            best_conf        = conf_val

                # Obstacle / player detection
                elif cls_val == obstacle_class:
                    obstacle_mask = mask_resized if obstacle_mask is None \
                                    else np.maximum(obstacle_mask, mask_resized)

        # ── 4. Update smoother ───────────────────
        if detected_corners is not None:
            smoother.update(detected_corners, confidence=best_conf)
            last_corners = smoother.get_smooth_corners()
        # else: reuse last_corners from previous frame

        # ── 5. Place ad if we have valid corners ─
        output_frame = frame.copy()

        if last_corners is not None:

            # Fade alpha after scene cut
            if fade_counter > 0:
                alpha = 1.0 - (fade_counter / fade_frames)
                fade_counter -= 1
            else:
                alpha = 1.0

            # Apply motion blur matching camera motion
            ad_blurred = apply_motion_blur(ad_img, blur_mag, blur_angle)

            # Apply subtle color/lighting match to scene
            ad_colored = match_color_lighting(ad_blurred, frame, last_corners)

            # Warp and blend into frame
            output_frame = warp_and_blend_ad(
                frame,
                ad_colored,
                last_corners,
                obstacle_mask=obstacle_mask,
                alpha=alpha
            )

        writer.write(output_frame)

        if frame_idx % 50 == 0:
            pct = (frame_idx / total * 100) if total > 0 else 0
            print(f"  {frame_idx}/{total} frames  ({pct:.1f}%)")

    cap.release()
    writer.release()
    print(f"\nDone! Output saved to: {output_path}")


# ─────────────────────────────────────────────
#  RUN — set your paths here
# ─────────────────────────────────────────────

if __name__ == "__main__":

    import os
    BASE = os.path.dirname(os.path.abspath(__file__))

    process_video(
        video_path      = os.path.join(BASE, "data",  "video3.MP4"),
        ad_path         = os.path.join(BASE, "data",  "ad.jpg"),
        model_path      = os.path.join(BASE, "model", "best.pt"),
        output_path     = os.path.join(BASE, "out",   "newcode_out.mp4"),

        # ── Tuning knobs ──────────────────────
        smooth_window   = 6,    # increase if still jerky, decrease if too laggy
        cut_threshold   = 0.4,  # increase if too many false cuts detected
        fade_frames     = 4,    # frames to fade in after a camera cut
        conf_threshold  = 0.4,  # min YOLO confidence to use a detection

        # ── Class indices from YOUR model ─────
        board_class     = 0,    # change if your ad board is a different class index
        obstacle_class  = 1,    # change if your obstacle/player is a different class index
    )