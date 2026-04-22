"""
live_detect.py

What this program does:
- Captures a cropped region from the center of your monitor
- Runs a trained YOLO model on that live screen region
- Detects three classes:
    - blue   = primary tracked target
    - green  = secondary / collectible target
    - red    = avoid target
- Tracks ALL targets persistently over time
- Uses averaged confidence (EMA) so random highs/lows are smoothed out
- Uses only positive score contributions:
    - avg confidence
    - near-center bonus
    - tracked-age bonus
    - optional marker bonus
    - recent-drop-memory bonus
- Only unmarked blue targets are allowed to become the active lock
"""

import os
import time
import math
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import ctypes
import win32api
import win32con


USER32 = ctypes.windll.user32

last_move_time = time.perf_counter()
last_vx = 0.0
last_vy = 0.0


RENDER_MODE = "full"   # "full", "boxes", "stats", "off"
PREVIEW_EVERY_N_FRAMES = 1   # set to 2 or 3 for less preview cost


# tune these
MAX_SPEED_PX_PER_SEC = 300.0      # absolute mouse speed cap
MAX_ACCEL_PX_PER_SEC2 = 500.0    # how quickly speed can change
DEADZONE_PX = 3.0                 # ignore tiny jitter

# examples:
# 0x10 = Shift
# 0x11 = Ctrl
# 0x12 = Alt
# 0x01 = Left mouse
# 0x02 = Right mouse
# 0x05 = Mouse button 4
# 0x0D = Enter Key

AIM_HOLD_KEY = 0x02   # legacy / same as raider lock key by default
RAIDER_LOCK_KEY = 0x02   # right mouse button
OTHER_LOCK_KEY = 0x05    # mouse button 4 / side button

# Global aim-region override hotkeys (function keys).
AIM_REGION_HOTKEYS = {
    0x74: "center",      # F5
    0x75: "head",        # F6
    0x76: "upper_body",  # F7
    0x77: "body",        # F8
    0x78: "thruster",    # F9
    0x79: None,          # F10 clears override back to per-class defaults
}

DRAW_APPLIED_VECTOR = True
APPLIED_VECTOR_SCALE = 6.0   # makes short moves easier to see
APPLIED_VECTOR_COLOR = (0, 0, 255)   # red in OpenCV BGR
APPLIED_VECTOR_THICKNESS = 2
APPLIED_VECTOR_MIN_DRAW = 1

# ============================================================
# 1) CLASS / BEHAVIOR CONFIG
# ============================================================

TRACK_CLASS = "raider"
COLLECT_CLASS = "pop"
AVOID_CLASS = "shredder"

# Manual box class selection
CLASS_NAMES = {
    0: "raider",
    1: "pop",
    2: "shredder",
    3: "wasp",
    4: "hornet",
    5: "spider",
    6: "comet",
    7: "fireball",
    8: "firefly",
}

# Per-class minimum raw confidence. Keep global YOLO conf low enough
# to let lower-confidence classes reach this Python-side filter.
CLASS_MIN_RAW_CONFIDENCE = {
    "raider": 0.55,
    "pop": 0.25,
    "shredder": 0.25,
    "wasp": 0.35,
    "hornet": 0.35,
    "spider": 0.35,
    "comet": 0.30,
    "fireball": 0.30,
    "firefly": 0.30,
}

# Per-class confidence contribution to score.
CLASS_CONFIDENCE_WEIGHT = {
    "raider": 1.00,
    "pop": 0.60,
    "shredder": 0.75,
    "wasp": 0.80,
    "hornet": 0.80,
    "spider": 0.70,
    "comet": 0.90,
    "fireball": 0.75,
    "firefly": 0.70,
}

# Per-class minimum average confidence for lock-style decisions.
CLASS_LOCK_MIN_AVG_CONFIDENCE = {
    "raider": 0.60,
    "pop": 0.30,
    "shredder": 0.30,
    "wasp": 0.40,
    "hornet": 0.40,
    "spider": 0.40,
    "comet": 0.40,
    "fireball": 0.35,
    "firefly": 0.35,
}

# Named aim regions inside each target box.
AIM_REGION_DEFS = {
    "center":        {"fx": 0.50, "fy": 0.50, "fw": 0.18, "fh": 0.18},
    "head":          {"fx": 0.50, "fy": 0.22, "fw": 0.22, "fh": 0.18},
    "upper_body":    {"fx": 0.50, "fy": 0.38, "fw": 0.28, "fh": 0.20},
    "body":          {"fx": 0.50, "fy": 0.52, "fw": 0.30, "fh": 0.26},
    "lower_body":    {"fx": 0.50, "fy": 0.68, "fw": 0.28, "fh": 0.20},
    "thruster":      {"fx": 0.50, "fy": 0.80, "fw": 0.24, "fh": 0.18},
    "left_thruster": {"fx": 0.35, "fy": 0.80, "fw": 0.18, "fh": 0.18},
    "right_thruster":{"fx": 0.65, "fy": 0.80, "fw": 0.18, "fh": 0.18},
}

# Which aim region each class should use.
CLASS_AIM_REGION = {
    "raider": "head",
    "pop": "center",
    "shredder": "body",
    "wasp": "head",
    "hornet": "head",
    "spider": "body",
    "comet": "thruster",
    "fireball": "center",
    "firefly": "center",
}

DRAW_AIM_REGIONS = True
AIM_REGION_COLOR = (255, 0, 255)  # magenta in BGR
AIM_REGION_THICKNESS = 1

TRACKABLE_CLASSES = set(CLASS_NAMES.values())

# Screen-change / motion gating for inference.
MOTION_SCAN_ENABLED = False
MOTION_DIFF_THRESHOLD = 20
MIN_CHANGED_PIXELS = 40
MIN_CHANGED_BLOB_AREA = 20
MOTION_RADIUS = 10
MIN_CHANGED_IN_RADIUS = 12
FORCE_INFERENCE_EVERY_N_FRAMES = 3


CROSSHAIR_OFFSET_X = 0     # + moves right, - moves left
CROSSHAIR_OFFSET_Y = -7     # + moves down, - moves up


# ============================================================
# 2) MODEL / CAPTURE CONFIG
# ============================================================

MODEL_PATH = r"runs/detect/runs/game_targets/weights/best.pt"

PYTORCH_MODEL_PATH = MODEL_PATH
TENSORRT_MODEL_PATH = MODEL_PATH.replace(".pt", ".engine")
PREFER_TENSORRT = True
EXPORT_TENSORRT_IF_MISSING = False
TENSORRT_HALF = True
TENSORRT_INT8 = False

TWO_STAGE_ENABLED = True
STAGE1_IMG_SIZE = 320
STAGE2_IMG_SIZE = 640
STAGE2_PADDING = 160
STAGE2_MIN_CONF = 0.10
STAGE2_REQUIRE_HOTKEY = True
STAGE2_CLASSES_MATCH_ONLY = True

MONITOR_INDEX = 1
CAPTURE_REGION = None

CONFIDENCE = 0.20
IMG_SIZE = 480
USE_GPU = True

WINDOW_NAME = "LightBurn"

CROP_WIDTH = 1200
CROP_HEIGHT = 500


# ============================================================
# 3) PERSISTENT TARGET TRACKING
# ============================================================

TRACK_MATCH_DISTANCE_PX = 120
TRACK_FORGET_FRAMES = 1
CONFIDENCE_SMOOTHING = 0.10   # lower = smoother, higher = reacts faster

LOCK_ON_MIN_AVG_CONFIDENCE = 0.6

# ============================================================
# 12) DYNAMIC VELOCITY LIMITER
# ============================================================
BASE_MAX_SPEED = 500.0         # px/sec minimum allowed speed
MAX_MAX_SPEED = 4900.0         # px/sec maximum allowed speed

NEAR_TARGET_RADIUS = 10.0      # inside this, slow way down
FAR_TARGET_RADIUS = 500.0      # outside this, allow near max speed

LOW_CONF_SPEED_MULT = 0.50     # low confidence slows movement
HIGH_CONF_SPEED_MULT = 1.00    # high confidence allows full movement

MAX_ACCEL_PER_SEC = 2500.0     # how quickly velocity is allowed to change

DEADZONE_RADIUS = 10.0
STOP_AWAY_MOTION = True


# ============================================================
# 4) POSITIVE-ONLY SCORING
# ============================================================

BASE_CONFIDENCE_WEIGHT = 1.00
CENTER_BONUS_MAX = 0.50
TRACKED_AGE_SCORE_BONUS = 0.02
MAX_TRACK_AGE_BONUS = 0.40
MARKER_SCORE_BONUS = 0.00


# ============================================================
# 5) RECENTLY DROPPED TARGET MEMORY
# ============================================================

DROP_MEMORY_ENABLED = False
DROP_MEMORY_FRAMES = 1
DROP_MEMORY_RADIUS_PX = 180
DROP_MEMORY_SCORE_BONUS = 0.35


# ============================================================
# 6) OPTIONAL MARKER / DOT CHECK ABOVE TARGET
# ============================================================

CHECK_MARKER_ABOVE_TARGET = False
MARKER_BOX_HEIGHT = 80
MARKER_BOX_WIDTH_RATIO = 0.50
MIN_MARKER_PIXELS = 15

MARKER_HSV_RANGES = [
    # red
    ((0, 120, 120), (10, 255, 255)),
    ((170, 120, 120), (179, 255, 255)),
    # green
    ((35, 50, 100), (85, 255, 255)),
    # blue
    ((90, 50, 100), (130, 255, 255)),
    # orange
    ((5, 60, 120), (25, 255, 255)),
]


def move_mouse_relative(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

def is_key_down(vk_code):
    return (USER32.GetAsyncKeyState(vk_code) & 0x8000) != 0

# ============================================================
# 7) BASIC GEOMETRY / UTILITY HELPERS
# ============================================================

def get_center_crop_region(monitor, crop_width=CROP_WIDTH, crop_height=CROP_HEIGHT):
    screen_left = monitor["left"]
    screen_top = monitor["top"]
    screen_width = monitor["width"]
    screen_height = monitor["height"]

    center_x = screen_left + screen_width // 2
    center_y = screen_top + screen_height // 2

    left = center_x - crop_width // 2
    top = center_y - crop_height // 2

    return {
        "left": left,
        "top": top,
        "width": crop_width,
        "height": crop_height,
    }


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def box_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_aim_region_box(x1, y1, x2, y2, region_name="center"):
    region = AIM_REGION_DEFS.get(region_name, AIM_REGION_DEFS["center"])

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    cx = x1 + int(w * region["fx"])
    cy = y1 + int(h * region["fy"])

    half_w = max(1, int(w * region["fw"] / 2.0))
    half_h = max(1, int(h * region["fh"] / 2.0))

    ax1 = clamp(cx - half_w, x1, x2)
    ay1 = clamp(cy - half_h, y1, y2)
    ax2 = clamp(cx + half_w, x1, x2)
    ay2 = clamp(cy + half_h, y1, y2)

    if ax2 <= ax1:
        ax2 = min(x2, ax1 + 1)
    if ay2 <= ay1:
        ay2 = min(y2, ay1 + 1)

    return ax1, ay1, ax2, ay2


def get_aim_point(x1, y1, x2, y2, region_name="center"):
    ax1, ay1, ax2, ay2 = get_aim_region_box(x1, y1, x2, y2, region_name)
    cx = (ax1 + ax2) // 2
    cy = (ay1 + ay2) // 2
    return cx, cy, ax1, ay1, ax2, ay2


def build_detection_entry(frame_bgr, class_name, x1, y1, x2, y2, raw_conf, aim_region_override=None):
    min_conf_for_class = CLASS_MIN_RAW_CONFIDENCE.get(class_name, CONFIDENCE)
    if raw_conf < min_conf_for_class:
        return None

    aim_region_name = aim_region_override or CLASS_AIM_REGION.get(class_name, "center")
    cx, cy, aim_x1, aim_y1, aim_x2, aim_y2 = get_aim_point(x1, y1, x2, y2, aim_region_name)

    has_marker = False
    marker_rect = None
    if CHECK_MARKER_ABOVE_TARGET:
        has_marker, marker_rect = marker_present_above_target(frame_bgr, x1, y1, x2, y2)

    return {
        "class_name": class_name,
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "cx": int(cx),
        "cy": int(cy),
        "raw_conf": float(raw_conf),
        "has_marker": has_marker,
        "marker_rect": marker_rect,
        "aim_region_name": aim_region_name,
        "aim_x1": int(aim_x1),
        "aim_y1": int(aim_y1),
        "aim_x2": int(aim_x2),
        "aim_y2": int(aim_y2),
    }


def remove_away_component(error_x, error_y, vx, vy):
    """
    Remove only the component of velocity that points away from the target.
    Keeps sideways motion, kills backwards drift.
    """
    err_mag_sq = error_x * error_x + error_y * error_y
    if err_mag_sq <= 1e-6:
        return 0.0, 0.0

    # projection of velocity onto target-error direction
    proj = (vx * error_x + vy * error_y) / err_mag_sq

    # if projection is negative, some motion is pointing away
    if proj < 0:
        away_vx = proj * error_x
        away_vy = proj * error_y
        vx -= away_vx
        vy -= away_vy

    return vx, vy

def limit_mouse_delta(raw_dx, raw_dy):
    global last_move_time, last_vx, last_vy

    now = time.perf_counter()
    dt = now - last_move_time
    last_move_time = now

    if dt <= 0:
        dt = 1 / 240.0

    # ignore tiny jitter
    mag = math.hypot(raw_dx, raw_dy)
    if mag < DEADZONE_PX:
        last_vx = 0.0
        last_vy = 0.0
        return 0, 0

    # desired velocity from this frame's requested move
    desired_vx = raw_dx / dt
    desired_vy = raw_dy / dt

    # 1) clamp acceleration
    dvx = desired_vx - last_vx
    dvy = desired_vy - last_vy
    dv_mag = math.hypot(dvx, dvy)
    max_dv = MAX_ACCEL_PX_PER_SEC2 * dt

    if dv_mag > max_dv and dv_mag > 0:
        scale = max_dv / dv_mag
        desired_vx = last_vx + dvx * scale
        desired_vy = last_vy + dvy * scale

    # 2) clamp absolute speed
    v_mag = math.hypot(desired_vx, desired_vy)
    if v_mag > MAX_SPEED_PX_PER_SEC and v_mag > 0:
        scale = MAX_SPEED_PX_PER_SEC / v_mag
        desired_vx *= scale
        desired_vy *= scale

    # save limited velocity
    last_vx = desired_vx
    last_vy = desired_vy

    # convert back to per-frame movement
    limited_dx = desired_vx * dt
    limited_dy = desired_vy * dt

    return int(round(limited_dx)), int(round(limited_dy))


def normalized_center_distance(cx, cy, frame_w, frame_h):
    screen_cx = frame_w / 2.0
    screen_cy = frame_h / 2.0
    dx = cx - screen_cx
    dy = cy - screen_cy
    dist = math.hypot(dx, dy)

    max_dist = math.hypot(frame_w / 2.0, frame_h / 2.0)
    return dist / max_dist if max_dist > 0 else 1.0


def positive_center_bonus(cx, cy, frame_w, frame_h, max_bonus=CENTER_BONUS_MAX):
    dist_norm = normalized_center_distance(cx, cy, frame_w, frame_h)
    return (1.0 - dist_norm) * max_bonus


def ema_update(old_value, new_value, alpha):
    if old_value is None:
        return new_value
    return (alpha * new_value) + ((1.0 - alpha) * old_value)


def make_track_id_generator():
    next_id = 1
    while True:
        yield next_id
        next_id += 1


def point_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def near_drop_memory(cx, cy, drop_memory):
    if drop_memory is None:
        return False
    dist = point_distance(cx, cy, drop_memory["cx"], drop_memory["cy"])
    return dist <= DROP_MEMORY_RADIUS_PX

# ============================================================
# 11) MOUSE POSITION / VELOCITY
# ============================================================

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def get_mouse_position():
    """
    Returns global mouse position in screen coordinates.
    """
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def maybe_export_tensorrt_engine():
    if not PREFER_TENSORRT or not EXPORT_TENSORRT_IF_MISSING:
        return
    if os.path.exists(TENSORRT_MODEL_PATH) or not os.path.exists(PYTORCH_MODEL_PATH):
        return

    export_model = YOLO(PYTORCH_MODEL_PATH)
    export_model.export(
        format="engine",
        imgsz=STAGE2_IMG_SIZE,
        device=0 if USE_GPU else "cpu",
        half=TENSORRT_HALF,
        int8=TENSORRT_INT8,
    )


def resolve_runtime_model_path():
    if PREFER_TENSORRT and os.path.exists(TENSORRT_MODEL_PATH):
        return TENSORRT_MODEL_PATH
    return PYTORCH_MODEL_PATH


def compute_motion_metrics(prev_gray, frame_bgr):
    small = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if prev_gray is None or prev_gray.shape != gray.shape:
        return gray, 0, 0, 0, True

    diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    changed_pixels = int(cv2.countNonZero(motion_mask))
    largest_blob_area = changed_pixels

    radius_kernel_size = max(1, (MOTION_RADIUS * 2) + 1)
    radius_kernel = np.ones((radius_kernel_size, radius_kernel_size), dtype=np.uint8)
    local_changed = cv2.filter2D((motion_mask > 0).astype(np.uint8), cv2.CV_32S, radius_kernel)
    max_changed_in_radius = int(local_changed.max()) if local_changed.size > 0 else 0

    should_infer = (
        changed_pixels >= MIN_CHANGED_PIXELS
        or largest_blob_area >= MIN_CHANGED_BLOB_AREA
        or max_changed_in_radius >= MIN_CHANGED_IN_RADIUS
    )

    return gray, changed_pixels, largest_blob_area, max_changed_in_radius, should_infer


def lerp(a, b, t):
    return a + (b - a) * t


def clamp01(x):
    return max(0.0, min(1.0, x))


def dynamic_speed_limit(distance_to_target, avg_conf):
    """
    Returns a max allowed speed in px/sec.
    Positive-only logic:
    - far target => faster allowed speed
    - near target => slower allowed speed
    - high confidence => faster allowed speed
    """

    # distance contribution
    dist_t = clamp01((distance_to_target - NEAR_TARGET_RADIUS) / max(FAR_TARGET_RADIUS - NEAR_TARGET_RADIUS, 1e-6))
    dist_speed = lerp(BASE_MAX_SPEED, MAX_MAX_SPEED, dist_t)

    # confidence contribution
    conf_t = clamp01(avg_conf)
    conf_mult = lerp(LOW_CONF_SPEED_MULT, HIGH_CONF_SPEED_MULT, conf_t)

    return dist_speed * conf_mult


def limit_velocity(desired_vx, desired_vy, max_speed):
    """
    Clamp a velocity vector to a max magnitude.
    """
    speed = math.hypot(desired_vx, desired_vy)
    if speed <= max_speed or speed <= 1e-6:
        return desired_vx, desired_vy

    scale = max_speed / speed
    return desired_vx * scale, desired_vy * scale


def accel_limit(prev_vx, prev_vy, target_vx, target_vy, dt):
    """
    Limit how fast velocity can change.
    """
    max_delta = MAX_ACCEL_PER_SEC * dt

    dvx = target_vx - prev_vx
    dvy = target_vy - prev_vy
    delta_mag = math.hypot(dvx, dvy)

    if delta_mag <= max_delta or delta_mag <= 1e-6:
        return target_vx, target_vy

    scale = max_delta / delta_mag
    return prev_vx + dvx * scale, prev_vy + dvy * scale


# ============================================================
# 8) MARKER CHECK ABOVE TARGET
# ============================================================

def marker_present_above_target(frame_bgr, x1, y1, x2, y2):
    target_w = x2 - x1
    if target_w <= 0:
        return False, None

    marker_w = int(target_w * MARKER_BOX_WIDTH_RATIO)

    target_h = y2 - y1
    marker_h = int(target_h * 0.75)
    marker_h = clamp(marker_h, 30, 120)

    cx = (x1 + x2) // 2

    rx1 = clamp(cx - marker_w // 2, 0, frame_bgr.shape[1] - 1)
    rx2 = clamp(cx + marker_w // 2, 0, frame_bgr.shape[1] - 1)
    ry2 = clamp(y1 - 5, 0, frame_bgr.shape[0] - 1)
    ry1 = clamp(ry2 - marker_h, 0, frame_bgr.shape[0] - 1)

    if rx2 <= rx1 or ry2 <= ry1:
        return False, None

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    total_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    for lower, upper in MARKER_HSV_RANGES:
        mask = cv2.inRange(
            hsv,
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8),
        )
        total_mask = cv2.bitwise_or(total_mask, mask)

    kernel = np.ones((3, 3), np.uint8)
    total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, kernel)
    total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(total_mask, connectivity=8)

    largest_blob_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_blob_area:
            largest_blob_area = area

    found = largest_blob_area >= MIN_MARKER_PIXELS
    return found, (rx1, ry1, rx2, ry2)


# ============================================================
# 9) VIEW / OVERLAY HELPERS
# ============================================================

def draw_crosshair(frame):
    h, w = frame.shape[:2]
    cx = w // 2
    cy = h // 2

    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 1)
    return cx, cy


# ============================================================
# 10) MAIN LOOP
# ============================================================

def main():
    maybe_export_tensorrt_engine()
    runtime_model_path = resolve_runtime_model_path()
    model = YOLO(runtime_model_path)
    print("Loaded model:", runtime_model_path)
    print("Model classes:", model.names)

    name_to_id = {name: idx for idx, name in model.names.items()} if isinstance(model.names, dict) else {name: idx for idx, name in enumerate(model.names)}
    trackable_class_ids = [name_to_id[name] for name in TRACKABLE_CLASSES if name in name_to_id]
    raider_class_ids = [name_to_id[TRACK_CLASS]] if TRACK_CLASS in name_to_id else []
    other_class_ids = [cid for name, cid in name_to_id.items() if name in TRACKABLE_CLASSES and name != TRACK_CLASS]

    with mss() as sct:
        monitor = sct.monitors[MONITOR_INDEX]
        region = CAPTURE_REGION if CAPTURE_REGION else get_center_crop_region(
            monitor,
            crop_width=CROP_WIDTH,
            crop_height=CROP_HEIGHT,
        )

        print("Capture region:", region)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1200, 800)

        last_fps_time = time.time()
        frames = 0
        fps = 0.0

        track_id_gen = make_track_id_generator()
        persistent_tracks = {}   # track_id -> dict
        active_lock_id = None
        drop_memory = None

        applied_dx = 0
        applied_dy = 0
        applied_force = 0.0

        mouse_x = 0
        mouse_y = 0
        mouse_vx = 0.0
        mouse_vy = 0.0
        mouse_speed = 0.0
        last_mouse_x = None
        last_mouse_y = None
        last_mouse_time = time.time()

        limited_vx = 0.0
        limited_vy = 0.0
        last_control_time = time.time()

        mouse_frac_x = 0.0
        mouse_frac_y = 0.0

        render_mode = RENDER_MODE
        preview_every_n_frames = PREVIEW_EVERY_N_FRAMES

        active_lock_mode = None
        current_aim_region_override = None

        prev_raider_key_down = False
        prev_other_key_down = False
        prev_aim_hotkey_down = {vk: False for vk in AIM_REGION_HOTKEYS}

        prev_motion_gray = None
        frames_since_infer = FORCE_INFERENCE_EVERY_N_FRAMES
        cached_detections = []
        motion_changed_pixels = 0
        motion_largest_blob_area = 0
        motion_changed_in_radius = 0
        did_run_inference = True

        while True:
            # ------------------------------------------------
            # Capture frame
            # ------------------------------------------------
            shot = np.array(sct.grab(region))
            frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

            # ------------------------------------------------
            # Model inference (optionally gated by screen-change detection)
            # ------------------------------------------------
            motion_gray, motion_changed_pixels, motion_largest_blob_area, motion_changed_in_radius, motion_should_infer = compute_motion_metrics(
                prev_motion_gray, frame
            )
            prev_motion_gray = motion_gray

            did_run_inference = (
                not MOTION_SCAN_ENABLED
                or motion_should_infer
                or frames_since_infer >= FORCE_INFERENCE_EVERY_N_FRAMES
            )

            stage1_classes = trackable_class_ids if trackable_class_ids else None

            results = None
            if did_run_inference:
                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE,
                    imgsz=STAGE1_IMG_SIZE if TWO_STAGE_ENABLED else IMG_SIZE,
                    device=0 if USE_GPU else "cpu",
                    verbose=False,
                    half=USE_GPU,
                    rect=True,
                    max_det=40,
                    classes=stage1_classes,
                )
                frames_since_infer = 0
            else:
                frames_since_infer += 1

                        # ------------------------------------------------
            # Mouse position + velocity
            # ------------------------------------------------
            now_mouse = time.time()
            mouse_x, mouse_y = get_mouse_position()

            dt_mouse = now_mouse - last_mouse_time
            if last_mouse_x is not None and dt_mouse > 0:
                mouse_vx = (mouse_x - last_mouse_x) / dt_mouse
                mouse_vy = (mouse_y - last_mouse_y) / dt_mouse
                mouse_speed = math.hypot(mouse_vx, mouse_vy)

            last_mouse_x = mouse_x
            last_mouse_y = mouse_y
            last_mouse_time = now_mouse

            # Mouse relative to the watched capture region
            mouse_rx = mouse_x - region["left"]
            mouse_ry = mouse_y - region["top"]

            frame_h, frame_w = frame.shape[:2]

            if render_mode == "full":
                annotated = frame.copy()
            elif render_mode in ("boxes", "stats"):
                annotated = np.zeros_like(frame)
            else:
                annotated = None

            base_cx = frame_w // 2
            base_cy = frame_h // 2
            screen_cx = int(base_cx + CROSSHAIR_OFFSET_X)
            screen_cy = int(base_cy + CROSSHAIR_OFFSET_Y)

            if annotated is not None and render_mode in ("full", "boxes"):
                draw_crosshair(annotated)

                # draw the shifted aim origin
                cv2.circle(annotated, (screen_cx, screen_cy), 6, (0, 255, 255), 2)
                cv2.line(annotated, (screen_cx - 12, screen_cy), (screen_cx + 12, screen_cy), (0, 255, 255), 2)
                cv2.line(annotated, (screen_cx, screen_cy - 12), (screen_cx, screen_cy + 12), (0, 255, 255), 2)

            # ------------------------------------------------
            # Global hotkeys
            # ------------------------------------------------
            raider_key_down = is_key_down(RAIDER_LOCK_KEY)
            other_key_down = is_key_down(OTHER_LOCK_KEY)

            raider_key_pressed = raider_key_down and not prev_raider_key_down
            other_key_pressed = other_key_down and not prev_other_key_down

            if raider_key_pressed:
                active_lock_mode = "raider"
                active_lock_id = None
            elif other_key_pressed:
                active_lock_mode = "other"
                active_lock_id = None
            elif not raider_key_down and not other_key_down:
                active_lock_mode = None
                active_lock_id = None

            prev_raider_key_down = raider_key_down
            prev_other_key_down = other_key_down

            for vk, region_name in AIM_REGION_HOTKEYS.items():
                hotkey_down = is_key_down(vk)
                if hotkey_down and not prev_aim_hotkey_down[vk]:
                    current_aim_region_override = region_name
                prev_aim_hotkey_down[vk] = hotkey_down

            # ------------------------------------------------
            # Build raw detections for classes we care about
            # ------------------------------------------------
            current_detections = []

            if results:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    for i in range(len(xyxy)):
                        cls_id = int(cls[i])
                        class_name = model.names[cls_id]

                        if class_name not in TRACKABLE_CLASSES:
                            continue

                        x1, y1, x2, y2 = xyxy[i].astype(int)
                        raw_conf = float(confs[i])

                        entry = build_detection_entry(
                            frame,
                            class_name,
                            x1,
                            y1,
                            x2,
                            y2,
                            raw_conf,
                            current_aim_region_override,
                        )
                        if entry is not None:
                            current_detections.append(entry)

            if did_run_inference and TWO_STAGE_ENABLED and (not STAGE2_REQUIRE_HOTKEY or active_lock_mode in ("raider", "other")):
                if active_lock_mode == "raider":
                    eligible_stage2_classes = {TRACK_CLASS}
                    stage2_class_ids = raider_class_ids if raider_class_ids else None
                elif active_lock_mode == "other":
                    eligible_stage2_classes = TRACKABLE_CLASSES - {TRACK_CLASS}
                    stage2_class_ids = other_class_ids if other_class_ids else None
                else:
                    eligible_stage2_classes = TRACKABLE_CLASSES
                    stage2_class_ids = trackable_class_ids if trackable_class_ids else None

                eligible_stage1 = [
                    (idx, det) for idx, det in enumerate(current_detections)
                    if det["class_name"] in eligible_stage2_classes
                ]

                if eligible_stage1:
                    stage1_idx, stage1_candidate = min(
                        eligible_stage1,
                        key=lambda item: math.hypot(item[1]["cx"] - screen_cx, item[1]["cy"] - screen_cy)
                    )

                    rx1 = max(0, stage1_candidate["x1"] - STAGE2_PADDING)
                    ry1 = max(0, stage1_candidate["y1"] - STAGE2_PADDING)
                    rx2 = min(frame_w, stage1_candidate["x2"] + STAGE2_PADDING)
                    ry2 = min(frame_h, stage1_candidate["y2"] + STAGE2_PADDING)

                    if rx2 > rx1 and ry2 > ry1:
                        roi = frame[ry1:ry2, rx1:rx2]

                        if STAGE2_CLASSES_MATCH_ONLY and stage1_candidate["class_name"] in name_to_id:
                            stage2_predict_classes = [name_to_id[stage1_candidate["class_name"]]]
                        else:
                            stage2_predict_classes = stage2_class_ids

                        stage2_results = model.predict(
                            source=roi,
                            conf=min(CONFIDENCE, STAGE2_MIN_CONF),
                            imgsz=STAGE2_IMG_SIZE,
                            device=0 if USE_GPU else "cpu",
                            verbose=False,
                            half=USE_GPU,
                            rect=True,
                            max_det=10,
                            classes=stage2_predict_classes,
                        )

                        refined_detections = []
                        if stage2_results:
                            stage2_result = stage2_results[0]
                            stage2_boxes = stage2_result.boxes

                            if stage2_boxes is not None and len(stage2_boxes) > 0:
                                s_xyxy = stage2_boxes.xyxy.cpu().numpy()
                                s_cls = stage2_boxes.cls.cpu().numpy()
                                s_confs = stage2_boxes.conf.cpu().numpy()

                                for j in range(len(s_xyxy)):
                                    s_cls_id = int(s_cls[j])
                                    s_class_name = model.names[s_cls_id]

                                    gx1, gy1, gx2, gy2 = s_xyxy[j].astype(int)
                                    gx1 += rx1
                                    gx2 += rx1
                                    gy1 += ry1
                                    gy2 += ry1

                                    refined_entry = build_detection_entry(
                                        frame,
                                        s_class_name,
                                        gx1,
                                        gy1,
                                        gx2,
                                        gy2,
                                        float(s_confs[j]),
                                        current_aim_region_override,
                                    )
                                    if refined_entry is not None:
                                        refined_detections.append(refined_entry)

                        if refined_detections:
                            refined_best = min(
                                refined_detections,
                                key=lambda det: math.hypot(det["cx"] - stage1_candidate["cx"], det["cy"] - stage1_candidate["cy"])
                            )
                            current_detections[stage1_idx] = refined_best

            if did_run_inference:
                cached_detections = [dict(det) for det in current_detections]
            else:
                current_detections = [dict(det) for det in cached_detections]

            # ------------------------------------------------
            # Mark all persistent tracks unmatched initially
            # ------------------------------------------------
            for track in persistent_tracks.values():
                track["matched_this_frame"] = False

            # ------------------------------------------------
            # Match detections to existing tracks by class + proximity
            # ------------------------------------------------
            for det in current_detections:
                best_track = None
                best_dist = float("inf")

                for track in persistent_tracks.values():
                    if track["class_name"] != det["class_name"]:
                        continue

                    dist = math.hypot(det["cx"] - track["cx"], det["cy"] - track["cy"])
                    if dist < best_dist and dist <= TRACK_MATCH_DISTANCE_PX:
                        best_dist = dist
                        best_track = track

                if best_track is not None:
                    best_track["x1"] = det["x1"]
                    best_track["y1"] = det["y1"]
                    best_track["x2"] = det["x2"]
                    best_track["y2"] = det["y2"]
                    best_track["cx"] = det["cx"]
                    best_track["cy"] = det["cy"]
                    best_track["raw_conf"] = det["raw_conf"]
                    best_track["avg_conf"] = ema_update(best_track["avg_conf"], det["raw_conf"], CONFIDENCE_SMOOTHING)
                    best_track["has_marker"] = det["has_marker"]
                    best_track["marker_rect"] = det["marker_rect"]
                    best_track["aim_region_name"] = det["aim_region_name"]
                    best_track["aim_x1"] = det["aim_x1"]
                    best_track["aim_y1"] = det["aim_y1"]
                    best_track["aim_x2"] = det["aim_x2"]
                    best_track["aim_y2"] = det["aim_y2"]
                    best_track["age"] += 1
                    best_track["missed"] = 0
                    best_track["matched_this_frame"] = True
                else:
                    track_id = next(track_id_gen)
                    persistent_tracks[track_id] = {
                        "track_id": track_id,
                        "class_name": det["class_name"],
                        "x1": det["x1"],
                        "y1": det["y1"],
                        "x2": det["x2"],
                        "y2": det["y2"],
                        "cx": det["cx"],
                        "cy": det["cy"],
                        "raw_conf": det["raw_conf"],
                        "avg_conf": det["raw_conf"],
                        "has_marker": det["has_marker"],
                        "marker_rect": det["marker_rect"],
                        "aim_region_name": det["aim_region_name"],
                        "aim_x1": det["aim_x1"],
                        "aim_y1": det["aim_y1"],
                        "aim_x2": det["aim_x2"],
                        "aim_y2": det["aim_y2"],
                        "age": 1,
                        "missed": 0,
                        "matched_this_frame": True,
                        "score": 0.0,
                    }

            # ------------------------------------------------
            # Age unmatched tracks and forget stale ones
            # ------------------------------------------------
            to_delete = []

            for track_id, track in persistent_tracks.items():
                if not track["matched_this_frame"]:
                    track["missed"] += 1

                if track["missed"] > TRACK_FORGET_FRAMES:
                    to_delete.append(track_id)

            for track_id in to_delete:
                if active_lock_id == track_id:
                    # remember last active lock position before deleting
                    if DROP_MEMORY_ENABLED:
                        old_track = persistent_tracks[track_id]
                        drop_memory = {
                            "cx": old_track["cx"],
                            "cy": old_track["cy"],
                            "frames_left": DROP_MEMORY_FRAMES,
                        }
                    active_lock_id = None

                del persistent_tracks[track_id]

            # ------------------------------------------------
            # Score all tracks using positive-only bonuses
            # ------------------------------------------------
            for track in persistent_tracks.values():
                class_conf_weight = CLASS_CONFIDENCE_WEIGHT.get(track["class_name"], BASE_CONFIDENCE_WEIGHT)
                base_score = track["avg_conf"] * class_conf_weight
                center_bonus = positive_center_bonus(track["cx"], track["cy"], frame_w, frame_h)
                age_bonus = min(track["age"] * TRACKED_AGE_SCORE_BONUS, MAX_TRACK_AGE_BONUS)
                marker_bonus = MARKER_SCORE_BONUS if track["has_marker"] else 0.0

                drop_bonus = 0.0
                if track["class_name"] == TRACK_CLASS and DROP_MEMORY_ENABLED and near_drop_memory(track["cx"], track["cy"], drop_memory):
                    drop_bonus = DROP_MEMORY_SCORE_BONUS

                track["score"] = base_score + center_bonus + age_bonus + marker_bonus + drop_bonus

            # ------------------------------------------------
            # Choose active lock from held activation key and nearest eligible target
            # ------------------------------------------------
            if active_lock_mode == "raider":
                eligible_classes = {TRACK_CLASS}
            elif active_lock_mode == "other":
                eligible_classes = TRACKABLE_CLASSES - {TRACK_CLASS}
            else:
                eligible_classes = set()

            eligible_tracks = [
                t for t in persistent_tracks.values()
                if t["class_name"] in eligible_classes
                and (not t["has_marker"])
                and t["avg_conf"] >= CLASS_LOCK_MIN_AVG_CONFIDENCE.get(t["class_name"], LOCK_ON_MIN_AVG_CONFIDENCE)
            ]

            if active_lock_id is not None:
                current_lock = persistent_tracks.get(active_lock_id)
                if (
                    current_lock is None
                    or current_lock["class_name"] not in eligible_classes
                    or current_lock["has_marker"]
                    or current_lock["avg_conf"] < CLASS_LOCK_MIN_AVG_CONFIDENCE.get(current_lock["class_name"], LOCK_ON_MIN_AVG_CONFIDENCE)
                ):
                    active_lock_id = None

            if active_lock_id is None and eligible_tracks:
                nearest_track = min(
                    eligible_tracks,
                    key=lambda t: math.hypot(t["cx"] - screen_cx, t["cy"] - screen_cy)
                )
                active_lock_id = nearest_track["track_id"]

            active_lock = persistent_tracks.get(active_lock_id) if active_lock_id is not None else None



            # ------------------------------------------------
            # Dynamic mouse/controller velocity suggestion
            # ------------------------------------------------
            now_control = time.time()
            dt_control = max(now_control - last_control_time, 1e-6)
            last_control_time = now_control

            applied_dx = 0
            applied_dy = 0
            applied_force = 0.0

            desired_vx = 0.0
            desired_vy = 0.0
            max_allowed_speed = 0.0
            direction_dot = 0.0

            if active_lock is not None:
                error_x = active_lock["cx"] - screen_cx
                error_y = active_lock["cy"] - screen_cy
                distance_to_target = math.hypot(error_x, error_y)

                # deadzone near center prevents tiny jitter
                if distance_to_target < DEADZONE_RADIUS:
                    desired_vx = 0.0
                    desired_vy = 0.0
                else:
                    # proportional desired velocity
                    Kp = 4.0
                    desired_vx = error_x * Kp
                    desired_vy = error_y * Kp

                # dynamic max speed based on distance + avg confidence
                max_allowed_speed = dynamic_speed_limit(distance_to_target, active_lock["avg_conf"])

                # first clamp to dynamic speed
                desired_vx, desired_vy = limit_velocity(desired_vx, desired_vy, max_allowed_speed)

                # then limit acceleration so it doesn't jump abruptly
                limited_vx, limited_vy = accel_limit(
                    limited_vx, limited_vy, desired_vx, desired_vy, dt_control
                )

                # remove any velocity component that points away from the target
                if STOP_AWAY_MOTION:
                    limited_vx, limited_vy = remove_away_component(
                        error_x, error_y, limited_vx, limited_vy
                    )

                # final hard stop if very close
                if distance_to_target < DEADZONE_RADIUS:
                    limited_vx, limited_vy = 0.0, 0.0

                direction_dot = limited_vx * error_x + limited_vy * error_y

            else:
                # decay toward zero if no active lock
                limited_vx, limited_vy = accel_limit(
                    limited_vx, limited_vy, 0.0, 0.0, dt_control
                )
                direction_dot = 0.0

            # ------------------------------------------------
            # Apply limited mouse movement only while hold key is down
            # ------------------------------------------------


            send_dx = 0
            send_dy = 0

            if active_lock is not None and is_key_down(AIM_HOLD_KEY):
                mouse_frac_x += limited_vx * dt_control
                mouse_frac_y += limited_vy * dt_control

                send_dx = int(mouse_frac_x)
                send_dy = int(mouse_frac_y)

                mouse_frac_x -= send_dx
                mouse_frac_y -= send_dy

                if send_dx != 0 or send_dy != 0:
                    move_mouse_relative(send_dx, send_dy)

                    applied_dx = send_dx
                    applied_dy = send_dy
                    applied_force = math.hypot(send_dx, send_dy)
            else:
                # optional: dump stored fractional movement so it doesn't jump later
                mouse_frac_x = 0.0
                mouse_frac_y = 0.0

            # ------------------------------------------------
            # Derive best green and nearest red from persistent tracks
            # ------------------------------------------------
            green_tracks = [t for t in persistent_tracks.values() if t["class_name"] == COLLECT_CLASS]
            best_green = max(green_tracks, key=lambda t: t["score"]) if green_tracks else None

            red_tracks = [t for t in persistent_tracks.values() if t["class_name"] == AVOID_CLASS]
            nearest_red = min(
                red_tracks,
                key=lambda t: math.hypot(t["cx"] - screen_cx, t["cy"] - screen_cy)
            ) if red_tracks else None

            # ------------------------------------------------
            # Draw all persistent tracks
            # ------------------------------------------------
            if annotated is not None and render_mode in ("full", "boxes"):
                for track in persistent_tracks.values():
                    class_name = track["class_name"]

                    if class_name == TRACK_CLASS:
                        color = (255, 0, 0)      # blue
                    elif class_name == COLLECT_CLASS:
                        color = (0, 255, 0)      # green
                    elif class_name == AVOID_CLASS:
                        color = (0, 0, 255)      # red
                    else:
                        color = (180, 180, 180)

                    thickness = 2
                    if active_lock is not None and track["track_id"] == active_lock["track_id"]:
                        color = (0, 255, 255)
                        thickness = 4

                    cv2.rectangle(annotated, (track["x1"], track["y1"]), (track["x2"], track["y2"]), color, thickness)

                    if DRAW_AIM_REGIONS:
                        cv2.rectangle(
                            annotated,
                            (int(track["aim_x1"]), int(track["aim_y1"])),
                            (int(track["aim_x2"]), int(track["aim_y2"])),
                            AIM_REGION_COLOR,
                            AIM_REGION_THICKNESS,
                        )

                    cv2.circle(annotated, (int(track["cx"]), int(track["cy"])), 4, color, -1)

                    label = (
                        f'{track["class_name"]} '
                        f'raw={track["raw_conf"]:.2f} '
                        f'avg={track["avg_conf"]:.2f} '
                        f's={track["score"]:.2f} '
                        f'age={track["age"]} '
                        f'aim={track.get("aim_region_name", "center")}'
                    )
                    if track["has_marker"]:
                        label += " marker"

                    cv2.putText(
                        annotated,
                        label,
                        (track["x1"], max(20, track["y1"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    if track["marker_rect"] is not None:
                        rx1, ry1, rx2, ry2 = track["marker_rect"]
                        marker_color = (255, 0, 255) if track["has_marker"] else (128, 128, 128)
                        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), marker_color, 2)

                if active_lock is not None:
                    cv2.line(
                        annotated,
                        (screen_cx, screen_cy),
                        (int(active_lock["cx"]), int(active_lock["cy"])),
                        (255, 255, 0),
                        2,
                    )

                if DRAW_APPLIED_VECTOR and applied_force >= APPLIED_VECTOR_MIN_DRAW:
                    arrow_end_x = int(screen_cx + applied_dx * APPLIED_VECTOR_SCALE)
                    arrow_end_y = int(screen_cy + applied_dy * APPLIED_VECTOR_SCALE)
                    cv2.arrowedLine(
                        annotated,
                        (int(screen_cx), int(screen_cy)),
                        (arrow_end_x, arrow_end_y),
                        APPLIED_VECTOR_COLOR,
                        APPLIED_VECTOR_THICKNESS,
                        cv2.LINE_AA,
                        tipLength=0.25,
                    )

                # Draw mouse if it is inside the watched region
                if 0 <= mouse_rx < frame_w and 0 <= mouse_ry < frame_h:
                    cv2.circle(annotated, (int(mouse_rx), int(mouse_ry)), 6, (255, 255, 255), 2)
                    cv2.line(annotated, (int(mouse_rx) - 10, int(mouse_ry)), (int(mouse_rx) + 10, int(mouse_ry)), (255, 255, 255), 1)
                    cv2.line(annotated, (int(mouse_rx), int(mouse_ry) - 10), (int(mouse_rx), int(mouse_ry) + 10), (255, 255, 255), 1)

            # ------------------------------------------------
            # Drop-memory decay
            # ------------------------------------------------
            if drop_memory is not None:
                if annotated is not None and render_mode in ("full", "boxes"):
                    cv2.circle(
                        annotated,
                        (int(drop_memory["cx"]), int(drop_memory["cy"])),
                        DROP_MEMORY_RADIUS_PX,
                        (255, 0, 255),
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"DROP MEMORY {drop_memory['frames_left']}",
                        (int(drop_memory["cx"]) + 8, int(drop_memory["cy"]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                drop_memory["frames_left"] -= 1
                if drop_memory["frames_left"] <= 0:
                    drop_memory = None

            # ------------------------------------------------
            # FPS update
            # ------------------------------------------------
            frames += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frames / (now - last_fps_time)
                frames = 0
                last_fps_time = now

            # ------------------------------------------------
            # Text / stats overlay
            # ------------------------------------------------
            if annotated is not None and render_mode in ("full", "boxes", "stats"):
                stats_lines = [
                    f"FPS: {fps:.1f}",
                    f"Render: {render_mode}  keys: 1=full 2=boxes 3=stats 4=off",
                    f"Mouse global: ({mouse_x}, {mouse_y})",
                    f"Mouse region: ({mouse_rx}, {mouse_ry})",
                    f"Mouse velocity: vx={mouse_vx:.1f}  vy={mouse_vy:.1f}  speed={mouse_speed:.1f} px/s",
                    f"Desired vel: ({desired_vx:.1f}, {desired_vy:.1f}) px/s",
                    f"Limited vel: ({limited_vx:.1f}, {limited_vy:.1f}) px/s",
                    f"Dynamic max speed: {max_allowed_speed:.1f} px/s",
                    f"Direction dot: {direction_dot:.1f}",
                    f"Tracks total: {len(persistent_tracks)}",
                    f"Lock mode: {active_lock_mode or 'none'}  raider_key={raider_key_down} other_key={other_key_down}",
                    f"Aim region override: {current_aim_region_override or 'per-class default'}  F5-F10",
                    f"Motion scan: enabled={MOTION_SCAN_ENABLED} infer={did_run_inference} changed={motion_changed_pixels} blob={motion_largest_blob_area} radius={motion_changed_in_radius}",
                    f"Backend: {'TensorRT' if str(runtime_model_path).endswith('.engine') else 'PyTorch'}  two_stage={TWO_STAGE_ENABLED}",
                    f"Stage1 imgsz={STAGE1_IMG_SIZE}  Stage2 imgsz={STAGE2_IMG_SIZE}  global_conf={CONFIDENCE:.2f}",
                ]

                if active_lock is not None:
                    stats_lines.append(
                        f'LOCK BLUE id={active_lock["track_id"]} avg={active_lock["avg_conf"]:.2f} score={active_lock["score"]:.2f} aim={active_lock.get("aim_region_name", "center")}'
                    )
                else:
                    stats_lines.append("LOCK BLUE: none")

                if best_green is not None:
                    stats_lines.append(
                        f'COLLECT GREEN id={best_green["track_id"]} avg={best_green["avg_conf"]:.2f} score={best_green["score"]:.2f}'
                    )
                else:
                    stats_lines.append("COLLECT GREEN: none")

                if nearest_red is not None:
                    stats_lines.append(
                        f'AVOID RED id={nearest_red["track_id"]} center=({int(nearest_red["cx"])},{int(nearest_red["cy"])})'
                    )
                else:
                    stats_lines.append("AVOID RED: none")

                y = 35
                for line in stats_lines:
                    cv2.putText(
                        annotated,
                        line,
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    y += 30

            # ------------------------------------------------
            # Show preview
            # ------------------------------------------------
            if annotated is not None and (frames % preview_every_n_frames == 0):
                cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):
                render_mode = "full"
            elif key == ord("2"):
                render_mode = "boxes"
            elif key == ord("3"):
                render_mode = "stats"
            elif key == ord("4"):
                render_mode = "off"
            elif key == ord("q") or key == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()