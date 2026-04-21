from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
VIDEO_PATH = Path("auto_person_labels/input_videos/inputtfue.mp4")
OUTPUT_ROOT = Path("auto_person_labels")

MODEL_NAME = "yolov8n.pt"
TARGET_CLASS_NAME = "person"

CONF_THRESHOLD = 0.37
FRAME_STRIDE = 10
VAL_SPLIT = 0.20
SAVE_EMPTY_FRAMES = False
IMG_EXT = ".jpg"
RANDOM_SEED = 42

IGNORE_ZONES = [
    #left, top right bottom
     (0.05, 0.55, 1.00, 1.00),
     (0.70, 0.00, 1.00, 0.30),
    ]

SHOW_PREVIEW = True
PREVIEW_WINDOW_NAME = "Auto Person Label Preview"
PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 720

IGNORE_SELF = False
CENTER_RADIUS_RATIO = 0.45
MIN_SELF_HITS = 6
# =========================

track_hits = defaultdict(int)
ignored_track_ids = set()

def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def norm_zone_to_pixels(zone, img_w, img_h):
    left, top, right, bottom = zone
    return (
        int(left * img_w),
        int(top * img_h),
        int(right * img_w),
        int(bottom * img_h),
    )


def boxes_overlap(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    return not (
        x2 <= a1 or
        x1 >= a2 or
        y2 <= b1 or
        y1 >= b2
    )


def is_in_ignore_zone(box, img_w, img_h):
    for zone in IGNORE_ZONES:
        zone_px = norm_zone_to_pixels(zone, img_w, img_h)
        if boxes_overlap(box, zone_px):
            return True
    return False

def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> str:
    box_w = x2 - x1
    box_h = y2 - y1
    center_x = x1 + box_w / 2.0
    center_y = y1 + box_h / 2.0
    return (
        f"0 "
        f"{center_x / img_w:.6f} "
        f"{center_y / img_h:.6f} "
        f"{box_w / img_w:.6f} "
        f"{box_h / img_h:.6f}"
    )


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "train_images": root / "images" / "train",
        "val_images": root / "images" / "val",
        "train_labels": root / "labels" / "train",
        "val_labels": root / "labels" / "val",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return dirs


def choose_split() -> str:
    return "val" if random.random() < VAL_SPLIT else "train"


def resize_for_preview(frame, max_width: int, max_height: int):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return frame
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def box_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def is_near_center(cx, cy, img_w, img_h, radius_ratio=CENTER_RADIUS_RATIO):
    center_x = img_w / 2.0
    center_y = img_h / 2.0
    radius = min(img_w, img_h) * radius_ratio
    dx = cx - center_x
    dy = cy - center_y
    return (dx * dx + dy * dy) <= (radius * radius)


def draw_preview(
    frame,
    person_boxes,
    frame_idx: int,
    saved_idx: int,
    conf_threshold: float,
    frame_stride: int,
    paused: bool,
    ):
    preview = frame.copy()

    h, w = preview.shape[:2]

    for zone in IGNORE_ZONES:
        zx1, zy1, zx2, zy2 = norm_zone_to_pixels(zone, w, h)
        cv2.rectangle(preview, (zx1, zy1), (zx2, zy2), (0, 165, 255), 2)
        cv2.putText(
            preview,
            "IGNORE",
            (zx1, max(20, zy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    radius = int(min(w, h) * CENTER_RADIUS_RATIO)
    cv2.circle(preview, (w // 2, h // 2), radius, (255, 0, 0), 2)

    for x1, y1, x2, y2, conf, ignored, track_id in person_boxes:
        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

        color = (0, 255, 0) if not ignored else (0, 0, 255)
        label = f"person {conf:.2f}"
        if track_id is not None:
            label += f" id={track_id}"
        if ignored:
            label += " IGNORE"

        cv2.rectangle(preview, (x1_i, y1_i), (x2_i, y2_i), color, 2)
        cv2.putText(
            preview,
            label,
            (x1_i, max(20, y1_i - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    status_1 = (
        f"frame={frame_idx} saved={saved_idx} stride={frame_stride} "
        f"conf={conf_threshold:.2f}"
    )
    status_2 = f"[q] quit   [space] pause/resume   paused={paused}"

    cv2.putText(preview, status_1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(preview, status_2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    return resize_for_preview(preview, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)


def main():
    random.seed(RANDOM_SEED)
    dirs = ensure_dirs(OUTPUT_ROOT)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    if SHOW_PREVIEW:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW_NAME, 1100, 700)

    frame_idx = 0
    saved_idx = 0
    paused = False
    last_preview_frame = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            last_preview_frame = frame.copy()
        else:
            if last_preview_frame is None:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord(" "):
                    paused = not paused
                continue
            frame = last_preview_frame.copy()

        if not paused and frame_idx % FRAME_STRIDE != 0:
            if SHOW_PREVIEW:
                small = resize_for_preview(frame, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)
                cv2.imshow(PREVIEW_WINDOW_NAME, small)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord(" "):
                    paused = not paused
            frame_idx += 1
            continue

        img_h, img_w = frame.shape[:2]

        results = model.track(
            source=frame,
            conf=CONF_THRESHOLD,
            device=0,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        person_lines = []
        person_boxes = []

        if results:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                track_ids = None
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()

                for i in range(len(xyxy)):
                    cls_id = int(cls[i])
                    class_name = model.names[cls_id]

                    if class_name != TARGET_CLASS_NAME:
                        continue

                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i])

                cx, cy = box_center(x1, y1, x2, y2)

                ignore = False

                for zone_i, zone in enumerate(IGNORE_ZONES):
                    zx1, zy1, zx2, zy2 = norm_zone_to_pixels(zone, img_w, img_h)
                    zone_px = (zx1, zy1, zx2, zy2)

                    #print(f"det center=({cx:.1f}, {cy:.1f}) | zone #{zone_i + 1} px={zone_px}")

                    if point_in_box(cx, cy, zone_px):
                        #print(f"Ignoring box center ({cx:.1f}, {cy:.1f}) in zone #{zone_i + 1}: {zone_px}")
                        ignore = True
                        break

                if ignore:
                    continue

                track_id = None
                if track_ids is not None:
                    track_id = int(track_ids[i])

                if IGNORE_SELF and track_id is not None:
                    if is_near_center(cx, cy, img_w, img_h):
                        track_hits[track_id] += 1

                    if track_hits[track_id] >= MIN_SELF_HITS:
                        ignored_track_ids.add(track_id)

                    if track_id in ignored_track_ids:
                        person_boxes.append((x1, y1, x2, y2, conf, True, track_id))
                        continue

                person_boxes.append((x1, y1, x2, y2, conf, False, track_id))
                person_lines.append(xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h))


        if person_lines or SAVE_EMPTY_FRAMES:
            split = choose_split()
            image_dir = dirs[f"{split}_images"]
            label_dir = dirs[f"{split}_labels"]

            stem = f"frame_{saved_idx:06d}"
            image_path = image_dir / f"{stem}{IMG_EXT}"
            label_path = label_dir / f"{stem}.txt"

            cv2.imwrite(str(image_path), frame)
            label_path.write_text("\n".join(person_lines), encoding="utf-8")
            saved_idx += 1

        if SHOW_PREVIEW:
            preview = draw_preview(
                frame=frame,
                person_boxes=person_boxes,
                frame_idx=frame_idx,
                saved_idx=saved_idx,
                conf_threshold=CONF_THRESHOLD,
                frame_stride=FRAME_STRIDE,
                paused=paused,
            )
            cv2.imshow(PREVIEW_WINDOW_NAME, preview)

            wait_time = 0 if paused else 1
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused

        if not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    yaml_text = """path: auto_person_labels
train: images/train
val: images/val

names:
  0: person
"""
    (OUTPUT_ROOT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


if __name__ == "__main__":
    main()