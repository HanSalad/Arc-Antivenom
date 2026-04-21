from __future__ import annotations

import random
import time
from pathlib import Path

from datetime import datetime
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
#MODEL_PATH = "yolov8n.pt"
#TARGET_CLASS_NAME = "person"
MODEL_PATH = r"C:\yolo_game_tracker\runs\detect\runs\game_targets\weights\best.pt"
TARGET_CLASS_NAME = "raider"

OUTPUT_ROOT = Path("screen_auto_labels")

MONITOR_INDEX = 1
CAPTURE_REGION = None
# Example:
# CAPTURE_REGION = {"left": 500, "top": 200, "width": 900, "height": 700}

CONF_THRESHOLD = 0.55
SAVE_INTERVAL_SECONDS = 0.75
VAL_SPLIT = 0.20
SAVE_EMPTY_FRAMES = False
IMG_EXT = ".jpg"
RANDOM_SEED = 42

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

DEFAULT_MANUAL_CLASS_ID = 1  # blue

SHOW_PREVIEW = True
PREVIEW_WINDOW_NAME = "Screen Auto Label Preview"
PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 720

DRAW_IGNORE_ZONES = False
IGNORE_ZONES = [
    # normalized: left, top, right, bottom
     (0.05, 0.65, 1.00, 1.00),
     (0.70, 0.00, 1.00, 0.30),
]
# =========================


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

def xyxy_to_yolo_with_class(cls_id: int, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> str:
    box_w = x2 - x1
    box_h = y2 - y1
    center_x = x1 + box_w / 2.0
    center_y = y1 + box_h / 2.0
    return (
        f"{cls_id} "
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
        return frame, 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

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
    return not (x2 <= a1 or x1 >= a2 or y2 <= b1 or y1 >= b2)


def is_in_ignore_zone(box, img_w, img_h):
    for zone in IGNORE_ZONES:
        zone_px = norm_zone_to_pixels(zone, img_w, img_h)
        if boxes_overlap(box, zone_px):
            return True
    return False


class ManualBoxTool:
    def __init__(self):
        self.draw_mode = False
        self.drawing = False

        self.start_x = 0
        self.start_y = 0
        self.current_x = 0
        self.current_y = 0

        self.preview_scale = 1.0

        # each box = (x1, y1, x2, y2, cls_id)
        self.manual_boxes: list[tuple[int, int, int, int, int]] = []

        self.class_ids = sorted(CLASS_NAMES.keys())
        self.manual_class_id = DEFAULT_MANUAL_CLASS_ID if DEFAULT_MANUAL_CLASS_ID in self.class_ids else self.class_ids[0]

    def set_scale(self, scale: float):
        self.preview_scale = scale if scale > 0 else 1.0

    def set_class_id(self, cls_id: int):
        if cls_id in self.class_ids:
            self.manual_class_id = cls_id

    def next_class(self):
        idx = self.class_ids.index(self.manual_class_id)
        idx = (idx + 1) % len(self.class_ids)
        self.manual_class_id = self.class_ids[idx]

    def prev_class(self):
        idx = self.class_ids.index(self.manual_class_id)
        idx = (idx - 1) % len(self.class_ids)
        self.manual_class_id = self.class_ids[idx]

    def mouse_callback(self, event, x, y, flags, param):
        if not self.draw_mode:
            return

        real_x = int(round(x / self.preview_scale))
        real_y = int(round(y / self.preview_scale))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = real_x, real_y
            self.current_x, self.current_y = real_x, real_y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_x, self.current_y = real_x, real_y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_x, self.current_y = real_x, real_y

            x1 = min(self.start_x, self.current_x)
            y1 = min(self.start_y, self.current_y)
            x2 = max(self.start_x, self.current_x)
            y2 = max(self.start_y, self.current_y)

            if x2 - x1 > 3 and y2 - y1 > 3:
                self.manual_boxes.append((x1, y1, x2, y2, self.manual_class_id))

    def undo(self):
        if self.manual_boxes:
            self.manual_boxes.pop()

    def clear(self):
        self.manual_boxes.clear()

def draw_preview(frame, kept_boxes, saved_count, last_save_time, paused, tool: ManualBoxTool):
    preview = frame.copy()
    h, w = preview.shape[:2]

    if DRAW_IGNORE_ZONES:
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

    for x1, y1, x2, y2, conf in kept_boxes:
        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(preview, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"{TARGET_CLASS_NAME} {conf:.2f}",
            (x1_i, max(20, y1_i - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for i, (x1, y1, x2, y2, cls_id) in enumerate(tool.manual_boxes):
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 0), 2)
        class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        cv2.putText(
            preview,
            f"manual {i+1} [{cls_id}:{class_name}]",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )



    if tool.draw_mode and tool.drawing:
        x1 = min(tool.start_x, tool.current_x)
        y1 = min(tool.start_y, tool.current_y)
        x2 = max(tool.start_x, tool.current_x)
        y2 = max(tool.start_y, tool.current_y)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 255), 1)

    current_class_name = CLASS_NAMES.get(tool.manual_class_id, f"class_{tool.manual_class_id}")

    status_1 = (
        f"saved={saved_count} conf={CONF_THRESHOLD:.2f} "
        f"paused={paused} draw_mode={tool.draw_mode}"
    )
    status_2 = (
        "[q] quit [space] pause [d] draw mode [u] undo [c] clear "
        "[s] save [[ / ]] class  [, / .] class"
    )
    status_3 = (
        f"last save: {last_save_time:.2f} manual_boxes={len(tool.manual_boxes)} "
        f"manual_class={tool.manual_class_id}:{current_class_name}"
    )

    cv2.putText(preview, status_1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(preview, status_2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(preview, status_3, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

    preview_small, scale = resize_for_preview(preview, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)
    tool.set_scale(scale)
    return preview_small


def save_frame_and_labels(frame, label_lines, dirs, saved_count, run_id):
    split = choose_split()
    image_dir = dirs[f"{split}_images"]
    label_dir = dirs[f"{split}_labels"]

    stem = f"screen_{run_id}_{saved_count:06d}"
    image_path = image_dir / f"{stem}{IMG_EXT}"
    label_path = label_dir / f"{stem}.txt"

    cv2.imwrite(str(image_path), frame)
    label_path.write_text("\n".join(label_lines), encoding="utf-8")

    print(f"Saved {image_path.name} | split={split} | boxes={len(label_lines)}")
    return saved_count + 1


def main():
    random.seed(RANDOM_SEED)
    dirs = ensure_dirs(OUTPUT_ROOT)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Run ID:", run_id)
    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("Classes:", model.names)

    tool = ManualBoxTool()

    if SHOW_PREVIEW:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW_NAME, 1100, 700)
        cv2.setMouseCallback(PREVIEW_WINDOW_NAME, tool.mouse_callback)

    saved_count = 0
    paused = False
    last_saved_at = 0.0

    frame = None
    kept_boxes: list[tuple[float, float, float, float, float]] = []
    label_lines: list[str] = []

    with mss() as sct:
        region = CAPTURE_REGION if CAPTURE_REGION else sct.monitors[MONITOR_INDEX]
        print("Capture region:", region)

        while True:
            if not paused:
                shot = np.array(sct.grab(region))
                frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

                img_h, img_w = frame.shape[:2]

                results = model.predict(
                    source=frame,
                    conf=CONF_THRESHOLD,
                    device=0,
                    verbose=False,
                )

                label_lines = []
                kept_boxes = []

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

                            if class_name != TARGET_CLASS_NAME:
                                continue

                            x1, y1, x2, y2 = xyxy[i]
                            conf = float(confs[i])

                            if IGNORE_ZONES and is_in_ignore_zone((x1, y1, x2, y2), img_w, img_h):
                                continue

                            kept_boxes.append((x1, y1, x2, y2, conf))
                            label_lines.append(xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h))

                now = time.time()
                should_save = (now - last_saved_at) >= SAVE_INTERVAL_SECONDS

                if should_save and (label_lines or SAVE_EMPTY_FRAMES):
                    saved_count = save_frame_and_labels(frame, label_lines, dirs, saved_count, run_id)
                    last_saved_at = now

            if SHOW_PREVIEW and frame is not None:
                preview = draw_preview(frame, kept_boxes, saved_count, last_saved_at, paused, tool)
                cv2.imshow(PREVIEW_WINDOW_NAME, preview)

            key = cv2.waitKey(1 if not paused else 30) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord(" "):
                paused = not paused
                if paused:
                    tool.clear()
                    tool.draw_mode = False

            elif key == ord("d") and paused:
                tool.draw_mode = not tool.draw_mode
                print(f"Draw mode: {tool.draw_mode}")

            elif key == ord("u") and paused:
                tool.undo()

            elif key == ord("c") and paused:
                tool.clear()

            elif key in (ord("]"), ord(".")) and paused:
                tool.next_class()
                print(f"Manual class set to {tool.manual_class_id}:{CLASS_NAMES.get(tool.manual_class_id, 'unknown')}")

            elif key in (ord("["), ord(",")) and paused:
                tool.prev_class()
                print(f"Manual class set to {tool.manual_class_id}:{CLASS_NAMES.get(tool.manual_class_id, 'unknown')}")

            elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8"), ord("9")) and paused:
                cls_id = int(chr(key)) - 1
                if cls_id in CLASS_NAMES:
                    tool.set_class_id(cls_id)
                    print(f"Manual class set to {tool.manual_class_id}:{CLASS_NAMES.get(tool.manual_class_id, 'unknown')}")

            elif key == ord("s") and paused and frame is not None:
                img_h, img_w = frame.shape[:2]

                if tool.manual_boxes:
                    # Manual boxes are saved exactly as drawn, with their selected class IDs.
                    # They are NOT restricted by ignore zones.
                    manual_lines = [
                        xyxy_to_yolo_with_class(cls_id, x1, y1, x2, y2, img_w, img_h)
                        for (x1, y1, x2, y2, cls_id) in tool.manual_boxes
                    ]
                    saved_count = save_frame_and_labels(frame, manual_lines, dirs, saved_count, run_id)
                    last_saved_at = time.time()
                    print(f"Manual save with {len(tool.manual_boxes)} boxes")
                elif label_lines or SAVE_EMPTY_FRAMES:
                    saved_count = save_frame_and_labels(frame, label_lines, dirs, saved_count, run_id)
                    last_saved_at = time.time()
                    print(f"Saved paused frame with auto labels: {len(label_lines)}")
                else:
                    print("Nothing to save.")


    cv2.destroyAllWindows()

    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val

names:
  0: {TARGET_CLASS_NAME}
"""
    (OUTPUT_ROOT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    print("Done.")
    print("Dataset yaml:", OUTPUT_ROOT / "dataset.yaml")


if __name__ == "__main__":
    main()