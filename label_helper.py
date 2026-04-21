from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import cv2


RAW_DIR = Path("screenshots/raw")
TRAIN_IMG_DIR = Path("screenshots/images/train")
VAL_IMG_DIR = Path("screenshots/images/val")
TRAIN_LABEL_DIR = Path("screenshots/labels/train")
VAL_LABEL_DIR = Path("screenshots/labels/val")

CLASS_ID = 0
WINDOW_NAME = "YOLO Label Helper"
AUTO_SPLIT_ON_SAVE = True
VAL_EVERY_N = 5


class LabelTool:
    def __init__(self, image_paths: List[Path]):
        if not image_paths:
            raise ValueError("No images found in screenshots/raw")

        self.image_paths = image_paths
        self.index = 0
        self.boxes: List[Tuple[int, int, int, int]] = []

        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_x = 0
        self.current_y = 0
        self.saved_count = 0

        self._ensure_dirs()
        self._load_existing_for_current()

    def _ensure_dirs(self) -> None:
        for folder in [
            TRAIN_IMG_DIR,
            VAL_IMG_DIR,
            TRAIN_LABEL_DIR,
            VAL_LABEL_DIR,
        ]:
            folder.mkdir(parents=True, exist_ok=True)

    def current_image_path(self) -> Path:
        return self.image_paths[self.index]

    def load_current_image(self):
        path = self.current_image_path()
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img

    def find_existing_label_path(self) -> Path | None:
        stem = self.current_image_path().stem
        for folder in [TRAIN_LABEL_DIR, VAL_LABEL_DIR]:
            candidate = folder / f"{stem}.txt"
            if candidate.exists():
                return candidate
        return None

    def _load_existing_for_current(self):
        self.boxes.clear()
        img = self.load_current_image()
        img_h, img_w = img.shape[:2]

        label_path = self.find_existing_label_path()
        if not label_path:
            return

        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            return

        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            _, cx, cy, bw, bh = map(float, parts)

            box_w = bw * img_w
            box_h = bh * img_h
            center_x = cx * img_w
            center_y = cy * img_h

            x1 = int(round(center_x - box_w / 2))
            y1 = int(round(center_y - box_h / 2))
            x2 = int(round(center_x + box_w / 2))
            y2 = int(round(center_y + box_h / 2))

            self.boxes.append((x1, y1, x2, y2))

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.current_x, self.current_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_x, self.current_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_x, self.current_y = x, y

            x1 = min(self.start_x, self.current_x)
            y1 = min(self.start_y, self.current_y)
            x2 = max(self.start_x, self.current_x)
            y2 = max(self.start_y, self.current_y)

            if x2 - x1 > 3 and y2 - y1 > 3:
                self.boxes.append((x1, y1, x2, y2))

    @staticmethod
    def box_to_yolo(box: Tuple[int, int, int, int], img_w: int, img_h: int) -> str:
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        center_x = x1 + box_w / 2
        center_y = y1 + box_h / 2

        return (
            f"{CLASS_ID} "
            f"{center_x / img_w:.6f} "
            f"{center_y / img_h:.6f} "
            f"{box_w / img_w:.6f} "
            f"{box_h / img_h:.6f}"
        )

    def remove_old_copies(self, stem: str):
        for folder in [TRAIN_IMG_DIR, VAL_IMG_DIR]:
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                p = folder / f"{stem}{ext}"
                if p.exists():
                    p.unlink()

        for folder in [TRAIN_LABEL_DIR, VAL_LABEL_DIR]:
            p = folder / f"{stem}.txt"
            if p.exists():
                p.unlink()

    def get_split_dirs(self):
        self.saved_count += 1

        if not AUTO_SPLIT_ON_SAVE:
            return TRAIN_IMG_DIR, TRAIN_LABEL_DIR

        if self.saved_count % VAL_EVERY_N == 0:
            return VAL_IMG_DIR, VAL_LABEL_DIR

        return TRAIN_IMG_DIR, TRAIN_LABEL_DIR

    def save_labels(self, empty: bool = False) -> None:
        img_path = self.current_image_path()
        img = self.load_current_image()
        img_h, img_w = img.shape[:2]

        self.remove_old_copies(img_path.stem)

        out_img_dir, out_label_dir = self.get_split_dirs()
        out_img_path = out_img_dir / img_path.name
        out_label_path = out_label_dir / f"{img_path.stem}.txt"

        shutil.copy2(img_path, out_img_path)

        if empty:
            out_label_path.write_text("", encoding="utf-8")
            print(f"Saved EMPTY label: {out_label_path}")
            return

        lines = [self.box_to_yolo(box, img_w, img_h) for box in self.boxes]
        out_label_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved label: {out_label_path}")

    def draw_overlay(self, image):
        display = image.copy()

        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"{i+1}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if self.drawing:
            x1 = min(self.start_x, self.current_x)
            y1 = min(self.start_y, self.current_y)
            x2 = max(self.start_x, self.current_x)
            y2 = max(self.start_y, self.current_y)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 1)

        status = (
            f"Image {self.index + 1}/{len(self.image_paths)} | "
            f"Boxes: {len(self.boxes)} | "
            f"[s] save [e] empty [n] next [b] back [u] undo [q] quit"
        )
        cv2.putText(
            display,
            status,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display

    def next_image(self):
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self._load_existing_for_current()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self._load_existing_for_current()

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            image = self.load_current_image()
            display = self.draw_overlay(image)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("u"):
                if self.boxes:
                    self.boxes.pop()
            elif key == ord("n"):
                self.next_image()
            elif key == ord("b"):
                self.prev_image()
            elif key == ord("e"):
                self.save_labels(empty=True)
                self.next_image()
            elif key == ord("s"):
                if not self.boxes:
                    print("No boxes drawn. Use [e] if empty.")
                else:
                    self.save_labels(empty=False)
                    self.next_image()

        cv2.destroyAllWindows()


def main():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in RAW_DIR.iterdir() if p.suffix.lower() in exts])
    tool = LabelTool(image_paths)
    tool.run()


if __name__ == "__main__":
    main()