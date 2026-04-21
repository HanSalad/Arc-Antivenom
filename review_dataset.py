from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageTk, ImageDraw, ImageFont

try:
    from send2trash import send2trash
    HAS_TRASH = True
except Exception:
    HAS_TRASH = False


# =========================
# CONFIG
# =========================
DATASET_ROOT = Path("screen_auto_labels")
#DATASET_ROOT = Path("screenshots")
# Example alternate:
# DATASET_ROOT = Path("auto_person_labels")

USE_TRASH = True
WINDOW_W = 1500
WINDOW_H = 950

BOX_COLORS = {
    0: "red",
    1: "blue",
    2: "lime",
    3: "orange",
    4: "cyan",
    5: "green",
    6: "purple",
    
}
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

BOX_WIDTH = 3
SELECTED_BOX_COLOR = "yellow"
TEXT_BG = "black"
TEXT_FG = "white"
# =========================


@dataclass
class YoloBox:
    cls_id: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Sample:
    image_path: Path
    label_path: Optional[Path]
    split: str


def find_label_for_image(image_path: Path, split: str) -> Optional[Path]:
    label_dir = DATASET_ROOT / "labels" / split
    candidate = label_dir / f"{image_path.stem}.txt"
    return candidate if candidate.exists() else None


def collect_samples() -> List[Sample]:
    samples: List[Sample] = []

    for split in ("train", "val"):
        image_dir = DATASET_ROOT / "images" / split
        if not image_dir.exists():
            continue

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            label_path = find_label_for_image(image_path, split)
            samples.append(Sample(image_path=image_path, label_path=label_path, split=split))

    return samples


def load_yolo_boxes(label_path: Optional[Path], img_w: int, img_h: int) -> List[YoloBox]:
    boxes: List[YoloBox] = []
    if label_path is None or not label_path.exists():
        return boxes

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            cls_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except ValueError:
            continue

        box_w = bw * img_w
        box_h = bh * img_h
        center_x = cx * img_w
        center_y = cy * img_h

        x1 = int(round(center_x - box_w / 2))
        y1 = int(round(center_y - box_h / 2))
        x2 = int(round(center_x + box_w / 2))
        y2 = int(round(center_y + box_h / 2))

        boxes.append(YoloBox(cls_id, x1, y1, x2, y2))

    return boxes


def box_to_yolo_line(box: YoloBox, img_w: int, img_h: int) -> str:
    box_w = box.x2 - box.x1
    box_h = box.y2 - box.y1
    center_x = box.x1 + box_w / 2
    center_y = box.y1 + box_h / 2

    return (
        f"{box.cls_id} "
        f"{center_x / img_w:.6f} "
        f"{center_y / img_h:.6f} "
        f"{box_w / img_w:.6f} "
        f"{box_h / img_h:.6f}"
    )


def fit_image_size(img_w: int, img_h: int, max_w: int, max_h: int):
    scale = min(max_w / img_w, max_h / img_h, 1.0)
    return int(img_w * scale), int(img_h * scale), scale


class DatasetReviewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Dataset Reviewer / Editor")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")

        self.samples = collect_samples()
        self.index = 0
        self.tk_img = None

        self.current_boxes: List[YoloBox] = []
        self.selected_box_index: Optional[int] = None
        self.boxes_dirty = False

        self.class_ids = sorted(CLASS_NAMES.keys())
        self.current_class_id = 1 if 1 in self.class_ids else self.class_ids[0]

        self.img_w = 1
        self.img_h = 1
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0

        self.drawing = False
        self.draw_start_x = 0
        self.draw_start_y = 0
        self.draw_current_x = 0
        self.draw_current_y = 0

        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=8)

        self.info_var = tk.StringVar(value="")
        self.info_label = tk.Label(top, textvariable=self.info_var, anchor="w", justify="left", font=("Segoe UI", 10))
        self.info_label.pack(fill="x")

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=8, pady=4)

        tk.Button(controls, text="Prev [A / ←]", command=self.prev_sample, width=15).pack(side="left", padx=3)
        tk.Button(controls, text="Next [D / →]", command=self.next_sample, width=15).pack(side="left", padx=3)
        tk.Button(controls, text="Save Labels [S]", command=self.save_labels, width=15).pack(side="left", padx=3)
        tk.Button(controls, text="Delete Pair [X]", command=self.delete_current_pair, width=15).pack(side="left", padx=3)
        tk.Button(controls, text="Refresh [R]", command=self.refresh_samples, width=15).pack(side="left", padx=3)

        class_frame = tk.Frame(root)
        class_frame.pack(fill="x", padx=8, pady=4)

        self.class_var = tk.StringVar(value=f"Current add class: {self.current_class_id} ({CLASS_NAMES.get(self.current_class_id, 'unknown')})")
        tk.Label(class_frame, textvariable=self.class_var, font=("Segoe UI", 10, "bold")).pack(side="left", padx=4)

        self.canvas = tk.Canvas(root, bg="gray15", highlightthickness=0, cursor="cross")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_down)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_up)

        root.bind("<Left>", lambda e: self.prev_sample())
        root.bind("<Right>", lambda e: self.next_sample())
        root.bind("<a>", lambda e: self.prev_sample())
        root.bind("<d>", lambda e: self.next_sample())
        root.bind("<x>", lambda e: self.delete_current_pair())
        root.bind("<r>", lambda e: self.refresh_samples())
        root.bind("<s>", lambda e: self.save_labels())
        root.bind("<Delete>", lambda e: self.delete_selected_box())
        root.bind("<BackSpace>", lambda e: self.delete_selected_box())
        root.bind("[", lambda e: self.prev_class())
        root.bind("]", lambda e: self.next_class())
        root.bind(",", lambda e: self.prev_class())
        root.bind(".", lambda e: self.next_class())

        root.bind("1", lambda e: self.set_current_class(0))
        root.bind("2", lambda e: self.set_current_class(1))
        root.bind("3", lambda e: self.set_current_class(2))
        root.bind("4", lambda e: self.set_current_class(3))
        root.bind("5", lambda e: self.set_current_class(4))
        root.bind("6", lambda e: self.set_current_class(5))
        root.bind("7", lambda e: self.set_current_class(6))
        root.bind("8", lambda e: self.set_current_class(7))
        root.bind("9", lambda e: self.set_current_class(8))

        self.show_current()

    def load_current_sample_boxes(self):
        sample = self.current_sample()
        if sample is None:
            self.current_boxes = []
            self.selected_box_index = None
            self.boxes_dirty = False
            return

        try:
            img = Image.open(sample.image_path).convert("RGB")
            self.img_w, self.img_h = img.size
        except Exception:
            self.current_boxes = []
            self.selected_box_index = None
            self.boxes_dirty = False
            return

        self.current_boxes = load_yolo_boxes(sample.label_path, self.img_w, self.img_h)
        self.selected_box_index = None
        self.boxes_dirty = False
    def set_current_class(self, cls_id: int):
        self.current_class_id = cls_id
        self.class_var.set(f"Current add class: {self.current_class_id} ({CLASS_NAMES.get(self.current_class_id, 'unknown')})")
        self.show_current()
    
    def next_class(self):
        idx = self.class_ids.index(self.current_class_id)
        idx = (idx + 1) % len(self.class_ids)
        self.set_current_class(self.class_ids[idx])

    def prev_class(self):
        idx = self.class_ids.index(self.current_class_id)
        idx = (idx - 1) % len(self.class_ids)
        self.set_current_class(self.class_ids[idx])

    def refresh_samples(self):
        current_path = None
        if self.samples and 0 <= self.index < len(self.samples):
            current_path = self.samples[self.index].image_path

        self.samples = collect_samples()

        if not self.samples:
            self.index = 0
            self.current_boxes = []
            self.canvas.delete("all")
            self.info_var.set("No samples found.")
            return

        if current_path is not None:
            for i, s in enumerate(self.samples):
                if s.image_path == current_path:
                    self.index = i
                    break
            else:
                self.index = min(self.index, len(self.samples) - 1)
        else:
            self.index = 0

        self.load_current_sample_boxes()
        self.show_current()

    def move_to_trash_or_delete(self, path: Path):
        if not path.exists():
            return
        if USE_TRASH and HAS_TRASH:
            send2trash(str(path))
        else:
            path.unlink(missing_ok=True)

    def current_sample(self) -> Optional[Sample]:
        if not self.samples:
            return None
        return self.samples[self.index]

    def ensure_label_path(self, sample: Sample) -> Path:
        if sample.label_path is not None:
            return sample.label_path
        label_dir = DATASET_ROOT / "labels" / sample.split
        label_dir.mkdir(parents=True, exist_ok=True)
        sample.label_path = label_dir / f"{sample.image_path.stem}.txt"
        return sample.label_path

    def delete_current_pair(self):
        sample = self.current_sample()
        if sample is None:
            return

        msg = f"Delete image and label?\n\n{sample.image_path.name}"
        if sample.label_path:
            msg += f"\n{sample.label_path.name}"

        if not messagebox.askyesno("Delete Pair", msg):
            return

        self.move_to_trash_or_delete(sample.image_path)
        if sample.label_path and sample.label_path.exists():
            self.move_to_trash_or_delete(sample.label_path)

        del self.samples[self.index]
        if self.index >= len(self.samples):
            self.index = max(0, len(self.samples) - 1)

        self.show_current()

    def prev_sample(self):
        if not self.samples:
            return
        self.index = (self.index - 1) % len(self.samples)
        self.load_current_sample_boxes()
        self.show_current()

    def next_sample(self):
        if not self.samples:
            return
        self.index = (self.index + 1) % len(self.samples)
        self.load_current_sample_boxes()
        self.show_current()

    def image_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        cx = int(x * self.display_scale + self.display_offset_x)
        cy = int(y * self.display_scale + self.display_offset_y)
        return cx, cy

    def canvas_to_image(self, x: int, y: int) -> tuple[int, int]:
        ix = int(round((x - self.display_offset_x) / self.display_scale))
        iy = int(round((y - self.display_offset_y) / self.display_scale))
        ix = max(0, min(self.img_w - 1, ix))
        iy = max(0, min(self.img_h - 1, iy))
        return ix, iy

    def point_in_box(self, x: int, y: int, box: YoloBox) -> bool:
        return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2

    def find_box_at_point(self, x: int, y: int) -> Optional[int]:
        for i in range(len(self.current_boxes) - 1, -1, -1):
            if self.point_in_box(x, y, self.current_boxes[i]):
                return i
        return None

    def on_left_click(self, event):
        if not self.samples:
            return
        ix, iy = self.canvas_to_image(event.x, event.y)
        self.selected_box_index = self.find_box_at_point(ix, iy)
        self.show_current()

    def on_right_down(self, event):
        if not self.samples:
            return
        self.drawing = True
        self.draw_start_x, self.draw_start_y = self.canvas_to_image(event.x, event.y)
        self.draw_current_x, self.draw_current_y = self.draw_start_x, self.draw_start_y
        self.show_current()

    def on_right_drag(self, event):
        if not self.drawing:
            return
        self.draw_current_x, self.draw_current_y = self.canvas_to_image(event.x, event.y)
        self.show_current()

    def on_right_up(self, event):
        if not self.drawing:
            return

        self.drawing = False
        self.draw_current_x, self.draw_current_y = self.canvas_to_image(event.x, event.y)

        x1 = min(self.draw_start_x, self.draw_current_x)
        y1 = min(self.draw_start_y, self.draw_current_y)
        x2 = max(self.draw_start_x, self.draw_current_x)
        y2 = max(self.draw_start_y, self.draw_current_y)

        if x2 - x1 > 3 and y2 - y1 > 3:
            self.current_boxes.append(YoloBox(self.current_class_id, x1, y1, x2, y2))
            self.selected_box_index = len(self.current_boxes) - 1
            self.boxes_dirty = True

        self.show_current()

    def delete_selected_box(self):
        if self.selected_box_index is None:
            return
        if 0 <= self.selected_box_index < len(self.current_boxes):
            del self.current_boxes[self.selected_box_index]
            self.selected_box_index = None
            self.boxes_dirty = True
            self.show_current()

    def save_labels(self):
        sample = self.current_sample()
        if sample is None:
            return

        label_path = self.ensure_label_path(sample)
        lines = [box_to_yolo_line(box, self.img_w, self.img_h) for box in self.current_boxes]
        label_path.write_text("\n".join(lines), encoding="utf-8")
        self.boxes_dirty = False
        self.show_current()
        messagebox.showinfo("Saved", f"Saved labels to:\n{label_path}")

    def show_current(self):
        self.canvas.delete("all")

        sample = self.current_sample()
        if sample is None:
            self.info_var.set("No samples found.")
            return

        try:
            img = Image.open(sample.image_path).convert("RGB")
        except Exception as e:
            self.info_var.set(f"Failed to open image: {sample.image_path}\n{e}")
            return

        self.img_w, self.img_h = img.size
        

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for i, box in enumerate(self.current_boxes):
            color = BOX_COLORS.get(box.cls_id, "white")
            width = BOX_WIDTH

            if self.selected_box_index == i:
                color = SELECTED_BOX_COLOR
                width = BOX_WIDTH + 1

            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=width)

            label_text = f"{box.cls_id}:{CLASS_NAMES.get(box.cls_id, 'cls')} #{i+1}"
            tx = box.x1
            ty = max(0, box.y1 - 16)
            tw = 120
            draw.rectangle((tx, ty, tx + tw, ty + 16), fill=TEXT_BG)
            draw.text((tx + 2, ty + 1), label_text, fill=color, font=font)

        if self.drawing:
            x1 = min(self.draw_start_x, self.draw_current_x)
            y1 = min(self.draw_start_y, self.draw_current_y)
            x2 = max(self.draw_start_x, self.draw_current_x)
            y2 = max(self.draw_start_y, self.draw_current_y)
            draw.rectangle((x1, y1, x2, y2), outline="cyan", width=2)

        canvas_w = max(self.canvas.winfo_width(), 200)
        canvas_h = max(self.canvas.winfo_height(), 200)

        new_w, new_h, scale = fit_image_size(self.img_w, self.img_h, canvas_w - 20, canvas_h - 20)
        self.display_scale = scale
        self.display_offset_x = (canvas_w - new_w) // 2
        self.display_offset_y = (canvas_h - new_h) // 2

        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_resized)

        self.canvas.create_image(self.display_offset_x, self.display_offset_y, image=self.tk_img, anchor="nw")

        label_name = sample.label_path.name if sample.label_path else "(missing)"
        selected_text = (
            f"selected={self.selected_box_index + 1}"
            if self.selected_box_index is not None and self.selected_box_index < len(self.current_boxes)
            else "selected=none"
        )

        self.info_var.set(
            f"[{self.index + 1}/{len(self.samples)}] split={sample.split} "
            f"image={sample.image_path.name} label={label_name} "
            f"boxes={len(self.current_boxes)} size={self.img_w}x{self.img_h} "
            f"{selected_text} add_class={self.current_class_id}:{CLASS_NAMES.get(self.current_class_id, 'unknown')} | "
            f"Left click select | Right drag add | Del remove | [ ] or , . scroll class | 1-9 direct class | S save"
        )


def main():
    root = tk.Tk()
    app = DatasetReviewer(root)
    app.load_current_sample_boxes()
    app.show_current()

    def on_resize(event):
            pass

    root.bind("<Configure>", on_resize)
    root.mainloop()


if __name__ == "__main__":
    main()