"""
launcher.py

Simple Tkinter launcher for a YOLO tooling suite.

- Creates/loads a central workspace
- Opens common workspace folders
- Launches companion tools/scripts from a single UI
- Passes YOLO_SUITE_WORKSPACE to child processes
"""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from workspace import (
    APP_NAME,
    app_root,
    bundle_root,
    ensure_workspace,
    find_first_existing,
    load_settings,
    open_in_file_manager,
    save_settings,
    set_workspace_env,
)


TOOL_TARGETS: dict[str, list[str]] = {
    "Live Detect": [
        "live_detect_two_stage_tensorrt_filters_trimmed.py",
        "live_detect_two_stage_tensorrt_filters_fixed.py",
        "live_detect_two_stage_tensorrt_filters.py",
        "live_detect_two_stage_tensorrt.py",
        "live_detect.py",
    ],
    "Train Model": [
        "train.py",
        "yolo_train.py",
        "train_model.py",
    ],
    "Review Dataset": [
        "review_dataset.py",
        "review.py",
        "dataset_review.py",
    ],
    "Export TensorRT": [
        "export_engine.py",
        "export_tensorrt.py",
        "yolo_export.py",
    ],
    "Label / Annotate": [
        "label_tool.py",
        "annotate.py",
        "dataset_tool.py",
    ],
}


class LauncherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_NAME} Launcher")
        self.geometry("860x620")
        self.minsize(760, 540)

        self.settings = load_settings()
        self.paths = ensure_workspace(self.settings.get("workspace_root"))

        self.workspace_var = tk.StringVar(value=str(self.paths.root))
        self.status_var = tk.StringVar(value="Ready.")
        self.last_tool_var = tk.StringVar(value=self.settings.get("last_tool", ""))

        self.tool_buttons: dict[str, ttk.Button] = {}
        self.tool_paths: dict[str, Path | None] = {}

        self._build_ui()
        self.refresh_tool_paths()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        title = ttk.Label(root, text=f"{APP_NAME} Launcher", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")

        subtitle = ttk.Label(
            root,
            text="One place to manage workspace folders and launch your YOLO tools.",
        )
        subtitle.pack(anchor="w", pady=(0, 10))

        ws_frame = ttk.LabelFrame(root, text="Workspace", padding=10)
        ws_frame.pack(fill="x", pady=(0, 12))

        ttk.Label(ws_frame, text="Root").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(ws_frame, textvariable=self.workspace_var)
        entry.grid(row=0, column=1, sticky="ew", padx=8)
        ws_frame.columnconfigure(1, weight=1)

        ttk.Button(ws_frame, text="Browse…", command=self.choose_workspace).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(ws_frame, text="Apply", command=self.apply_workspace).grid(row=0, column=3)
        ttk.Button(ws_frame, text="Open Root", command=lambda: open_in_file_manager(self.paths.root)).grid(row=1, column=1, sticky="w", pady=(8, 0))

        folders = ttk.LabelFrame(root, text="Workspace Folders", padding=10)
        folders.pack(fill="x", pady=(0, 12))

        folder_buttons = [
            ("Datasets", "datasets"),
            ("Projects", "projects"),
            ("Models", "models"),
            ("Engines", "engines"),
            ("Runs", "runs"),
            ("Exports", "exports"),
            ("Captures", "captures"),
            ("Configs", "configs"),
            ("Logs", "logs"),
        ]

        for idx, (label, attr) in enumerate(folder_buttons):
            btn = ttk.Button(folders, text=label, command=lambda a=attr: open_in_file_manager(getattr(self.paths, a)))
            btn.grid(row=idx // 3, column=idx % 3, padx=6, pady=6, sticky="ew")

        for i in range(3):
            folders.columnconfigure(i, weight=1)

        tools = ttk.LabelFrame(root, text="Tools", padding=10)
        tools.pack(fill="both", expand=True, pady=(0, 12))

        grid = ttk.Frame(tools)
        grid.pack(fill="x")
        for i in range(2):
            grid.columnconfigure(i, weight=1)

        for idx, tool_name in enumerate(TOOL_TARGETS.keys()):
            btn = ttk.Button(grid, text=tool_name, command=lambda n=tool_name: self.launch_tool(n))
            btn.grid(row=idx // 2, column=idx % 2, sticky="ew", padx=6, pady=6)
            self.tool_buttons[tool_name] = btn

        actions = ttk.Frame(tools)
        actions.pack(fill="x", pady=(8, 8))
        ttk.Button(actions, text="Refresh Tool Scan", command=self.refresh_tool_paths).pack(side="left")
        ttk.Button(actions, text="Open App Folder", command=lambda: open_in_file_manager(app_root())).pack(side="left", padx=8)
        ttk.Button(actions, text="Open Bundle Folder", command=lambda: open_in_file_manager(bundle_root())).pack(side="left")

        info = ttk.LabelFrame(tools, text="Detected Targets", padding=10)
        info.pack(fill="both", expand=True)

        self.info_text = tk.Text(info, height=14, wrap="word")
        self.info_text.pack(fill="both", expand=True)
        self.info_text.configure(state="disabled")

        status = ttk.Frame(root)
        status.pack(fill="x")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        ttk.Label(status, textvariable=self.last_tool_var).pack(side="right")

    def choose_workspace(self) -> None:
        folder = filedialog.askdirectory(
            title="Choose workspace folder",
            initialdir=self.workspace_var.get() or str(Path.home()),
        )
        if folder:
            self.workspace_var.set(folder)

    def apply_workspace(self) -> None:
        try:
            self.paths = ensure_workspace(self.workspace_var.get())
            self.workspace_var.set(str(self.paths.root))
            self.settings["workspace_root"] = str(self.paths.root)
            save_settings(self.settings, self.paths.root)
            self.status_var.set(f"Workspace set to {self.paths.root}")
        except Exception as exc:
            messagebox.showerror("Workspace Error", str(exc))

    def refresh_tool_paths(self) -> None:
        search_roots = [
            app_root(),
            bundle_root(),
            Path.cwd(),
        ]

        self.tool_paths = {}
        lines: list[str] = [f"Workspace: {self.paths.root}", ""]
        for tool_name, targets in TOOL_TARGETS.items():
            candidates: list[Path] = []
            for root in search_roots:
                for target in targets:
                    candidates.append(root / target)

            match = find_first_existing(candidates)
            self.tool_paths[tool_name] = match

            if match is None:
                self.tool_buttons[tool_name].configure(state="disabled")
                lines.append(f"[missing] {tool_name}")
            else:
                self.tool_buttons[tool_name].configure(state="normal")
                lines.append(f"[ok] {tool_name} -> {match}")

        self._set_info_text("\n".join(lines))
        self.status_var.set("Tool scan refreshed.")

    def launch_tool(self, tool_name: str) -> None:
        target = self.tool_paths.get(tool_name)
        if target is None:
            messagebox.showwarning("Missing Tool", f"No script or executable was found for {tool_name}.")
            return

        self.apply_workspace()
        env = set_workspace_env(None, self.paths.root)

        try:
            if target.suffix.lower() == ".py":
                cmd = [sys.executable, str(target)]
            else:
                cmd = [str(target)]

            subprocess.Popen(
                cmd,
                cwd=str(target.parent),
                env=env,
            )
            self.settings["last_tool"] = tool_name
            save_settings(self.settings, self.paths.root)
            self.last_tool_var.set(f"Last launched: {tool_name}")
            self.status_var.set(f"Launched {tool_name}")
        except Exception as exc:
            messagebox.showerror("Launch Failed", f"Could not launch {tool_name}.\n\n{exc}")

    def _set_info_text(self, text: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")


def main() -> None:
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
