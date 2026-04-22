"""
workspace.py

Central workspace/path manager for a YOLO tooling suite.

Use this module everywhere instead of hardcoding paths.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
from typing import Any


APP_NAME = "Salad's Sauce"
WORKSPACE_ENV_VAR = "ARC_ANTI_VENOM"


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    datasets: Path
    projects: Path
    models: Path
    engines: Path
    runs: Path
    exports: Path
    captures: Path
    configs: Path
    logs: Path
    temp: Path

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


def bundle_root() -> Path:
    """
    Folder where bundled app resources live when packaged,
    or the current file folder during normal development.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def app_root() -> Path:
    """
    Folder containing the running program.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def default_workspace_root(app_name: str = APP_NAME) -> Path:
    """
    Reasonable per-user default workspace location.
    """
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / app_name

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / app_name

    return Path.home() / f".{app_name.lower()}"


def resolve_workspace_root(explicit_root: str | Path | None = None) -> Path:
    """
    Priority:
    1. explicit_root argument
    2. YOLO_SUITE_WORKSPACE env var
    3. default per-user location
    """
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    env_root = os.environ.get(WORKSPACE_ENV_VAR)
    if env_root:
        return Path(env_root).expanduser().resolve()

    return default_workspace_root().resolve()


def build_workspace_paths(root: str | Path | None = None) -> WorkspacePaths:
    root_path = resolve_workspace_root(root)
    return WorkspacePaths(
        root=root_path,
        datasets=root_path / "datasets",
        projects=root_path / "projects",
        models=root_path / "models",
        engines=root_path / "engines",
        runs=root_path / "runs",
        exports=root_path / "exports",
        captures=root_path / "captures",
        configs=root_path / "configs",
        logs=root_path / "logs",
        temp=root_path / "temp",
    )


def ensure_workspace(root: str | Path | None = None) -> WorkspacePaths:
    paths = build_workspace_paths(root)
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def launcher_settings_path(root: str | Path | None = None) -> Path:
    return ensure_workspace(root).configs / "launcher_settings.json"


def default_settings(root: str | Path | None = None) -> dict[str, Any]:
    paths = ensure_workspace(root)
    return {
        "workspace_root": str(paths.root),
        "last_tool": "",
        "prefer_tensorrt": True,
        "stage1_imgsz": 320,
        "stage2_imgsz": 640,
        "filters_enabled": True,
        "gamma": 1.0,
        "brightness": 0,
        "contrast": 1.0,
    }


def load_settings(root: str | Path | None = None) -> dict[str, Any]:
    path = launcher_settings_path(root)
    if not path.exists():
        return default_settings(root)

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_settings(root)


def save_settings(data: dict[str, Any], root: str | Path | None = None) -> Path:
    settings_root = root
    if settings_root is None:
        settings_root = data.get("workspace_root") or None

    path = launcher_settings_path(settings_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def set_workspace_env(env: dict[str, str] | None, root: str | Path | None = None) -> dict[str, str]:
    out = dict(env or os.environ)
    out[WORKSPACE_ENV_VAR] = str(resolve_workspace_root(root))
    return out


def open_in_file_manager(path: str | Path) -> None:
    target = str(Path(path).resolve())

    system = platform.system()
    if system == "Windows":
        os.startfile(target)  # type: ignore[attr-defined]
        return
    if system == "Darwin":
        subprocess.Popen(["open", target])
        return
    subprocess.Popen(["xdg-open", target])


def find_first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
