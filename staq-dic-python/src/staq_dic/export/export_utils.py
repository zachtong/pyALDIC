"""Shared utility functions for all export backends."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def frame_tag(i: int, n_frames: int) -> str:
    """Return a 1-based, zero-padded frame label.

    Examples:
        frame_tag(0, 10)   -> "frame_01"
        frame_tag(9, 10)   -> "frame_10"
        frame_tag(0, 100)  -> "frame_001"
    """
    width = len(str(n_frames))
    return f"frame_{i + 1:0{width}d}"


def make_prefix(image_folder: Path | None) -> str:
    """Derive a safe export filename prefix from the image folder name.

    Replaces characters forbidden in Windows filenames (and whitespace) with
    underscores so the prefix can be used on all platforms.
    """
    if image_folder is None:
        return "dic"
    stem = image_folder.name or "dic"
    return re.sub(r'[<>:"/\\|?*\s]', "_", stem)


def make_timestamp() -> str:
    """Return the current local time as a ``YYYYMMDDHHMMSS`` string."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def ensure_dir(path: Path) -> Path:
    """Create *path* (and any parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
