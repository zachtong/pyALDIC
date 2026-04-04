"""Image loading controller with natural sort and Unicode-safe I/O.

Scans a folder for supported image files, sorts them naturally,
and provides cached reading in both float64 grayscale and RGB uint8.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from staq_dic.gui.app_state import AppState

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}
)


def _natural_sort_key(name: str) -> list[str | int]:
    """Split a filename into text/numeric parts for natural ordering.

    "img_2.tif" -> ["img_", 2, ".tif"]
    "img_10.tif" -> ["img_", 10, ".tif"]
    """
    parts: list[str | int] = []
    for token in re.split(r"(\d+)", name):
        if token.isdigit():
            parts.append(int(token))
        else:
            parts.append(token.lower())
    return parts


def _read_image_raw(path: str) -> NDArray[np.uint8]:
    """Read an image using Unicode-safe I/O (np.fromfile + cv2.imdecode).

    cv2.imread fails on Windows paths containing CJK or other non-ASCII
    characters.  This approach works universally.

    Args:
        path: Absolute path to the image file.

    Returns:
        BGR uint8 image array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    buf = np.fromfile(str(file_path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


class ImageController:
    """Loads images from a folder and provides cached reading."""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._cache: dict[int, NDArray[np.float64]] = {}
        self._cache_rgb: dict[int, NDArray[np.uint8]] = {}

    def load_folder(self, folder: str) -> None:
        """Scan folder for supported image files, natural-sort, update state.

        Args:
            folder: Path to the image folder.
        """
        self._cache.clear()
        self._cache_rgb.clear()

        folder_path = Path(folder)
        if not folder_path.is_dir():
            self._state.image_folder = None
            self._state.set_image_files([])
            return

        files: list[str] = []
        for child in folder_path.iterdir():
            if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(child))

        # Natural sort by filename
        files.sort(key=lambda p: _natural_sort_key(Path(p).name))

        self._state.image_folder = folder_path
        self._state.set_image_files(files)

    def read_image(self, idx: int) -> NDArray[np.float64]:
        """Read image at index as float64 grayscale [0, 1].

        Results are cached per index.

        Args:
            idx: Frame index into state.image_files.

        Returns:
            (H, W) float64 array with values in [0.0, 1.0].

        Raises:
            IndexError: If idx is out of range.
        """
        if idx < 0 or idx >= len(self._state.image_files):
            raise IndexError(f"Image index {idx} out of range")

        if idx in self._cache:
            return self._cache[idx]

        raw = _read_image_raw(self._state.image_files[idx])

        # Convert to grayscale if needed
        if raw.ndim == 3:
            if raw.shape[2] == 4:
                gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            gray = raw

        result = gray.astype(np.float64) / 255.0
        self._cache[idx] = result
        return result

    def read_image_rgb(self, idx: int) -> NDArray[np.uint8]:
        """Read image at index as RGB uint8 for display.

        Results are cached per index.

        Args:
            idx: Frame index into state.image_files.

        Returns:
            (H, W, 3) uint8 RGB array.

        Raises:
            IndexError: If idx is out of range.
        """
        if idx < 0 or idx >= len(self._state.image_files):
            raise IndexError(f"Image index {idx} out of range")

        if idx in self._cache_rgb:
            return self._cache_rgb[idx]

        raw = _read_image_raw(self._state.image_files[idx])

        # Convert to RGB
        if raw.ndim == 2:
            rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        elif raw.shape[2] == 4:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        self._cache_rgb[idx] = rgb
        return rgb

    def image_dimensions(self, idx: int) -> tuple[int, int]:
        """Get image dimensions without full caching.

        Args:
            idx: Frame index into state.image_files.

        Returns:
            (height, width) tuple.
        """
        img = self.read_image(idx)
        return img.shape[0], img.shape[1]

    def clear_cache(self) -> None:
        """Clear the image cache."""
        self._cache.clear()
        self._cache_rgb.clear()
