"""Image loading controller with dual sort modes and Unicode-safe I/O.

Scans a folder for supported image files, sorts them (lexicographic
by default, or natural sort), and provides cached reading in both
float64 grayscale and RGB uint8.

Sort modes:
- Lexicographic (default): plain string sort — works correctly for
  zero-padded names like image00001, image00002, ..., image00010.
- Natural sort: treats embedded numbers as integers — image1, image2,
  ..., image9, image10 (instead of image1, image10, image11, ...).
"""

from __future__ import annotations

import re
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


def _lexicographic_sort_key(name: str) -> str:
    """Case-insensitive lexicographic key."""
    return name.lower()


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
        self._natural_sort: bool = False
        self._raw_files: list[str] = []  # unsorted file list

    @property
    def natural_sort(self) -> bool:
        """Whether natural sort is enabled."""
        return self._natural_sort

    def set_natural_sort(self, enabled: bool) -> None:
        """Change sort mode and re-sort the current file list."""
        if enabled == self._natural_sort:
            return
        self._natural_sort = enabled
        if self._raw_files:
            self._sort_and_update()

    def load_folder(self, folder: str) -> None:
        """Scan folder for supported image files, sort, update state.

        Args:
            folder: Path to the image folder.
        """
        self._cache.clear()
        self._cache_rgb.clear()

        folder_path = Path(folder)
        if not folder_path.is_dir():
            self._state.image_folder = None
            self._raw_files = []
            self._state.set_image_files([])
            return

        files: list[str] = []
        for child in folder_path.iterdir():
            if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(child))

        self._raw_files = files
        self._state.image_folder = folder_path

        # Sort and validate
        self._sort_and_update()

    def _sort_and_update(self) -> None:
        """Sort _raw_files with current mode, validate sizes, update state."""
        self._cache.clear()
        self._cache_rgb.clear()

        files = list(self._raw_files)
        if self._natural_sort:
            files.sort(key=lambda p: _natural_sort_key(Path(p).name))
        else:
            files.sort(key=lambda p: _lexicographic_sort_key(Path(p).name))

        self._state.set_image_files(files)

        # Validate image dimensions (first vs all others)
        if len(files) >= 2:
            self._validate_dimensions(files)

    def _validate_dimensions(self, files: list[str]) -> None:
        """Check that all images have the same dimensions as the first."""
        try:
            ref = _read_image_raw(files[0])
            ref_shape = ref.shape[:2]  # (H, W)
            mismatched: list[str] = []
            for f in files[1:]:
                img = _read_image_raw(f)
                if img.shape[:2] != ref_shape:
                    mismatched.append(
                        f"{Path(f).name} ({img.shape[1]}x{img.shape[0]})"
                    )
            if mismatched:
                ref_dim = f"{ref_shape[1]}x{ref_shape[0]}"
                self._state.log_message.emit(
                    f"Image size mismatch! Reference is {ref_dim}, "
                    f"but {len(mismatched)} image(s) differ: "
                    + ", ".join(mismatched[:5]),
                    "error",
                )
        except Exception:
            pass  # validation is best-effort

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
