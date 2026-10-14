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
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from numpy.typing import NDArray

from al_dic.gui.app_state import AppState

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg", ".jp2", ".webp"}
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


def _read_image_raw(path: str) -> NDArray:
    """Read an image using Unicode-safe I/O (np.fromfile + cv2.imdecode).

    cv2.imread fails on Windows paths containing CJK or other non-ASCII
    characters.  This approach works universally.

    The image is read with IMREAD_UNCHANGED so the original bit depth
    (uint8, uint16, float32) and channel count are preserved.

    Args:
        path: Absolute path to the image file.

    Returns:
        Image array in its native dtype (may be uint8, uint16, or float32).

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


def _read_image_size(path: str) -> tuple[int, int]:
    """Return (H, W) for *path* without decoding the pixels.

    Pillow parses the header and stops, which is what the folder scan needs:
    reading 3000 frames to compare their dimensions used to decode every one
    of them in full, costing minutes (far worse when the images live on a
    network or cloud-synced drive, where a full read pulls the whole file
    down). Falls back to a full decode for anything Pillow cannot parse, so
    no supported format is lost.
    """
    try:
        with Image.open(path) as im:
            w, h = im.size
        return h, w
    except Exception:
        return _read_image_raw(path).shape[:2]


def _normalize_to_float64(img: NDArray) -> NDArray[np.float64]:
    """Normalize an image of any bit depth to float64 in [0.0, 1.0].

    Handles uint8 (0-255), uint16 (0-65535), uint32 (0-4294967295),
    and floating-point images (assumed already in [0, 1] or clamped).
    """
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0
    if img.dtype == np.uint32:
        return img.astype(np.float64) / 4294967295.0
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img.astype(np.float64), 0.0, 1.0)
    # Fallback: assume unsigned int, scale by max possible value
    info = np.iinfo(img.dtype)
    return img.astype(np.float64) / info.max


def _to_display_uint8(img: NDArray) -> NDArray[np.uint8]:
    """Convert an image of any bit depth to uint8 for display.

    Applies the same logic as _normalize_to_float64 but outputs [0, 255].
    """
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    if img.dtype == np.uint32:
        return (img / 16777216).astype(np.uint8)
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    info = np.iinfo(img.dtype)
    return (img.astype(np.float64) / info.max * 255).astype(np.uint8)


def _to_gray_uint8(img: NDArray) -> NDArray[np.uint8]:
    """Convert any-depth image to single-channel uint8 grayscale.

    Handles multi-channel (BGR/BGRA) and multi-depth (uint16, float) inputs.
    """
    # Convert to single channel first (if multi-channel)
    if img.ndim == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return _to_display_uint8(gray)


def _decode_grayscale_float64(path: str) -> NDArray[np.float64]:
    """Decode an image file to (H, W) float64 grayscale in [0, 1].

    Single source of truth for the compute-path decode: used by both
    ImageController.read_image and the streaming compute provider, so the
    two produce byte-identical frames.
    """
    raw = _read_image_raw(path)
    if raw.ndim == 3:
        if raw.shape[2] == 4:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw
    return _normalize_to_float64(gray)


_RGB_CACHE_SIZE = 8  # max RGB preview frames kept resident (bounded LRU)


class ImageController:
    """Loads images from a folder and provides cached reading."""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._cache: dict[int, NDArray[np.float64]] = {}
        # Bounded LRU (OrderedDict): scrubbing a long sequence must not
        # pin every RGB preview frame in RAM; misses re-decode from disk.
        self._cache_rgb: "OrderedDict[int, NDArray[np.uint8]]" = OrderedDict()
        self._natural_sort: bool = False
        self._raw_files: list[str] = []  # unsorted file list
        # Invalidate caches whenever the image file list changes. Each
        # ImageController instance (MainWindow + StrainWindow each own
        # one) caches by integer index, so if the underlying files
        # change without clearing, index 0 could still serve the
        # previous sequence's frame 0. MainWindow's controller clears
        # on load_folder() explicitly; StrainWindow's controller
        # only saw the new files via state.image_files, and without
        # this hook would keep serving stale frames.
        state.images_changed.connect(self._on_images_changed)

    def _on_images_changed(self) -> None:
        self._cache.clear()
        self._cache_rgb.clear()

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

    def set_files(self, files: list[str]) -> None:
        """Replace the controller's file list without rescanning the folder.

        Callers that mutate ``state.image_files`` directly (deleting frames
        from the image list) must go through this, or the controller keeps
        the original list and a later sort toggle resurrects the deleted
        frames -- and rescans every file again.
        """
        self._raw_files = list(files)
        self._cache.clear()
        self._cache_rgb.clear()

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
            ref_shape = _read_image_size(files[0])  # (H, W)
            mismatched: list[str] = []
            for f in files[1:]:
                shape = _read_image_size(f)
                if shape != ref_shape:
                    mismatched.append(
                        f"{Path(f).name} ({shape[1]}x{shape[0]})"
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

        result = _decode_grayscale_float64(self._state.image_files[idx])
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
            self._cache_rgb.move_to_end(idx)  # mark most-recently-used
            return self._cache_rgb[idx]

        raw = _read_image_raw(self._state.image_files[idx])

        # Convert non-uint8 to uint8 first (16-bit → 8-bit, etc.)
        if raw.dtype != np.uint8:
            if raw.ndim == 3:
                # Multi-channel: convert each channel
                raw = _to_display_uint8(raw)
            else:
                raw = _to_display_uint8(raw)

        # Convert to RGB
        if raw.ndim == 2:
            rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        elif raw.shape[2] == 4:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        self._cache_rgb[idx] = rgb
        if len(self._cache_rgb) > _RGB_CACHE_SIZE:
            self._cache_rgb.popitem(last=False)  # evict least-recently-used
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
