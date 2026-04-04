"""ROI mask controller — boolean add/cut operations using OpenCV.

Manages a 2-D boolean mask and exposes add_rectangle, add_polygon,
add_circle, import_mask, and clear operations.  Each shape can be
applied in "add" (union) or "cut" (subtract) mode.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


class ROIController:
    """Boolean ROI mask with add/cut shape operations."""

    def __init__(self, img_shape: tuple[int, int]) -> None:
        """Initialize an empty mask.

        Args:
            img_shape: (height, width) of the image.
        """
        if len(img_shape) != 2 or img_shape[0] <= 0 or img_shape[1] <= 0:
            raise ValueError(
                f"img_shape must be (H, W) with positive dimensions, got {img_shape}"
            )
        self._shape = img_shape
        self.mask: NDArray[np.bool_] = np.zeros(img_shape, dtype=bool)

    @property
    def shape(self) -> tuple[int, int]:
        """Return (height, width) of the mask."""
        return self._shape

    def add_rectangle(
        self, x1: int, y1: int, x2: int, y2: int, mode: str = "add"
    ) -> None:
        """Add or cut a filled rectangle.

        Args:
            x1, y1: Top-left corner (column, row).
            x2, y2: Bottom-right corner (column, row).
            mode: "add" to union, "cut" to subtract.
        """
        canvas = np.zeros(self._shape, dtype=np.uint8)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, thickness=-1)
        self._apply(canvas, mode)

    def add_polygon(
        self, points: list[tuple[int, int]], mode: str = "add"
    ) -> None:
        """Add or cut a filled polygon.

        Args:
            points: List of (x, y) vertices.
            mode: "add" to union, "cut" to subtract.
        """
        if len(points) < 3:
            return
        canvas = np.zeros(self._shape, dtype=np.uint8)
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], 255)
        self._apply(canvas, mode)

    def add_circle(
        self, cx: int, cy: int, radius: int, mode: str = "add"
    ) -> None:
        """Add or cut a filled circle.

        Args:
            cx, cy: Center (column, row).
            radius: Circle radius in pixels.
            mode: "add" to union, "cut" to subtract.
        """
        if radius <= 0:
            return
        canvas = np.zeros(self._shape, dtype=np.uint8)
        cv2.circle(canvas, (cx, cy), radius, 255, thickness=-1)
        self._apply(canvas, mode)

    def import_mask(self, path: str) -> None:
        """Import an external mask image (grayscale).

        Pixels > 127 become True.  The image is resized if dimensions
        do not match.

        Args:
            path: Filesystem path to the mask image.

        Raises:
            IOError: If the file cannot be read as an image.
        """
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read mask: {path}")
        if img.shape != self._shape:
            img = cv2.resize(img, (self._shape[1], self._shape[0]))
        self.mask = img > 127

    def clear(self) -> None:
        """Reset the mask to all False."""
        self.mask = np.zeros(self._shape, dtype=bool)

    def _apply(self, canvas: NDArray[np.uint8], mode: str) -> None:
        """Apply a rasterized shape to the mask.

        Args:
            canvas: uint8 image with 255 inside the shape.
            mode: "add" or "cut".
        """
        region = canvas > 0
        if mode == "add":
            self.mask = self.mask | region
        elif mode == "cut":
            self.mask = self.mask & ~region
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Expected 'add' or 'cut'.")
