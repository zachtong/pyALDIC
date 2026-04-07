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

    def stroke_segment(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        radius: int,
        mode: str = "add",
    ) -> None:
        """Paint or erase a thick line segment (freehand brush stroke).

        Used by the ``+ Refine`` brush sub-tool to paint material-point
        refinement zones with a draggable cursor.  Each call rasterizes
        a single segment between consecutive mouse positions.

        Args:
            x1, y1: Start of the segment (column, row).
            x2, y2: End of the segment (column, row). When equal to
                start, the stroke degenerates to a single filled disc.
            radius: Brush radius in pixels (>= 1).
            mode: ``"add"`` to union into the mask, ``"cut"`` to subtract.
        """
        if radius <= 0:
            return
        canvas = np.zeros(self._shape, dtype=np.uint8)
        if x1 == x2 and y1 == y2:
            cv2.circle(canvas, (x1, y1), radius, 255, thickness=-1)
        else:
            cv2.line(
                canvas,
                (x1, y1),
                (x2, y2),
                255,
                thickness=2 * radius,
                lineType=cv2.LINE_8,
            )
        self._apply(canvas, mode)

    def import_mask(self, path: str) -> None:
        """Import an external mask image.

        Supports all bit depths (uint8, uint16, float) and common
        formats (tif, png, bmp, jpg, jp2, webp).  Pixels brighter
        than 50% become True.  The image is resized if dimensions
        do not match.

        Args:
            path: Filesystem path to the mask image.

        Raises:
            IOError: If the file cannot be read as an image.
        """
        from staq_dic.io.io_utils import read_mask_as_bool

        self.mask = read_mask_as_bool(path, target_shape=self._shape)

    def clear(self) -> None:
        """Reset the mask to all False."""
        self.mask = np.zeros(self._shape, dtype=bool)

    def invert(self) -> None:
        """Invert the mask (True <-> False)."""
        self.mask = ~self.mask

    def save_mask(self, path: str) -> None:
        """Save the current mask as a grayscale PNG (255=True, 0=False).

        Args:
            path: Filesystem path to write the image.
        """
        img = (self.mask.astype(np.uint8)) * 255
        success, buf = cv2.imencode(".png", img)
        if not success:
            raise IOError(f"Failed to encode mask to PNG: {path}")
        buf.tofile(path)

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
