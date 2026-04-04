"""Visualization controller -- two-level cache for field rendering.

Tier 1 (interp cache): scatter_to_grid output arrays.
    Key: (frame_idx, field_name)
    Invalidated: when results change.
    Survives: colormap/range changes.

Tier 2 (pixmap cache): colored QPixmap ready for display.
    Key: (frame_idx, field_name, cmap, vmin, vmax)
    Invalidated: when colormap or range changes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QImage, QPixmap
from matplotlib import colormaps

from staq_dic.utils.interpolation import FieldInterpolator, scatter_to_grid


def apply_colormap(
    data: NDArray[np.float64],
    vmin: float,
    vmax: float,
    cmap: str = "jet",
) -> NDArray[np.uint8]:
    """Apply matplotlib colormap to 2D float array -> RGBA uint8.

    NaN pixels get alpha=0 (transparent).
    """
    if vmax <= vmin:
        vmax = vmin + 1e-10

    normalized = (data - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    cm = colormaps[cmap]
    rgba = (cm(normalized) * 255).astype(np.uint8)

    # NaN pixels -> transparent
    nan_mask = np.isnan(data)
    rgba[nan_mask] = 0

    return rgba


class VizController:
    """Manages field visualization with two-level caching."""

    def __init__(self) -> None:
        # Tier 1: interpolation results  {(frame, field) -> (data, x_grid, y_grid)}
        self._interp_cache: dict[tuple, tuple] = {}
        # Tier 2: colored pixmaps  {(frame, field, cmap, vmin, vmax) -> QPixmap}
        self._pixmap_cache: dict[tuple, QPixmap] = {}
        self._interpolator: FieldInterpolator | None = None

    def clear_all(self) -> None:
        """Clear both cache tiers (results changed)."""
        self._interp_cache.clear()
        self._pixmap_cache.clear()
        self._interpolator = None

    def clear_pixmap_cache(self) -> None:
        """Clear Tier 2 only (colormap/range changed)."""
        self._pixmap_cache.clear()

    def store_interp_result(
        self,
        key: tuple,
        data: NDArray[np.float64],
        x_grid: NDArray[np.float64] | None,
        y_grid: NDArray[np.float64] | None,
    ) -> None:
        """Store an interpolation result in Tier 1 cache."""
        self._interp_cache[key] = (data, x_grid, y_grid)

    def get_interp_result(
        self, key: tuple
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None] | None:
        """Get an interpolation result from Tier 1 cache. Returns None on miss."""
        return self._interp_cache.get(key)

    def render_field(
        self,
        frame_idx: int,
        field_name: str,
        nodes: NDArray[np.float64],
        values: NDArray[np.float64],
        img_shape: tuple[int, int],
        mesh_step: int,
        cmap: str = "jet",
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> tuple[QPixmap, NDArray[np.float64] | None, NDArray[np.float64] | None]:
        """Render a displacement field to a QPixmap overlay.

        Uses two-level cache:
        1. Check Tier 2 (pixmap) -- exact match on all params.
        2. Check Tier 1 (interp) -- reuse if only color params changed.
        3. Full compute -- scatter_to_grid + colormap.

        Returns:
            (pixmap, x_grid, y_grid) -- pixmap for display, grids for positioning.
        """
        interp_key = (frame_idx, field_name)
        pixmap_key = (frame_idx, field_name, cmap, round(vmin, 6), round(vmax, 6))

        # Tier 2: exact pixmap hit
        cached_interp = self._interp_cache.get(interp_key)
        if pixmap_key in self._pixmap_cache and cached_interp is not None:
            _, xg, yg = cached_interp
            return self._pixmap_cache[pixmap_key], xg, yg

        # Tier 1: interpolation result hit
        if cached_interp is not None:
            grid_data, xg, yg = cached_interp
        else:
            # Full compute
            if self._interpolator is None:
                self._interpolator = FieldInterpolator(nodes)
            grid_data, info = scatter_to_grid(
                nodes,
                values,
                img_shape=img_shape,
                mesh_step=mesh_step,
                output_mode="auto",
                oversample=4,
                interpolator=self._interpolator,
            )
            xg = info["x_grid"]
            yg = info["y_grid"]
            self._interp_cache[interp_key] = (grid_data, xg, yg)

        # Apply colormap
        rgba = apply_colormap(grid_data, vmin, vmax, cmap)

        # Convert to QPixmap
        h, w = rgba.shape[:2]
        rgba_contiguous = np.ascontiguousarray(rgba)
        qimg = QImage(
            rgba_contiguous.data, w, h, w * 4, QImage.Format.Format_RGBA8888
        )
        pixmap = QPixmap.fromImage(qimg.copy())  # .copy() detaches from numpy

        self._pixmap_cache[pixmap_key] = pixmap
        return pixmap, xg, yg
