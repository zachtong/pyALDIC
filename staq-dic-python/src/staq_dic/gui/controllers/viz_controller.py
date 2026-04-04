"""Visualization controller -- two-level cache for field rendering.

Tier 1 (interp cache): scatter_to_grid output arrays.
    Key: (frame_idx, field_name)
    Invalidated: when results change.
    Survives: colormap/range changes.

Tier 2 (pixmap cache): colored QPixmap ready for display.
    Key: (frame_idx, field_name, cmap, vmin, vmax)
    Invalidated: when colormap or range changes.

Warped mask cache: deformed-coordinate ROI masks.
    Key: frame_idx
    Computed during full compute for deformed mode.
    Reused across field switches (disp_u <-> disp_v) at same frame.
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
        # Tier 1: interpolation results  {(frame, field) -> (data, x_grid, y_grid, out_step)}
        self._interp_cache: dict[tuple, tuple] = {}
        # Tier 2: colored pixmaps  {(frame, field, cmap, vmin, vmax, masked) -> QPixmap}
        self._pixmap_cache: dict[tuple, QPixmap] = {}
        self._interpolator: FieldInterpolator | None = None
        # Warped mask cache for deformed mode: {frame_idx -> outside_bool_mask}
        self._warp_cache: dict[int, NDArray[np.bool_]] = {}

    def clear_all(self) -> None:
        """Clear both cache tiers (results changed)."""
        self._interp_cache.clear()
        self._pixmap_cache.clear()
        self._interpolator = None
        self._warp_cache.clear()

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
        self._interp_cache[key] = (data, x_grid, y_grid, 1)

    def get_interp_result(
        self, key: tuple
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None] | None:
        """Get an interpolation result from Tier 1 cache. Returns None on miss."""
        cached = self._interp_cache.get(key)
        if cached is None:
            return None
        data, xg, yg, _step = cached
        return (data, xg, yg)

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
        roi_mask: NDArray[np.bool_] | None = None,
        deformed: bool = False,
        ref_uv: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
        deformed_mask: NDArray[np.bool_] | None = None,
    ) -> tuple[QPixmap, NDArray[np.float64] | None, NDArray[np.float64] | None, int]:
        """Render a displacement field to a QPixmap overlay.

        Uses two-level cache:
        1. Check Tier 2 (pixmap) -- exact match on all params.
        2. Check Tier 1 (interp) -- reuse if only color params changed.
        3. Full compute -- scatter_to_grid + colormap.

        Args:
            deformed: If True, nodes are in deformed configuration (different
                per frame). The cached FieldInterpolator is NOT reused since
                the Delaunay triangulation changes every frame.
            ref_uv: (u_node_values, v_node_values) — node-level accumulated
                displacements for warping the ROI mask from reference to
                deformed coordinates.  Required when deformed=True and
                roi_mask is provided.
            deformed_mask: Per-frame ground-truth mask in deformed coordinates.
                When provided with deformed=True, used directly instead of
                warping the reference roi_mask via inverse displacement.

        Returns:
            (pixmap, x_grid, y_grid, output_step) -- pixmap for display,
            grids for positioning, output_step for scaling.
        """
        interp_key = (frame_idx, field_name, deformed)
        has_mask = roi_mask is not None
        has_def_mask = deformed_mask is not None
        pixmap_key = (frame_idx, field_name, cmap, round(vmin, 6), round(vmax, 6), has_mask, deformed, has_def_mask)

        # Tier 2: exact pixmap hit
        cached_interp = self._interp_cache.get(interp_key)
        if pixmap_key in self._pixmap_cache and cached_interp is not None:
            _, xg, yg, out_step = cached_interp
            return self._pixmap_cache[pixmap_key], xg, yg, out_step

        # Tier 1: interpolation result hit
        if cached_interp is not None:
            grid_data, xg, yg, out_step = cached_interp
        else:
            # Full compute.
            # For deformed mode, nodes change every frame — create a fresh
            # interpolator instead of reusing the cached reference one.
            if deformed:
                interpolator = FieldInterpolator(nodes)
            else:
                if self._interpolator is None:
                    self._interpolator = FieldInterpolator(nodes)
                interpolator = self._interpolator
            grid_data, info = scatter_to_grid(
                nodes,
                values,
                img_shape=img_shape,
                mesh_step=mesh_step,
                output_mode="auto",
                oversample=4,
                interpolator=interpolator,
            )
            xg = info["x_grid"]
            yg = info["y_grid"]
            out_step = int(info.get("output_step", 1))
            self._interp_cache[interp_key] = (grid_data, xg, yg, out_step)

            # Compute warped mask for deformed mode: map deformed grid points
            # back to reference coordinates and look up the reference mask.
            if deformed and ref_uv is not None and roi_mask is not None:
                u_vals, v_vals = ref_uv
                u_grid = interpolator.interpolate(u_vals, xg, yg)
                v_grid = interpolator.interpolate(v_vals, xg, yg)
                # Reference positions of deformed grid points
                xr = xg - u_grid
                yr = yg - v_grid
                # NaN from interpolation outside convex hull — mark as outside
                nan_warp = np.isnan(xr) | np.isnan(yr)
                xr_safe = np.nan_to_num(xr, nan=0.0)
                yr_safe = np.nan_to_num(yr, nan=0.0)
                xi = np.clip(np.round(xr_safe).astype(int), 0, roi_mask.shape[1] - 1)
                yi = np.clip(np.round(yr_safe).astype(int), 0, roi_mask.shape[0] - 1)
                outside = ~roi_mask[yi, xi] | nan_warp
                self._warp_cache[frame_idx] = outside

        # Apply ROI mask: NaN for grid points outside the ROI.
        # Three masking paths in priority order:
        #   1. deformed + deformed_mask provided -> direct lookup on GT mask
        #   2. deformed + warped mask cached     -> inverse-displacement warp
        #   3. reference mode (or fallback)      -> direct lookup on roi_mask
        render_data = grid_data
        mask_to_use = None
        if deformed and deformed_mask is not None and xg is not None and yg is not None:
            # Path 1: per-frame GT mask — direct pixel lookup in deformed coords
            xi = np.clip(np.round(xg).astype(int), 0, deformed_mask.shape[1] - 1)
            yi = np.clip(np.round(yg).astype(int), 0, deformed_mask.shape[0] - 1)
            mask_to_use = ~deformed_mask[yi, xi]
        elif roi_mask is not None and xg is not None and yg is not None:
            if deformed and frame_idx in self._warp_cache:
                # Path 2: warped mask (reference -> deformed via inverse disp)
                mask_to_use = self._warp_cache[frame_idx]
            else:
                # Path 3: reference mode — direct pixel lookup on roi_mask
                xi = np.clip(np.round(xg).astype(int), 0, roi_mask.shape[1] - 1)
                yi = np.clip(np.round(yg).astype(int), 0, roi_mask.shape[0] - 1)
                mask_to_use = ~roi_mask[yi, xi]
        if mask_to_use is not None and np.any(mask_to_use):
            render_data = grid_data.copy()
            render_data[mask_to_use] = np.nan

        # Apply colormap
        rgba = apply_colormap(render_data, vmin, vmax, cmap)

        # Convert to QPixmap
        h, w = rgba.shape[:2]
        rgba_contiguous = np.ascontiguousarray(rgba)
        qimg = QImage(
            rgba_contiguous.data, w, h, w * 4, QImage.Format.Format_RGBA8888
        )
        pixmap = QPixmap.fromImage(qimg.copy())  # .copy() detaches from numpy

        self._pixmap_cache[pixmap_key] = pixmap
        return pixmap, xg, yg, out_step
