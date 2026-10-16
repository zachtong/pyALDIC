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

import math

from al_dic.core.colormaps import resolve
import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QImage, QPixmap
from scipy.interpolate import NearestNDInterpolator

from al_dic.utils.crack_barrier import crack_aware_nearest_fill
from al_dic.utils.interpolation import FieldInterpolator, scatter_to_grid
from al_dic.utils.tracked_region import (
    paint_tracked_region,
    tracked_reach_px,
    trimmed_cells,
)


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

    cm = resolve(cmap)
    rgba = (cm(normalized) * 255).astype(np.uint8)

    # NaN pixels -> transparent
    nan_mask = np.isnan(data)
    rgba[nan_mask] = 0

    return rgba


def _invalid_node_grid(
    nodes: NDArray[np.float64],
    values: NDArray[np.float64],
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
) -> NDArray[np.bool_] | None:
    """Grid mask (True = blank) for cells whose nearest node has a NaN value.

    Strain post-processing NaNs edge-trimmed / plane-fit-failed nodes to mark
    them unreliable, but ``scattered_interpolant`` drops NaN nodes and
    back-fills their grid cells from valid neighbours -- so the trim vanishes
    and the trimmed band renders as extrapolated (abnormal) values instead of
    the intended transparent boundary band.  Rasterising node finite-ness by
    nearest neighbour restores that band: a grid cell is blanked iff its
    closest node is invalid.  Being nearest-node based, it is independent of
    mesh density and matches the deformed / reference node positions passed in.

    Returns ``None`` when every node is finite (nothing to blank).
    """
    # A node with a non-finite POSITION (crack-destroyed node in deformed
    # coordinates) has no place on screen: exclude it from claiming cells --
    # and from the NN build, which would raise on NaN coordinates.
    node_ok = np.isfinite(nodes).all(axis=1)
    finite = np.isfinite(values) & node_ok
    if bool(finite.all()):
        return None
    if not bool(finite[node_ok].any()) or not bool(node_ok.any()):
        return np.ones(x_grid.shape, dtype=bool)
    nn = NearestNDInterpolator(
        nodes[node_ok], finite[node_ok].astype(np.float64),
    )
    query = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    keep = nn(query).reshape(x_grid.shape) >= 0.5
    return ~keep


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
        # Crack-aware blank cache: {interp_key -> bool grid | None}.  Depends on
        # node positions + the barrier (ROI/crack) mask, so it is cleared
        # whenever results or mask content change.
        self._crack_cache: dict[tuple, NDArray[np.bool_] | None] = {}
        # Hide-untracked, deformed view without a per-frame ROI: the reference
        # ROI warped forward at image resolution, exactly as the export does.
        self._warped_roi_cache: dict[int, NDArray[np.bool_]] = {}

    def clear_all(self) -> None:
        """Clear both cache tiers (results changed)."""
        self._interp_cache.clear()
        self._pixmap_cache.clear()
        self._interpolator = None
        self._warp_cache.clear()
        self._crack_cache.clear()
        self._warped_roi_cache.clear()

    def clear_pixmap_cache(self) -> None:
        """Clear Tier 2 only (colormap/range changed)."""
        self._pixmap_cache.clear()

    def invalidate_masks(self) -> None:
        """Clear caches that depend on ROI mask content (pixmap + warp + crack),
        and the interpolated grids that were filled up to a mask edge
        (extrapolated "fill trimmed edges" and hide-untracked renders)."""
        self._pixmap_cache.clear()
        self._warp_cache.clear()
        self._crack_cache.clear()
        self._warped_roi_cache.clear()
        for key in [k for k in self._interp_cache if k[3] is True or k[3] == "tracked"]:
            del self._interp_cache[key]

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
        blank_invalid_nodes: bool = False,
        hide_untracked: bool = False,
        tracked: NDArray[np.bool_] | None = None,
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
            blank_invalid_nodes: When True, grid cells whose nearest node has a
                NaN value are made transparent (strain edge-trim / plane-fit
                failures).  Prevents the interpolator's back-fill from hiding
                the trim.  Leave False for displacement fields.
            hide_untracked: "Hide untracked regions" -- paint a cell only if a
                tracked node lies within one grid cell (diagonal included);
                fill the rest of that reach up to the ROI / crack edge from the
                same side, leave everything farther transparent
                (:mod:`al_dic.utils.tracked_region`).  False keeps the
                historical rendering byte for byte.
            tracked: Per-node "has displacement" flags used by hide_untracked
                (independent of strain trimming).  None = nodes with a finite
                position and a finite value.

        Returns:
            (pixmap, x_grid, y_grid, output_step) -- pixmap for display,
            grids for positioning, output_step for scaling.
        """
        # "Fill trimmed edges" (blank_invalid_nodes False) on a field that
        # actually has trimmed nodes: extrapolate beyond the shrunken
        # interpolation hull so the fill reaches the full ROI, not just the
        # kept-node hull.  Displacement fields (no NaN) are unaffected.
        # Hide-untracked in the deformed view needs the deformed ROI as its
        # fill boundary.  Without a per-frame ROI, warp the reference ROI the
        # way the export does (export_png._compute_warped_mask), so the screen
        # and the exported file are clipped and filled identically.
        if (hide_untracked and deformed and deformed_mask is None
                and roi_mask is not None and ref_uv is not None):
            deformed_mask = self._warped_roi_cache.get(frame_idx)
            if deformed_mask is None:
                from al_dic.export.export_png import _compute_warped_mask
                ref_nodes = nodes - np.column_stack(
                    [np.asarray(ref_uv[0], dtype=np.float64),
                     np.asarray(ref_uv[1], dtype=np.float64)])
                deformed_mask = _compute_warped_mask(
                    ref_nodes, nodes, np.asarray(roi_mask, dtype=bool), img_shape)
                self._warped_roi_cache[frame_idx] = deformed_mask

        extrapolate = (
            not hide_untracked
            and not blank_invalid_nodes
            and bool(np.isnan(np.asarray(values, dtype=np.float64)).any())
        )
        # Hide-untracked renders replace the extrapolation with the tracked-
        # region paint, so they cache under their own key; legacy keys are
        # unchanged.
        interp_key = (frame_idx, field_name, deformed,
                      "tracked" if hide_untracked else extrapolate)
        has_mask = roi_mask is not None
        has_def_mask = deformed_mask is not None
        pixmap_key = (frame_idx, field_name, cmap, round(vmin, 6), round(vmax, 6), has_mask, deformed, has_def_mask, blank_invalid_nodes, hide_untracked)

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
            reach = tracked_reach_px(mesh_step)
            grid_data, info = scatter_to_grid(
                nodes,
                values,
                img_shape=img_shape,
                mesh_step=mesh_step,
                output_mode="auto",
                oversample=4,
                interpolator=interpolator,
                margin=int(math.ceil(reach)) if hide_untracked else None,
            )
            xg = info["x_grid"]
            yg = info["y_grid"]
            out_step = int(info.get("output_step", 1))

            # Crack-aware rendering: the Delaunay behind scatter_to_grid
            # reconnects nodes across a crack the mesh split.  Flag cells inside
            # a barrier-crossing triangle so the overlay respects the crack.
            # Barrier is in the render-coordinate space: deformed_mask in
            # deformed mode, else roi_mask.  None (bit-exact) when no triangle
            # crosses a barrier.
            barrier = (
                deformed_mask if (deformed and deformed_mask is not None)
                else (roi_mask if not deformed else None)
            )
            crack_grid = None
            if xg is not None and yg is not None:
                crack_grid = interpolator.cross_crack_grid(
                    values, xg, yg, barrier,
                )

            # "Fill trimmed edges" (fill ON, strain with trim): recover both the
            # ROI outer edge (beyond the shrunken hull) AND the crack inner edge
            # (just flagged) from reliable nodes, each cell from the nearest node
            # reachable WITHOUT crossing the crack -- so the two faces fill from
            # their own side and the crack stays a sharp line.  Bake it into
            # grid_data; the crack flag is then consumed (not re-blanked).
            if extrapolate and xg is not None and yg is not None:
                gd = np.array(grid_data, copy=True)
                if crack_grid is not None:
                    gd[crack_grid] = np.nan
                need = np.isnan(gd)
                if barrier is not None:
                    b = np.asarray(barrier)
                    bx = np.clip(np.round(xg).astype(np.int64), 0, b.shape[1] - 1)
                    by = np.clip(np.round(yg).astype(np.int64), 0, b.shape[0] - 1)
                    need &= b[by, bx] > 0.5
                if need.any():
                    vals_arr = np.asarray(values, dtype=np.float64)
                    vmask = ~np.isnan(vals_arr) & np.isfinite(nodes).all(axis=1)
                    fill = crack_aware_nearest_fill(
                        xg, yg, need, nodes[vmask], vals_arr[vmask], barrier,
                    )
                    gd = np.where(need, fill, gd)
                grid_data = gd
                crack_grid = None  # consumed by the fill; do not blank again

            # "Hide untracked regions": paint only within reach of a tracked
            # node, filling that reach up to the ROI / crack edge from the same
            # side; the crack band is refilled here, so its flag is consumed.
            if hide_untracked and xg is not None and yg is not None:
                gd = np.array(grid_data, dtype=np.float64, copy=True)
                if crack_grid is not None:
                    gd[crack_grid] = np.nan
                vals_arr = np.asarray(values, dtype=np.float64)
                pos_ok = np.isfinite(nodes).all(axis=1)
                is_tracked = (np.isfinite(vals_arr) if tracked is None
                              else np.asarray(tracked, dtype=bool))
                # No mask = no edge to fill toward: only cells the
                # interpolation already reached may stay (no halo).
                inside = np.isfinite(gd)
                if barrier is not None:
                    b = np.asarray(barrier)
                    bx = np.clip(np.round(xg).astype(np.int64), 0, b.shape[1] - 1)
                    by = np.clip(np.round(yg).astype(np.int64), 0, b.shape[0] - 1)
                    inside = b[by, bx] > 0.5
                fill_ok = pos_ok & np.isfinite(vals_arr)
                grid_data = paint_tracked_region(
                    gd, xg, yg, inside, nodes[pos_ok & is_tracked],
                    nodes[fill_ok], vals_arr[fill_ok], barrier, reach,
                )
                crack_grid = None

            self._interp_cache[interp_key] = (grid_data, xg, yg, out_step)
            self._crack_cache[interp_key] = crack_grid

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

        # Re-apply node-level invalidity (NaN values = edge-trimmed or
        # plane-fit-failed strain nodes) at grid resolution.  Without this the
        # scattered interpolator would back-fill trimmed cells from valid
        # neighbours, making the trim invisible and the trimmed band show
        # extrapolated (abnormal) values instead of a transparent boundary.
        if blank_invalid_nodes and xg is not None and yg is not None:
            if hide_untracked:
                # Only TRACKED nodes without a value are trimmed; untracked
                # ones were already handled by the reach rule.
                invalid_grid = trimmed_cells(
                    xg, yg, nodes, values,
                    np.isfinite(np.asarray(values, dtype=np.float64))
                    if tracked is None else tracked,
                )
            else:
                invalid_grid = _invalid_node_grid(nodes, values, xg, yg)
            if invalid_grid is not None:
                mask_to_use = (
                    invalid_grid if mask_to_use is None
                    else (mask_to_use | invalid_grid)
                )

        # Crack-aware blank (cached per interp_key; independent of the color/
        # trim toggles, so it always applies -- rendering must respect cracks).
        crack_grid = self._crack_cache.get(interp_key)
        if crack_grid is not None:
            mask_to_use = (
                crack_grid if mask_to_use is None
                else (mask_to_use | crack_grid)
            )

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
