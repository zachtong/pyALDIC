"""Interpolation utilities wrapping scipy.

Provides consistent interfaces for scattered interpolation and
NaN-filling, replacing MATLAB's scatteredInterpolant and
ba_interp2_spline.

IMPORTANT — Image interpolation note:
    MATLAB uses ba_interp2_spline (bicubic spline, mex).
    Python uses scipy.ndimage.map_coordinates(order=3).
    Expected single-point difference: ~1e-4.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay

from al_dic.utils.crack_barrier import cross_crack_cell_mask


def scattered_interpolant(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    query_points: NDArray[np.float64],
    method: str = "linear",
    fill_method: str = "nearest",
) -> NDArray[np.float64]:
    """Scattered data interpolation, mimicking MATLAB's scatteredInterpolant.

    Args:
        points: (N, 2) array of known data point coordinates.
        values: (N,) array of known values.
        query_points: (M, 2) array of query coordinates.
        method: 'linear' or 'nearest'. MATLAB's 'natural' maps to 'linear'.
        fill_method: How to fill extrapolated points. 'nearest' or 'nan'.

    Returns:
        (M,) array of interpolated values.
    """
    # Remove NaN entries (values, and points with non-finite coordinates --
    # e.g. crack-destroyed nodes whose deformed position is undefined; a NaN
    # point would make scipy's Delaunay raise).
    valid = ~np.isnan(values) & np.isfinite(points).all(axis=1)
    if not np.any(valid):
        return np.full(len(query_points), np.nan)

    pts = points[valid]
    vals = values[valid]

    if method == "nearest":
        interp = NearestNDInterpolator(pts, vals)
        return interp(query_points)

    # Linear interpolation with nearest-neighbor extrapolation
    interp = LinearNDInterpolator(pts, vals, fill_value=np.nan)
    result = interp(query_points)

    if fill_method == "nearest":
        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            nn = NearestNDInterpolator(pts, vals)
            result[nan_mask] = nn(query_points[nan_mask])

    return result


def scattered_interpolant_uv(
    points: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    query_points: NDArray[np.float64],
    fill_method: str = "nearest",
    tri_cache: dict | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate two co-located scalar fields (u, v) at ``query_points``,
    building the triangulation ONCE for both.

    Byte-identical to two ``scattered_interpolant`` calls (when u and v share
    the same NaN pattern) but avoids the redundant Delaunay rebuild: when u
    and v share the same valid mask, ONE ``LinearNDInterpolator`` over a
    2-column value array builds a single Delaunay and evaluates both columns.
    When the masks differ, falls back to two independent
    ``scattered_interpolant`` calls.

    NOTE: a tempting further shortcut -- returning ``(u, v)`` directly when
    the query points equal the data points (the accumulative-mode case) --
    is NOT byte-identical: scipy's barycentric evaluation at a vertex differs
    from the raw value by ~1e-16, so a node query is not the exact identity.
    That faster-but-not-byte-identical path is intentionally omitted here.

    Args:
        points: (N, 2) data-point coordinates.
        u, v: (N,) co-located scalar fields (may contain NaN).
        query_points: (M, 2) query coordinates.
        fill_method: 'nearest' or 'nan' for out-of-hull queries.
        tri_cache: optional {points.tobytes(): Delaunay} cache to reuse the
            triangulation across calls sharing the same points + valid mask
            (e.g. every frame of the cumulative transform on a uniform mesh).
            Byte-identical; only avoids the redundant Delaunay build.

    Returns:
        (disp_u, disp_v), each (M,), identical to the per-field calls.
    """
    disp_u, disp_v, _, _ = scattered_interpolant_uv_diag(
        points, u, v, query_points, fill_method=fill_method,
        tri_cache=tri_cache,
    )
    return disp_u, disp_v


def scattered_interpolant_uv_diag(
    points: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    query_points: NDArray[np.float64],
    fill_method: str = "nearest",
    tri_cache: dict | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_],
    NDArray[np.float64],
]:
    """:func:`scattered_interpolant_uv` that also reports extrapolation.

    The values are computed exactly as before; the extra outputs only
    observe them.

    Returns:
        ``(disp_u, disp_v, outside, nn_dist)``, each (M,):

        - ``outside``: the query lies outside the convex hull of the valid
          nodes (the linear interpolant returned NaN there). When u and v
          drop different nodes, it is the OR of the two fields' hulls.
        - ``nn_dist``: distance in px from an outside query to the node its
          nearest-neighbor value was copied from (the larger of the two
          fields' distances when they drop different nodes); 0 for inside
          queries and when ``fill_method`` is not ``'nearest'``.
    """
    finite_pts = np.isfinite(points).all(axis=1)
    valid_u = ~np.isnan(u) & finite_pts
    valid_v = ~np.isnan(v) & finite_pts

    # Shared triangulation is only valid when u and v drop the same nodes.
    if not np.array_equal(valid_u, valid_v):
        disp_u, out_u, dist_u = _scattered_interpolant_diag(
            points, u, query_points, fill_method
        )
        disp_v, out_v, dist_v = _scattered_interpolant_diag(
            points, v, query_points, fill_method
        )
        return disp_u, disp_v, out_u | out_v, np.maximum(dist_u, dist_v)

    valid = valid_u
    if not np.any(valid):
        nan = np.full(len(query_points), np.nan)
        return nan, nan.copy(), *_no_extrapolation(len(query_points))

    pts = points[valid]
    vals = np.column_stack([u[valid], v[valid]])
    # One LinearNDInterpolator over a 2-column value array builds a single
    # Delaunay internally (identical to the per-field path) and interpolates
    # both columns through it. With a tri_cache the Delaunay is reused across
    # calls sharing the same (points, valid mask) -- the build dominates the
    # cost, so this is ~Nx on a uniform mesh -- while staying byte-identical.
    pts_or_tri = pts
    if tri_cache is not None:
        key = pts.tobytes()
        tri = tri_cache.get(key)
        if tri is None:
            tri = Delaunay(pts)
            tri_cache[key] = tri
        pts_or_tri = tri
    interp = LinearNDInterpolator(pts_or_tri, vals, fill_value=np.nan)
    result = interp(query_points)

    outside = np.isnan(result[:, 0])
    nn_dist = np.zeros(len(query_points))
    if fill_method == "nearest":
        nan_mask = np.isnan(result[:, 0])
        if np.any(nan_mask):
            nn = NearestNDInterpolator(pts, vals)
            result[nan_mask] = nn(query_points[nan_mask])
            # The interpolator's own KD-tree, queried as __call__ does.
            nn_dist[nan_mask] = nn.tree.query(query_points[nan_mask])[0]

    return result[:, 0], result[:, 1], outside, nn_dist


def _no_extrapolation(
    n: int,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """``(outside, nn_dist)`` for ``n`` queries that were not extrapolated."""
    return np.zeros(n, dtype=bool), np.zeros(n)


def _scattered_interpolant_diag(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    query_points: NDArray[np.float64],
    fill_method: str,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """One field of :func:`scattered_interpolant_uv_diag`'s fallback.

    Values are those of ``scattered_interpolant(points, values,
    query_points, fill_method=fill_method)`` (same operations, in order);
    also returns ``(outside, nn_dist)`` as documented there.
    """
    valid = ~np.isnan(values) & np.isfinite(points).all(axis=1)
    if not np.any(valid):
        return np.full(len(query_points), np.nan), *_no_extrapolation(
            len(query_points)
        )

    pts = points[valid]
    vals = values[valid]
    interp = LinearNDInterpolator(pts, vals, fill_value=np.nan)
    result = interp(query_points)

    outside = np.isnan(result)
    nn_dist = np.zeros(len(query_points))
    if fill_method == "nearest":
        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            nn = NearestNDInterpolator(pts, vals)
            result[nan_mask] = nn(query_points[nan_mask])
            nn_dist[nan_mask] = nn.tree.query(query_points[nan_mask])[0]

    return result, outside, nn_dist


def fill_nan_scattered(
    coords: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fill NaN values using nearest-neighbor scattered interpolation.

    Args:
        coords: (N, 2) node coordinates.
        values: (N,) values with potential NaNs.

    Returns:
        (N,) values with NaNs replaced by nearest valid neighbor.
    """
    result = values.copy()
    nan_mask = np.isnan(result)
    if not np.any(nan_mask) or np.all(nan_mask):
        return result

    valid = ~nan_mask
    nn = NearestNDInterpolator(coords[valid], result[valid])
    result[nan_mask] = nn(coords[nan_mask])
    return result


def interp2_bicubic(
    image: NDArray[np.float64],
    x_query: NDArray[np.float64],
    y_query: NDArray[np.float64],
    fill_value: float = 0.0,
) -> NDArray[np.float64]:
    """Bicubic image interpolation, replacing MATLAB's ba_interp2_spline.

    Uses scipy.ndimage.map_coordinates with order=3 (cubic spline).

    Args:
        image: (H, W) float64 image.
        x_query: x-coordinates (columns) to sample, any shape.
        y_query: y-coordinates (rows) to sample, same shape as x_query.
        fill_value: Value for out-of-bounds queries.

    Returns:
        Interpolated values, same shape as x_query.
    """
    # map_coordinates expects (row, col) = (y, x) coordinates
    coords = np.array([y_query.ravel(), x_query.ravel()])
    result = map_coordinates(
        image, coords, order=3, mode="constant", cval=fill_value
    )
    return result.reshape(x_query.shape)


# ---------------------------------------------------------------------------
# FieldInterpolator — precomputed Delaunay for multi-field visualization
# ---------------------------------------------------------------------------


class FieldInterpolator:
    """Precomputed Delaunay-based interpolator for DIC field visualization.

    Builds a Delaunay triangulation once from node positions, then reuses it
    for interpolating multiple scalar fields (u, v, exx, eyy, exy, ...).

    Parameters
    ----------
    nodes : (N, 2) array
        Node coordinates (x, y).
    method : {"linear", "clough_tocher"}
        "linear": C0 bilinear on Delaunay triangles (fast).
        "clough_tocher": C1 cubic Clough-Tocher (smooth, ~2x slower eval).
    """

    def __init__(
        self,
        nodes: NDArray[np.float64],
        method: str = "linear",
    ) -> None:
        if method not in ("linear", "clough_tocher"):
            raise ValueError(
                f"Unknown method '{method}'. Use 'linear' or 'clough_tocher'."
            )
        self._method = method
        self._nodes = np.asarray(nodes, dtype=np.float64)
        # Nodes with non-finite coordinates (crack-destroyed nodes in a
        # deformed configuration) cannot participate: Delaunay raises on NaN.
        self._finite = np.isfinite(self._nodes).all(axis=1)
        self._nodes_f = self._nodes[self._finite]
        if self._nodes_f.shape[0] < 3:
            raise ValueError(
                "FieldInterpolator needs at least 3 nodes with finite "
                f"coordinates (got {self._nodes_f.shape[0]})."
            )
        self._tri = Delaunay(self._nodes_f)

    @property
    def method(self) -> str:
        return self._method

    def interpolate(
        self,
        values: NDArray[np.float64],
        x_grid: NDArray[np.float64],
        y_grid: NDArray[np.float64],
        fill_outside: str = "nan",
    ) -> NDArray[np.float64]:
        """Interpolate a scalar field onto a 2D grid.

        Parameters
        ----------
        values : (N,) array
            Scalar values at each node. NaN nodes are excluded.
        x_grid, y_grid : (H, W) arrays
            Meshgrid of query coordinates.
        fill_outside : {"nan", "nearest"}
            How to handle points outside the node convex hull.

        Returns
        -------
        result : (H, W) array
            Interpolated field. NaN where undefined (if fill_outside="nan").
        """
        vals = np.asarray(values, dtype=np.float64)
        valid = ~np.isnan(vals) & self._finite

        if not np.any(valid):
            return np.full(x_grid.shape, np.nan, dtype=np.float64)

        if np.array_equal(valid, self._finite):
            # All finite-position nodes carry values: reuse the prebuilt
            # triangulation (identical to the historical all-valid fast path
            # when every node is finite).
            pts, v, tri = self._nodes_f, vals[self._finite], self._tri
        else:
            pts = self._nodes[valid]
            v = vals[valid]
            tri = Delaunay(pts)

        if self._method == "linear":
            interp = LinearNDInterpolator(tri, v)
        else:
            interp = CloughTocher2DInterpolator(tri, v)

        result = interp(x_grid, y_grid)

        if fill_outside == "nearest":
            nan_mask = np.isnan(result)
            if np.any(nan_mask):
                nn = NearestNDInterpolator(pts, v)
                result[nan_mask] = nn(x_grid[nan_mask], y_grid[nan_mask])

        return result

    def cross_crack_grid(
        self,
        values: NDArray[np.float64],
        x_grid: NDArray[np.float64],
        y_grid: NDArray[np.float64],
        barrier_mask: NDArray | None,
    ) -> NDArray[np.bool_] | None:
        """Grid mask (True = blank) for cells whose interpolating triangle
        crosses a crack / hole barrier.

        Uses the SAME triangulation :meth:`interpolate` builds for *values*
        (value-reduced when some nodes are NaN), so the crack-aware blanking
        matches the interpolated field exactly.  Returns ``None`` when there is
        no barrier or no triangle crosses it (render stays bit-exact).
        """
        if barrier_mask is None:
            return None
        vals = np.asarray(values, dtype=np.float64)
        valid = ~np.isnan(vals) & self._finite
        if int(valid.sum()) < 3:
            return None
        if np.array_equal(valid, self._finite):
            pts, tri = self._nodes_f, self._tri
        else:
            pts = self._nodes[valid]
            tri = Delaunay(pts)
        return cross_crack_cell_mask(tri, pts, x_grid, y_grid, barrier_mask)


# ---------------------------------------------------------------------------
# scatter_to_grid — smart output grid sizing for visualization
# ---------------------------------------------------------------------------


def scatter_to_grid(
    nodes: NDArray[np.float64],
    values: NDArray[np.float64],
    img_shape: tuple[int, int],
    mesh_step: int,
    output_mode: str = "auto",
    method: str = "clough_tocher",
    oversample: int = 4,
    max_output_pixels: int = 0,
    fill_outside: str = "nan",
    interpolator: FieldInterpolator | None = None,
    margin: int | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """Interpolate scattered node values onto a regular pixel grid.

    Smart output grid sizing: the output grid density is proportional to the
    mesh density, not the image pixel density.  This avoids redundant
    computation for large images with coarse meshes.

    Rendering strategy (benchmarked on 4000x3000 images):
        - Preview: output_mode="auto", oversample=4.  CloughTocher (C1) by
          default — only ~15% slower than Linear, with noticeably smoother
          contours.  The real bottleneck is Delaunay construction, not the
          interpolation method.
        - Export:  output_mode="full".  CloughTocher for publication-quality
          smoothness.  ~7s for 6 fields at step=8 — acceptable for a
          background save operation.
        - For deformed-config preview with >50k nodes, the caller should
          subsample nodes (take every 3rd) to stay under 0.5s latency.

    Parameters
    ----------
    nodes : (N, 2) array
        Node coordinates (x, y). Can be reference or deformed positions.
    values : (N,) array
        Scalar field values at each node.
    img_shape : (H, W)
        Original image shape in pixels.
    mesh_step : int
        Mesh node spacing in pixels.
    output_mode : {"auto", "preview", "full"}
        "auto": output_step = max(1, mesh_step // oversample).
        "preview": same as auto, but capped by max_output_pixels.
        "full": output_step = 1 (pixel-level).
    method : {"linear", "clough_tocher"}
        Interpolation method.  Default is "clough_tocher" (C1 smooth) for
        visually superior contour plots.  Use "linear" only if speed is
        critical and visual quality is secondary.
    oversample : int
        Oversampling factor for auto/preview modes.  Higher = denser output.
        Default 4 means output_step = mesh_step // 4.
    max_output_pixels : int
        Maximum output pixels for preview mode.  0 = no cap.
    fill_outside : {"nan", "nearest"}
        How to fill points outside the node convex hull.
    interpolator : FieldInterpolator or None
        Pre-built interpolator to reuse.  If None, one is created internally.
    margin : int or None
        Pixels added around the node bounding box.  None (default) keeps the
        historical ``mesh_step // 2``; "Hide untracked regions" widens it to
        its reach so the field can be painted up to the ROI edge.

    Returns
    -------
    result : (H_out, W_out) array
        Interpolated field image.
    info : dict
        Metadata: x_grid, y_grid (meshgrid arrays), output_step,
        img_shape, method.
    """
    h, w = img_shape
    nodes = np.asarray(nodes, dtype=np.float64)

    # --- Determine output grid step ---
    if output_mode == "full":
        out_step = 1
    else:
        out_step = max(1, mesh_step // oversample)

    # --- Build output grid covering node bounding box ---
    # nanmin/nanmax: crack-destroyed nodes carry NaN deformed coordinates and
    # must not blow up the extent computation (identical when all finite).
    margin = mesh_step // 2 if margin is None else int(margin)
    x_min = max(0, int(math.floor(np.nanmin(nodes[:, 0]))) - margin)
    x_max = min(w, int(math.ceil(np.nanmax(nodes[:, 0]))) + margin)
    y_min = max(0, int(math.floor(np.nanmin(nodes[:, 1]))) - margin)
    y_max = min(h, int(math.ceil(np.nanmax(nodes[:, 1]))) + margin)

    grid_xs = np.arange(x_min, x_max, out_step, dtype=np.float64)
    grid_ys = np.arange(y_min, y_max, out_step, dtype=np.float64)

    # --- Preview: enforce pixel cap ---
    if output_mode == "preview" and max_output_pixels > 0:
        n_pixels = len(grid_xs) * len(grid_ys)
        if n_pixels > max_output_pixels:
            scale = math.sqrt(max_output_pixels / n_pixels)
            new_step = max(1, int(math.ceil(out_step / scale)))
            grid_xs = np.arange(x_min, x_max, new_step, dtype=np.float64)
            grid_ys = np.arange(y_min, y_max, new_step, dtype=np.float64)
            out_step = new_step

    x_grid, y_grid = np.meshgrid(grid_xs, grid_ys)

    # --- Interpolate ---
    if interpolator is None:
        interpolator = FieldInterpolator(nodes, method=method)

    result = interpolator.interpolate(
        values, x_grid, y_grid, fill_outside=fill_outside
    )

    info = {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "output_step": out_step,
        "img_shape": img_shape,
        "method": method,
        "n_nodes": len(nodes),
        "grid_shape": x_grid.shape,
    }
    return result, info
