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
    # Remove NaN entries
    valid = ~np.isnan(values)
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
        self._tri = Delaunay(self._nodes)

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
        valid = ~np.isnan(vals)

        if not np.any(valid):
            return np.full(x_grid.shape, np.nan, dtype=np.float64)

        if np.all(valid):
            pts, v, tri = self._nodes, vals, self._tri
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
    margin = mesh_step // 2
    x_min = max(0, int(math.floor(nodes[:, 0].min())) - margin)
    x_max = min(w, int(math.ceil(nodes[:, 0].max())) + margin)
    y_min = max(0, int(math.floor(nodes[:, 1].min())) - margin)
    y_max = min(h, int(math.ceil(nodes[:, 1].max())) + margin)

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
