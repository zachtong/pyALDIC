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

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import map_coordinates


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
