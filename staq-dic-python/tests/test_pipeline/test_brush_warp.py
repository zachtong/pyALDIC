"""Tests for material-point brush mask warping.

The brush refinement mask is painted in frame-0 image coordinates. When the
pipeline switches to a new reference frame K (incremental mode), the mask
must be warped to K's image coordinates so the *same material points* stay
inside the refinement zone, not the same pixel locations.

These tests exercise the helper ``warp_brush_mask_to_ref`` directly, using
synthetic node displacement fields with known closed-form behaviour.
"""

from __future__ import annotations

import numpy as np

from staq_dic.core._brush_warp import warp_brush_mask_to_ref


def _grid_nodes(h: int, w: int, step: int) -> np.ndarray:
    """Build an (N, 2) array of (x, y) nodes on a regular grid."""
    xs = np.arange(0, w, step)
    ys = np.arange(0, h, step)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)


def test_zero_displacement_is_identity() -> None:
    """Zero node displacement must return the input mask unchanged."""
    h, w = 64, 64
    brush = np.zeros((h, w), dtype=bool)
    brush[20:40, 20:40] = True
    coords = _grid_nodes(h, w, step=4)
    n = coords.shape[0]
    cumulative_U = np.zeros(2 * n, dtype=np.float64)

    warped = warp_brush_mask_to_ref(brush, cumulative_U, coords, (h, w))

    assert warped.shape == brush.shape
    assert warped.dtype == np.bool_
    np.testing.assert_array_equal(warped, brush)


def test_uniform_translation_warps_brush() -> None:
    """A +10 px translation in x must shift the painted region right by 10."""
    h, w = 64, 64
    brush = np.zeros((h, w), dtype=bool)
    brush[20:30, 20:30] = True
    coords = _grid_nodes(h, w, step=4)
    n = coords.shape[0]

    # Interleaved [u0, v0, u1, v1, ...] with +10 px in x everywhere.
    cumulative_U = np.zeros(2 * n, dtype=np.float64)
    cumulative_U[0::2] = 10.0  # u
    cumulative_U[1::2] = 0.0   # v

    warped = warp_brush_mask_to_ref(brush, cumulative_U, coords, (h, w))

    # Painted box now centred around x=35 (was 25), y unchanged.
    assert warped[25, 35]
    # Original location now vacated.
    assert not warped[25, 15]
    # Outside the warped box stays outside.
    assert not warped[5, 5]


def test_uniform_translation_y_axis() -> None:
    """A +8 px translation in y must shift the painted region down by 8."""
    h, w = 64, 64
    brush = np.zeros((h, w), dtype=bool)
    brush[10:20, 25:35] = True
    coords = _grid_nodes(h, w, step=4)
    n = coords.shape[0]

    cumulative_U = np.zeros(2 * n, dtype=np.float64)
    cumulative_U[1::2] = 8.0  # +8 px in y

    warped = warp_brush_mask_to_ref(brush, cumulative_U, coords, (h, w))

    # Box centre was at (y=15, x=30); now at (y=23, x=30).
    assert warped[23, 30]
    assert not warped[15, 30]


def test_returns_bool_dtype() -> None:
    """Helper must always return a boolean array regardless of input type."""
    h, w = 32, 32
    brush = np.zeros((h, w), dtype=bool)
    brush[8:16, 8:16] = True
    coords = _grid_nodes(h, w, step=4)
    cumulative_U = np.zeros(2 * coords.shape[0], dtype=np.float64)

    warped = warp_brush_mask_to_ref(brush, cumulative_U, coords, (h, w))
    assert warped.dtype == np.bool_
