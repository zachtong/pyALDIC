"""Tests for interpolation utilities."""

import numpy as np
import pytest

from staq_dic.utils.interpolation import (
    FieldInterpolator,
    fill_nan_scattered,
    interp2_bicubic,
    scatter_to_grid,
    scattered_interpolant,
)


class TestScatteredInterpolant:
    def test_linear_field(self):
        """Interpolation of a linear field should be exact."""
        rng = np.random.default_rng(0)
        points = rng.random((50, 2)) * 10
        values = 2 * points[:, 0] + 3 * points[:, 1] + 1

        query = rng.random((20, 2)) * 8 + 1  # interior points
        result = scattered_interpolant(points, values, query)
        expected = 2 * query[:, 0] + 3 * query[:, 1] + 1

        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_nearest_extrapolation(self):
        """Points outside the convex hull should get nearest-neighbor values."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        values = np.array([1, 2, 3, 4], dtype=np.float64)

        query = np.array([[-1, -1]], dtype=np.float64)  # outside convex hull
        result = scattered_interpolant(points, values, query)
        assert not np.isnan(result[0])

    def test_all_nan_values(self):
        points = np.array([[0, 0], [1, 1]], dtype=np.float64)
        values = np.array([np.nan, np.nan])
        query = np.array([[0.5, 0.5]], dtype=np.float64)
        result = scattered_interpolant(points, values, query)
        assert np.isnan(result[0])

    def test_nearest_method(self):
        points = np.array([[0, 0], [10, 10]], dtype=np.float64)
        values = np.array([1.0, 2.0])
        query = np.array([[1, 1]], dtype=np.float64)
        result = scattered_interpolant(points, values, query, method="nearest")
        assert result[0] == 1.0


class TestFillNanScattered:
    def test_no_nans(self):
        coords = np.array([[0, 0], [1, 1]], dtype=np.float64)
        values = np.array([1.0, 2.0])
        result = fill_nan_scattered(coords, values)
        np.testing.assert_array_equal(result, values)

    def test_fills_nans(self):
        coords = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        values = np.array([1.0, np.nan, 3.0])
        result = fill_nan_scattered(coords, values)
        assert not np.any(np.isnan(result))
        # NaN at index 1 should be filled by nearest neighbor (either 1.0 or 3.0)
        assert result[1] in (1.0, 3.0)


class TestInterp2Bicubic:
    def test_identity_at_grid_points(self):
        """Interpolation at integer grid points should return exact values."""
        img = np.random.default_rng(42).random((32, 32))
        x = np.array([5.0, 10.0, 15.0])
        y = np.array([5.0, 10.0, 15.0])
        result = interp2_bicubic(img, x, y)
        expected = np.array([img[5, 5], img[10, 10], img[15, 15]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_out_of_bounds_fill(self):
        img = np.ones((10, 10))
        x = np.array([-5.0])
        y = np.array([-5.0])
        result = interp2_bicubic(img, x, y, fill_value=0.0)
        assert result[0] == pytest.approx(0.0, abs=0.1)

    def test_2d_query(self):
        img = np.random.default_rng(1).random((32, 32))
        xx, yy = np.meshgrid(np.arange(5, 25), np.arange(5, 25))
        result = interp2_bicubic(img, xx.astype(float), yy.astype(float))
        assert result.shape == (20, 20)


# ---- Helpers ----

def _make_grid_nodes(img_size=128, step=8):
    """Uniform grid nodes inside image."""
    half = step // 2
    xs = np.arange(half, img_size - half + 1, step, dtype=np.float64)
    ys = np.arange(half, img_size - half + 1, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


# ---- FieldInterpolator tests ----


class TestFieldInterpolator:
    """Tests for FieldInterpolator with precomputed Delaunay."""

    def test_linear_field_exact(self):
        """Linear interpolation of a linear field should be exact."""
        nodes = _make_grid_nodes(128, 8)
        values = 2.5 * nodes[:, 0] - 1.3 * nodes[:, 1] + 0.7
        interp = FieldInterpolator(nodes)

        query_x, query_y = np.meshgrid(
            np.arange(10, 110, dtype=np.float64),
            np.arange(10, 110, dtype=np.float64),
        )
        result = interp.interpolate(values, query_x, query_y)
        expected = 2.5 * query_x - 1.3 * query_y + 0.7

        valid = ~np.isnan(result)
        assert valid.sum() > 0
        np.testing.assert_allclose(result[valid], expected[valid], atol=1e-10)

    def test_reuse_across_fields(self):
        """Same interpolator should work for multiple value arrays."""
        nodes = _make_grid_nodes(128, 8)
        interp = FieldInterpolator(nodes)
        query_x, query_y = np.meshgrid(
            np.arange(10, 110, dtype=np.float64),
            np.arange(10, 110, dtype=np.float64),
        )
        r1 = interp.interpolate(nodes[:, 0], query_x, query_y)
        r2 = interp.interpolate(nodes[:, 1], query_x, query_y)

        valid = ~np.isnan(r1) & ~np.isnan(r2)
        np.testing.assert_allclose(r1[valid], query_x[valid], atol=1e-10)
        np.testing.assert_allclose(r2[valid], query_y[valid], atol=1e-10)

    def test_nan_outside_convex_hull(self):
        """Points outside node convex hull should be NaN (linear method)."""
        nodes = np.array(
            [[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float64
        )
        values = np.array([1.0, 2.0, 3.0, 4.0])
        interp = FieldInterpolator(nodes)
        qx = np.array([[0.0, 30.0]])
        qy = np.array([[0.0, 30.0]])
        result = interp.interpolate(values, qx, qy)
        assert np.isnan(result[0, 0])       # outside
        assert not np.isnan(result[0, 1])    # inside

    def test_clough_tocher_more_accurate(self):
        """CloughTocher should be more accurate than Linear for quadratic."""
        nodes = _make_grid_nodes(256, 16)
        cx, cy = 128.0, 128.0
        values = 0.01 * (nodes[:, 0] - cx) ** 2 + 0.008 * (nodes[:, 1] - cy) ** 2

        query_x, query_y = np.meshgrid(
            np.arange(20, 230, dtype=np.float64),
            np.arange(20, 230, dtype=np.float64),
        )
        gt = 0.01 * (query_x - cx) ** 2 + 0.008 * (query_y - cy) ** 2

        r_lin = FieldInterpolator(nodes, method="linear").interpolate(
            values, query_x, query_y
        )
        r_ct = FieldInterpolator(nodes, method="clough_tocher").interpolate(
            values, query_x, query_y
        )

        valid = ~np.isnan(r_lin) & ~np.isnan(r_ct)
        rmse_lin = np.sqrt(np.mean((r_lin[valid] - gt[valid]) ** 2))
        rmse_ct = np.sqrt(np.mean((r_ct[valid] - gt[valid]) ** 2))
        assert rmse_ct < rmse_lin

    def test_handles_nan_values(self):
        """Nodes with NaN values should be excluded, not crash."""
        nodes = _make_grid_nodes(64, 8)
        values = np.ones(len(nodes))
        values[0] = np.nan
        values[3] = np.nan
        interp = FieldInterpolator(nodes)
        qx, qy = np.meshgrid(
            np.arange(10, 50, dtype=np.float64),
            np.arange(10, 50, dtype=np.float64),
        )
        result = interp.interpolate(values, qx, qy)
        assert np.nansum(~np.isnan(result)) > 0

    def test_fill_outside_nearest(self):
        """fill_outside='nearest' should fill NaN outside convex hull."""
        nodes = _make_grid_nodes(64, 8)
        values = np.ones(len(nodes)) * 5.0
        interp = FieldInterpolator(nodes)
        # Query includes points outside the node range
        qx, qy = np.meshgrid(
            np.arange(0, 64, dtype=np.float64),
            np.arange(0, 64, dtype=np.float64),
        )
        result = interp.interpolate(values, qx, qy, fill_outside="nearest")
        assert not np.any(np.isnan(result))

    def test_invalid_method_raises(self):
        nodes = _make_grid_nodes(64, 8)
        with pytest.raises(ValueError, match="Unknown method"):
            FieldInterpolator(nodes, method="cubic_spline")


# ---- scatter_to_grid tests ----


class TestScatterToGrid:
    """Tests for scatter_to_grid convenience function."""

    def test_output_shape_auto(self):
        """Auto mode: output step = max(1, mesh_step // oversample)."""
        nodes = _make_grid_nodes(256, 16)
        values = np.ones(len(nodes))
        result, info = scatter_to_grid(
            nodes, values, img_shape=(256, 256), mesh_step=16,
            output_mode="auto", oversample=4,
        )
        assert info["output_step"] == 4
        # Grid dimension should be roughly img_size / output_step
        assert result.shape[0] <= 70
        assert result.shape[1] <= 70

    def test_output_shape_full(self):
        """Full mode: 1:1 with image pixels."""
        nodes = _make_grid_nodes(128, 16)
        values = np.ones(len(nodes))
        result, info = scatter_to_grid(
            nodes, values, img_shape=(128, 128), mesh_step=16,
            output_mode="full",
        )
        assert info["output_step"] == 1
        assert result.shape[0] >= 100

    def test_output_shape_preview(self):
        """Preview mode with max_output_pixels cap."""
        nodes = _make_grid_nodes(1024, 8)
        values = np.ones(len(nodes))
        result, info = scatter_to_grid(
            nodes, values, img_shape=(1024, 1024), mesh_step=8,
            output_mode="preview", max_output_pixels=512 * 512,
        )
        assert result.size <= 512 * 512 + 1024  # small margin

    def test_accuracy_preserved(self):
        """Interpolated values should match ground truth for linear field."""
        nodes = _make_grid_nodes(256, 8)
        values = 2.0 * nodes[:, 0] - 1.5 * nodes[:, 1]
        result, info = scatter_to_grid(
            nodes, values, img_shape=(256, 256), mesh_step=8,
            output_mode="auto", oversample=4,
        )
        xg, yg = info["x_grid"], info["y_grid"]
        expected = 2.0 * xg - 1.5 * yg
        valid = ~np.isnan(result)
        assert valid.sum() > 0
        # CloughTocher (default) introduces ~1e-7 floating-point noise on
        # linear fields due to C1 cubic patch fitting; 1e-6 is safe.
        np.testing.assert_allclose(result[valid], expected[valid], atol=1e-6)

    def test_returns_grid_coords(self):
        """Info dict must contain x_grid, y_grid for overlay alignment."""
        nodes = _make_grid_nodes(128, 8)
        values = np.ones(len(nodes))
        _, info = scatter_to_grid(
            nodes, values, img_shape=(128, 128), mesh_step=8,
        )
        assert "x_grid" in info
        assert "y_grid" in info
        assert "output_step" in info
        assert info["x_grid"].shape == info["y_grid"].shape

    def test_deformed_config(self):
        """Shifted nodes should shift grid coverage accordingly."""
        nodes = _make_grid_nodes(256, 16)
        shifted = nodes.copy()
        shifted[:, 0] += 20.0
        values = np.ones(len(nodes))
        _, info = scatter_to_grid(
            shifted, values, img_shape=(256, 256), mesh_step=16,
        )
        assert info["x_grid"].min() >= 15

    def test_reuse_interpolator(self):
        """Pre-built interpolator should be reusable across calls."""
        nodes = _make_grid_nodes(128, 8)
        interp = FieldInterpolator(nodes, method="linear")
        v1 = nodes[:, 0]
        v2 = nodes[:, 1]

        r1, _ = scatter_to_grid(
            nodes, v1, img_shape=(128, 128), mesh_step=8,
            interpolator=interp,
        )
        r2, _ = scatter_to_grid(
            nodes, v2, img_shape=(128, 128), mesh_step=8,
            interpolator=interp,
        )
        assert not np.array_equal(r1, r2)
