"""Tests for interpolation utilities."""

import numpy as np
import pytest

from staq_dic.utils.interpolation import (
    scattered_interpolant,
    fill_nan_scattered,
    interp2_bicubic,
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
