"""Tests for smooth_field_sparse (sparse Gaussian smoothing)."""

import numpy as np
import pytest

from al_dic.strain.smooth_field import smooth_field_sparse
from al_dic.utils.region_analysis import NodeRegionMap


def _make_region_map(n_nodes):
    """Create a single-region map with all nodes."""
    return NodeRegionMap(
        region_node_lists=[np.arange(n_nodes, dtype=np.int64)],
        n_regions=1,
    )


class TestSmoothFieldSparse:
    def test_zero_sigma_noop(self):
        """sigma=0 should return input unchanged."""
        coords = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        region_map = _make_region_map(3)

        result = smooth_field_sparse(values, coords, sigma=0.0, region_map=region_map)
        np.testing.assert_array_equal(result, values)

    def test_constant_field_unchanged(self):
        """Smoothing a constant field should return the same constant."""
        coords = np.array([
            [0, 0], [16, 0], [32, 0],
            [0, 16], [16, 16], [32, 16],
        ], dtype=np.float64)
        n = coords.shape[0]
        values = np.ones(2 * n)  # constant u=1, v=1
        region_map = _make_region_map(n)

        result = smooth_field_sparse(values, coords, sigma=10.0, region_map=region_map)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_smoothing_reduces_noise(self):
        """Smoothed values should have smaller variance than input."""
        coords = np.array([
            [0, 0], [16, 0], [32, 0], [48, 0],
            [0, 16], [16, 16], [32, 16], [48, 16],
        ], dtype=np.float64)
        n = coords.shape[0]
        region_map = _make_region_map(n)

        rng = np.random.RandomState(42)
        values = rng.randn(2 * n)

        result = smooth_field_sparse(values, coords, sigma=20.0, region_map=region_map)

        assert np.std(result[0::2]) < np.std(values[0::2])
        assert np.std(result[1::2]) < np.std(values[1::2])

    def test_no_nan_propagation(self):
        """NaN values in non-NaN nodes should not propagate."""
        coords = np.array([
            [0, 0], [16, 0], [32, 0],
            [0, 16], [16, 16], [32, 16],
        ], dtype=np.float64)
        n = coords.shape[0]
        values = np.ones(2 * n)
        values[4] = np.nan  # Node 2 u is NaN
        values[5] = np.nan  # Node 2 v is NaN
        region_map = _make_region_map(n)

        result = smooth_field_sparse(values, coords, sigma=20.0, region_map=region_map)

        # Non-NaN nodes should still have finite values
        for i in [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]:
            assert np.isfinite(result[i])

    def test_four_components(self):
        """Should work with 4-component (gradient) fields."""
        coords = np.array([
            [0, 0], [16, 0], [32, 0],
            [0, 16], [16, 16], [32, 16],
        ], dtype=np.float64)
        n = coords.shape[0]
        values = np.ones(4 * n)
        region_map = _make_region_map(n)

        result = smooth_field_sparse(
            values, coords, sigma=10.0, region_map=region_map, n_components=4,
        )
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_two_regions_independent(self):
        """Nodes in separate regions should not affect each other."""
        coords = np.array([
            [0, 0], [16, 0],       # Region A
            [1000, 0], [1016, 0],   # Region B (far away)
        ], dtype=np.float64)
        region_map = NodeRegionMap(
            region_node_lists=[
                np.array([0, 1], dtype=np.int64),
                np.array([2, 3], dtype=np.int64),
            ],
            n_regions=2,
        )
        values = np.array([1.0, 0.0, 1.0, 0.0, 10.0, 0.0, 10.0, 0.0])

        result = smooth_field_sparse(values, coords, sigma=50.0, region_map=region_map)

        # Region A averages should be ~1, region B ~10 (no cross-contamination)
        assert result[0] < 5.0  # Region A u-values
        assert result[4] > 5.0  # Region B u-values

    def test_output_shape(self):
        """Output shape should match input."""
        coords = np.array([[0, 0], [16, 0]], dtype=np.float64)
        values = np.zeros(4)
        region_map = _make_region_map(2)
        result = smooth_field_sparse(values, coords, sigma=10.0, region_map=region_map)
        assert result.shape == (4,)
