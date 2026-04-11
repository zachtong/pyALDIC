"""Tests for outlier_detection (detect_bad_points + fill_nan_idw)."""

import numpy as np
import pytest

from al_dic.utils.outlier_detection import detect_bad_points, fill_nan_idw


class TestDetectBadPoints:
    def _make_coords(self, n=10):
        return np.column_stack([np.arange(n) * 16.0, np.zeros(n)])

    def test_all_good(self):
        """All converged → no bad points."""
        coords = self._make_coords(10)
        conv = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5, 3], dtype=np.int64)
        bad_pts, bad_num = detect_bad_points(conv, 100, coords)
        assert len(bad_pts) == 0
        assert bad_num == 0

    def test_negative_convergence(self):
        """Negative convergence count → bad point."""
        coords = self._make_coords(5)
        conv = np.array([3, -1, 4, 5, 3], dtype=np.int64)
        bad_pts, _ = detect_bad_points(conv, 100, coords)
        assert 1 in bad_pts

    def test_exceeded_max_iter(self):
        """conv_iter > max_iter-1 → bad point."""
        coords = self._make_coords(5)
        conv = np.array([3, 4, 100, 5, 3], dtype=np.int64)
        bad_pts, _ = detect_bad_points(conv, 100, coords)
        assert 2 in bad_pts

    def test_mask_only_excluded_from_count(self):
        """conv_iter == max_iter+2 should be in bad_pts but NOT in bad_pt_num."""
        coords = self._make_coords(5)
        conv = np.array([3, 4, 102, 5, 3], dtype=np.int64)
        bad_pts, bad_num = detect_bad_points(conv, 100, coords)
        assert 2 in bad_pts
        assert bad_num == 0  # mask-only failure excluded

    def test_statistical_outlier(self):
        """A node with high iteration count should be flagged."""
        coords = self._make_coords(10)
        conv = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 50], dtype=np.int64)
        bad_pts, _ = detect_bad_points(conv, 100, coords, sigma_factor=1.0, min_threshold=6)
        assert 9 in bad_pts

    def test_empty(self):
        """Empty input → empty output."""
        coords = np.empty((0, 2))
        conv = np.empty(0, dtype=np.int64)
        bad_pts, bad_num = detect_bad_points(conv, 100, coords)
        assert len(bad_pts) == 0


class TestFillNanRbf:
    def test_no_nan(self):
        """No NaN values → return unchanged."""
        coords = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float64)
        V = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 3 nodes, 2 components
        result = fill_nan_idw(V, coords, n_components=2)
        np.testing.assert_array_equal(result, V)

    def test_fills_nan(self):
        """NaN values should be filled by interpolation."""
        # Use a 2D grid (non-collinear) so LinearNDInterpolator works
        coords = np.array([
            [0, 0], [10, 0], [20, 0],
            [0, 10], [10, 10], [20, 10],
        ], dtype=np.float64)
        # 6 nodes, 2 components. Node 2 is NaN.
        V = np.array([1.0, 0.0, 2.0, 0.0, np.nan, np.nan,
                       1.5, 0.0, 2.5, 0.0, 3.0, 0.0])
        result = fill_nan_idw(V, coords, n_components=2)
        assert not np.any(np.isnan(result))
        # Interpolated u at node 2 should be reasonable (between neighbor values)
        assert 0.5 < result[4] < 4.0

    def test_fills_nan_collinear(self):
        """Collinear points should fall back to nearest-neighbor."""
        coords = np.array([
            [0, 0], [10, 0], [20, 0], [30, 0]
        ], dtype=np.float64)
        V = np.array([1.0, 0.0, np.nan, np.nan, 3.0, 0.0, 4.0, 0.0])
        result = fill_nan_idw(V, coords, n_components=2)
        assert not np.any(np.isnan(result))

    def test_four_components(self):
        """Should work with 4-component (F gradient) vectors."""
        coords = np.array([
            [0, 0], [10, 0], [20, 0],
            [0, 10], [10, 10], [20, 10],
        ], dtype=np.float64)
        V = np.zeros(24)  # 6 nodes * 4 components
        V[0:4] = [1.0, 2.0, 3.0, 4.0]
        V[4:8] = [2.0, 3.0, 4.0, 5.0]
        V[8:12] = [np.nan, np.nan, np.nan, np.nan]  # Node 2 NaN
        V[12:16] = [1.5, 2.5, 3.5, 4.5]
        V[16:20] = [2.5, 3.5, 4.5, 5.5]
        V[20:24] = [3.0, 4.0, 5.0, 6.0]

        result = fill_nan_idw(V, coords, n_components=4)
        assert not np.any(np.isnan(result))
        # Node 2 F11 should be interpolated reasonably
        assert abs(result[8] - 3.0) < 1.5

    def test_all_nan_returns_zeros(self):
        """All NaN → should return zeros with warning."""
        coords = np.array([[0, 0], [10, 0]], dtype=np.float64)
        V = np.array([np.nan, np.nan, np.nan, np.nan])
        with pytest.warns(UserWarning, match="All nodes are NaN"):
            result = fill_nan_idw(V, coords, n_components=2)
        np.testing.assert_array_equal(result, 0.0)
