"""Tests for init_disp (FFT displacement cleanup + interleaved assembly)."""

import numpy as np
import pytest

from staq_dic.solver.init_disp import (
    init_disp,
    _inpaint_nans,
    _nan_neighbor_mean,
    _outlier_pass,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid_coords(ny: int, nx: int, step: float = 16.0):
    """Return 1-D x0, y0 coordinate vectors for a (ny, nx) grid."""
    x0 = np.arange(nx, dtype=np.float64) * step
    y0 = np.arange(ny, dtype=np.float64) * step
    return x0, y0


def _uniform_fields(ny: int, nx: int, u_val: float, v_val: float):
    """Create uniform displacement grids with no NaN or outliers."""
    u = np.full((ny, nx), u_val, dtype=np.float64)
    v = np.full((ny, nx), v_val, dtype=np.float64)
    cc = np.ones((ny, nx), dtype=np.float64)
    return u, v, cc


# ===========================================================================
# _nan_neighbor_mean
# ===========================================================================

class TestNanNeighborMean:
    def test_center_surrounded(self):
        """Center pixel with 4 known neighbors gets their mean."""
        field = np.full((3, 3), np.nan, dtype=np.float64)
        field[0, 1] = 2.0  # up
        field[2, 1] = 4.0  # down
        field[1, 0] = 6.0  # left
        field[1, 2] = 8.0  # right
        result = _nan_neighbor_mean(field)
        assert result[1, 1] == pytest.approx(5.0)

    def test_corner_only_two_neighbors(self):
        """Corner pixel (0,0) has only right and down neighbors."""
        field = np.array([
            [np.nan, 3.0, np.nan],
            [7.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ])
        result = _nan_neighbor_mean(field)
        assert result[0, 0] == pytest.approx(5.0)

    def test_all_nan_neighbors(self):
        """Pixel surrounded by all NaN → NaN output."""
        field = np.full((3, 3), np.nan, dtype=np.float64)
        result = _nan_neighbor_mean(field)
        assert np.isnan(result[1, 1])


# ===========================================================================
# _inpaint_nans
# ===========================================================================

class TestInpaintNans:
    def test_no_nans_unchanged(self):
        """Field without NaN should be returned unchanged."""
        field = np.arange(12, dtype=np.float64).reshape(3, 4)
        result = _inpaint_nans(field)
        np.testing.assert_array_equal(result, field)

    def test_single_nan_center(self):
        """Single NaN in center of uniform field fills to that value."""
        field = np.full((5, 5), 7.0, dtype=np.float64)
        field[2, 2] = np.nan
        result = _inpaint_nans(field)
        assert not np.any(np.isnan(result))
        assert result[2, 2] == pytest.approx(7.0)

    def test_nan_hole_fills(self):
        """A 2x2 NaN hole in a uniform field should converge to field value."""
        field = np.full((6, 6), 3.0, dtype=np.float64)
        field[2:4, 2:4] = np.nan
        result = _inpaint_nans(field)
        assert not np.any(np.isnan(result))
        np.testing.assert_allclose(result[2:4, 2:4], 3.0, atol=1e-10)

    def test_convergence_linear_gradient(self):
        """NaN in center of a linear gradient should interpolate smoothly."""
        field = np.arange(25, dtype=np.float64).reshape(5, 5)
        original_val = field[2, 2]
        field[2, 2] = np.nan
        result = _inpaint_nans(field)
        # Neighbor mean of {7, 17, 11, 13} = 12.0, which equals original
        assert result[2, 2] == pytest.approx(original_val, abs=0.5)

    def test_all_nan_stays_nan(self):
        """All-NaN field cannot be inpainted; remains NaN."""
        field = np.full((4, 4), np.nan, dtype=np.float64)
        result = _inpaint_nans(field, max_iter=10)
        assert np.all(np.isnan(result))

    def test_max_iter_respected(self):
        """With max_iter=0, no iteration occurs and NaN remains."""
        field = np.full((3, 3), 1.0, dtype=np.float64)
        field[1, 1] = np.nan
        result = _inpaint_nans(field, max_iter=0)
        assert np.isnan(result[1, 1])


# ===========================================================================
# _outlier_pass
# ===========================================================================

class TestOutlierPass:
    def test_uniform_field_no_change(self):
        """Uniform field has no outliers; borders still get NaN-ed."""
        ny, nx = 5, 6
        u = np.full((ny, nx), 2.0, dtype=np.float64)
        v = np.full((ny, nx), -1.0, dtype=np.float64)
        u_out, v_out = _outlier_pass(u, v, use_8_neighbors=True, n_sigma=3.0)

        # Interior should survive
        assert not np.isnan(u_out[2, 3])
        assert not np.isnan(v_out[2, 3])
        # Borders should be NaN
        assert np.all(np.isnan(u_out[0, :]))
        assert np.all(np.isnan(u_out[-1, :]))
        assert np.all(np.isnan(u_out[:, 0]))
        assert np.all(np.isnan(u_out[:, -1]))

    def test_single_outlier_detected_8_neighbor(self):
        """A single extreme outlier pixel should be NaN-ed (8-neighbor)."""
        ny, nx = 6, 6
        u = np.full((ny, nx), 5.0, dtype=np.float64)
        v = np.full((ny, nx), 5.0, dtype=np.float64)
        u[3, 3] = 5000.0  # extreme outlier
        u_out, v_out = _outlier_pass(u, v, use_8_neighbors=True, n_sigma=3.0)
        assert np.isnan(u_out[3, 3])

    def test_single_outlier_detected_4_neighbor(self):
        """A single extreme outlier should be NaN-ed (4-neighbor)."""
        ny, nx = 6, 6
        u = np.full((ny, nx), 0.0, dtype=np.float64)
        v = np.full((ny, nx), 0.0, dtype=np.float64)
        v[3, 3] = -999.0  # extreme outlier in v
        u_out, v_out = _outlier_pass(u, v, use_8_neighbors=False, n_sigma=2.0)
        assert np.isnan(v_out[3, 3])

    def test_outlier_nans_neighbors(self):
        """When outlier detected, its 3x3 neighborhood should all be NaN."""
        ny, nx = 7, 7
        u = np.zeros((ny, nx), dtype=np.float64)
        v = np.zeros((ny, nx), dtype=np.float64)
        u[3, 3] = 1000.0
        u_out, _ = _outlier_pass(u, v, use_8_neighbors=True, n_sigma=2.0)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                assert np.isnan(u_out[3 + dy, 3 + dx]), (
                    f"Neighbor ({3+dy}, {3+dx}) should be NaN"
                )

    def test_small_grid_no_crash(self):
        """2x2 grid is too small for interior loop; should return unchanged."""
        u = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([[5.0, 6.0], [7.0, 8.0]])
        u_out, v_out = _outlier_pass(u, v, use_8_neighbors=True, n_sigma=3.0)
        np.testing.assert_array_equal(u_out, u)
        np.testing.assert_array_equal(v_out, v)

    def test_3x3_grid_only_borders(self):
        """3x3 grid has exactly one interior pixel (1,1); borders NaN-ed."""
        u = np.ones((3, 3), dtype=np.float64)
        v = np.ones((3, 3), dtype=np.float64)
        u_out, v_out = _outlier_pass(u, v, use_8_neighbors=False, n_sigma=2.0)
        # Borders should be NaN
        assert np.all(np.isnan(u_out[0, :]))
        # Interior pixel (1,1) in uniform field should survive (no outlier)
        # But border NaN-ing may NaN it too since it IS on the border
        # Actually (1,1) is NOT on the border: row 0, row 2, col 0, col 2 are borders
        # (1,1) is interior. For a 3x3, row 1, col 1 is the only interior pixel.
        # In a uniform field std=0, so |val - avg| = 0 which is not > 0.
        # So interior survives the outlier check.
        # But border NaN-ing at the end sets rows 0,-1 and cols 0,-1 to NaN.
        # For a 3x3: row 0 = row 0, row -1 = row 2, col 0 = col 0, col -1 = col 2
        # So only (1,1) survives.
        assert not np.isnan(u_out[1, 1])


# ===========================================================================
# init_disp (integration)
# ===========================================================================

class TestInitDisp:
    def test_clean_uniform_passthrough(self):
        """Uniform displacement with no NaN/outliers passes through."""
        ny, nx = 4, 5
        u_val, v_val = 2.5, -1.3
        u, v, cc = _uniform_fields(ny, nx, u_val, v_val)
        x0, y0 = _make_grid_coords(ny, nx)

        U0 = init_disp(u, v, cc, x0, y0, method=1)

        n_nodes = ny * nx
        assert U0.shape == (2 * n_nodes,)
        # All u-components should be u_val, all v-components should be v_val
        np.testing.assert_allclose(U0[0::2], u_val, atol=1e-10)
        np.testing.assert_allclose(U0[1::2], v_val, atol=1e-10)

    def test_nan_inpainting_method0(self):
        """Method 0 inpaints NaN but skips outlier removal."""
        ny, nx = 5, 5
        u = np.full((ny, nx), 4.0, dtype=np.float64)
        v = np.full((ny, nx), -2.0, dtype=np.float64)
        u[2, 2] = np.nan
        v[2, 2] = np.nan
        x0, y0 = _make_grid_coords(ny, nx)
        cc = np.ones((ny, nx))

        U0 = init_disp(u, v, cc, x0, y0, method=0)

        assert not np.any(np.isnan(U0))
        # The inpainted value should be close to the uniform value
        # Node (2,2) in .T.ravel() ordering: index = ix*ny + iy = 2*5+2 = 12
        assert U0[2 * 12] == pytest.approx(4.0, abs=0.1)
        assert U0[2 * 12 + 1] == pytest.approx(-2.0, abs=0.1)

    def test_method0_vs_method1_outlier(self):
        """Method 0 should NOT remove outliers; method 1 should."""
        ny, nx = 6, 6
        u_base = np.full((ny, nx), 1.0, dtype=np.float64)
        v_base = np.full((ny, nx), 1.0, dtype=np.float64)
        # Place an extreme outlier at interior point
        u_base[3, 3] = 5000.0
        x0, y0 = _make_grid_coords(ny, nx)
        cc = np.ones((ny, nx))

        U0_m0 = init_disp(u_base.copy(), v_base.copy(), cc, x0, y0, method=0)
        U0_m1 = init_disp(u_base.copy(), v_base.copy(), cc, x0, y0, method=1)

        # In method 0, the outlier value should remain
        # Node (3,3): index = 3*6+3 = 21
        assert U0_m0[2 * 21] == pytest.approx(5000.0, rel=1e-6)

        # In method 1, the outlier should be replaced by something closer to 1.0
        assert abs(U0_m1[2 * 21] - 1.0) < 100.0  # much less than 5000

    def test_interleaving_format(self):
        """Output should be [u0, v0, u1, v1, ...] interleaved."""
        ny, nx = 3, 4
        u = np.arange(12, dtype=np.float64).reshape(ny, nx) + 100
        v = np.arange(12, dtype=np.float64).reshape(ny, nx) + 200
        x0, y0 = _make_grid_coords(ny, nx)
        cc = np.ones((ny, nx))

        U0 = init_disp(u, v, cc, x0, y0, method=0)

        assert U0.shape == (2 * ny * nx,)
        # Even indices are u, odd indices are v
        u_flat = U0[0::2]
        v_flat = U0[1::2]
        assert u_flat.shape == (ny * nx,)
        assert v_flat.shape == (ny * nx,)

    def test_node_ordering_transpose_ravel(self):
        """Verify .T.ravel() produces node index = ix * ny + iy ordering.

        For a (ny=3, nx=4) grid:
          u_clean[iy, ix]  -->  node_index = ix * ny + iy
          So u_clean[0, 0] -> node 0, u_clean[1, 0] -> node 1,
             u_clean[2, 0] -> node 2, u_clean[0, 1] -> node 3, etc.
        """
        ny, nx = 3, 4
        # Assign unique values to verify ordering
        u = np.arange(12, dtype=np.float64).reshape(ny, nx)
        v = np.zeros((ny, nx), dtype=np.float64)
        x0, y0 = _make_grid_coords(ny, nx)
        cc = np.ones((ny, nx))

        U0 = init_disp(u, v, cc, x0, y0, method=0)
        u_flat = U0[0::2]

        # Check: node at (iy=1, ix=2) has node_index = 2*3 + 1 = 7
        expected_val = u[1, 2]  # = 1*4 + 2 = 6 (from arange)
        assert u_flat[7] == pytest.approx(expected_val)

        # Check: node at (iy=0, ix=3) has node_index = 3*3 + 0 = 9
        expected_val2 = u[0, 3]  # = 0*4 + 3 = 3
        assert u_flat[9] == pytest.approx(expected_val2)

    def test_output_length(self):
        """Output vector length should be 2 * ny * nx."""
        ny, nx = 4, 5
        u, v, cc = _uniform_fields(ny, nx, 0.0, 0.0)
        x0, y0 = _make_grid_coords(ny, nx)
        U0 = init_disp(u, v, cc, x0, y0, method=1)
        assert U0.shape == (2 * ny * nx,)
        assert U0.dtype == np.float64

    def test_does_not_mutate_input(self):
        """Input arrays should not be modified."""
        ny, nx = 4, 5
        u = np.full((ny, nx), 3.0, dtype=np.float64)
        v = np.full((ny, nx), -1.0, dtype=np.float64)
        u[2, 2] = np.nan
        u_orig = u.copy()
        v_orig = v.copy()
        x0, y0 = _make_grid_coords(ny, nx)
        cc = np.ones((ny, nx))

        init_disp(u, v, cc, x0, y0, method=1)

        np.testing.assert_array_equal(u, u_orig)
        np.testing.assert_array_equal(v, v_orig)

    def test_scattered_nans_all_filled(self):
        """Scattered NaN values should all be filled in output."""
        rng = np.random.default_rng(42)
        ny, nx = 6, 6
        u = np.full((ny, nx), 2.0, dtype=np.float64)
        v = np.full((ny, nx), -1.0, dtype=np.float64)
        # Scatter ~25% NaN
        nan_mask = rng.random((ny, nx)) < 0.25
        u[nan_mask] = np.nan
        v[nan_mask] = np.nan
        x0, y0 = _make_grid_coords(ny, nx)
        cc = np.ones((ny, nx))

        U0 = init_disp(u, v, cc, x0, y0, method=1)
        assert not np.any(np.isnan(U0))

    def test_all_nan_input(self):
        """All-NaN input should produce all-NaN output (cannot inpaint)."""
        ny, nx = 4, 5
        u = np.full((ny, nx), np.nan, dtype=np.float64)
        v = np.full((ny, nx), np.nan, dtype=np.float64)
        x0, y0 = _make_grid_coords(ny, nx)
        cc = np.ones((ny, nx))

        U0 = init_disp(u, v, cc, x0, y0, method=0)
        # All NaN input with no valid neighbors -> NaN propagates
        assert np.all(np.isnan(U0))

    def test_2x2_grid_no_crash(self):
        """2x2 grid (too small for outlier interior) should not crash."""
        ny, nx = 2, 2
        u, v, cc = _uniform_fields(ny, nx, 1.0, 2.0)
        x0, y0 = _make_grid_coords(ny, nx)

        U0 = init_disp(u, v, cc, x0, y0, method=1)
        assert U0.shape == (8,)
        # With method=1, outlier pass borders everything NaN, then inpaint
        # fills from (empty) neighbors -> all NaN.
        # Actually for 2x2, _outlier_pass returns unchanged (ny<3 guard).
        # So values survive.
        assert not np.any(np.isnan(U0))

    def test_3x3_grid_method1(self):
        """3x3 grid with method=1 should not crash and produce finite output."""
        ny, nx = 3, 3
        u = np.full((ny, nx), 5.0, dtype=np.float64)
        v = np.full((ny, nx), -3.0, dtype=np.float64)
        x0, y0 = _make_grid_coords(ny, nx)
        cc = np.ones((ny, nx))

        U0 = init_disp(u, v, cc, x0, y0, method=1)
        assert U0.shape == (18,)
        # _outlier_pass NaNs borders but interior (1,1) survives.
        # Then inpaint fills the 8 border values from center.
        assert not np.any(np.isnan(U0))
        # All should converge back to uniform values
        np.testing.assert_allclose(U0[0::2], 5.0, atol=1e-8)
        np.testing.assert_allclose(U0[1::2], -3.0, atol=1e-8)

    def test_cc_max_unused(self):
        """cc_max values should not affect the result (API compatibility arg)."""
        ny, nx = 4, 5
        u, v, _ = _uniform_fields(ny, nx, 1.0, 2.0)
        x0, y0 = _make_grid_coords(ny, nx)
        cc_ones = np.ones((ny, nx))
        cc_random = np.random.default_rng(0).random((ny, nx))

        U0_a = init_disp(u.copy(), v.copy(), cc_ones, x0, y0, method=0)
        U0_b = init_disp(u.copy(), v.copy(), cc_random, x0, y0, method=0)
        np.testing.assert_array_equal(U0_a, U0_b)
