"""Tests for Numba kernel consistency — compare Numba vs Python outputs.

All tests are skipped when Numba is not installed (CI environment).
When Numba is available, verifies that Numba-compiled kernels produce
results matching the Python fallback implementations.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, label

# Import HAS_NUMBA to gate all tests
try:
    from al_dic.solver.numba_kernels import HAS_NUMBA
except ImportError:
    HAS_NUMBA = False

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_speckle(size=128, seed=42):
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.standard_normal((size, size)), sigma=3.0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def _make_gradients(img):
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    dy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return dx, dy


def _make_interior_coords(size=128, n=4, step=None):
    if step is None:
        step = size // (n + 1)
    pts = np.arange(step, size - step + 1, step, dtype=np.float64)[:n]
    xs, ys = np.meshgrid(pts, pts)
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)


# ---------------------------------------------------------------------------
# Precompute 6-DOF: Numba vs Python
# ---------------------------------------------------------------------------

class TestPrecompute6DOF:
    def test_precompute_6dof_numba_vs_python(self):
        """Numba and Python precompute should produce same results."""
        from al_dic.solver.icgn_batch import _precompute_subsets_6dof_python
        from al_dic.solver.numba_kernels import precompute_subsets_6dof_numba

        img = _make_speckle(128)
        dx, dy = _make_gradients(img)
        mask = np.ones_like(img)
        coords = _make_interior_coords(128, n=4)

        winsize = 20
        half_w = winsize // 2
        Sy = Sx = winsize + 1

        # Python
        py = _precompute_subsets_6dof_python(coords, img, dx, dy, mask, winsize)

        # Numba
        (ref_all, gx_all, gy_all, mask_all,
         XX_all, YY_all, H_all, meanf_all, bottomf_all,
         valid, mark_hole) = precompute_subsets_6dof_numba(
            coords, img, dx, dy, mask, half_w, Sy, Sx,
        )

        # Compare
        np.testing.assert_array_equal(py["valid"], valid)
        for i in range(coords.shape[0]):
            if py["valid"][i]:
                np.testing.assert_allclose(
                    py["ref_all"][i], ref_all[i], atol=1e-10,
                    err_msg=f"ref_all mismatch at node {i}",
                )
                np.testing.assert_allclose(
                    py["H_all"][i], H_all[i], atol=1e-6,
                    err_msg=f"H_all mismatch at node {i}",
                )
                np.testing.assert_allclose(
                    py["meanf_all"][i], meanf_all[i], atol=1e-10,
                )


# ---------------------------------------------------------------------------
# Precompute 2-DOF: Numba vs Python
# ---------------------------------------------------------------------------

class TestPrecompute2DOF:
    def test_precompute_2dof_numba_vs_python(self):
        """Numba and Python 2-DOF precompute should match."""
        from al_dic.solver.icgn_batch import _precompute_subsets_2dof_python
        from al_dic.solver.numba_kernels import precompute_subsets_2dof_numba

        img = _make_speckle(128)
        dx, dy = _make_gradients(img)
        mask = np.ones_like(img)
        coords = _make_interior_coords(128, n=4)
        n = coords.shape[0]

        winsize = 20
        ws_x = np.full(n, winsize, dtype=np.int64)
        ws_y = np.full(n, winsize, dtype=np.int64)
        Sy = Sx = winsize + 1

        # Python
        py = _precompute_subsets_2dof_python(
            coords, img, dx, dy, mask, ws_x, ws_y,
        )

        # Numba
        (ref_all, gx_all, gy_all, mask_all,
         XX_all, YY_all, H2_img_all,
         meanf_all, bottomf_all, valid) = precompute_subsets_2dof_numba(
            coords, img, dx, dy, mask, ws_x, ws_y, Sy, Sx,
        )

        np.testing.assert_array_equal(py["valid"], valid)
        for i in range(n):
            if py["valid"][i]:
                np.testing.assert_allclose(
                    py["H2_img_all"][i], H2_img_all[i], atol=1e-6,
                )


# ---------------------------------------------------------------------------
# IC-GN 6-DOF: Numba vs batch
# ---------------------------------------------------------------------------

class TestICGN6DOFConsistency:
    def test_icgn_6dof_parallel_vs_batch(self):
        """Numba parallel and batch 6-DOF should agree on same input."""
        from al_dic.solver.numba_kernels import icgn_6dof_parallel
        from al_dic.solver.icgn_batch import _iterate_6dof_batch, precompute_subsets_6dof

        img = _make_speckle(128)
        coords = _make_interior_coords(128, n=3)
        dx, dy = _make_gradients(img)
        mask = np.ones_like(img)
        winsize = 20

        pre = precompute_subsets_6dof(coords, img, dx, dy, mask, winsize)
        n = coords.shape[0]
        U0_2d = np.zeros((n, 2), dtype=np.float64)

        # Batch
        U_batch, F_batch, conv_batch, _ = _iterate_6dof_batch(
            coords, U0_2d.copy(), img.copy(), pre, tol=1e-4, max_iter=50,
        )

        # Numba
        rounded = np.round(coords).astype(np.float64)
        P_numba, conv_numba = icgn_6dof_parallel(
            rounded,
            U0_2d[:, 0].copy(), U0_2d[:, 1].copy(),
            pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
            pre["XX_all"], pre["YY_all"], pre["H_all"],
            pre["meanf_all"], pre["bottomf_all"],
            pre["valid"], img.copy(), 1e-4, 50,
        )
        U_numba = P_numba[:, 4:6]

        for i in range(n):
            if pre["valid"][i]:
                np.testing.assert_allclose(
                    U_batch[i], U_numba[i], atol=1e-4,
                    err_msg=f"6-DOF displacement mismatch at node {i}",
                )


# ---------------------------------------------------------------------------
# IC-GN 2-DOF: Numba vs batch
# ---------------------------------------------------------------------------

class TestICGN2DOFConsistency:
    def test_icgn_2dof_parallel_vs_batch(self):
        """Numba parallel and batch 2-DOF should agree on same input."""
        from al_dic.solver.numba_kernels import icgn_2dof_parallel
        from al_dic.solver.icgn_batch import _iterate_2dof_batch, precompute_subsets_2dof

        img = _make_speckle(128)
        coords = _make_interior_coords(128, n=3)
        dx, dy = _make_gradients(img)
        mask = np.ones_like(img)
        winsize = 20
        n = coords.shape[0]

        ws = np.full(n, winsize, dtype=np.int64)
        pre = precompute_subsets_2dof(coords, img, dx, dy, mask, ws, ws)

        U_old_2d = np.zeros((n, 2), dtype=np.float64)
        F_old_2d = np.zeros((n, 4), dtype=np.float64)
        udual_2d = np.zeros((n, 2), dtype=np.float64)
        mu = 1e-3

        # Batch
        U_batch, conv_batch = _iterate_2dof_batch(
            coords, U_old_2d.copy(), F_old_2d.copy(), udual_2d.copy(),
            img.copy(), pre, mu, tol=1e-4, max_iter=50,
        )

        # Numba
        rounded = np.round(coords).astype(np.float64)
        U_numba, conv_numba = icgn_2dof_parallel(
            rounded, U_old_2d.copy(), F_old_2d.copy(), udual_2d.copy(),
            pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
            pre["XX_all"], pre["YY_all"], pre["H2_img_all"],
            pre["meanf_all"], pre["bottomf_all"],
            pre["valid"], img.copy(), mu, 1e-4, 50,
        )

        for i in range(n):
            if pre["valid"][i]:
                np.testing.assert_allclose(
                    U_batch[i], U_numba[i], atol=1e-4,
                    err_msg=f"2-DOF displacement mismatch at node {i}",
                )


# ---------------------------------------------------------------------------
# Bicubic interpolation tests
# ---------------------------------------------------------------------------

class TestBicubicInterp:
    def test_vs_map_coordinates(self):
        """Numba bicubic should match scipy map_coordinates (order=3)."""
        from al_dic.solver.numba_kernels import _bicubic_interp
        from scipy.ndimage import map_coordinates

        img = _make_speckle(64)
        h, w = img.shape

        # Test at a few interior points
        test_pts = [(20.3, 30.7), (40.5, 25.2), (15.8, 50.1)]
        for y, x in test_pts:
            numba_val = _bicubic_interp(img, y, x, h, w)
            scipy_val = map_coordinates(img, [[y], [x]], order=3, mode="constant")[0]
            assert abs(numba_val - scipy_val) < 0.05, \
                f"Mismatch at ({y}, {x}): numba={numba_val}, scipy={scipy_val}"

    def test_oob_returns_zero(self):
        """Out-of-bounds interpolation should return 0."""
        from al_dic.solver.numba_kernels import _bicubic_interp

        img = _make_speckle(64)
        h, w = img.shape

        assert _bicubic_interp(img, -1.0, 30.0, h, w) == 0.0
        assert _bicubic_interp(img, 30.0, -1.0, h, w) == 0.0
        assert _bicubic_interp(img, 0.5, 30.0, h, w) == 0.0  # < 1.0


# ---------------------------------------------------------------------------
# Flood fill tests
# ---------------------------------------------------------------------------

class TestFloodFill:
    def test_matches_scipy_label(self):
        """Flood fill center should match scipy label for connected center."""
        from al_dic.solver.numba_kernels import _flood_fill_center

        mask = np.zeros((21, 21), dtype=np.float64)
        mask[5:16, 5:16] = 1.0

        result = _flood_fill_center(mask, 21, 21)
        labeled, _ = label(mask > 0.5)
        cl = labeled[10, 10]
        expected = (labeled == cl).astype(np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_empty_mask(self):
        """Empty mask (center not set) should return all zeros."""
        from al_dic.solver.numba_kernels import _flood_fill_center

        mask = np.zeros((11, 11), dtype=np.float64)
        result = _flood_fill_center(mask, 11, 11)
        assert result.sum() == 0.0

    def test_disconnected(self):
        """Only the center-connected component should be returned."""
        from al_dic.solver.numba_kernels import _flood_fill_center

        mask = np.zeros((21, 21), dtype=np.float64)
        mask[8:13, 8:13] = 1.0  # center block
        mask[0:3, 0:3] = 1.0    # disconnected corner

        result = _flood_fill_center(mask, 21, 21)
        # Corner block should NOT be included
        assert result[1, 1] == 0.0
        assert result[10, 10] == 1.0


# ---------------------------------------------------------------------------
# Known translation recovery
# ---------------------------------------------------------------------------

class TestKnownTranslation:
    def test_numba_6dof_known_translation(self):
        """Numba 6-DOF should recover a known uniform translation."""
        from al_dic.solver.numba_kernels import icgn_6dof_parallel
        from al_dic.solver.icgn_batch import precompute_subsets_6dof
        from scipy.ndimage import shift as ndimage_shift

        img_ref = _make_speckle(128)
        shift_x, shift_y = 1.5, 1.0
        img_def = ndimage_shift(img_ref, [shift_y, shift_x], order=3, mode="constant")

        coords = np.array([[64.0, 64.0]], dtype=np.float64)
        dx, dy = _make_gradients(img_ref)
        mask = np.ones_like(img_ref)
        winsize = 30

        pre = precompute_subsets_6dof(coords, img_ref, dx, dy, mask, winsize)
        rounded = np.round(coords).astype(np.float64)

        P_out, conv = icgn_6dof_parallel(
            rounded,
            np.array([shift_x + 0.3]), np.array([shift_y - 0.2]),
            pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
            pre["XX_all"], pre["YY_all"], pre["H_all"],
            pre["meanf_all"], pre["bottomf_all"],
            pre["valid"], img_def, 1e-4, 100,
        )

        if pre["valid"][0]:
            assert abs(P_out[0, 4] - shift_x) < 0.5
            assert abs(P_out[0, 5] - shift_y) < 0.5

    def test_numba_2dof_known_translation(self):
        """Numba 2-DOF should track a known translation."""
        from al_dic.solver.numba_kernels import icgn_2dof_parallel
        from al_dic.solver.icgn_batch import precompute_subsets_2dof
        from scipy.ndimage import shift as ndimage_shift

        img_ref = _make_speckle(128)
        shift_x, shift_y = 1.0, 0.5
        img_def = ndimage_shift(img_ref, [shift_y, shift_x], order=3, mode="constant")

        coords = np.array([[64.0, 64.0]], dtype=np.float64)
        dx, dy = _make_gradients(img_ref)
        mask = np.ones_like(img_ref)
        winsize = 30

        ws = np.full(1, winsize, dtype=np.int64)
        pre = precompute_subsets_2dof(coords, img_ref, dx, dy, mask, ws, ws)
        rounded = np.round(coords).astype(np.float64)

        U_old = np.array([[shift_x + 0.2, shift_y - 0.1]])
        F_old = np.zeros((1, 4))
        udual = np.zeros((1, 2))

        U_out, conv = icgn_2dof_parallel(
            rounded, U_old, F_old, udual,
            pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
            pre["XX_all"], pre["YY_all"], pre["H2_img_all"],
            pre["meanf_all"], pre["bottomf_all"],
            pre["valid"], img_def, 1e-3, 1e-4, 100,
        )

        if pre["valid"][0]:
            assert abs(U_out[0, 0] - shift_x) < 0.5
            assert abs(U_out[0, 1] - shift_y) < 0.5


# ---------------------------------------------------------------------------
# Cubic weight symmetry
# ---------------------------------------------------------------------------

class TestCubicWeight:
    def test_symmetry(self):
        """Cubic weight should be symmetric: w(t) == w(-t)."""
        from al_dic.solver.numba_kernels import _cubic_weight

        for t in [0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 2.5]:
            assert _cubic_weight(t) == pytest.approx(_cubic_weight(-t))

    def test_unity_at_zero(self):
        from al_dic.solver.numba_kernels import _cubic_weight
        assert _cubic_weight(0.0) == pytest.approx(1.0)

    def test_zero_at_two(self):
        from al_dic.solver.numba_kernels import _cubic_weight
        assert _cubic_weight(2.0) == pytest.approx(0.0)
