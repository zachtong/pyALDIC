"""Tests for local_icgn (parallel dispatcher for per-node IC-GN)."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as ndimage_shift

from al_dic.core.data_structures import DICPara, ImageGradients
from al_dic.solver.local_icgn import (
    local_icgn,
    local_icgn_postprocess,
    local_icgn_precompute,
    local_icgn_solve_subset,
)


def _make_speckle_pair(size=128, shift_x=0.0, shift_y=0.0):
    """Create a synthetic speckle pair and gradients."""
    rng = np.random.RandomState(42)
    img = rng.rand(size, size).astype(np.float64)
    img = gaussian_filter(img, sigma=3.0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    img_ref = img.copy()
    img_def = ndimage_shift(img_ref, [shift_y, shift_x], order=3, mode="constant")

    df_dx = np.zeros_like(img_ref)
    df_dy = np.zeros_like(img_ref)
    df_dx[:, 1:-1] = (img_ref[:, 2:] - img_ref[:, :-2]) / 2.0
    df_dy[1:-1, :] = (img_ref[2:, :] - img_ref[:-2, :]) / 2.0

    mask = np.ones_like(img_ref)
    Df = ImageGradients(df_dx=df_dx, df_dy=df_dy, img_ref_mask=mask, img_size=img_ref.shape)
    return img_ref, img_def, Df


class TestLocalICGN:
    def test_zero_displacement(self):
        """Identical images should yield near-zero displacement."""
        img_ref, _, Df = _make_speckle_pair(shift_x=0, shift_y=0)
        img_def = img_ref.copy()

        # 2 nodes well inside the image
        coords = np.array([[48.0, 48.0], [80.0, 80.0]])
        n = coords.shape[0]
        U0 = np.zeros(2 * n)

        para = DICPara(winsize=20, icgn_max_iter=50)

        U, F, local_time, conv_iter, bad_pt_num, mark_hole = local_icgn(
            U0, coords, Df, img_ref, img_def, para, tol=1e-4,
        )

        assert U.shape == (2 * n,)
        assert F.shape == (4 * n,)
        assert conv_iter.shape == (n,)
        assert local_time >= 0
        assert bad_pt_num >= 0
        np.testing.assert_allclose(U, 0.0, atol=0.2)

    def test_output_shapes(self):
        """All output arrays should have correct shapes."""
        img_ref, _, Df = _make_speckle_pair()
        img_def = img_ref.copy()

        coords = np.array([[48.0, 48.0], [80.0, 80.0], [64.0, 64.0]])
        n = coords.shape[0]
        U0 = np.zeros(2 * n)

        para = DICPara(winsize=20, icgn_max_iter=50)

        U, F, local_time, conv_iter, bad_pt_num, mark_hole = local_icgn(
            U0, coords, Df, img_ref, img_def, para, tol=1e-4,
        )

        assert U.shape == (2 * n,)
        assert F.shape == (4 * n,)
        assert conv_iter.shape == (n,)
        assert isinstance(local_time, float)
        assert isinstance(bad_pt_num, (int, np.integer))
        assert mark_hole.dtype == np.int64

    def test_known_translation(self):
        """Should recover a known rigid translation."""
        shift_x, shift_y = 1.5, 1.0
        img_ref, img_def, Df = _make_speckle_pair(shift_x=shift_x, shift_y=shift_y)

        # Single node well inside the image, initial guess near truth
        coords = np.array([[64.0, 64.0]])
        n = 1
        U0 = np.array([shift_x + 0.3, shift_y - 0.2])

        para = DICPara(winsize=30, icgn_max_iter=100)

        U, F, local_time, conv_iter, bad_pt_num, mark_hole = local_icgn(
            U0, coords, Df, img_ref, img_def, para, tol=1e-4,
        )

        np.testing.assert_allclose(U[0], shift_x, atol=0.5)
        np.testing.assert_allclose(U[1], shift_y, atol=0.5)

    def test_edge_node_flagged(self):
        """Node too close to image edge should be flagged as hole."""
        img_ref, _, Df = _make_speckle_pair(size=128)
        img_def = img_ref.copy()

        # Node 0 is near edge, node 1 is well inside
        coords = np.array([[5.0, 5.0], [64.0, 64.0]])
        n = coords.shape[0]
        U0 = np.zeros(2 * n)

        para = DICPara(winsize=30, icgn_max_iter=50)

        U, F, local_time, conv_iter, bad_pt_num, mark_hole = local_icgn(
            U0, coords, Df, img_ref, img_def, para, tol=1e-4,
        )

        # Node 0 should be in mark_hole_strain (near edge, out-of-bounds)
        assert 0 in mark_hole

    def test_no_nan_in_output(self):
        """Output should not contain NaN after fill_nan_idw."""
        img_ref, _, Df = _make_speckle_pair()
        img_def = img_ref.copy()

        # Multiple nodes, a 2D grid so interpolation works
        xs = [40, 60, 80]
        ys = [40, 60, 80]
        coords = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
        n = coords.shape[0]
        U0 = np.zeros(2 * n)

        para = DICPara(winsize=16, icgn_max_iter=50)

        U, F, local_time, conv_iter, bad_pt_num, mark_hole = local_icgn(
            U0, coords, Df, img_ref, img_def, para, tol=1e-4,
        )

        assert not np.any(np.isnan(U))
        assert not np.any(np.isnan(F))


class TestLocalICGNComposition:
    """Verify that precompute+solve_subset+postprocess matches monolithic
    local_icgn() exactly. Required by the seed-propagation BFS, which
    builds ctx once and calls solve_subset per layer.
    """

    @staticmethod
    def _make_case(n_grid=4, shift_x=0.8, shift_y=0.4):
        img_ref, img_def, Df = _make_speckle_pair(shift_x=shift_x, shift_y=shift_y)
        xs = np.linspace(32, 96, n_grid)
        ys = np.linspace(32, 96, n_grid)
        coords = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
        n = coords.shape[0]
        U0 = np.full(2 * n, 0.5, dtype=np.float64)
        para = DICPara(winsize=20, icgn_max_iter=50)
        return img_ref, img_def, Df, coords, U0, para

    def test_full_mesh_parity(self):
        """Composed full-mesh call must bit-match monolithic local_icgn."""
        img_ref, img_def, Df, coords, U0, para = self._make_case()

        U_ref, F_ref, _, conv_ref, bad_ref, hole_ref = local_icgn(
            U0, coords, Df, img_ref, img_def, para, tol=1e-4,
        )

        ctx = local_icgn_precompute(coords, Df, img_ref, para)
        U0_2d = U0.reshape(-1, 2)
        U_2d, F_2d, conv_iter = local_icgn_solve_subset(
            ctx, None, U0_2d, img_def, tol=1e-4,
        )
        U_c, F_c, _, conv_c, bad_c, hole_c = local_icgn_postprocess(
            ctx, U_2d, F_2d, conv_iter,
        )

        np.testing.assert_array_equal(U_c, U_ref)
        np.testing.assert_array_equal(F_c, F_ref)
        np.testing.assert_array_equal(conv_c, conv_ref)
        np.testing.assert_array_equal(hole_c, hole_ref)
        assert bad_c == bad_ref

    def test_split_subset_parity(self):
        """Two half-subset solves must produce the same full-mesh result."""
        img_ref, img_def, Df, coords, U0, para = self._make_case()

        U_ref, F_ref, _, conv_ref, bad_ref, hole_ref = local_icgn(
            U0, coords, Df, img_ref, img_def, para, tol=1e-4,
        )

        ctx = local_icgn_precompute(coords, Df, img_ref, para)
        n = ctx.n_nodes
        half = n // 2
        idx_a = np.arange(half, dtype=np.int64)
        idx_b = np.arange(half, n, dtype=np.int64)
        U0_2d = U0.reshape(-1, 2)

        U_2d_full = np.zeros((n, 2), dtype=np.float64)
        F_2d_full = np.zeros((n, 4), dtype=np.float64)
        conv_full = np.zeros(n, dtype=np.int64)

        U_a, F_a, conv_a = local_icgn_solve_subset(
            ctx, idx_a, U0_2d[idx_a], img_def, tol=1e-4,
        )
        U_b, F_b, conv_b = local_icgn_solve_subset(
            ctx, idx_b, U0_2d[idx_b], img_def, tol=1e-4,
        )
        U_2d_full[idx_a] = U_a
        U_2d_full[idx_b] = U_b
        F_2d_full[idx_a] = F_a
        F_2d_full[idx_b] = F_b
        conv_full[idx_a] = conv_a
        conv_full[idx_b] = conv_b

        U_c, F_c, _, conv_c, bad_c, hole_c = local_icgn_postprocess(
            ctx, U_2d_full, F_2d_full, conv_full,
        )

        np.testing.assert_array_equal(U_c, U_ref)
        np.testing.assert_array_equal(F_c, F_ref)
        np.testing.assert_array_equal(conv_c, conv_ref)
        np.testing.assert_array_equal(hole_c, hole_ref)
        assert bad_c == bad_ref
