"""Tests for subpb1_solver (ADMM subproblem 1 dispatcher)."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as ndimage_shift

from staq_dic.core.data_structures import DICPara, ImageGradients
from staq_dic.solver.subpb1_solver import subpb1_solver


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


class TestSubpb1Solver:
    def test_zero_displacement(self):
        """Identical images should yield near-zero displacement."""
        img_ref, _, Df = _make_speckle_pair(shift_x=0, shift_y=0)
        img_def = img_ref.copy()

        # 2D grid of nodes for interpolation
        coords = np.array([
            [40.0, 40.0], [60.0, 40.0], [80.0, 40.0],
            [40.0, 60.0], [60.0, 60.0], [80.0, 60.0],
        ])
        n = coords.shape[0]
        USubpb2 = np.zeros(2 * n)
        FSubpb2 = np.zeros(4 * n)
        udual = np.zeros(2 * n)
        vdual = np.zeros(4 * n)

        para = DICPara(winsize=20, icgn_max_iter=50, mu=1e-3)

        U, solve_time, conv_iter, bad_pt_num = subpb1_solver(
            USubpb2, FSubpb2, udual, vdual,
            coords, Df, img_ref, img_def,
            mu=1e-3, beta=1e-2, para=para, tol=1e-4,
        )

        assert U.shape == (2 * n,)
        assert solve_time >= 0
        assert conv_iter.shape == (n,)
        np.testing.assert_allclose(U, 0.0, atol=0.5)

    def test_output_shapes(self):
        """All output arrays should have correct shapes."""
        img_ref, _, Df = _make_speckle_pair()
        img_def = img_ref.copy()

        coords = np.array([
            [40.0, 40.0], [60.0, 40.0],
            [40.0, 60.0], [60.0, 60.0],
        ])
        n = coords.shape[0]

        para = DICPara(winsize=20, icgn_max_iter=50)

        U, solve_time, conv_iter, bad_pt_num = subpb1_solver(
            np.zeros(2 * n), np.zeros(4 * n),
            np.zeros(2 * n), np.zeros(4 * n),
            coords, Df, img_ref, img_def,
            mu=1e-3, beta=1e-2, para=para, tol=1e-4,
        )

        assert U.shape == (2 * n,)
        assert isinstance(solve_time, float)
        assert conv_iter.shape == (n,)
        assert isinstance(bad_pt_num, (int, np.integer))

    def test_with_per_node_winsize(self):
        """Should work with per-node window sizes."""
        img_ref, _, Df = _make_speckle_pair()
        img_def = img_ref.copy()

        coords = np.array([
            [48.0, 48.0], [80.0, 48.0],
            [48.0, 80.0], [80.0, 80.0],
        ])
        n = coords.shape[0]

        winsize_list = np.array([
            [20, 20], [16, 16], [20, 20], [16, 16],
        ], dtype=np.float64)

        para = DICPara(winsize=20, icgn_max_iter=50, winsize_list=winsize_list)

        U, solve_time, conv_iter, bad_pt_num = subpb1_solver(
            np.zeros(2 * n), np.zeros(4 * n),
            np.zeros(2 * n), np.zeros(4 * n),
            coords, Df, img_ref, img_def,
            mu=1e-3, beta=1e-2, para=para, tol=1e-4,
        )

        assert U.shape == (2 * n,)
        assert not np.any(np.isnan(U))

    def test_no_nan_in_output(self):
        """Output should not contain NaN after fill_nan_idw."""
        img_ref, _, Df = _make_speckle_pair()
        img_def = img_ref.copy()

        coords = np.array([
            [40.0, 40.0], [60.0, 40.0], [80.0, 40.0],
            [40.0, 60.0], [60.0, 60.0], [80.0, 60.0],
        ])
        n = coords.shape[0]

        para = DICPara(winsize=16, icgn_max_iter=50)

        U, _, _, _ = subpb1_solver(
            np.zeros(2 * n), np.zeros(4 * n),
            np.zeros(2 * n), np.zeros(4 * n),
            coords, Df, img_ref, img_def,
            mu=1e-3, beta=1e-2, para=para, tol=1e-4,
        )

        assert not np.any(np.isnan(U))
