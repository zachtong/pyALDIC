"""Tests for icgn_solver (6-DOF IC-GN) and icgn_subpb1 (2-DOF IC-GN)."""

import numpy as np
import pytest
from scipy.ndimage import shift as ndimage_shift

from al_dic.solver.icgn_solver import icgn_solver, _build_hessian_6dof
from al_dic.solver.icgn_subpb1 import icgn_subpb1


def _make_speckle_pair(size=128, shift_x=2.0, shift_y=1.5):
    """Create a synthetic speckle reference/deformed image pair.

    Returns (img_ref, img_def, df_dx, df_dy, mask).
    """
    rng = np.random.RandomState(42)
    # Generate speckle pattern
    img = rng.rand(size, size).astype(np.float64)
    # Smooth to make gradients meaningful
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=3.0)

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    img_ref = img.copy()

    # Create shifted deformed image
    img_def = ndimage_shift(img_ref, [shift_y, shift_x], order=3, mode='constant')

    # Compute gradients (central difference)
    df_dx = np.zeros_like(img_ref)
    df_dy = np.zeros_like(img_ref)
    df_dx[:, 1:-1] = (img_ref[:, 2:] - img_ref[:, :-2]) / 2.0
    df_dy[1:-1, :] = (img_ref[2:, :] - img_ref[:-2, :]) / 2.0

    mask = np.ones_like(img_ref)

    return img_ref, img_def, df_dx, df_dy, mask


class TestBuildHessian:
    def test_symmetric(self):
        """Hessian should be symmetric."""
        rng = np.random.RandomState(0)
        XX = rng.randn(10, 10)
        YY = rng.randn(10, 10)
        gx = rng.randn(10, 10)
        gy = rng.randn(10, 10)

        H = _build_hessian_6dof(XX, YY, gx, gy)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_positive_semidefinite(self):
        """Hessian should be positive semi-definite."""
        rng = np.random.RandomState(1)
        XX = rng.randn(10, 10)
        YY = rng.randn(10, 10)
        gx = rng.randn(10, 10)
        gy = rng.randn(10, 10)

        H = _build_hessian_6dof(XX, YY, gx, gy)
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals >= -1e-10)


class TestICGNSolver:
    def test_zero_displacement(self):
        """Identical images should converge to zero displacement."""
        img_ref, _, df_dx, df_dy, mask = _make_speckle_pair(shift_x=0, shift_y=0)
        img_def = img_ref.copy()

        U0 = np.array([0.0, 0.0])
        U, F, step = icgn_solver(
            U0, x0=64.0, y0=64.0,
            df_dx=df_dx, df_dy=df_dy, img_ref_mask=mask,
            img_ref=img_ref, img_def=img_def,
            winsize=40, tol=1e-4,
        )

        assert step <= 100  # Should converge
        np.testing.assert_allclose(U, [0.0, 0.0], atol=0.05)
        np.testing.assert_allclose(F, [0.0, 0.0, 0.0, 0.0], atol=0.01)

    def test_known_translation(self):
        """Should recover a known rigid translation."""
        shift_x, shift_y = 2.0, 1.5
        img_ref, img_def, df_dx, df_dy, mask = _make_speckle_pair(
            shift_x=shift_x, shift_y=shift_y
        )

        # Start with a close initial guess
        U0 = np.array([shift_x + 0.5, shift_y - 0.3])
        U, F, step = icgn_solver(
            U0, x0=64.0, y0=64.0,
            df_dx=df_dx, df_dy=df_dy, img_ref_mask=mask,
            img_ref=img_ref, img_def=img_def,
            winsize=40, tol=1e-4,
        )

        assert step <= 100
        np.testing.assert_allclose(U[0], shift_x, atol=0.3)
        np.testing.assert_allclose(U[1], shift_y, atol=0.3)

    def test_out_of_bounds(self):
        """Node near edge should return max_iter+2."""
        img_ref, img_def, df_dx, df_dy, mask = _make_speckle_pair()

        U0 = np.array([0.0, 0.0])
        U, F, step = icgn_solver(
            U0, x0=5.0, y0=5.0,  # Too close to edge for winsize=40
            df_dx=df_dx, df_dy=df_dy, img_ref_mask=mask,
            img_ref=img_ref, img_def=img_def,
            winsize=40, tol=1e-4,
        )

        assert step >= 100 + 2  # Out-of-bounds failure

    def test_output_shapes(self):
        """Output shapes should be correct."""
        img_ref, img_def, df_dx, df_dy, mask = _make_speckle_pair()

        U0 = np.array([0.0, 0.0])
        U, F, step = icgn_solver(
            U0, x0=64.0, y0=64.0,
            df_dx=df_dx, df_dy=df_dy, img_ref_mask=mask,
            img_ref=img_ref, img_def=img_def,
            winsize=40, tol=1e-4,
        )

        assert U.shape == (2,)
        assert F.shape == (4,)
        assert isinstance(step, (int, np.integer))


class TestICGNSubpb1:
    def test_zero_displacement(self):
        """Identical images → zero displacement update."""
        img_ref, _, df_dx, df_dy, mask = _make_speckle_pair(shift_x=0, shift_y=0)
        img_def = img_ref.copy()

        U_old = np.array([0.0, 0.0])
        F_old = np.array([0.0, 0.0, 0.0, 0.0])
        udual = np.array([0.0, 0.0])

        U, step = icgn_subpb1(
            U_old, F_old, x0=64.0, y0=64.0,
            df_dx=df_dx, df_dy=df_dy, img_ref_mask=mask,
            img_ref=img_ref, img_def=img_def,
            winsize_x=40, winsize_y=40,
            mu=1e-3, udual=udual, tol=1e-4,
        )

        assert step <= 100
        np.testing.assert_allclose(U, [0.0, 0.0], atol=0.1)

    def test_output_shape(self):
        """Output U should have shape (2,)."""
        img_ref, img_def, df_dx, df_dy, mask = _make_speckle_pair()

        U, step = icgn_subpb1(
            np.array([0.0, 0.0]), np.zeros(4),
            x0=64.0, y0=64.0,
            df_dx=df_dx, df_dy=df_dy, img_ref_mask=mask,
            img_ref=img_ref, img_def=img_def,
            winsize_x=40, winsize_y=40,
            mu=1e-3, udual=np.zeros(2), tol=1e-4,
        )

        assert U.shape == (2,)
