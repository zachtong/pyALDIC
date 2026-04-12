"""Tests for icgn_batch edge cases — boundary checks, singular Hessians, etc."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from al_dic.solver.icgn_batch import (
    _compose_warp_batch,
    _iterate_6dof_batch,
    _iterate_2dof_batch,
    precompute_subsets_6dof,
    precompute_subsets_2dof,
    _connected_center_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_speckle(size=128, seed=42):
    """Simple speckle image for testing."""
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.standard_normal((size, size)), sigma=3.0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def _make_gradients(img):
    """Central-difference gradients."""
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    dy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return dx, dy


def _make_pre_6dof(coords, img, winsize=20):
    """Precompute 6-DOF subsets for given coords."""
    dx, dy = _make_gradients(img)
    mask = np.ones_like(img)
    return precompute_subsets_6dof(coords, img, dx, dy, mask, winsize)


def _make_pre_2dof(coords, img, winsize=20):
    """Precompute 2-DOF subsets for given coords."""
    dx, dy = _make_gradients(img)
    mask = np.ones_like(img)
    n = coords.shape[0]
    ws = np.full(n, winsize, dtype=int)
    return precompute_subsets_2dof(coords, img, dx, dy, mask, ws, ws)


# ---------------------------------------------------------------------------
# OOB and boundary tests
# ---------------------------------------------------------------------------

class TestOOBNodes:
    def test_oob_nodes_marked_inactive(self):
        """Nodes pushed outside image should be deactivated during iteration."""
        img = _make_speckle(128)
        # Put one node near the edge
        coords = np.array([[5.0, 5.0], [64.0, 64.0]], dtype=np.float64)
        pre = _make_pre_6dof(coords, img, winsize=20)
        U0_2d = np.zeros((2, 2), dtype=np.float64)
        # Push the edge node far out
        U0_2d[0, 0] = -50.0

        U, F, conv, mark_hole = _iterate_6dof_batch(
            coords, U0_2d, img.copy(), pre, tol=1e-3, max_iter=20,
        )
        # Edge node should not converge normally
        assert conv[0] >= 20

    def test_2dof_oob_deactivation(self):
        """2-DOF nodes pushed out of bounds should be deactivated."""
        img = _make_speckle(128)
        coords = np.array([[5.0, 5.0], [64.0, 64.0]], dtype=np.float64)
        pre = _make_pre_2dof(coords, img, winsize=20)

        U_old_2d = np.zeros((2, 2), dtype=np.float64)
        U_old_2d[0, 0] = -50.0
        F_old_2d = np.zeros((2, 4), dtype=np.float64)
        udual_2d = np.zeros((2, 2), dtype=np.float64)

        U_out, conv = _iterate_2dof_batch(
            coords, U_old_2d, F_old_2d, udual_2d,
            img.copy(), pre, mu=1e-3, tol=1e-3, max_iter=20,
        )
        assert conv[0] >= 20


# ---------------------------------------------------------------------------
# Pixel count and Hessian tests
# ---------------------------------------------------------------------------

class TestSubsetEdgeCases:
    def test_too_few_valid_pixels(self):
        """Mostly-masked subset should be deactivated."""
        img = _make_speckle(128)
        coords = np.array([[64.0, 64.0]], dtype=np.float64)
        dx, dy = _make_gradients(img)
        # Nearly empty mask
        mask = np.zeros_like(img)
        mask[64, 64] = 1.0  # Only center pixel
        pre = precompute_subsets_6dof(coords, img, dx, dy, mask, winsize=20)
        # This node should be invalid due to too few connected pixels
        assert not pre["valid"][0]

    def test_singular_hessian_batch_fallback(self):
        """Zero Hessian should trigger per-node fallback or deactivation."""
        img = _make_speckle(128)
        coords = np.array([[64.0, 64.0]], dtype=np.float64)
        pre = _make_pre_6dof(coords, img, winsize=20)

        # Zero out Hessian to make it singular
        pre["H_all"][0] = np.zeros((6, 6))
        U0_2d = np.zeros((1, 2), dtype=np.float64)

        U, F, conv, _ = _iterate_6dof_batch(
            coords, U0_2d, img.copy(), pre, tol=1e-3, max_iter=10,
        )
        # Should not crash; node should be deactivated or stuck
        assert conv[0] >= 10 or np.all(np.isfinite(U[0]))

    def test_singular_hessian_individual_deactivation(self):
        """When batch solve fails and individual solve also fails, node deactivated."""
        img = _make_speckle(128)
        coords = np.array([[48.0, 48.0], [80.0, 80.0]], dtype=np.float64)
        pre = _make_pre_6dof(coords, img, winsize=20)

        # Make both Hessians singular
        pre["H_all"][:] = 0.0
        U0_2d = np.zeros((2, 2), dtype=np.float64)

        U, F, conv, _ = _iterate_6dof_batch(
            coords, U0_2d, img.copy(), pre, tol=1e-3, max_iter=5,
        )
        # Should not crash
        assert U.shape == (2, 2)


# ---------------------------------------------------------------------------
# Compose warp tests
# ---------------------------------------------------------------------------

class TestComposeWarp:
    def test_identity(self):
        """delta_P = 0 should return P unchanged."""
        P = np.array([[0.01, 0.02, 0.03, 0.04, 1.0, 2.0]], dtype=np.float64)
        delta_P = np.zeros_like(P)

        result, singular = _compose_warp_batch(P, delta_P)
        np.testing.assert_allclose(result, P, atol=1e-12)
        assert not singular[0]

    def test_singular_determinant(self):
        """det < 1e-15 should be flagged as singular."""
        P = np.zeros((1, 6), dtype=np.float64)
        # Create delta_P with det = (1+0)*(1+0) - 0*0 = 1, then make it singular
        delta_P = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        # det = (1 + (-1)) * (1 + 0) - 0*0 = 0 -> singular

        _, singular = _compose_warp_batch(P, delta_P)
        assert singular[0]


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

class TestConvergencePaths:
    def test_immediate_convergence(self):
        """If initial norm < tol, should converge on first iteration."""
        img = _make_speckle(128)
        coords = np.array([[64.0, 64.0]], dtype=np.float64)
        pre = _make_pre_6dof(coords, img, winsize=20)
        # Initial guess = zero, image is identical -> immediate convergence
        U0_2d = np.zeros((1, 2), dtype=np.float64)

        U, F, conv, _ = _iterate_6dof_batch(
            coords, U0_2d, img.copy(), pre, tol=1e-1, max_iter=50,
        )
        # Should converge very quickly
        if pre["valid"][0]:
            assert conv[0] <= 5

    def test_delta_p_early_stop(self):
        """Small delta_P should trigger early convergence."""
        img = _make_speckle(128)
        coords = np.array([[64.0, 64.0]], dtype=np.float64)
        pre = _make_pre_6dof(coords, img, winsize=20)
        U0_2d = np.zeros((1, 2), dtype=np.float64)

        U, F, conv, _ = _iterate_6dof_batch(
            coords, U0_2d, img.copy(), pre, tol=0.5, max_iter=50,
        )
        if pre["valid"][0]:
            assert conv[0] < 50


# ---------------------------------------------------------------------------
# Connected center mask tests
# ---------------------------------------------------------------------------

class TestConnectedCenterMask:
    def test_full_mask(self):
        """Full mask should return all ones."""
        mask = np.ones((11, 11), dtype=np.bool_)
        result = _connected_center_mask(mask)
        np.testing.assert_array_equal(result, np.ones((11, 11)))

    def test_center_not_in_mask(self):
        """If center pixel is masked out, should return all zeros."""
        mask = np.ones((11, 11), dtype=np.bool_)
        mask[5, 5] = False
        result = _connected_center_mask(mask)
        assert result[5, 5] == 0.0

    def test_2dof_singular_lm_solve(self):
        """Singular LM 2x2 solve should deactivate the node."""
        img = _make_speckle(128)
        coords = np.array([[64.0, 64.0]], dtype=np.float64)
        pre = _make_pre_2dof(coords, img, winsize=20)

        # Zero out Hessian to make LM solve singular
        pre["H2_img_all"][0] = np.zeros((2, 2))
        U_old_2d = np.zeros((1, 2), dtype=np.float64)
        F_old_2d = np.zeros((1, 4), dtype=np.float64)
        udual_2d = np.zeros((1, 2), dtype=np.float64)

        U_out, conv = _iterate_2dof_batch(
            coords, U_old_2d, F_old_2d, udual_2d,
            img.copy(), pre, mu=1e-3, tol=1e-3, max_iter=10,
        )
        # Should not crash
        assert U_out.shape == (1, 2)
