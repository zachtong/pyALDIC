"""Tests for comp_def_grad (local plane-fitting deformation gradient)."""

import numpy as np
import pytest

from staq_dic.strain.comp_def_grad import comp_def_grad


def _make_grid_coords(nx=5, ny=5, spacing=16):
    """Create a regular grid of coordinates."""
    x = np.arange(nx) * spacing
    y = np.arange(ny) * spacing
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)


class TestCompDefGrad:
    def test_zero_displacement(self):
        """Zero U should give zero gradient."""
        coords = _make_grid_coords()
        n = coords.shape[0]
        U = np.zeros(2 * n)
        elems = np.empty((0, 8), dtype=np.int64)

        F = comp_def_grad(U, coords, elems, rad=25.0)

        # All finite values should be ~0
        finite_mask = np.isfinite(F)
        if finite_mask.any():
            np.testing.assert_allclose(F[finite_mask], 0.0, atol=1e-10)

    def test_uniform_translation(self):
        """Constant displacement should give zero gradient."""
        coords = _make_grid_coords()
        n = coords.shape[0]
        U = np.empty(2 * n)
        U[0::2] = 5.0
        U[1::2] = 3.0
        elems = np.empty((0, 8), dtype=np.int64)

        F = comp_def_grad(U, coords, elems, rad=25.0)

        finite_mask = np.isfinite(F)
        np.testing.assert_allclose(F[finite_mask], 0.0, atol=1e-10)

    def test_linear_u_x(self):
        """u = 0.01*x should give du/dx = 0.01."""
        coords = _make_grid_coords(nx=5, ny=5, spacing=16)
        n = coords.shape[0]
        dudx = 0.01
        U = np.zeros(2 * n)
        U[0::2] = dudx * coords[:, 0]
        elems = np.empty((0, 8), dtype=np.int64)

        F = comp_def_grad(U, coords, elems, rad=25.0)

        # Interior nodes should recover du/dx ≈ 0.01
        # Skip edge nodes which may have fewer neighbors
        interior = (coords[:, 0] > 8) & (coords[:, 0] < 56) & \
                   (coords[:, 1] > 8) & (coords[:, 1] < 56)
        interior_idx = np.where(interior)[0]

        for i in interior_idx:
            if np.isfinite(F[4 * i]):
                np.testing.assert_allclose(F[4 * i], dudx, atol=0.005)

    def test_linear_v_y(self):
        """v = 0.02*y should give dv/dy = 0.02."""
        coords = _make_grid_coords(nx=5, ny=5, spacing=16)
        n = coords.shape[0]
        dvdy = 0.02
        U = np.zeros(2 * n)
        U[1::2] = dvdy * coords[:, 1]
        elems = np.empty((0, 8), dtype=np.int64)

        F = comp_def_grad(U, coords, elems, rad=25.0)

        interior = (coords[:, 0] > 8) & (coords[:, 0] < 56) & \
                   (coords[:, 1] > 8) & (coords[:, 1] < 56)
        interior_idx = np.where(interior)[0]

        for i in interior_idx:
            if np.isfinite(F[4 * i + 3]):
                np.testing.assert_allclose(F[4 * i + 3], dvdy, atol=0.005)

    def test_output_shape(self):
        """Output should have shape (4*n_nodes,)."""
        coords = _make_grid_coords(3, 3)
        n = coords.shape[0]
        U = np.zeros(2 * n)
        elems = np.empty((0, 8), dtype=np.int64)

        F = comp_def_grad(U, coords, elems, rad=25.0)
        assert F.shape == (4 * n,)

    def test_too_few_neighbors_gives_nan(self):
        """Isolated nodes with <3 neighbors should get NaN."""
        # Single node: no plane fit possible
        coords = np.array([[0.0, 0.0]])
        U = np.zeros(2)
        elems = np.empty((0, 8), dtype=np.int64)

        F = comp_def_grad(U, coords, elems, rad=10.0)
        assert np.all(np.isnan(F))

    def test_empty_input(self):
        """Empty coordinates should return empty F."""
        coords = np.empty((0, 2))
        U = np.empty(0)
        elems = np.empty((0, 8), dtype=np.int64)

        F = comp_def_grad(U, coords, elems, rad=10.0)
        assert F.shape == (0,)
