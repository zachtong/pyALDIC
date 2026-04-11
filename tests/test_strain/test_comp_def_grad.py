"""Tests for comp_def_grad (local plane-fitting deformation gradient)."""

import numpy as np
import pytest

from al_dic.strain.comp_def_grad import comp_def_grad


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


class TestCompDefGradFiltering:
    """Tests for invalid-neighbor filtering (NaN displacement + mask)."""

    def _make_linear_case(self, nx=7, ny=7, spacing=16, dudx=0.01):
        """Grid with exact linear u = dudx * x displacement."""
        x = np.arange(nx) * spacing
        y = np.arange(ny) * spacing
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
        n = coords.shape[0]
        U = np.zeros(2 * n)
        U[0::2] = dudx * coords[:, 0]
        elems = np.empty((0, 8), dtype=np.int64)
        return coords, U, elems, n

    def test_nan_displacement_excluded_from_fit(self):
        """Nodes with NaN displacement must not corrupt valid neighbors' gradient.

        Setup: linear u = 0.01*x field, NaN injected at one interior node.
        Expected: valid nodes adjacent to the NaN node still recover du/dx ≈ 0.01.
        Old behavior (before fix): those neighbors return NaN because lstsq
        propagates NaN from the bad node into the design matrix.
        """
        coords, U, elems, n = self._make_linear_case(dudx=0.01)

        # Inject NaN at the grid center node
        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
        center = np.argmin((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
        U[2 * center] = np.nan
        U[2 * center + 1] = np.nan

        F = comp_def_grad(U, coords, elems, rad=25.0)

        # Nodes adjacent to center (within 20px) but not NaN themselves
        adj = (
            (np.abs(coords[:, 0] - cx) < 20) &
            (np.abs(coords[:, 1] - cy) < 20) &
            (np.arange(n) != center)
        )
        for i in np.where(adj)[0]:
            assert np.isfinite(F[4 * i]), (
                f"Node {i} adjacent to NaN node has F=NaN — "
                "NaN neighbors not properly excluded"
            )
            np.testing.assert_allclose(F[4 * i], 0.01, atol=0.005,
                                       err_msg=f"du/dx wrong at node {i}")

    def test_mask_excludes_outside_nodes_from_fit(self):
        """Nodes outside the mask must not contaminate inside neighbors.

        Setup: linear u = 0.01*x field, one column of nodes has WRONG large
        displacement but is outside the mask. Valid interior nodes should still
        recover du/dx ≈ 0.01.
        """
        coords, U, elems, n = self._make_linear_case(nx=7, ny=7, spacing=16, dudx=0.01)

        # Corrupt the rightmost column with wrong large displacement
        right_col = coords[:, 0] == coords[:, 0].max()
        U[0::2][right_col] = 999.0   # grossly wrong, but finite (not caught by isfinite alone)

        # Mask: exclude rightmost column (nodes at x=96, cut off at col=90)
        H, W = 130, 130
        mask = np.ones((H, W), dtype=np.float64)
        mask[:, 90:] = 0.0  # nodes at x=96 → col=96 ≥ 90 → excluded

        F_no_mask = comp_def_grad(U, coords, elems, rad=25.0, mask=None)
        F_masked  = comp_def_grad(U, coords, elems, rad=25.0, mask=mask)

        # Second-to-last column nodes (at x=80, adjacent to corrupt column)
        second_right = (coords[:, 0] == coords[:, 0].max() - 16)
        for i in np.where(second_right)[0]:
            # Without mask: corrupted by wrong neighbor
            assert not np.isclose(F_no_mask[4 * i], 0.01, atol=0.05), (
                "Expected no-mask result to be wrong (sanity check)"
            )
            # With mask: corrupt column excluded → correct result
            np.testing.assert_allclose(
                F_masked[4 * i], 0.01, atol=0.01,
                err_msg=f"Node {i} contaminated by out-of-mask neighbor",
            )

    def test_all_nan_returns_all_nan(self):
        """If all nodes have NaN displacement, all F should be NaN."""
        coords, U, elems, n = self._make_linear_case()
        U[:] = np.nan

        F = comp_def_grad(U, coords, elems, rad=25.0)
        assert np.all(np.isnan(F))

    def test_mask_all_excluded_returns_all_nan(self):
        """If mask excludes all nodes, all F should be NaN."""
        coords, U, elems, n = self._make_linear_case()
        mask = np.zeros((200, 200), dtype=np.float64)  # all zeros — exclude everything

        F = comp_def_grad(U, coords, elems, rad=25.0, mask=mask)
        assert np.all(np.isnan(F))
