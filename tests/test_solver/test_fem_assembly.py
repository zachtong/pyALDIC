"""Tests for fem_assembly (shape functions + Gauss quadrature)."""

import numpy as np
import pytest

from al_dic.solver.fem_assembly import (
    compute_all_elements_gp,
    gauss_points,
    GAUSS_PTS_2x2,
    GAUSS_WTS_2x2,
)


class TestGaussPoints:
    def test_order_1(self):
        pts, wts = gauss_points(1)
        assert pts.shape == (1, 2)
        np.testing.assert_allclose(pts[0], [0.0, 0.0])
        np.testing.assert_allclose(wts[0], 4.0)

    def test_order_2(self):
        pts, wts = gauss_points(2)
        assert pts.shape == (4, 2)
        assert wts.shape == (4,)
        np.testing.assert_allclose(wts, 1.0)

    def test_order_3(self):
        pts, wts = gauss_points(3)
        assert pts.shape == (9, 2)
        assert wts.shape == (9,)
        # Weights should sum to 4 (area of [-1,1]^2)
        np.testing.assert_allclose(wts.sum(), 4.0)

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            gauss_points(5)


class TestComputeAllElementsGP:
    def _make_unit_square(self, n_ele=1):
        """Single Q4 element: unit square [0,1]^2, no hanging nodes."""
        # Node order: BL(0,0), BR(1,0), TR(1,1), TL(0,1)
        ptx = np.zeros((n_ele, 8))
        pty = np.zeros((n_ele, 8))
        ptx[:, 0] = 0.0; pty[:, 0] = 0.0
        ptx[:, 1] = 1.0; pty[:, 1] = 0.0
        ptx[:, 2] = 1.0; pty[:, 2] = 1.0
        ptx[:, 3] = 0.0; pty[:, 3] = 1.0
        delta = np.zeros((n_ele, 4))
        return ptx, pty, delta

    def test_output_shapes(self):
        """Output shapes should match specification."""
        n_ele = 3
        ptx, pty, delta = self._make_unit_square(n_ele)
        N, DN, Jdet = compute_all_elements_gp(0.0, 0.0, ptx, pty, delta, n_ele)

        assert N.shape == (n_ele, 2, 16)
        assert DN.shape == (n_ele, 4, 16)
        assert Jdet.shape == (n_ele,)

    def test_partition_of_unity(self):
        """Shape functions should sum to 1 at any point."""
        ptx, pty, delta = self._make_unit_square()

        for ksi, eta in [(-0.5, -0.5), (0.0, 0.0), (0.5, 0.5), (-0.8, 0.3)]:
            N, _, _ = compute_all_elements_gp(ksi, eta, ptx, pty, delta, 1)
            # N[0, 0, :] has shape function values at u-DOFs (even cols)
            n_vals = N[0, 0, 0::2]  # N1, N2, N3, N4 (N5-N8 are 0)
            np.testing.assert_allclose(n_vals.sum(), 1.0, atol=1e-14)

    def test_partition_of_unity_with_hanging_node(self):
        """Shape functions should sum to 1 even with hanging nodes."""
        ptx, pty, delta = self._make_unit_square()
        # Add hanging node on edge (n1, n2) → midside node 4
        ptx[0, 4] = 1.0  # midpoint x
        pty[0, 4] = 0.5  # midpoint y
        delta[0, 0] = 1.0  # activate node 5 (col 4)

        for ksi, eta in [(0.0, 0.0), (0.5, 0.5), (-0.5, -0.5)]:
            N, _, _ = compute_all_elements_gp(ksi, eta, ptx, pty, delta, 1)
            n_vals = N[0, 0, 0::2]  # All 8 shape function values
            np.testing.assert_allclose(n_vals.sum(), 1.0, atol=1e-14)

    def test_jacobian_unit_square(self):
        """Jacobian of unit square should be 0.25 (area scaling)."""
        ptx, pty, delta = self._make_unit_square()
        _, _, Jdet = compute_all_elements_gp(0.0, 0.0, ptx, pty, delta, 1)

        # For [0,1]^2 mapped from [-1,1]^2: J = [[0.5,0],[0,0.5]], det = 0.25
        np.testing.assert_allclose(Jdet[0], 0.25, atol=1e-14)

    def test_jacobian_16x16_square(self):
        """16x16 element should have Jdet = 64."""
        ptx = np.zeros((1, 8))
        pty = np.zeros((1, 8))
        ptx[0, :4] = [0.0, 16.0, 16.0, 0.0]
        pty[0, :4] = [0.0, 0.0, 16.0, 16.0]
        delta = np.zeros((1, 4))

        _, _, Jdet = compute_all_elements_gp(0.0, 0.0, ptx, pty, delta, 1)
        # J = [[8,0],[0,8]], det = 64
        np.testing.assert_allclose(Jdet[0], 64.0, atol=1e-12)

    def test_corner_values(self):
        """At corner nodes, shape functions should be 1 or 0."""
        ptx, pty, delta = self._make_unit_square()

        # At (ksi, eta) = (-1, -1) → node 0 (BL)
        N, _, _ = compute_all_elements_gp(-1.0, -1.0, ptx, pty, delta, 1)
        n_vals = N[0, 0, 0::2]
        np.testing.assert_allclose(n_vals[0], 1.0, atol=1e-14)
        np.testing.assert_allclose(n_vals[1:4], 0.0, atol=1e-14)

        # At (ksi, eta) = (+1, -1) → node 1 (BR)
        N, _, _ = compute_all_elements_gp(1.0, -1.0, ptx, pty, delta, 1)
        n_vals = N[0, 0, 0::2]
        np.testing.assert_allclose(n_vals[1], 1.0, atol=1e-14)

    def test_N_structure(self):
        """N_all should interleave: N[0, 2k]=Nk for u, N[1, 2k+1]=Nk for v."""
        ptx, pty, delta = self._make_unit_square()
        N, _, _ = compute_all_elements_gp(0.3, -0.2, ptx, pty, delta, 1)

        # Row 0 should have non-zeros only at even columns
        np.testing.assert_allclose(N[0, 0, 1::2], 0.0, atol=1e-14)
        # Row 1 should have non-zeros only at odd columns
        np.testing.assert_allclose(N[0, 1, 0::2], 0.0, atol=1e-14)
        # Matching values
        np.testing.assert_allclose(N[0, 0, 0::2], N[0, 1, 1::2], atol=1e-14)

    def test_DN_structure(self):
        """DN_all should have correct interleaving pattern."""
        ptx, pty, delta = self._make_unit_square()
        _, DN, _ = compute_all_elements_gp(0.3, -0.2, ptx, pty, delta, 1)

        # Rows 0,1 (du/dx, du/dy) → non-zero at even columns only
        np.testing.assert_allclose(DN[0, 0, 1::2], 0.0, atol=1e-14)
        np.testing.assert_allclose(DN[0, 1, 1::2], 0.0, atol=1e-14)
        # Rows 2,3 (dv/dx, dv/dy) → non-zero at odd columns only
        np.testing.assert_allclose(DN[0, 2, 0::2], 0.0, atol=1e-14)
        np.testing.assert_allclose(DN[0, 3, 0::2], 0.0, atol=1e-14)

    def test_linear_field_exact(self):
        """DN should recover exact gradient for a linear displacement field.

        If u(x,y) = 2x + 3y, then du/dx = 2, du/dy = 3 everywhere.
        """
        ptx = np.zeros((1, 8))
        pty = np.zeros((1, 8))
        ptx[0, :4] = [0.0, 10.0, 10.0, 0.0]
        pty[0, :4] = [0.0, 0.0, 10.0, 10.0]
        delta = np.zeros((1, 4))

        _, DN, _ = compute_all_elements_gp(0.0, 0.0, ptx, pty, delta, 1)

        # Nodal u-values for u = 2x + 3y
        coords = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        u_nodal = 2.0 * coords[:, 0] + 3.0 * coords[:, 1]

        # Build DOF vector (u only, v=0): [u0, 0, u1, 0, u2, 0, u3, 0, 0, ...]
        dof = np.zeros(16)
        for k in range(4):
            dof[2 * k] = u_nodal[k]

        # DN @ dof should give [du/dx, du/dy, dv/dx, dv/dy]
        grad = DN[0] @ dof
        np.testing.assert_allclose(grad[0], 2.0, atol=1e-12)  # du/dx
        np.testing.assert_allclose(grad[1], 3.0, atol=1e-12)  # du/dy

    def test_quadrature_integration(self):
        """2x2 Gauss quadrature should integrate constant function exactly."""
        ptx = np.zeros((1, 8))
        pty = np.zeros((1, 8))
        ptx[0, :4] = [0.0, 4.0, 4.0, 0.0]
        pty[0, :4] = [0.0, 0.0, 4.0, 4.0]
        delta = np.zeros((1, 4))

        gp, wt = gauss_points(2)
        area = 0.0
        for i in range(4):
            _, _, Jdet = compute_all_elements_gp(
                gp[i, 0], gp[i, 1], ptx, pty, delta, 1
            )
            area += Jdet[0] * wt[i]

        # Area of 4x4 square = 16
        np.testing.assert_allclose(area, 16.0, atol=1e-12)
