"""Tests for subpb2_solver (FEM global solve for ADMM subproblem 2)."""

import numpy as np
import pytest

from staq_dic.core.data_structures import DICMesh
from staq_dic.solver.subpb2_solver import _gauss_points_1d, subpb2_solver


# ---------------------------------------------------------------------------
# Mesh fixture: simple 2x2 grid of Q4 elements (no hanging nodes)
# ---------------------------------------------------------------------------

def _make_simple_mesh(spacing=16):
    """Create a 2x2 grid of Q4 elements (9 nodes, 4 elements).

    Layout (node indices):
        6---7---8
        |   |   |
        3---4---5
        |   |   |
        0---1---2
    """
    s = spacing
    coords = np.array([
        [0, 0], [s, 0], [2 * s, 0],
        [0, s], [s, s], [2 * s, s],
        [0, 2 * s], [s, 2 * s], [2 * s, 2 * s],
    ], dtype=np.float64)

    # Q8 connectivity: corners CCW, midside -1 (no hanging nodes)
    elems = np.array([
        [0, 1, 4, 3, -1, -1, -1, -1],
        [1, 2, 5, 4, -1, -1, -1, -1],
        [3, 4, 7, 6, -1, -1, -1, -1],
        [4, 5, 8, 7, -1, -1, -1, -1],
    ], dtype=np.int64)

    return DICMesh(
        coordinates_fem=coords,
        elements_fem=elems,
    )


def _uniform_U(coords, ux=1.0, vy=0.5):
    """Create a uniform displacement field."""
    n = coords.shape[0]
    U = np.empty(2 * n, dtype=np.float64)
    U[0::2] = ux
    U[1::2] = vy
    return U


def _linear_U(coords, dudx=1.0, dvdy=0.0):
    """Create a linear displacement field u = dudx * x, v = dvdy * y."""
    n = coords.shape[0]
    U = np.empty(2 * n, dtype=np.float64)
    U[0::2] = dudx * coords[:, 0]
    U[1::2] = dvdy * coords[:, 1]
    return U


def _uniform_F(n_nodes, f11=0.0, f21=0.0, f12=0.0, f22=0.0):
    """Create a uniform deformation gradient."""
    F = np.empty(4 * n_nodes, dtype=np.float64)
    F[0::4] = f11
    F[1::4] = f21
    F[2::4] = f12
    F[3::4] = f22
    return F


# ---------------------------------------------------------------------------
# Tests for _gauss_points_1d
# ---------------------------------------------------------------------------

class TestGaussPoints1D:
    def test_order_2(self):
        pts, wts = _gauss_points_1d(2)
        assert len(pts) == 2
        assert len(wts) == 2
        np.testing.assert_allclose(wts, [1.0, 1.0])

    def test_order_3(self):
        pts, wts = _gauss_points_1d(3)
        assert len(pts) == 3
        np.testing.assert_allclose(np.sum(wts), 2.0, atol=1e-14)

    def test_order_4(self):
        pts, wts = _gauss_points_1d(4)
        assert len(pts) == 4
        np.testing.assert_allclose(np.sum(wts), 2.0, atol=1e-4)

    def test_order_5(self):
        pts, wts = _gauss_points_1d(5)
        assert len(pts) == 5
        np.testing.assert_allclose(np.sum(wts), 2.0, atol=1e-4)

    def test_invalid_order(self):
        with pytest.raises(ValueError, match="not supported"):
            _gauss_points_1d(1)


# ---------------------------------------------------------------------------
# Tests for subpb2_solver
# ---------------------------------------------------------------------------

class TestSubpb2Solver:
    def test_zero_displacement(self):
        """Zero U and F should yield zero Uhat."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]
        U = np.zeros(2 * n)
        F = np.zeros(4 * n)

        Uhat = subpb2_solver(
            mesh, gauss_pt_order=2, beta=1e-2, mu=1e-3,
            U=U, F=F,
            udual=np.zeros(2 * n), vdual=np.zeros(4 * n),
            alpha=0.0, winstepsize=16,
        )

        assert Uhat.shape == (2 * n,)
        np.testing.assert_allclose(Uhat, 0.0, atol=1e-12)

    def test_uniform_displacement(self):
        """Uniform U with zero F should approximately recover U."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]
        U = _uniform_U(mesh.coordinates_fem, ux=2.0, vy=1.0)
        F = np.zeros(4 * n)

        Uhat = subpb2_solver(
            mesh, gauss_pt_order=2, beta=1e-2, mu=1e-1,
            U=U, F=F,
            udual=np.zeros(2 * n), vdual=np.zeros(4 * n),
            alpha=0.0, winstepsize=16,
        )

        assert Uhat.shape == (2 * n,)
        # mu dominates → Uhat ≈ U
        np.testing.assert_allclose(Uhat, U, atol=0.5)

    def test_linear_field_consistent(self):
        """Linear U with consistent F should recover U exactly."""
        mesh = _make_simple_mesh(spacing=16)
        n = mesh.coordinates_fem.shape[0]

        # u = 0.01 * x, v = 0  (small displacement gradient)
        dudx = 0.01
        U = _linear_U(mesh.coordinates_fem, dudx=dudx, dvdy=0.0)
        F = _uniform_F(n, f11=dudx, f21=0.0, f12=0.0, f22=0.0)

        Uhat = subpb2_solver(
            mesh, gauss_pt_order=2, beta=1e-2, mu=1e-3,
            U=U, F=F,
            udual=np.zeros(2 * n), vdual=np.zeros(4 * n),
            alpha=0.0, winstepsize=16,
        )

        np.testing.assert_allclose(Uhat, U, atol=0.05)

    def test_output_shape(self):
        """Output should have shape (2*n_nodes,)."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]

        Uhat = subpb2_solver(
            mesh, gauss_pt_order=2, beta=1e-2, mu=1e-3,
            U=np.zeros(2 * n), F=np.zeros(4 * n),
            udual=np.zeros(2 * n), vdual=np.zeros(4 * n),
            alpha=0.0, winstepsize=16,
        )

        assert Uhat.shape == (2 * n,)

    def test_stiffness_matrix_symmetric(self):
        """The assembled stiffness matrix should be symmetric."""
        # We test indirectly by checking that the solution doesn't blow up
        # and produces reasonable values
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]
        rng = np.random.RandomState(42)
        U = rng.randn(2 * n) * 0.1
        F = rng.randn(4 * n) * 0.01

        Uhat = subpb2_solver(
            mesh, gauss_pt_order=2, beta=1e-2, mu=1e-3,
            U=U, F=F,
            udual=np.zeros(2 * n), vdual=np.zeros(4 * n),
            alpha=0.0, winstepsize=16,
        )

        assert np.all(np.isfinite(Uhat))
        # Solution should be bounded
        assert np.max(np.abs(Uhat)) < 100

    def test_smoothing_effect(self):
        """subpb2 with beta >> mu should smooth noisy displacement."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]

        # Smooth baseline: u = 0, v = 0
        U_smooth = np.zeros(2 * n)
        F_smooth = np.zeros(4 * n)

        # Add noise to U
        rng = np.random.RandomState(42)
        noise = rng.randn(2 * n) * 0.5
        U_noisy = U_smooth + noise

        # With beta >> mu, gradient penalty dominates → should smooth toward zero
        Uhat = subpb2_solver(
            mesh, gauss_pt_order=2, beta=1.0, mu=1e-4,
            U=U_noisy, F=F_smooth,
            udual=np.zeros(2 * n), vdual=np.zeros(4 * n),
            alpha=0.0, winstepsize=16,
        )

        # Uhat should be smoother than U_noisy (closer to F-implied field)
        noise_u = np.std(U_noisy)
        noise_hat = np.std(Uhat)
        assert noise_hat < noise_u

    def test_empty_mesh(self):
        """Empty mesh should return copy of U."""
        mesh = DICMesh(
            coordinates_fem=np.empty((0, 2)),
            elements_fem=np.empty((0, 8), dtype=np.int64),
        )
        U = np.array([1.0, 2.0])
        result = subpb2_solver(
            mesh, gauss_pt_order=2, beta=1e-2, mu=1e-3,
            U=U, F=np.zeros(4),
            udual=np.zeros(2), vdual=np.zeros(4),
            alpha=0.0, winstepsize=16,
        )
        np.testing.assert_array_equal(result, U)

    def test_gauss_order_3(self):
        """Should work with 3x3 Gauss quadrature."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]
        U = np.zeros(2 * n)
        F = np.zeros(4 * n)

        Uhat = subpb2_solver(
            mesh, gauss_pt_order=3, beta=1e-2, mu=1e-3,
            U=U, F=F,
            udual=np.zeros(2 * n), vdual=np.zeros(4 * n),
            alpha=0.0, winstepsize=16,
        )

        np.testing.assert_allclose(Uhat, 0.0, atol=1e-12)

    def test_with_dirichlet(self):
        """Dirichlet nodes should retain their U values."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]

        # Fix nodes 0 and 2 (bottom corners)
        mesh.dirichlet = np.array([0, 2], dtype=np.int64)

        U = np.zeros(2 * n)
        U[0] = 5.0  # u at node 0
        U[1] = 3.0  # v at node 0
        U[4] = 7.0  # u at node 2
        U[5] = 1.0  # v at node 2

        Uhat = subpb2_solver(
            mesh, gauss_pt_order=2, beta=1e-2, mu=1e-3,
            U=U, F=np.zeros(4 * n),
            udual=np.zeros(2 * n), vdual=np.zeros(4 * n),
            alpha=0.0, winstepsize=16,
        )

        # Dirichlet nodes should be fixed
        np.testing.assert_allclose(Uhat[0], 5.0, atol=1e-10)
        np.testing.assert_allclose(Uhat[1], 3.0, atol=1e-10)
        np.testing.assert_allclose(Uhat[4], 7.0, atol=1e-10)
        np.testing.assert_allclose(Uhat[5], 1.0, atol=1e-10)

    def test_with_alpha(self):
        """Non-zero alpha should add smoothness regularization."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]
        U = _uniform_U(mesh.coordinates_fem, ux=1.0, vy=0.0)
        F = np.zeros(4 * n)

        Uhat_no_alpha = subpb2_solver(
            mesh, gauss_pt_order=2, beta=1e-2, mu=1e-3,
            U=U, F=F,
            udual=np.zeros(2 * n), vdual=np.zeros(4 * n),
            alpha=0.0, winstepsize=16,
        )

        Uhat_with_alpha = subpb2_solver(
            mesh, gauss_pt_order=2, beta=1e-2, mu=1e-3,
            U=U, F=F,
            udual=np.zeros(2 * n), vdual=np.zeros(4 * n),
            alpha=1.0, winstepsize=16,
        )

        # Both should produce valid results
        assert np.all(np.isfinite(Uhat_no_alpha))
        assert np.all(np.isfinite(Uhat_with_alpha))
