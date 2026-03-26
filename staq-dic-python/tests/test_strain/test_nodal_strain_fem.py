"""Tests for nodal_strain_fem (FEM-based nodal strain computation)."""

import numpy as np
import pytest

from staq_dic.core.data_structures import DICMesh, DICPara
from staq_dic.strain.nodal_strain_fem import global_nodal_strain_fem


def _make_simple_mesh(spacing=16):
    """Create a 2x2 grid of Q4 elements (9 nodes, 4 elements)."""
    s = spacing
    coords = np.array([
        [0, 0], [s, 0], [2 * s, 0],
        [0, s], [s, s], [2 * s, s],
        [0, 2 * s], [s, 2 * s], [2 * s, 2 * s],
    ], dtype=np.float64)
    elems = np.array([
        [0, 1, 4, 3, -1, -1, -1, -1],
        [1, 2, 5, 4, -1, -1, -1, -1],
        [3, 4, 7, 6, -1, -1, -1, -1],
        [4, 5, 8, 7, -1, -1, -1, -1],
    ], dtype=np.int64)
    return DICMesh(coordinates_fem=coords, elements_fem=elems)


class TestGlobalNodalStrainFEM:
    def test_zero_displacement(self):
        """Zero U should give zero strain."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]
        para = DICPara(winstepsize=16)
        U = np.zeros(2 * n)

        F = global_nodal_strain_fem(mesh, para, U)

        assert F.shape == (4 * n,)
        np.testing.assert_allclose(F, 0.0, atol=1e-12)

    def test_uniform_translation(self):
        """Uniform translation (u=const) should give zero gradient."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]
        para = DICPara(winstepsize=16)
        U = np.empty(2 * n)
        U[0::2] = 5.0  # u = 5 everywhere
        U[1::2] = 3.0  # v = 3 everywhere

        F = global_nodal_strain_fem(mesh, para, U)

        # Gradients of constant field should be zero
        np.testing.assert_allclose(F, 0.0, atol=1e-10)

    def test_linear_displacement(self):
        """u = 0.01*x should give du/dx = 0.01."""
        mesh = _make_simple_mesh(spacing=16)
        n = mesh.coordinates_fem.shape[0]
        para = DICPara(winstepsize=16)

        dudx = 0.01
        U = np.zeros(2 * n)
        U[0::2] = dudx * mesh.coordinates_fem[:, 0]

        F = global_nodal_strain_fem(mesh, para, U)

        # F11 = du/dx should be 0.01
        np.testing.assert_allclose(F[0::4], dudx, atol=1e-10)
        # Other components should be ~0
        np.testing.assert_allclose(F[1::4], 0.0, atol=1e-10)
        np.testing.assert_allclose(F[2::4], 0.0, atol=1e-10)
        np.testing.assert_allclose(F[3::4], 0.0, atol=1e-10)

    def test_shear_displacement(self):
        """u = 0.01*y should give du/dy = 0.01."""
        mesh = _make_simple_mesh(spacing=16)
        n = mesh.coordinates_fem.shape[0]
        para = DICPara(winstepsize=16)

        dudy = 0.01
        U = np.zeros(2 * n)
        U[0::2] = dudy * mesh.coordinates_fem[:, 1]

        F = global_nodal_strain_fem(mesh, para, U)

        np.testing.assert_allclose(F[0::4], 0.0, atol=1e-10)   # du/dx
        np.testing.assert_allclose(F[2::4], dudy, atol=1e-10)   # du/dy
        np.testing.assert_allclose(F[1::4], 0.0, atol=1e-10)    # dv/dx
        np.testing.assert_allclose(F[3::4], 0.0, atol=1e-10)    # dv/dy

    def test_output_shape(self):
        """Output should have shape (4*n_nodes,)."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]
        para = DICPara(winstepsize=16)

        F = global_nodal_strain_fem(mesh, para, np.zeros(2 * n))
        assert F.shape == (4 * n,)

    def test_no_nan_in_output(self):
        """Output should not contain NaN for a well-connected mesh."""
        mesh = _make_simple_mesh()
        n = mesh.coordinates_fem.shape[0]
        para = DICPara(winstepsize=16)
        U = 0.01 * mesh.coordinates_fem[:, 0].repeat(2)

        F = global_nodal_strain_fem(mesh, para, np.zeros(2 * n))
        assert not np.any(np.isnan(F))

    def test_empty_mesh(self):
        """Empty mesh should return zeros."""
        mesh = DICMesh(
            coordinates_fem=np.zeros((3, 2)),
            elements_fem=np.empty((0, 8), dtype=np.int64),
        )
        para = DICPara(winstepsize=16)
        F = global_nodal_strain_fem(mesh, para, np.zeros(6))
        np.testing.assert_array_equal(F, np.zeros(12))
