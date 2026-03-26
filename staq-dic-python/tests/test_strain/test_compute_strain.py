"""Tests for compute_strain (strain computation router)."""

import numpy as np
import pytest

from staq_dic.core.data_structures import DICMesh, DICPara, StrainResult
from staq_dic.strain.compute_strain import compute_strain, _compute_derived_strains
from staq_dic.utils.region_analysis import NodeRegionMap


def _make_mesh_and_map(spacing=16):
    """Create a 2x2 mesh with region map."""
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
    mesh = DICMesh(coordinates_fem=coords, elements_fem=elems)
    region_map = NodeRegionMap(
        region_node_lists=[np.arange(9, dtype=np.int64)],
        n_regions=1,
    )
    return mesh, region_map


class TestComputeDerivedStrains:
    def test_uniaxial(self):
        """Uniaxial tension: exx > 0, exy = 0, eyy = 0."""
        exx = np.array([0.01])
        exy = np.array([0.0])
        eyy = np.array([0.0])

        pmax, pmin, maxshear, vm = _compute_derived_strains(exx, exy, eyy)

        np.testing.assert_allclose(pmax, 0.01)
        np.testing.assert_allclose(pmin, 0.0)
        np.testing.assert_allclose(maxshear, 0.005)
        assert vm[0] > 0

    def test_pure_shear(self):
        """Pure shear: exx = eyy = 0, exy = gamma/2."""
        exx = np.array([0.0])
        exy = np.array([0.005])
        eyy = np.array([0.0])

        pmax, pmin, maxshear, vm = _compute_derived_strains(exx, exy, eyy)

        np.testing.assert_allclose(pmax, 0.005)
        np.testing.assert_allclose(pmin, -0.005)
        np.testing.assert_allclose(maxshear, 0.005)

    def test_hydrostatic(self):
        """Equal biaxial: principal strains both equal, zero shear."""
        eps = 0.01
        exx = np.array([eps])
        exy = np.array([0.0])
        eyy = np.array([eps])

        pmax, pmin, maxshear, vm = _compute_derived_strains(exx, exy, eyy)

        np.testing.assert_allclose(pmax, eps, atol=1e-14)
        np.testing.assert_allclose(pmin, eps, atol=1e-14)
        np.testing.assert_allclose(maxshear, 0.0, atol=1e-14)


class TestComputeStrain:
    def test_zero_displacement_method3(self):
        """Zero U with FEM method should give zero strain."""
        mesh, region_map = _make_mesh_and_map()
        n = mesh.coordinates_fem.shape[0]
        para = DICPara(
            winstepsize=16, method_to_compute_strain=3,
            strain_type=0, strain_smoothness=0.0,
        )
        U = np.zeros(2 * n)

        result = compute_strain(mesh, para, U, region_map)

        assert isinstance(result, StrainResult)
        np.testing.assert_allclose(result.strain_exx, 0.0, atol=1e-10)
        np.testing.assert_allclose(result.strain_eyy, 0.0, atol=1e-10)
        np.testing.assert_allclose(result.strain_exy, 0.0, atol=1e-10)

    def test_linear_field_method3(self):
        """u = 0.01*x with FEM method should give exx ≈ 0.01."""
        mesh, region_map = _make_mesh_and_map(spacing=16)
        n = mesh.coordinates_fem.shape[0]
        dudx = 0.01
        U = np.zeros(2 * n)
        U[0::2] = dudx * mesh.coordinates_fem[:, 0]

        para = DICPara(
            winstepsize=16, method_to_compute_strain=3,
            strain_type=0, strain_smoothness=0.0,
        )

        result = compute_strain(mesh, para, U, region_map)

        np.testing.assert_allclose(result.strain_exx, dudx, atol=0.005)

    def test_method2_plane_fitting(self):
        """Plane fitting method should give reasonable results."""
        mesh, region_map = _make_mesh_and_map(spacing=16)
        n = mesh.coordinates_fem.shape[0]
        dudx = 0.01
        U = np.zeros(2 * n)
        U[0::2] = dudx * mesh.coordinates_fem[:, 0]

        para = DICPara(
            winstepsize=16, method_to_compute_strain=2,
            strain_type=0, strain_smoothness=0.0,
            strain_plane_fit_rad=25.0,
        )

        result = compute_strain(mesh, para, U, region_map)

        assert isinstance(result, StrainResult)
        assert not np.any(np.isnan(result.strain_exx))

    def test_result_has_all_fields(self):
        """StrainResult should have all expected fields populated."""
        mesh, region_map = _make_mesh_and_map()
        n = mesh.coordinates_fem.shape[0]
        para = DICPara(
            winstepsize=16, method_to_compute_strain=3,
            strain_type=0, strain_smoothness=0.0,
        )

        result = compute_strain(mesh, para, np.zeros(2 * n), region_map)

        assert result.disp_u is not None
        assert result.disp_v is not None
        assert result.dudx is not None
        assert result.dvdx is not None
        assert result.dudy is not None
        assert result.dvdy is not None
        assert result.strain_exx is not None
        assert result.strain_exy is not None
        assert result.strain_eyy is not None
        assert result.strain_principal_max is not None
        assert result.strain_principal_min is not None
        assert result.strain_maxshear is not None
        assert result.strain_von_mises is not None

    def test_with_smoothing(self):
        """Non-zero smoothness should produce finite results."""
        mesh, region_map = _make_mesh_and_map()
        n = mesh.coordinates_fem.shape[0]
        para = DICPara(
            winstepsize=16, method_to_compute_strain=3,
            strain_type=0, strain_smoothness=1e-4,
        )

        result = compute_strain(mesh, para, np.zeros(2 * n), region_map)
        assert np.all(np.isfinite(result.strain_exx))

    def test_eulerian_almansi_type(self):
        """Eulerian-Almansi strain type should work."""
        mesh, region_map = _make_mesh_and_map()
        n = mesh.coordinates_fem.shape[0]
        para = DICPara(
            winstepsize=16, method_to_compute_strain=3,
            strain_type=1, strain_smoothness=0.0,
        )

        result = compute_strain(mesh, para, np.zeros(2 * n), region_map)
        np.testing.assert_allclose(result.strain_exx, 0.0, atol=1e-10)
