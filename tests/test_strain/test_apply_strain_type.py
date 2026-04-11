"""Tests for apply_strain_type (strain measure conversion)."""

import numpy as np
import pytest

from al_dic.core.data_structures import DICPara
from al_dic.strain.apply_strain_type import apply_strain_type


def _make_F(n, f11=0.0, f21=0.0, f12=0.0, f22=0.0):
    """Create uniform deformation gradient vector."""
    F = np.empty(4 * n, dtype=np.float64)
    F[0::4] = f11
    F[1::4] = f21
    F[2::4] = f12
    F[3::4] = f22
    return F


class TestApplyStrainType:
    def test_infinitesimal_identity(self):
        """strain_type=0 should return input unchanged."""
        F = _make_F(5, f11=0.01, f21=0.002, f12=0.003, f22=0.02)
        para = DICPara(strain_type=0)
        F_strain, F_world = apply_strain_type(F, para)

        np.testing.assert_array_equal(F_strain, F)
        # World: off-diag flipped
        np.testing.assert_allclose(F_world[0::4], F[0::4])
        np.testing.assert_allclose(F_world[1::4], -F[1::4])
        np.testing.assert_allclose(F_world[2::4], -F[2::4])
        np.testing.assert_allclose(F_world[3::4], F[3::4])

    def test_zero_strain(self):
        """Zero deformation gradient should give zero strain for all types."""
        F = np.zeros(20, dtype=np.float64)
        for st in [0, 1, 2]:
            para = DICPara(strain_type=st)
            F_strain, F_world = apply_strain_type(F, para)
            np.testing.assert_allclose(F_strain, 0.0, atol=1e-15)
            np.testing.assert_allclose(F_world, 0.0, atol=1e-15)

    def test_eulerian_almansi_small_strain(self):
        """For small strains, Eulerian-Almansi should be close to infinitesimal."""
        eps = 0.001
        F = _make_F(3, f11=eps, f21=0.0, f12=0.0, f22=eps)
        para = DICPara(strain_type=1)
        F_strain, _ = apply_strain_type(F, para)

        # For small dudx: 1/(1-dudx) - 1 ≈ dudx + dudx² + ... ≈ dudx
        np.testing.assert_allclose(F_strain[0::4], eps, rtol=0.01)
        np.testing.assert_allclose(F_strain[3::4], eps, rtol=0.01)

    def test_green_lagrangian_small_strain(self):
        """For small strains, Green-Lagrangian should be close to infinitesimal."""
        eps = 0.001
        F = _make_F(3, f11=eps, f21=0.0, f12=0.0, f22=eps)
        para = DICPara(strain_type=2)
        F_strain, _ = apply_strain_type(F, para)

        # For small dudx: dudx + 0.5*(dudx² + dvdx²) ≈ dudx
        np.testing.assert_allclose(F_strain[0::4], eps, rtol=0.01)
        np.testing.assert_allclose(F_strain[3::4], eps, rtol=0.01)

    def test_eulerian_almansi_finite(self):
        """Verify Eulerian-Almansi formula for finite strain."""
        dudx = 0.1
        dvdy = 0.2
        F = _make_F(1, f11=dudx, f21=0.0, f12=0.0, f22=dvdy)
        para = DICPara(strain_type=1)
        F_strain, _ = apply_strain_type(F, para)

        expected_exx = 1.0 / (1.0 - dudx) - 1.0
        expected_eyy = 1.0 / (1.0 - dvdy) - 1.0
        np.testing.assert_allclose(F_strain[0], expected_exx, atol=1e-14)
        np.testing.assert_allclose(F_strain[3], expected_eyy, atol=1e-14)

    def test_green_lagrangian_finite(self):
        """Verify Green-Lagrangian formula E=(F_cm^T F_cm - I)/2 for finite strain."""
        dudx, dvdx, dudy, dvdy = 0.1, 0.02, 0.03, 0.2
        F = _make_F(1, f11=dudx, f21=dvdx, f12=dudy, f22=dvdy)
        para = DICPara(strain_type=2)
        F_strain, _ = apply_strain_type(F, para)

        # F_cm = I + grad(u); E = (F_cm^T F_cm - I)/2
        expected_exx = dudx + 0.5 * (dudx**2 + dvdx**2)
        expected_eyy = dvdy + 0.5 * (dudy**2 + dvdy**2)
        expected_exy = 0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy)
        np.testing.assert_allclose(F_strain[0], expected_exx, atol=1e-14)
        np.testing.assert_allclose(F_strain[3], expected_eyy, atol=1e-14)
        np.testing.assert_allclose(F_strain[2], expected_exy, atol=1e-14)

    def test_green_lagrangian_rigid_rotation_invariant(self):
        """GL strain must be near-zero for rigid rotation (rotation-invariance)."""
        # Exact rotation by 10 deg: u=(cosθ-1)x - sinθ·y, v=sinθ·x + (cosθ-1)y
        theta = np.radians(10.0)
        dudx = np.cos(theta) - 1.0
        dvdx = np.sin(theta)
        dudy = -np.sin(theta)
        dvdy = np.cos(theta) - 1.0
        F = _make_F(1, f11=dudx, f21=dvdx, f12=dudy, f22=dvdy)
        para = DICPara(strain_type=2)
        F_strain, _ = apply_strain_type(F, para)
        # E should be ~0 for a rigid rotation (rotation-invariance of GL)
        np.testing.assert_allclose(F_strain[0], 0.0, atol=1e-12)  # Exx
        np.testing.assert_allclose(F_strain[3], 0.0, atol=1e-12)  # Eyy
        np.testing.assert_allclose(F_strain[2], 0.0, atol=1e-12)  # Exy

    def test_green_lagrangian_simple_shear_eyy_positive(self):
        """GL Eyy must be +γ²/2 for simple shear (fibres in y ARE stretched)."""
        gamma = 0.2
        F = _make_F(1, f11=0.0, f21=0.0, f12=gamma, f22=0.0)  # u=γy, v=0
        para = DICPara(strain_type=2)
        F_strain, _ = apply_strain_type(F, para)
        # Standard GL: Eyy = (gamma^2)/2 > 0
        np.testing.assert_allclose(F_strain[3], gamma**2 / 2.0, atol=1e-14)

    def test_world_coord_sign_flip(self):
        """World coordinates should flip F21 and F12 signs."""
        F = _make_F(2, f11=1.0, f21=2.0, f12=3.0, f22=4.0)
        para = DICPara(strain_type=0)
        _, F_world = apply_strain_type(F, para)

        np.testing.assert_allclose(F_world[0::4], 1.0)   # F11 unchanged
        np.testing.assert_allclose(F_world[1::4], -2.0)   # F21 flipped
        np.testing.assert_allclose(F_world[2::4], -3.0)   # F12 flipped
        np.testing.assert_allclose(F_world[3::4], 4.0)    # F22 unchanged

    def test_unknown_strain_type_warns(self):
        """Unknown strain_type should warn and return infinitesimal."""
        F = _make_F(2, f11=0.01, f22=0.01)
        para = DICPara(strain_type=99)
        with pytest.warns(UserWarning, match="Unknown strain_type"):
            F_strain, _ = apply_strain_type(F, para)
        np.testing.assert_array_equal(F_strain, F)

    def test_output_shapes(self):
        """Output shapes should match input."""
        F = np.zeros(20)
        para = DICPara(strain_type=0)
        F_strain, F_world = apply_strain_type(F, para)
        assert F_strain.shape == (20,)
        assert F_world.shape == (20,)

    def test_input_not_modified(self):
        """Input array should not be modified."""
        F = _make_F(3, f11=0.1, f22=0.2)
        F_orig = F.copy()
        para = DICPara(strain_type=2)
        apply_strain_type(F, para)
        np.testing.assert_array_equal(F, F_orig)
