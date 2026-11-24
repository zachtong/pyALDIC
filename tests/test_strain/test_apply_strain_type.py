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


def _almansi(dudx, dvdx, dudy, dvdy):
    """Euler-Almansi strain from its definition, as a 2x2 matrix."""
    Fm = np.array([[1.0 + dudx, dudy], [dvdx, 1.0 + dvdy]])
    Finv = np.linalg.inv(Fm)
    return 0.5 * (np.eye(2) - Finv.T @ Finv)


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

        # For small dudx: (1 - 1/(1+dudx)^2)/2 = dudx - 1.5*dudx^2 + ... ≈ dudx
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

    def test_eulerian_almansi_matches_the_matrix_definition(self):
        """e = (I - F^-T F^-1) / 2 with F = I + grad(u), for arbitrary gradients."""
        rng = np.random.default_rng(7)
        grads = rng.uniform(-0.3, 0.3, size=(50, 4))   # dudx, dvdx, dudy, dvdy
        F = grads.ravel()
        F_strain, _ = apply_strain_type(F, DICPara(strain_type=1))

        for k, (dudx, dvdx, dudy, dvdy) in enumerate(grads):
            expected = _almansi(dudx, dvdx, dudy, dvdy)
            np.testing.assert_allclose(F_strain[4 * k], expected[0, 0], atol=1e-13)
            np.testing.assert_allclose(F_strain[4 * k + 3], expected[1, 1], atol=1e-13)
            # the tensor is symmetric: both off-diagonal slots hold e_xy
            np.testing.assert_allclose(F_strain[4 * k + 1], expected[0, 1], atol=1e-13)
            np.testing.assert_allclose(F_strain[4 * k + 2], expected[0, 1], atol=1e-13)

    def test_eulerian_almansi_uniaxial_stretch(self):
        """A stretch of 1.1 along x gives e_xx = (1 - 1/1.1^2)/2 and nothing else."""
        F = _make_F(1, f11=0.1)
        F_strain, _ = apply_strain_type(F, DICPara(strain_type=1))

        np.testing.assert_allclose(F_strain[0], 0.5 * (1.0 - 1.0 / 1.1**2), atol=1e-14)
        np.testing.assert_allclose(F_strain[1:], 0.0, atol=1e-14)

    def test_eulerian_almansi_rigid_rotation_is_zero(self):
        """A rigid rotation deforms nothing, at any angle."""
        for degrees in (1.0, 10.0, 45.0, 120.0):
            theta = np.radians(degrees)
            F = _make_F(1, f11=np.cos(theta) - 1.0, f21=np.sin(theta),
                        f12=-np.sin(theta), f22=np.cos(theta) - 1.0)
            F_strain, _ = apply_strain_type(F, DICPara(strain_type=1))
            np.testing.assert_allclose(F_strain, 0.0, atol=1e-12)

    def test_eulerian_almansi_simple_shear(self):
        """u = gamma*y: e_xy = gamma/2 and e_yy = -gamma^2/2 (mirror of Green-Lagrange)."""
        gamma = 0.2
        F = _make_F(1, f12=gamma)
        F_strain, _ = apply_strain_type(F, DICPara(strain_type=1))

        np.testing.assert_allclose(F_strain[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(F_strain[2], gamma / 2.0, atol=1e-14)
        np.testing.assert_allclose(F_strain[3], -gamma**2 / 2.0, atol=1e-14)

    def test_eulerian_almansi_is_the_push_forward_of_green_lagrange(self):
        """The two finite measures describe one deformation: e = F^-T E F^-1."""
        dudx, dvdx, dudy, dvdy = 0.12, -0.05, 0.08, -0.03
        F = _make_F(1, f11=dudx, f21=dvdx, f12=dudy, f22=dvdy)
        almansi, _ = apply_strain_type(F, DICPara(strain_type=1))
        green, _ = apply_strain_type(F, DICPara(strain_type=2))

        Fm = np.array([[1.0 + dudx, dudy], [dvdx, 1.0 + dvdy]])
        E = np.array([[green[0], green[2]], [green[1], green[3]]])
        Finv = np.linalg.inv(Fm)
        expected = Finv.T @ E @ Finv
        got = np.array([[almansi[0], almansi[2]], [almansi[1], almansi[3]]])
        np.testing.assert_allclose(got, expected, atol=1e-13)

    def test_eulerian_almansi_world_flips_only_the_shear(self):
        """Turning the y axis upward changes the sign of e_xy and nothing else."""
        F = _make_F(1, f11=0.1, f21=0.02, f12=0.05, f22=-0.04)
        F_strain, F_world = apply_strain_type(F, DICPara(strain_type=1))

        np.testing.assert_allclose(F_world[0::4], F_strain[0::4])
        np.testing.assert_allclose(F_world[3::4], F_strain[3::4])
        np.testing.assert_allclose(F_world[1::4], -F_strain[1::4])
        np.testing.assert_allclose(F_world[2::4], -F_strain[2::4])

    def test_eulerian_almansi_undefined_where_the_material_is_inverted(self):
        """det F <= 0 is not a deformation; the strain there is NaN, not a number."""
        F = np.concatenate([
            _make_F(1, f11=-1.0),             # det F = 0: collapsed to a line
            _make_F(1, f11=-1.5),             # det F < 0: turned inside out
            _make_F(1, f11=0.1),              # a valid node next to them
        ])
        with np.errstate(all="raise"):        # no divide-by-zero leaks out
            F_strain, _ = apply_strain_type(F, DICPara(strain_type=1))

        assert np.all(np.isnan(F_strain[:8]))
        np.testing.assert_allclose(F_strain[8], 0.5 * (1.0 - 1.0 / 1.1**2), atol=1e-14)

    def test_eulerian_almansi_keeps_untracked_nodes_untracked(self):
        """A node without a gradient (NaN) stays NaN and does not affect others."""
        F = np.concatenate([_make_F(1, f11=np.nan, f22=0.1), _make_F(1, f22=0.1)])
        F_strain, _ = apply_strain_type(F, DICPara(strain_type=1))

        assert np.all(np.isnan(F_strain[:4]))
        np.testing.assert_allclose(F_strain[7], 0.5 * (1.0 - 1.0 / 1.1**2), atol=1e-14)

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
