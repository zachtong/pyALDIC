"""Tests for core data structures and configuration."""

from dataclasses import replace

import numpy as np
import pytest

from staq_dic.core.data_structures import (
    DICPara,
    DICMesh,
    GridxyROIRange,
    FrameResult,
    StrainResult,
    split_uv,
    merge_uv,
    split_F,
    merge_F,
)
from staq_dic.core.config import dicpara_default, validate_dicpara


# ---------------------------------------------------------------------------
# DICPara
# ---------------------------------------------------------------------------

class TestDICPara:
    def test_default_creation(self):
        p = dicpara_default()
        assert p.winsize == 40
        assert p.winstepsize == 16
        assert p.winsize_min == 8
        assert p.mu == 1e-3
        assert p.reference_mode == "incremental"

    def test_override(self):
        p = dicpara_default(winsize=32, winstepsize=8, winsize_min=4)
        assert p.winsize == 32
        assert p.winstepsize == 8
        assert p.winsize_min == 4

    def test_frozen(self):
        p = dicpara_default()
        with pytest.raises(AttributeError):
            p.winsize = 64  # type: ignore

    def test_replace(self):
        p = dicpara_default()
        p2 = replace(p, winsize=32)
        assert p2.winsize == 32
        assert p.winsize == 40  # original unchanged

    def test_invalid_winstepsize(self):
        with pytest.raises(ValueError, match="winstepsize"):
            dicpara_default(winstepsize=15)

    def test_invalid_winsize_min(self):
        with pytest.raises(ValueError, match="winsize_min"):
            dicpara_default(winsize_min=6)

    def test_winsize_min_gt_winstepsize(self):
        with pytest.raises(ValueError, match="winsize_min"):
            dicpara_default(winsize_min=32, winstepsize=16)

    def test_invalid_winsize_odd(self):
        with pytest.raises(ValueError, match="winsize"):
            dicpara_default(winsize=33)

    def test_invalid_mu(self):
        with pytest.raises(ValueError, match="mu"):
            dicpara_default(mu=-1)

    def test_invalid_tol(self):
        with pytest.raises(ValueError, match="tol"):
            dicpara_default(tol=0)
        with pytest.raises(ValueError, match="tol"):
            dicpara_default(tol=1.0)

    def test_invalid_reference_mode(self):
        with pytest.raises(ValueError, match="reference_mode"):
            dicpara_default(reference_mode="invalid")

    def test_invalid_gauss_pt_order(self):
        with pytest.raises(ValueError, match="gauss_pt_order"):
            dicpara_default(gauss_pt_order=4)

    def test_valid_accumulative_mode(self):
        p = dicpara_default(reference_mode="accumulative")
        assert p.reference_mode == "accumulative"


# ---------------------------------------------------------------------------
# Interleaved vector utilities
# ---------------------------------------------------------------------------

class TestInterleavedVectors:
    def test_split_merge_uv_roundtrip(self):
        n = 10
        U = np.arange(2 * n, dtype=np.float64)
        u, v = split_uv(U)
        assert u.shape == (n,)
        assert v.shape == (n,)
        np.testing.assert_array_equal(u, U[0::2])
        np.testing.assert_array_equal(v, U[1::2])

        U_rt = merge_uv(u, v)
        np.testing.assert_array_equal(U, U_rt)

    def test_split_merge_F_roundtrip(self):
        n = 5
        F = np.arange(4 * n, dtype=np.float64)
        F11, F21, F12, F22 = split_F(F)
        assert F11.shape == (n,)
        np.testing.assert_array_equal(F11, F[0::4])
        np.testing.assert_array_equal(F21, F[1::4])
        np.testing.assert_array_equal(F12, F[2::4])
        np.testing.assert_array_equal(F22, F[3::4])

        F_rt = merge_F(F11, F21, F12, F22)
        np.testing.assert_array_equal(F, F_rt)

    def test_split_uv_copies(self):
        """split_uv should return copies, not views."""
        U = np.ones(10, dtype=np.float64)
        u, v = split_uv(U)
        u[0] = 999
        assert U[0] == 1.0  # original unchanged


# ---------------------------------------------------------------------------
# DICMesh
# ---------------------------------------------------------------------------

class TestDICMesh:
    def test_simple_mesh(self, simple_quad_mesh):
        mesh = simple_quad_mesh
        assert mesh.coordinates_fem.shape == (9, 2)
        assert mesh.elements_fem.shape == (4, 8)
        assert mesh.irregular.shape == (0, 3)

    def test_mutable(self, simple_quad_mesh):
        """DICMesh is intentionally mutable (not frozen)."""
        mesh = simple_quad_mesh
        mesh.element_min_size = 4
        assert mesh.element_min_size == 4


# ---------------------------------------------------------------------------
# FrameResult / StrainResult
# ---------------------------------------------------------------------------

class TestResults:
    def test_frame_result_frozen(self):
        U = np.zeros(20)
        fr = FrameResult(U=U)
        with pytest.raises(AttributeError):
            fr.U = np.ones(20)  # type: ignore

    def test_strain_result_minimal(self):
        sr = StrainResult(
            disp_u=np.zeros(10),
            disp_v=np.zeros(10),
        )
        assert sr.dudx is None
        assert sr.strain_exx is None
