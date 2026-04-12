"""Tests for core data structures and configuration."""

from dataclasses import replace

import numpy as np
import pytest

from al_dic.core.data_structures import (
    DICPara,
    DICMesh,
    FrameSchedule,
    GridxyROIRange,
    FrameResult,
    StrainResult,
    split_uv,
    merge_uv,
    split_F,
    merge_F,
)
from al_dic.core.config import dicpara_default, validate_dicpara


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

    # --- Extended config validation tests (Phase 1.2) ---

    def test_invalid_admm_max_iter_zero(self):
        with pytest.raises(ValueError, match="admm_max_iter"):
            dicpara_default(admm_max_iter=0)

    def test_invalid_cluster_no_negative(self):
        with pytest.raises(ValueError, match="cluster_no"):
            dicpara_default(cluster_no=-1)

    def test_invalid_icgn_max_iter_zero(self):
        with pytest.raises(ValueError, match="icgn_max_iter"):
            dicpara_default(icgn_max_iter=0)

    def test_invalid_init_guess_mode(self):
        with pytest.raises(ValueError, match="init_guess_mode"):
            dicpara_default(init_guess_mode="invalid")

    def test_valid_init_guess_modes(self):
        for mode in ("auto", "fft", "previous"):
            p = dicpara_default(init_guess_mode=mode)
            assert p.init_guess_mode == mode

    def test_invalid_size_of_fft_search_region_zero(self):
        with pytest.raises(ValueError, match="size_of_fft_search_region"):
            dicpara_default(size_of_fft_search_region=0)

    def test_invalid_strain_plane_fit_rad_negative(self):
        with pytest.raises(ValueError, match="strain_plane_fit_rad"):
            dicpara_default(strain_plane_fit_rad=-1)

    def test_invalid_disp_smoothness_negative(self):
        with pytest.raises(ValueError, match="disp_smoothness"):
            dicpara_default(disp_smoothness=-0.1)

    def test_invalid_strain_smoothness_negative(self):
        with pytest.raises(ValueError, match="strain_smoothness"):
            dicpara_default(strain_smoothness=-0.1)

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha"):
            dicpara_default(alpha=-1)

    def test_frame_schedule_wrong_type(self):
        with pytest.raises(TypeError, match="FrameSchedule"):
            dicpara_default(frame_schedule="bad")

    def test_frame_schedule_non_incremental_warns(self):
        sched = FrameSchedule.from_mode("accumulative", 4)
        with pytest.warns(UserWarning, match="frame_schedule"):
            dicpara_default(
                frame_schedule=sched,
                reference_mode="accumulative",
            )

    def test_valid_frame_schedule(self):
        sched = FrameSchedule.from_mode("incremental", 4)
        p = dicpara_default(frame_schedule=sched)
        assert p.frame_schedule is sched


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


# ---------------------------------------------------------------------------
# FrameSchedule extensions
# ---------------------------------------------------------------------------

class TestFrameScheduleExtensions:
    # --- from_every_n ---

    def test_from_every_n_basic(self):
        """every_n=2 with 6 frames: refs at 0,2,4."""
        sched = FrameSchedule.from_every_n(n=2, n_frames=6)
        assert sched.ref_indices == (0, 0, 2, 2, 4)

    def test_from_every_n_every_frame(self):
        """every_n=1 is same as incremental."""
        sched = FrameSchedule.from_every_n(n=1, n_frames=4)
        assert sched.ref_indices == (0, 1, 2)

    def test_from_every_n_too_large(self):
        """n larger than n_frames-1: all frames reference frame 0."""
        sched = FrameSchedule.from_every_n(n=100, n_frames=4)
        assert sched.ref_indices == (0, 0, 0)

    def test_from_every_n_n3_frames7(self):
        """every_n=3 with 7 frames: refs at 0,3,6."""
        sched = FrameSchedule.from_every_n(n=3, n_frames=7)
        assert sched.ref_indices == (0, 0, 0, 3, 3, 3)

    def test_from_every_n_invalid_n(self):
        """n must be >= 1."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            FrameSchedule.from_every_n(n=0, n_frames=4)

    def test_from_every_n_invalid_n_frames(self):
        """n_frames must be >= 2."""
        with pytest.raises(ValueError, match="n_frames must be >= 2"):
            FrameSchedule.from_every_n(n=2, n_frames=1)

    # --- from_custom ---

    def test_from_custom_basic(self):
        """Custom ref frames: 0, 3 with 6 frames."""
        sched = FrameSchedule.from_custom(custom_refs=[0, 3], n_frames=6)
        assert sched.ref_indices == (0, 0, 0, 3, 3)

    def test_from_custom_single_ref(self):
        """Only frame 0 as ref = accumulative."""
        sched = FrameSchedule.from_custom(custom_refs=[0], n_frames=4)
        assert sched.ref_indices == (0, 0, 0)

    def test_from_custom_rejects_last_frame(self):
        """Last frame cannot be a ref."""
        with pytest.raises(ValueError, match="last frame"):
            FrameSchedule.from_custom(custom_refs=[0, 5], n_frames=6)

    def test_from_custom_always_includes_zero(self):
        """Frame 0 is always included even if not in list."""
        sched = FrameSchedule.from_custom(custom_refs=[3], n_frames=6)
        assert sched.parent(1) == 0

    def test_from_custom_unsorted_input(self):
        """custom_refs need not be sorted."""
        sched = FrameSchedule.from_custom(custom_refs=[3, 0], n_frames=6)
        assert sched.ref_indices == (0, 0, 0, 3, 3)

    def test_from_custom_invalid_n_frames(self):
        """n_frames must be >= 2."""
        with pytest.raises(ValueError, match="n_frames must be >= 2"):
            FrameSchedule.from_custom(custom_refs=[0], n_frames=1)

    def test_from_custom_negative_ref(self):
        """Negative ref frame indices are invalid."""
        with pytest.raises(ValueError, match="negative"):
            FrameSchedule.from_custom(custom_refs=[-1, 0], n_frames=4)

    def test_from_custom_out_of_range_ref(self):
        """Ref frame index >= n_frames is invalid."""
        with pytest.raises(ValueError, match="out of range"):
            FrameSchedule.from_custom(custom_refs=[0, 10], n_frames=6)

    # --- ref_frame_set ---

    def test_ref_frame_set_accumulative(self):
        sched = FrameSchedule.from_mode("accumulative", 5)
        assert sched.ref_frame_set == {0}

    def test_ref_frame_set_incremental(self):
        sched = FrameSchedule.from_mode("incremental", 5)
        assert sched.ref_frame_set == {0, 1, 2, 3}

    def test_ref_frame_set_every_n(self):
        sched = FrameSchedule.from_every_n(n=2, n_frames=6)
        assert sched.ref_frame_set == {0, 2, 4}
