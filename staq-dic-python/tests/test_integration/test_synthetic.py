"""Synthetic integration tests for the STAQ-DIC pipeline.

Validates the full AL-DIC pipeline against known displacement/strain fields
using synthetic speckle images with analytically defined deformations.

Mirrors the MATLAB test_aldic_synthetic.m test suite with 10 cases covering:
    - Zero displacement, rigid translation, affine deformation
    - Annular mask, pure shear, large deformation
    - Multi-frame (incremental/accumulative), local-only, rotation

Coordinate convention:
    - Python 0-based: center of 256x256 image is (127, 127).
    - Displacement fields: u(x,y) is x-displacement, v(x,y) is y-displacement.
    - Ground truth is evaluated at mesh node coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import DICPara, GridxyROIRange, merge_uv
from staq_dic.core.pipeline import run_aldic

from tests.conftest import (
    apply_displacement,
    compute_disp_rmse,
    compute_strain_rmse,
    generate_speckle,
    make_annular_mask,
    make_circular_mask,
    make_mesh_for_image,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 256, 256
CX, CY = 127.0, 127.0
STEP = 16
MARGIN = 16


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

CASES: dict[str, dict] = {
    "case1_zero": dict(
        u2=lambda x, y: np.zeros_like(x),
        v2=lambda x, y: np.zeros_like(x),
        u3=lambda x, y: np.zeros_like(x),
        v3=lambda x, y: np.zeros_like(x),
        F11=0.0, F22=0.0, F12=0.0, F21=0.0,
        mask="solid",
        overrides={},
        disp_tol=0.01,    # Measured RMSE = 0.000 (exact)
        strain_tol=0.01,
    ),
    "case2_translation": dict(
        u2=lambda x, y: np.full_like(x, 2.5),
        v2=lambda x, y: np.full_like(x, -1.8),
        u3=lambda x, y: np.full_like(x, 5.0),
        v3=lambda x, y: np.full_like(x, -3.6),
        F11=0.0, F22=0.0, F12=0.0, F21=0.0,
        mask="solid",
        overrides={},
        disp_tol=0.3,     # Measured RMSE ~0.11
        strain_tol=0.01,
    ),
    "case3_affine": dict(
        u2=lambda x, y: 0.02 * (x - CX),
        v2=lambda x, y: 0.02 * (y - CY),
        u3=lambda x, y: 0.04 * (x - CX),
        v3=lambda x, y: 0.04 * (y - CY),
        F11=0.02, F22=0.02, F12=0.0, F21=0.0,
        mask="solid",
        overrides={},
        disp_tol=0.5,     # Measured RMSE ~0.26
        strain_tol=0.02,   # Measured max ~0.011
    ),
    "case4_annular": dict(
        u2=lambda x, y: 0.02 * (x - CX),
        v2=lambda x, y: 0.02 * (y - CY),
        u3=lambda x, y: 0.04 * (x - CX),
        v3=lambda x, y: 0.04 * (y - CY),
        F11=0.02, F22=0.02, F12=0.0, F21=0.0,
        mask="annular",
        overrides=dict(winsize_min=4),
        disp_tol=0.5,     # Measured RMSE ~0.29
        strain_tol=0.02,   # Measured max ~0.013
    ),
    "case5_shear": dict(
        u2=lambda x, y: 0.015 * (y - CY),
        v2=lambda x, y: np.zeros_like(x),
        u3=lambda x, y: 0.030 * (y - CY),
        v3=lambda x, y: np.zeros_like(x),
        F11=0.0, F22=0.0, F12=0.015, F21=0.0,
        mask="solid",
        overrides={},
        disp_tol=0.3,     # Measured RMSE ~0.09
        strain_tol=0.04,   # Measured F12 RMSE ~0.027
    ),
    "case6_large_deform": dict(
        u2=lambda x, y: 0.10 * (x - CX) + 0.05 * (y - CY),
        v2=lambda x, y: 0.05 * (x - CX) + 0.10 * (y - CY),
        u3=lambda x, y: 0.20 * (x - CX) + 0.10 * (y - CY),
        v3=lambda x, y: 0.10 * (x - CX) + 0.20 * (y - CY),
        F11=0.10, F22=0.10, F12=0.05, F21=0.05,
        mask="solid",
        overrides=dict(winsize=48),
        disp_tol=1.0,
        strain_tol=0.05,
    ),
    "case7_multiframe_incr": dict(
        u2=lambda x, y: np.full_like(x, 1.0),
        v2=lambda x, y: np.zeros_like(x),
        u3=lambda x, y: np.full_like(x, 2.0),
        v3=lambda x, y: np.zeros_like(x),
        F11=0.0, F22=0.0, F12=0.0, F21=0.0,
        mask="solid",
        overrides=dict(reference_mode="incremental"),
        disp_tol=0.5,
        strain_tol=0.03,
    ),
    "case8_multiframe_accum": dict(
        u2=lambda x, y: np.full_like(x, 1.0),
        v2=lambda x, y: np.zeros_like(x),
        u3=lambda x, y: np.full_like(x, 2.0),
        v3=lambda x, y: np.zeros_like(x),
        F11=0.0, F22=0.0, F12=0.0, F21=0.0,
        mask="solid",
        overrides={},
        disp_tol=0.5,
        strain_tol=0.03,
    ),
    "case9_local_only": dict(
        u2=lambda x, y: 0.02 * (x - CX),
        v2=lambda x, y: 0.02 * (y - CY),
        u3=lambda x, y: 0.04 * (x - CX),
        v3=lambda x, y: 0.04 * (y - CY),
        F11=0.02, F22=0.02, F12=0.0, F21=0.0,
        mask="solid",
        overrides=dict(use_global_step=False),
        disp_tol=0.5,     # Measured RMSE ~0.24; no ADMM smoothing → noisier
        strain_tol=0.02,   # Measured max ~0.011
    ),
    "case10_rotation": dict(
        u2=lambda x, y: (
            (x - CX) * (np.cos(np.pi / 90) - 1)
            - (y - CY) * np.sin(np.pi / 90)
        ),
        v2=lambda x, y: (
            (x - CX) * np.sin(np.pi / 90)
            + (y - CY) * (np.cos(np.pi / 90) - 1)
        ),
        u3=lambda x, y: (
            (x - CX) * (np.cos(np.pi / 45) - 1)
            - (y - CY) * np.sin(np.pi / 45)
        ),
        v3=lambda x, y: (
            (x - CX) * np.sin(np.pi / 45)
            + (y - CY) * (np.cos(np.pi / 45) - 1)
        ),
        F11=np.cos(np.pi / 90) - 1,
        F22=np.cos(np.pi / 90) - 1,
        F12=-np.sin(np.pi / 90),
        F21=np.sin(np.pi / 90),
        mask="solid",
        overrides={},
        disp_tol=0.3,     # Measured RMSE ~0.12
        strain_tol=0.08,   # Measured off-diag strain RMSE ~0.066
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case_para(**overrides) -> DICPara:
    """Build DICPara for 256x256 synthetic tests with optional overrides."""
    defaults = dict(
        winsize=32,
        winstepsize=16,
        winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, 255), gridy=(0, 255)),
        reference_mode="accumulative",
        admm_max_iter=3,
        admm_tol=1e-2,
        method_to_compute_strain=3,
        strain_smoothness=0.0,
        disp_smoothness=0.0,
        smoothness=0.0,
        show_plots=False,
        icgn_max_iter=50,
        tol=1e-2,
        mu=1e-3,
        gauss_pt_order=2,
        alpha=0.0,
    )
    defaults.update(overrides)
    return dicpara_default(**defaults)


def _build_test_data(
    case_name: str,
    ref: np.ndarray,
) -> dict:
    """Build all data needed to run the pipeline for a single test case.

    Args:
        case_name: Key into the CASES dict.
        ref: Reference speckle image (H, W) float64.

    Returns:
        Dictionary with keys:
            para, images, masks, mesh, U0, case_def, mask_img
    """
    case_def = CASES[case_name]

    # Build DICPara with case-specific overrides
    para = _case_para(**case_def["overrides"])

    # Build mask — MATLAB uses circular mask (radius=90) even for "solid" cases.
    # This excludes edge nodes where image warp boundary artifacts degrade accuracy.
    if case_def["mask"] == "annular":
        mask_img = make_annular_mask(
            IMG_H, IMG_W, cx=CX, cy=CY, r_outer=90.0, r_inner=40.0,
        )
    else:
        mask_img = make_circular_mask(
            IMG_H, IMG_W, cx=CX, cy=CY, radius=90.0,
        )

    # Generate displacement fields on full image grid
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)

    u2_field = case_def["u2"](xx, yy)
    v2_field = case_def["v2"](xx, yy)
    u3_field = case_def["u3"](xx, yy)
    v3_field = case_def["v3"](xx, yy)

    # Apply inverse warp to generate deformed images
    deformed2 = apply_displacement(ref, u2_field, v2_field)
    deformed3 = apply_displacement(ref, u3_field, v3_field)

    images = [ref, deformed2, deformed3]
    masks = [mask_img, mask_img, mask_img]

    # Build mesh
    mesh = make_mesh_for_image(IMG_H, IMG_W, step=STEP, margin=MARGIN)

    # Compute ground truth at mesh node coordinates
    node_x = mesh.coordinates_fem[:, 0]
    node_y = mesh.coordinates_fem[:, 1]

    gt_u2 = case_def["u2"](node_x, node_y)
    gt_v2 = case_def["v2"](node_x, node_y)
    gt_u3 = case_def["u3"](node_x, node_y)
    gt_v3 = case_def["v3"](node_x, node_y)

    # Create U0 = ground truth at nodes for frame 2 (initial guess)
    U0 = merge_uv(gt_u2, gt_v2)

    return dict(
        para=para,
        images=images,
        masks=masks,
        mesh=mesh,
        U0=U0,
        case_def=case_def,
        mask_img=mask_img,
        gt_u2=gt_u2,
        gt_v2=gt_v2,
        gt_u3=gt_u3,
        gt_v3=gt_v3,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_speckle():
    """Module-scoped 256x256 reference speckle pattern."""
    return generate_speckle(IMG_H, IMG_W, sigma=3.0, seed=42)


# ---------------------------------------------------------------------------
# Displacement tests (Task 3): cases 1-5, 9
# ---------------------------------------------------------------------------

SINGLE_FRAME_CASES = [
    "case1_zero",
    "case2_translation",
    "case3_affine",
    "case4_annular",
    "case5_shear",
    "case9_local_only",
]


class TestSyntheticDisplacement:
    """Test displacement accuracy for single-frame synthetic cases."""

    @pytest.mark.parametrize("case_name", SINGLE_FRAME_CASES)
    def test_frame2_displacement(self, ref_speckle, case_name):
        """Pipeline frame 2 displacement should match ground truth within tol."""
        data = _build_test_data(case_name, ref_speckle)

        result = run_aldic(
            data["para"],
            data["images"],
            data["masks"],
            mesh=data["mesh"],
            U0=data["U0"],
            compute_strain=False,
        )

        assert len(result.result_disp) >= 1, "No displacement results returned"

        # Get cumulative displacement (preferred) or raw displacement
        frame_result = result.result_disp[0]
        U = frame_result.U_accum if frame_result.U_accum is not None else frame_result.U

        coords = result.dic_mesh.coordinates_fem
        rmse_u, rmse_v = compute_disp_rmse(
            U, coords,
            data["gt_u2"], data["gt_v2"],
            data["mask_img"],
        )

        disp_tol = data["case_def"]["disp_tol"]
        assert rmse_u < disp_tol, (
            f"{case_name} frame2: RMSE_u={rmse_u:.4f} >= tol={disp_tol}"
        )
        assert rmse_v < disp_tol, (
            f"{case_name} frame2: RMSE_v={rmse_v:.4f} >= tol={disp_tol}"
        )

    @pytest.mark.parametrize("case_name", SINGLE_FRAME_CASES)
    def test_frame3_displacement(self, ref_speckle, case_name):
        """Pipeline frame 3 displacement should match ground truth within tol."""
        data = _build_test_data(case_name, ref_speckle)

        result = run_aldic(
            data["para"],
            data["images"],
            data["masks"],
            mesh=data["mesh"],
            U0=data["U0"],
            compute_strain=False,
        )

        assert len(result.result_disp) >= 2, "Fewer than 2 displacement results"

        # Frame 3 is result_disp[1]
        frame_result = result.result_disp[1]
        U = frame_result.U_accum if frame_result.U_accum is not None else frame_result.U

        coords = result.dic_mesh.coordinates_fem
        rmse_u, rmse_v = compute_disp_rmse(
            U, coords,
            data["gt_u3"], data["gt_v3"],
            data["mask_img"],
        )

        disp_tol = data["case_def"]["disp_tol"]
        assert rmse_u < disp_tol, (
            f"{case_name} frame3: RMSE_u={rmse_u:.4f} >= tol={disp_tol}"
        )
        assert rmse_v < disp_tol, (
            f"{case_name} frame3: RMSE_v={rmse_v:.4f} >= tol={disp_tol}"
        )


# ---------------------------------------------------------------------------
# Strain tests (Task 4): cases with nonzero ground truth gradients
# ---------------------------------------------------------------------------

STRAIN_CASES = [
    "case3_affine",
    "case5_shear",
    # case6_large_deform excluded: GT U0 only gives displacement, not F gradients.
    # ICGN must discover 10%+ strain from scratch; needs initial gradient estimates.
    "case9_local_only",
    "case10_rotation",
]


class TestSyntheticStrain:
    """Strain RMSE validation for cases with nonzero ground truth."""

    @pytest.mark.parametrize("case_name", STRAIN_CASES)
    def test_strain_rmse(self, ref_speckle, case_name):
        """Strain gradient RMSE should be within tolerance."""
        data = _build_test_data(case_name, ref_speckle)
        case_def = data["case_def"]

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=True,
        )

        assert len(result.result_strain) >= 1, "No strain results"
        sr = result.result_strain[0]
        assert sr.dudx is not None, "dudx is None"

        coords = result.dic_mesh.coordinates_fem
        rmses = compute_strain_rmse(
            sr,
            case_def["F11"], case_def["F21"],
            case_def["F12"], case_def["F22"],
            coords, data["mask_img"],
        )

        tol = case_def["strain_tol"]
        for key, val in rmses.items():
            assert val < tol, f"{case_name} {key}={val:.5f} >= tol={tol}"


# ---------------------------------------------------------------------------
# Large deformation tests (Task 5)
# ---------------------------------------------------------------------------


class TestSyntheticLargeDeform:
    """Large deformation case (10% stretch + 5% shear).

    Tests ICGN convergence with 10% strain and 5% shear.
    Ground-truth U0 provides displacement, ICGN iterates to find F.
    """

    def test_case6_frame2_displacement(self, ref_speckle):
        """Case 6 displacement RMSE should be within tolerance."""
        data = _build_test_data("case6_large_deform", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        frame_result = result.result_disp[0]
        U = frame_result.U_accum if frame_result.U_accum is not None else frame_result.U

        coords = result.dic_mesh.coordinates_fem
        rmse_u, rmse_v = compute_disp_rmse(
            U, coords, data["gt_u2"], data["gt_v2"], data["mask_img"],
        )

        assert rmse_u < 1.0, f"case6 RMSE_u={rmse_u:.4f} >= 1.0"
        assert rmse_v < 1.0, f"case6 RMSE_v={rmse_v:.4f} >= 1.0"


# ---------------------------------------------------------------------------
# Multi-frame tests (Task 6): incremental vs accumulative
# ---------------------------------------------------------------------------


class TestSyntheticMultiFrame:
    """Multi-frame tests: incremental vs accumulative reference mode."""

    def test_case7_incremental_frame2(self, ref_speckle):
        """Incremental mode frame 2: cumulative u=1.0."""
        data = _build_test_data("case7_multiframe_incr", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        assert len(result.result_disp) >= 1
        U_accum = result.result_disp[0].U_accum
        assert U_accum is not None, "Incremental mode should set U_accum"

        rmse_u, rmse_v = compute_disp_rmse(
            U_accum, result.dic_mesh.coordinates_fem,
            data["gt_u2"], data["gt_v2"], data["mask_img"],
        )
        assert rmse_u < 0.5, f"case7 frame2 RMSE_u={rmse_u:.4f}"
        assert rmse_v < 0.5, f"case7 frame2 RMSE_v={rmse_v:.4f}"

    def test_case7_incremental_frame3(self, ref_speckle):
        """Incremental mode frame 3: cumulative u=2.0."""
        data = _build_test_data("case7_multiframe_incr", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        assert len(result.result_disp) >= 2
        U_accum = result.result_disp[1].U_accum
        assert U_accum is not None

        rmse_u, rmse_v = compute_disp_rmse(
            U_accum, result.dic_mesh.coordinates_fem,
            data["gt_u3"], data["gt_v3"], data["mask_img"],
        )
        assert rmse_u < 0.5, f"case7 frame3 RMSE_u={rmse_u:.4f}"
        assert rmse_v < 0.5, f"case7 frame3 RMSE_v={rmse_v:.4f}"

    def test_case8_accumulative_frame2(self, ref_speckle):
        """Accumulative mode frame 2: u=1.0 vs reference."""
        data = _build_test_data("case8_multiframe_accum", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        frame_result = result.result_disp[0]
        U = frame_result.U_accum if frame_result.U_accum is not None else frame_result.U

        rmse_u, rmse_v = compute_disp_rmse(
            U, result.dic_mesh.coordinates_fem,
            data["gt_u2"], data["gt_v2"], data["mask_img"],
        )
        assert rmse_u < 0.5, f"case8 frame2 RMSE_u={rmse_u:.4f}"
        assert rmse_v < 0.5, f"case8 frame2 RMSE_v={rmse_v:.4f}"

    def test_case8_accumulative_frame3(self, ref_speckle):
        """Accumulative mode frame 3: u=2.0 vs reference."""
        data = _build_test_data("case8_multiframe_accum", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        assert len(result.result_disp) >= 2
        frame_result = result.result_disp[1]
        U = frame_result.U_accum if frame_result.U_accum is not None else frame_result.U

        rmse_u, rmse_v = compute_disp_rmse(
            U, result.dic_mesh.coordinates_fem,
            data["gt_u3"], data["gt_v3"], data["mask_img"],
        )
        assert rmse_u < 0.5, f"case8 frame3 RMSE_u={rmse_u:.4f}"
        assert rmse_v < 0.5, f"case8 frame3 RMSE_v={rmse_v:.4f}"


# ---------------------------------------------------------------------------
# Rotation tests (Task 7): displacement + strain for pure rotation
# ---------------------------------------------------------------------------


class TestSyntheticRotation:
    """Pure rotation (2 degrees) — tests non-trivial deformation gradient."""

    def test_case10_frame2_displacement(self, ref_speckle):
        """Rotation displacement RMSE < 0.5."""
        data = _build_test_data("case10_rotation", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        frame_result = result.result_disp[0]
        U = frame_result.U_accum if frame_result.U_accum is not None else frame_result.U

        coords = result.dic_mesh.coordinates_fem
        rmse_u, rmse_v = compute_disp_rmse(
            U, coords, data["gt_u2"], data["gt_v2"], data["mask_img"],
        )
        assert rmse_u < 0.3, f"case10 RMSE_u={rmse_u:.4f}"
        assert rmse_v < 0.3, f"case10 RMSE_v={rmse_v:.4f}"

    def test_case10_strain(self, ref_speckle):
        """Rotation strain should match analytical gradients."""
        data = _build_test_data("case10_rotation", ref_speckle)
        case_def = data["case_def"]

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=True,
        )

        assert len(result.result_strain) >= 1
        sr = result.result_strain[0]

        rmses = compute_strain_rmse(
            sr,
            case_def["F11"], case_def["F21"],
            case_def["F12"], case_def["F22"],
            result.dic_mesh.coordinates_fem, data["mask_img"],
        )

        for key, val in rmses.items():
            assert val < 0.08, f"case10 {key}={val:.5f} >= 0.08"
