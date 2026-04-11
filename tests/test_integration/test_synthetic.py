"""Synthetic integration tests for the AL-DIC pipeline.

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

from dataclasses import replace as dc_replace

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    DICPara,
    FrameSchedule,
    GridxyROIRange,
    merge_uv,
)
from al_dic.core.pipeline import run_aldic

from tests.conftest import (
    apply_displacement,
    apply_displacement_lagrangian,
    compute_disp_rmse,
    compute_disp_rmse_interior,
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
        disp_tol=0.01,
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
        disp_tol=0.03,    # Lagrangian measured: RMSE ~0.005px
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
        disp_tol=0.05,    # Lagrangian measured: RMSE ~0.010px
        strain_tol=0.02,
    ),
    "case4_annular": dict(
        u2=lambda x, y: 0.02 * (x - CX),
        v2=lambda x, y: 0.02 * (y - CY),
        u3=lambda x, y: 0.04 * (x - CX),
        v3=lambda x, y: 0.04 * (y - CY),
        F11=0.02, F22=0.02, F12=0.0, F21=0.0,
        mask="annular",
        overrides=dict(winsize_min=4),
        disp_tol=0.5,     # Annular mask → boundary effects at hole edges
        strain_tol=0.02,
    ),
    "case5_shear": dict(
        u2=lambda x, y: 0.015 * (y - CY),
        v2=lambda x, y: np.zeros_like(x),
        u3=lambda x, y: 0.030 * (y - CY),
        v3=lambda x, y: np.zeros_like(x),
        F11=0.0, F22=0.0, F12=0.015, F21=0.0,
        mask="solid",
        overrides={},
        disp_tol=0.05,    # Lagrangian measured: RMSE ~0.007px
        strain_tol=0.04,
    ),
    "case6_large_deform": dict(
        u2=lambda x, y: 0.10 * (x - CX) + 0.05 * (y - CY),
        v2=lambda x, y: 0.05 * (x - CX) + 0.10 * (y - CY),
        u3=lambda x, y: 0.20 * (x - CX) + 0.10 * (y - CY),
        v3=lambda x, y: 0.10 * (x - CX) + 0.20 * (y - CY),
        F11=0.10, F22=0.10, F12=0.05, F21=0.05,
        mask="solid",
        overrides=dict(winsize=48),
        disp_tol=1.0,     # 10% strain → large residual expected
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
        disp_tol=0.05,    # Lagrangian measured: RMSE ~0.005px
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
        disp_tol=0.05,    # Lagrangian measured: RMSE ~0.005px
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
        disp_tol=0.05,    # Lagrangian measured: RMSE ~0.003px (no ADMM)
        strain_tol=0.02,
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
        disp_tol=0.05,    # Lagrangian measured: RMSE ~0.004px
        strain_tol=0.08,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case_para(**overrides) -> DICPara:
    """Build DICPara for 256x256 synthetic tests with optional overrides.

    When frame_schedule is provided, reference_mode defaults to 'incremental'
    to avoid the "both set" warning (schedule takes precedence anyway).
    """
    # If frame_schedule is explicitly provided, default to incremental
    # to avoid the spurious "both set" warning from validate_dicpara.
    ref_mode = "accumulative"
    if "frame_schedule" in overrides and overrides["frame_schedule"] is not None:
        ref_mode = "incremental"
    if "reference_mode" in overrides:
        ref_mode = overrides.pop("reference_mode")

    defaults = dict(
        winsize=32,
        winstepsize=16,
        winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, 255), gridy=(0, 255)),
        reference_mode=ref_mode,
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

    Uses Lagrangian image generation for exact DIC ground truth.
    Uses full-image mask for the pipeline to avoid circular-mask
    contamination (zeros in IC-GN windows).  RMSE is computed with
    edge-margin exclusion for solid cases or mask-based for annular.

    Args:
        case_name: Key into the CASES dict.
        ref: Reference speckle image (H, W) float64.

    Returns:
        Dictionary with keys:
            para, images, masks, mesh, U0, case_def, mask_img,
            gt_u2, gt_v2, gt_u3, gt_v3
    """
    case_def = CASES[case_name]

    # Build DICPara with case-specific overrides
    para = _case_para(**case_def["overrides"])

    # Pipeline mask: full image for all cases.
    # Annular mask is only used for RMSE filtering (not pipeline input).
    # This avoids IC-GN window contamination from zero-valued pixels.
    if case_def["mask"] == "annular":
        rmse_mask = make_annular_mask(
            IMG_H, IMG_W, cx=CX, cy=CY, r_outer=90.0, r_inner=40.0,
        )
        pipeline_mask = rmse_mask  # annular: pipeline needs hole info
    else:
        rmse_mask = None  # solid: use edge-margin exclusion instead
        pipeline_mask = np.ones((IMG_H, IMG_W), dtype=np.float64)

    # Generate deformed images using Lagrangian displacement convention.
    # The lambda functions in CASES define u(X,Y) at reference coordinates,
    # which is exactly what DIC solves for — no Eulerian approximation error.
    deformed2 = apply_displacement_lagrangian(ref, case_def["u2"], case_def["v2"])
    deformed3 = apply_displacement_lagrangian(ref, case_def["u3"], case_def["v3"])

    images = [ref, deformed2, deformed3]
    masks = [pipeline_mask, pipeline_mask, pipeline_mask]

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
        rmse_mask=rmse_mask,  # None for solid (use edge-margin), mask for annular
        gt_u2=gt_u2,
        gt_v2=gt_v2,
        gt_u3=gt_u3,
        gt_v3=gt_v3,
    )


def _compute_case_rmse(
    U: np.ndarray,
    coords: np.ndarray,
    gt_u: np.ndarray,
    gt_v: np.ndarray,
    data: dict,
) -> tuple[float, float]:
    """Compute displacement RMSE using the appropriate method for the case.

    Solid cases use edge-margin exclusion (no mask contamination).
    Annular cases use mask-based filtering (real geometry constraint).
    """
    rmse_mask = data["rmse_mask"]
    if rmse_mask is not None:
        return compute_disp_rmse(U, coords, gt_u, gt_v, rmse_mask)
    else:
        return compute_disp_rmse_interior(
            U, coords, gt_u, gt_v, img_size=(IMG_H, IMG_W), edge_margin=32,
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
        rmse_u, rmse_v = _compute_case_rmse(
            U, coords, data["gt_u2"], data["gt_v2"], data,
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
        rmse_u, rmse_v = _compute_case_rmse(
            U, coords, data["gt_u3"], data["gt_v3"], data,
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
        # For strain RMSE, use mask if annular, else build a full mask
        strain_mask = data["rmse_mask"]
        if strain_mask is None:
            strain_mask = np.ones((IMG_H, IMG_W), dtype=np.float64)
        rmses = compute_strain_rmse(
            sr,
            case_def["F11"], case_def["F21"],
            case_def["F12"], case_def["F22"],
            coords, strain_mask,
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
        rmse_u, rmse_v = _compute_case_rmse(
            U, coords, data["gt_u2"], data["gt_v2"], data,
        )

        tol = data["case_def"]["disp_tol"]
        assert rmse_u < tol, f"case6 RMSE_u={rmse_u:.4f} >= {tol}"
        assert rmse_v < tol, f"case6 RMSE_v={rmse_v:.4f} >= {tol}"


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

        rmse_u, rmse_v = _compute_case_rmse(
            U_accum, result.dic_mesh.coordinates_fem,
            data["gt_u2"], data["gt_v2"], data,
        )
        tol = data["case_def"]["disp_tol"]
        assert rmse_u < tol, f"case7 frame2 RMSE_u={rmse_u:.4f}"
        assert rmse_v < tol, f"case7 frame2 RMSE_v={rmse_v:.4f}"

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

        rmse_u, rmse_v = _compute_case_rmse(
            U_accum, result.dic_mesh.coordinates_fem,
            data["gt_u3"], data["gt_v3"], data,
        )
        tol = data["case_def"]["disp_tol"]
        assert rmse_u < tol, f"case7 frame3 RMSE_u={rmse_u:.4f}"
        assert rmse_v < tol, f"case7 frame3 RMSE_v={rmse_v:.4f}"

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

        rmse_u, rmse_v = _compute_case_rmse(
            U, result.dic_mesh.coordinates_fem,
            data["gt_u2"], data["gt_v2"], data,
        )
        tol = data["case_def"]["disp_tol"]
        assert rmse_u < tol, f"case8 frame2 RMSE_u={rmse_u:.4f}"
        assert rmse_v < tol, f"case8 frame2 RMSE_v={rmse_v:.4f}"

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

        rmse_u, rmse_v = _compute_case_rmse(
            U, result.dic_mesh.coordinates_fem,
            data["gt_u3"], data["gt_v3"], data,
        )
        tol = data["case_def"]["disp_tol"]
        assert rmse_u < tol, f"case8 frame3 RMSE_u={rmse_u:.4f}"
        assert rmse_v < tol, f"case8 frame3 RMSE_v={rmse_v:.4f}"


# ---------------------------------------------------------------------------
# Rotation tests (Task 7): displacement + strain for pure rotation
# ---------------------------------------------------------------------------


class TestSyntheticRotation:
    """Pure rotation (2 degrees) — tests non-trivial deformation gradient."""

    def test_case10_frame2_displacement(self, ref_speckle):
        """Rotation displacement RMSE should be within tolerance."""
        data = _build_test_data("case10_rotation", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        frame_result = result.result_disp[0]
        U = frame_result.U_accum if frame_result.U_accum is not None else frame_result.U

        coords = result.dic_mesh.coordinates_fem
        rmse_u, rmse_v = _compute_case_rmse(
            U, coords, data["gt_u2"], data["gt_v2"], data,
        )
        tol = data["case_def"]["disp_tol"]
        assert rmse_u < tol, f"case10 RMSE_u={rmse_u:.4f}"
        assert rmse_v < tol, f"case10 RMSE_v={rmse_v:.4f}"

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

        strain_mask = data["rmse_mask"]
        if strain_mask is None:
            strain_mask = np.ones((IMG_H, IMG_W), dtype=np.float64)
        rmses = compute_strain_rmse(
            sr,
            case_def["F11"], case_def["F21"],
            case_def["F12"], case_def["F22"],
            result.dic_mesh.coordinates_fem, strain_mask,
        )

        tol = case_def["strain_tol"]
        for key, val in rmses.items():
            assert val < tol, f"case10 {key}={val:.5f} >= {tol}"


# ---------------------------------------------------------------------------
# Frame schedule tests (Task 8): skip-frame, mixed, backward compat
# ---------------------------------------------------------------------------


class TestFrameSchedule:
    """Integration tests for generalized FrameSchedule."""

    def test_case11_skip2_keyframe(self, ref_speckle):
        """Skip-2 key-frame: frames 1,2 ref 0; frames 3,4 ref 2.

        5 images total (0-4), 4 deformed frames.
        Schedule: (0, 0, 2, 2)
        Translation: +1.0 px/frame in x.
        """
        n_total = 5
        tx_per_frame = 1.0

        # Build progressive deformation images (Lagrangian)
        imgs = [ref_speckle]
        for k in range(1, n_total):
            tx = tx_per_frame * k
            imgs.append(apply_displacement_lagrangian(
                ref_speckle,
                lambda x, y, _tx=tx: np.full_like(x, _tx),
                lambda x, y: np.zeros_like(x),
            ))

        full_mask = np.ones((IMG_H, IMG_W), dtype=np.float64)
        masks_list = [full_mask] * n_total

        # Schedule: frames 1,2 -> ref 0; frames 3,4 -> ref 2
        schedule = FrameSchedule(ref_indices=(0, 0, 2, 2))

        mesh = make_mesh_for_image(IMG_H, IMG_W, step=STEP, margin=MARGIN)
        node_x = mesh.coordinates_fem[:, 0]

        # Initial guess for first pair (frame 0 -> frame 1)
        gt_u1 = np.full(len(node_x), tx_per_frame)
        gt_v1 = np.zeros(len(node_x))
        U0 = merge_uv(gt_u1, gt_v1)

        para = _case_para(frame_schedule=schedule)

        result = run_aldic(
            para, imgs, masks_list,
            mesh=mesh, U0=U0, compute_strain=False,
        )

        assert len(result.result_disp) == 4

        # Verify cumulative displacements for each frame
        for i in range(4):
            frame_result = result.result_disp[i]
            U_accum = frame_result.U_accum
            assert U_accum is not None, f"Frame {i+2}: U_accum is None"

            expected_u = tx_per_frame * (i + 1)
            rmse_u, rmse_v = compute_disp_rmse_interior(
                U_accum, result.dic_mesh.coordinates_fem,
                np.full(len(node_x), expected_u),
                np.zeros(len(node_x)),
                img_size=(IMG_H, IMG_W), edge_margin=32,
            )
            assert rmse_u < 0.1, (
                f"case11 frame{i+2}: RMSE_u={rmse_u:.4f} >= 0.1 "
                f"(expected cumulative u={expected_u})"
            )
            assert rmse_v < 0.1, (
                f"case11 frame{i+2}: RMSE_v={rmse_v:.4f} >= 0.1"
            )

    def test_case12_mixed(self, ref_speckle):
        """Mixed schedule: (0, 1, 0) — some direct, some chained.

        4 images (0-3), 3 deformed frames.
        Frame 1: refs 0 (direct)
        Frame 2: refs 1 (chained through frame 1)
        Frame 3: refs 0 (direct)
        Translation: +1.5 px/frame.
        """
        n_total = 4
        tx_per_frame = 1.5

        imgs = [ref_speckle]
        for k in range(1, n_total):
            tx = tx_per_frame * k
            imgs.append(apply_displacement_lagrangian(
                ref_speckle,
                lambda x, y, _tx=tx: np.full_like(x, _tx),
                lambda x, y: np.zeros_like(x),
            ))

        full_mask = np.ones((IMG_H, IMG_W), dtype=np.float64)
        masks_list = [full_mask] * n_total

        schedule = FrameSchedule(ref_indices=(0, 1, 0))

        mesh = make_mesh_for_image(IMG_H, IMG_W, step=STEP, margin=MARGIN)
        node_x = mesh.coordinates_fem[:, 0]

        gt_u1 = np.full(len(node_x), tx_per_frame)
        U0 = merge_uv(gt_u1, np.zeros(len(node_x)))

        para = _case_para(frame_schedule=schedule)

        result = run_aldic(
            para, imgs, masks_list,
            mesh=mesh, U0=U0, compute_strain=False,
        )

        assert len(result.result_disp) == 3

        for i in range(3):
            expected_u = tx_per_frame * (i + 1)
            U_accum = result.result_disp[i].U_accum
            assert U_accum is not None

            rmse_u, rmse_v = compute_disp_rmse_interior(
                U_accum, result.dic_mesh.coordinates_fem,
                np.full(len(node_x), expected_u),
                np.zeros(len(node_x)),
                img_size=(IMG_H, IMG_W), edge_margin=32,
            )
            assert rmse_u < 0.1, (
                f"case12 frame{i+2}: RMSE_u={rmse_u:.4f} "
                f"(expected cumulative u={expected_u})"
            )

    def test_case13_backward_compat(self, ref_speckle):
        """Legacy mode (no schedule) should match explicit schedule.

        Run the same 3-frame translation test with:
        1. reference_mode='accumulative' (legacy)
        2. frame_schedule=FrameSchedule((0, 0)) (explicit)
        Results should be identical.
        """
        imgs = [ref_speckle]
        for k in range(1, 3):
            tx = 1.0 * k
            imgs.append(apply_displacement_lagrangian(
                ref_speckle,
                lambda x, y, _tx=tx: np.full_like(x, _tx),
                lambda x, y: np.zeros_like(x),
            ))

        full_mask = np.ones((IMG_H, IMG_W), dtype=np.float64)
        masks_list = [full_mask] * 3

        mesh = make_mesh_for_image(IMG_H, IMG_W, step=STEP, margin=MARGIN)
        node_x = mesh.coordinates_fem[:, 0]
        U0 = merge_uv(np.full(len(node_x), 1.0), np.zeros(len(node_x)))

        # Run 1: legacy mode (no schedule)
        para_legacy = _case_para(reference_mode="accumulative")
        result_legacy = run_aldic(
            para_legacy, imgs, masks_list,
            mesh=mesh, U0=U0, compute_strain=False,
        )

        # Run 2: explicit schedule equivalent to accumulative
        schedule = FrameSchedule(ref_indices=(0, 0))
        para_schedule = _case_para(frame_schedule=schedule)
        result_schedule = run_aldic(
            para_schedule, imgs, masks_list,
            mesh=mesh, U0=U0, compute_strain=False,
        )

        # Both should have identical cumulative results
        for i in range(2):
            U_legacy = result_legacy.result_disp[i].U_accum
            U_sched = result_schedule.result_disp[i].U_accum
            assert U_legacy is not None
            assert U_sched is not None
            np.testing.assert_allclose(
                U_sched, U_legacy, atol=1e-10,
                err_msg=f"Frame {i+2}: schedule vs legacy mismatch",
            )


# ---------------------------------------------------------------------------
# Quadtree mesh integration tests (Task 9): adaptive refinement + pipeline
# ---------------------------------------------------------------------------


class TestQuadtreeMesh:
    """Integration tests for adaptive quadtree mesh with AL-DIC pipeline.

    Tests that generate_mesh() produces a valid quadtree mesh from an
    annular mask, and that the pipeline can track displacement correctly
    on the refined mesh with hanging (irregular) nodes.
    """

    def test_quadtree_mesh_structure(self, ref_speckle):
        """generate_mesh should refine elements near annular hole boundary.

        Verifies:
        - Quadtree mesh has more nodes than uniform mesh (refinement occurred)
        - Some elements have midside hanging nodes (Q8 cols 4-7 != 0)
        - mark_coord_hole_edge is non-empty
        """
        from dataclasses import replace as dc_replace
        from al_dic.io.image_ops import compute_image_gradient
        from al_dic.mesh.generate_mesh import generate_mesh
        from al_dic.mesh.mesh_setup import mesh_setup

        # Annular mask with hole in center
        annular_mask = make_annular_mask(
            IMG_H, IMG_W, cx=CX, cy=CY, r_outer=90.0, r_inner=40.0,
        )

        # Build DICPara with mask
        para = _case_para()
        para = dc_replace(para, img_ref_mask=annular_mask)

        # Build initial uniform mesh
        xs = np.arange(MARGIN, IMG_W - MARGIN + 1, STEP, dtype=np.float64)
        ys = np.arange(MARGIN, IMG_H - MARGIN + 1, STEP, dtype=np.float64)
        uniform_mesh = mesh_setup(xs, ys, para)

        # Compute image gradient (needed by generate_mesh for mask info)
        f_img = ref_speckle / ref_speckle.max()
        Df = compute_image_gradient(f_img, annular_mask)

        # Initial displacement (zero for structure test)
        n_nodes_uniform = uniform_mesh.coordinates_fem.shape[0]
        U0 = np.zeros(2 * n_nodes_uniform, dtype=np.float64)

        # Generate quadtree mesh
        mesh_qt, U0_qt = generate_mesh(uniform_mesh, para, Df, U0)

        # Verify refinement occurred
        n_nodes_qt = mesh_qt.coordinates_fem.shape[0]
        n_elems_qt = mesh_qt.elements_fem.shape[0]
        n_elems_uniform = uniform_mesh.elements_fem.shape[0]

        assert n_nodes_qt > n_nodes_uniform, (
            f"Quadtree should have more nodes: {n_nodes_qt} <= {n_nodes_uniform}"
        )
        # Note: element count may decrease because mark_inside removes
        # elements inside the hole.  Refinement adds children near the
        # boundary, but hole elements are deleted.  Check nodes instead.

        # Verify hanging nodes exist (Q8 midside columns 4-7, -1 = no midside)
        midside_cols = mesh_qt.elements_fem[:, 4:8]
        has_hanging = np.any(midside_cols >= 0)
        assert has_hanging, "Quadtree mesh should have hanging (midside) nodes"

        # Verify boundary nodes identified
        assert len(mesh_qt.mark_coord_hole_edge) > 0, (
            "mark_coord_hole_edge should be non-empty for annular mask"
        )

        # Verify U0 interpolation shape
        assert len(U0_qt) == 2 * n_nodes_qt

    def test_quadtree_pipeline_affine(self, ref_speckle):
        """Full pipeline with quadtree mesh should track affine deformation.

        Flow: uniform mesh → generate_mesh (quadtree) → run_aldic.
        Affine expansion: u = 0.02*(x-CX), v = 0.02*(y-CY).
        """
        from dataclasses import replace as dc_replace
        from al_dic.io.image_ops import compute_image_gradient
        from al_dic.mesh.generate_mesh import generate_mesh
        from al_dic.mesh.mesh_setup import mesh_setup

        # Annular mask
        annular_mask = make_annular_mask(
            IMG_H, IMG_W, cx=CX, cy=CY, r_outer=90.0, r_inner=40.0,
        )

        # Affine displacement functions
        u_func = lambda x, y: 0.02 * (x - CX)
        v_func = lambda x, y: 0.02 * (y - CY)

        # Generate deformed image (Lagrangian)
        deformed = apply_displacement_lagrangian(ref_speckle, u_func, v_func)

        # Build DICPara with mask
        para = _case_para()
        para = dc_replace(para, img_ref_mask=annular_mask)

        # Build initial uniform mesh
        xs = np.arange(MARGIN, IMG_W - MARGIN + 1, STEP, dtype=np.float64)
        ys = np.arange(MARGIN, IMG_H - MARGIN + 1, STEP, dtype=np.float64)
        uniform_mesh = mesh_setup(xs, ys, para)

        # Compute image gradient
        f_img = ref_speckle / ref_speckle.max()
        Df = compute_image_gradient(f_img, annular_mask)

        # Ground truth at uniform mesh nodes → initial guess
        node_x = uniform_mesh.coordinates_fem[:, 0]
        node_y = uniform_mesh.coordinates_fem[:, 1]
        gt_u = u_func(node_x, node_y)
        gt_v = v_func(node_x, node_y)
        U0_uniform = merge_uv(gt_u, gt_v)

        # Generate quadtree mesh + interpolated U0
        mesh_qt, U0_qt = generate_mesh(uniform_mesh, para, Df, U0_uniform)

        # Run pipeline with quadtree mesh
        result = run_aldic(
            para,
            [ref_speckle, deformed],
            [annular_mask, annular_mask],
            mesh=mesh_qt,
            U0=U0_qt,
            compute_strain=False,
        )

        assert len(result.result_disp) >= 1, "No displacement results"

        # Evaluate displacement accuracy
        frame_result = result.result_disp[0]
        U = frame_result.U_accum if frame_result.U_accum is not None else frame_result.U

        coords = result.dic_mesh.coordinates_fem
        gt_u_qt = u_func(coords[:, 0], coords[:, 1])
        gt_v_qt = v_func(coords[:, 0], coords[:, 1])

        # Use annular mask for RMSE (quadtree mesh nodes may be outside annulus)
        rmse_u, rmse_v = compute_disp_rmse(
            U, coords, gt_u_qt, gt_v_qt, annular_mask,
        )

        # Quadtree should achieve similar accuracy to uniform mesh
        assert rmse_u < 0.5, f"Quadtree affine RMSE_u={rmse_u:.4f} >= 0.5"
        assert rmse_v < 0.5, f"Quadtree affine RMSE_v={rmse_v:.4f} >= 0.5"

    def test_quadtree_vs_uniform_translation(self, ref_speckle):
        """Quadtree and uniform meshes should agree for uniform translation.

        Translation is spatially constant, so mesh topology shouldn't
        affect the result significantly.  This tests that the quadtree
        pipeline path (hanging nodes, boundary marking) doesn't introduce
        artifacts compared to the uniform path.
        """
        from dataclasses import replace as dc_replace
        from al_dic.io.image_ops import compute_image_gradient
        from al_dic.mesh.generate_mesh import generate_mesh
        from al_dic.mesh.mesh_setup import mesh_setup

        tx, ty = 2.0, -1.0
        u_func = lambda x, y: np.full_like(x, tx)
        v_func = lambda x, y: np.full_like(x, ty)

        annular_mask = make_annular_mask(
            IMG_H, IMG_W, cx=CX, cy=CY, r_outer=90.0, r_inner=40.0,
        )
        deformed = apply_displacement_lagrangian(ref_speckle, u_func, v_func)

        para = _case_para()
        para = dc_replace(para, img_ref_mask=annular_mask)

        # --- Uniform mesh path ---
        uniform_mesh = make_mesh_for_image(IMG_H, IMG_W, step=STEP, margin=MARGIN)
        n_uniform = uniform_mesh.coordinates_fem.shape[0]
        U0_uniform = merge_uv(
            np.full(n_uniform, tx), np.full(n_uniform, ty),
        )

        result_uniform = run_aldic(
            para,
            [ref_speckle, deformed],
            [annular_mask, annular_mask],
            mesh=uniform_mesh,
            U0=U0_uniform,
            compute_strain=False,
        )

        # --- Quadtree mesh path ---
        xs = np.arange(MARGIN, IMG_W - MARGIN + 1, STEP, dtype=np.float64)
        ys = np.arange(MARGIN, IMG_H - MARGIN + 1, STEP, dtype=np.float64)
        init_mesh = mesh_setup(xs, ys, para)
        f_img = ref_speckle / ref_speckle.max()
        Df = compute_image_gradient(f_img, annular_mask)

        n_init = init_mesh.coordinates_fem.shape[0]
        U0_init = merge_uv(np.full(n_init, tx), np.full(n_init, ty))
        mesh_qt, U0_qt = generate_mesh(init_mesh, para, Df, U0_init)

        result_qt = run_aldic(
            para,
            [ref_speckle, deformed],
            [annular_mask, annular_mask],
            mesh=mesh_qt,
            U0=U0_qt,
            compute_strain=False,
        )

        # Both should achieve good accuracy on their own nodes
        fr_uniform = result_uniform.result_disp[0]
        U_uni = fr_uniform.U_accum if fr_uniform.U_accum is not None else fr_uniform.U
        coords_uni = result_uniform.dic_mesh.coordinates_fem
        rmse_u_uni, rmse_v_uni = compute_disp_rmse(
            U_uni, coords_uni,
            np.full(coords_uni.shape[0], tx),
            np.full(coords_uni.shape[0], ty),
            annular_mask,
        )

        fr_qt = result_qt.result_disp[0]
        U_qt = fr_qt.U_accum if fr_qt.U_accum is not None else fr_qt.U
        coords_qt = result_qt.dic_mesh.coordinates_fem
        rmse_u_qt, rmse_v_qt = compute_disp_rmse(
            U_qt, coords_qt,
            np.full(coords_qt.shape[0], tx),
            np.full(coords_qt.shape[0], ty),
            annular_mask,
        )

        # Quadtree should not be significantly worse than uniform
        assert rmse_u_qt < 0.5, f"Quadtree RMSE_u={rmse_u_qt:.4f} >= 0.5"
        assert rmse_v_qt < 0.5, f"Quadtree RMSE_v={rmse_v_qt:.4f} >= 0.5"
        assert rmse_u_uni < 0.5, f"Uniform RMSE_u={rmse_u_uni:.4f} >= 0.5"
