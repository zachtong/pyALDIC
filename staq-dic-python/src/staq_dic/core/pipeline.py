"""Full AL-DIC pipeline assembly.

Port of MATLAB run_aldic.m (Jin Yang, Caltech / Zach Tong refactoring).

Orchestrates the complete Augmented Lagrangian DIC workflow:
    Section 2b  — Normalize images, initialize storage
    Section 3   — Compute initial guess (FFT cross-correlation + mesh)
    Section 4   — Local ICGN (Subproblem 1 initial solve)
    Section 5   — Global FEM solve (Subproblem 2 initial solve)
    Section 6   — ADMM iterations (alternating Subpb1 + Subpb2)
    Section 7   — Convergence check and frame loop
    Cumulative  — Transform incremental -> cumulative displacements
    Section 8   — Compute strains (optional)

MATLAB/Python differences:
    - MATLAB uses struct-based data passing; Python uses typed dataclasses.
    - MATLAB ``parfor`` → Python ``concurrent.futures`` or ``joblib``.
    - MATLAB ``ba_interp2_spline`` (MEX C) → ``scipy.ndimage.map_coordinates``.
    - MATLAB persistent workspace variables → explicit function arguments.
    - MATLAB ``tic``/``toc`` → ``time.perf_counter``.
    - Progress reporting via callback functions (same pattern in both).
    - MATLAB ExportIntermediates saves .mat files; Python saves .npz files
      for cross-validation with the MATLAB version.

References:
    [1] J Yang, K Bhattacharya. Augmented Lagrangian Digital Image
        Correlation. Exp. Mech., 2019.
    [2] J Yang, K Bhattacharya. Fast adaptive mesh augmented Lagrangian
        Digital Image Correlation. Exp. Mech., 2021.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import replace
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .data_structures import (
    DICMesh,
    DICPara,
    FrameResult,
    PipelineResult,
    StrainResult,
)
from ..io.image_ops import compute_image_gradient, normalize_images
from ..mesh.mesh_setup import mesh_setup
from ..solver.init_disp import init_disp
from ..solver.integer_search import integer_search, integer_search_pyramid
from ..solver.local_icgn import local_icgn
from ..solver.subpb1_solver import subpb1_solver
from ..solver.subpb2_solver import subpb2_solver
from ..strain.compute_strain import compute_strain as _compute_strain_fn
from ..strain.nodal_strain_fem import global_nodal_strain_fem
from ..strain.smooth_field import smooth_field_sparse
from ..utils.interpolation import scattered_interpolant
from ..utils.region_analysis import NodeRegionMap, precompute_node_regions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _default_progress(frac: float, msg: str) -> None:
    """Print progress to console (fallback)."""
    logger.info("[%3.0f%%] %s", frac * 100, msg)


def _default_stop() -> bool:
    """Never stop (fallback)."""
    return False


def _smooth_disp(
    U: NDArray[np.float64],
    coords: NDArray[np.float64],
    para: DICPara,
    region_map: NodeRegionMap,
) -> NDArray[np.float64]:
    """Smooth displacement field via sparse Gaussian kernel."""
    sigma = para.winstepsize * max(0.3, 500.0 * para.disp_smoothness)
    return smooth_field_sparse(U, coords, sigma, region_map, n_components=2)


def _smooth_strain_field(
    F: NDArray[np.float64],
    coords: NDArray[np.float64],
    para: DICPara,
    region_map: NodeRegionMap,
) -> NDArray[np.float64]:
    """Smooth deformation gradient field via sparse Gaussian kernel."""
    sigma = para.winstepsize * max(0.3, 500.0 * para.strain_smoothness)
    return smooth_field_sparse(F, coords, sigma, region_map, n_components=4)


def _restore_at_nodes(
    target: NDArray[np.float64],
    source: NDArray[np.float64],
    node_indices: NDArray[np.int64],
    n_components: int,
) -> NDArray[np.float64]:
    """Restore values at specified nodes from source array.

    Used to preserve local ICGN results at hole/edge boundary nodes
    after global FEM smoothing (which may produce poor values there).
    """
    if len(node_indices) == 0:
        return target
    result = target.copy()
    for c in range(n_components):
        result[n_components * node_indices + c] = (
            source[n_components * node_indices + c]
        )
    return result


def _auto_tune_beta(
    mesh: DICMesh,
    para: DICPara,
    mu: float,
    U_subpb1: NDArray[np.float64],
    F_subpb1: NDArray[np.float64],
) -> float:
    """Auto-tune ADMM penalty beta via grid search + quadratic refinement.

    Sweeps ``para.beta_range * winstepsize^2 * mu``, solves subpb2 for each
    candidate, and finds the beta minimizing the normalized error sum.

    Args:
        mesh: DIC FE mesh.
        para: DIC parameters (uses beta_range, winstepsize, gauss_pt_order).
        mu: ADMM image-matching weight.
        U_subpb1: Displacement from subproblem 1 (2*n_nodes,).
        F_subpb1: Deformation gradient from subproblem 1 (4*n_nodes,).

    Returns:
        Optimal beta value.
    """
    beta_list = np.array(para.beta_range) * para.winstepsize ** 2 * mu
    n_beta = len(beta_list)
    n_nodes = mesh.coordinates_fem.shape[0]

    # Zero dual variables for tuning (alpha=0)
    udual_zero = np.zeros(4 * n_nodes, dtype=np.float64)
    vdual_zero = np.zeros(2 * n_nodes, dtype=np.float64)

    err1 = np.zeros(n_beta)
    err2 = np.zeros(n_beta)

    for k, beta_k in enumerate(beta_list):
        logger.info("Beta tuning: trying beta=%.4e (%d/%d)", beta_k, k + 1, n_beta)
        U_trial = subpb2_solver(
            mesh, para.gauss_pt_order, beta_k, mu,
            U_subpb1, F_subpb1, udual_zero, vdual_zero,
            0.0,  # alpha=0 for tuning
            para.winstepsize,
        )
        F_trial = global_nodal_strain_fem(mesh, para, U_trial)
        err1[k] = np.linalg.norm(U_subpb1 - U_trial)
        err2[k] = np.linalg.norm(F_subpb1 - F_trial)

    # Normalize errors and find minimum of combined score
    std1, std2 = np.std(err1), np.std(err2)
    if std1 > 1e-15 and std2 > 1e-15:
        err1_norm = (err1 - np.mean(err1)) / std1
        err2_norm = (err2 - np.mean(err2)) / std2
        err_sum = err1_norm + err2_norm
        idx_best = int(np.argmin(err_sum))

        # Quadratic refinement around minimum
        if 0 < idx_best < n_beta - 1:
            try:
                x = np.log10(beta_list[idx_best - 1 : idx_best + 2])
                y = err_sum[idx_best - 1 : idx_best + 2]
                p = np.polyfit(x, y, 2)
                if abs(p[0]) > 1e-15:
                    beta = 10.0 ** (-p[1] / (2.0 * p[0]))
                else:
                    beta = beta_list[idx_best]
            except Exception:
                beta = beta_list[idx_best]
        else:
            beta = beta_list[idx_best]
    else:
        beta = beta_list[n_beta // 2]

    logger.info("Auto-tuned beta = %.6e", beta)
    return float(beta)


def _apply_post_solve_corrections(
    U_subpb2: NDArray[np.float64],
    F_subpb2: NDArray[np.float64],
    U_subpb1: NDArray[np.float64],
    F_subpb1: NDArray[np.float64],
    mesh: DICMesh,
    para: DICPara,
    region_map: NodeRegionMap,
    mark_hole_strain: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply post-subpb2 smoothing and edge/hole restoration.

    Matches MATLAB's post-solve processing:
        1. Smooth displacement (if disp_smoothness > threshold).
        2. Restore F at hole/edge boundary nodes from subpb1.
        3. Blend 10% subpb2 + 90% subpb1, then smooth strain.
        4. Restore U and F at strain-hole nodes from subpb1.
    """
    coords = mesh.coordinates_fem

    # Smooth displacement
    if para.disp_smoothness > 1e-6:
        U_subpb2 = _smooth_disp(U_subpb2, coords, para, region_map)

    # Restore F at hole/edge nodes
    F_subpb2 = _restore_at_nodes(
        F_subpb2, F_subpb1, mesh.mark_coord_hole_edge, 4,
    )

    # Blend + smooth strain
    if para.strain_smoothness > 1e-6:
        F_blend = 0.1 * F_subpb2 + 0.9 * F_subpb1
        F_subpb2 = _smooth_strain_field(F_blend, coords, para, region_map)

    # Restore at strain-hole nodes
    U_subpb2 = _restore_at_nodes(U_subpb2, U_subpb1, mark_hole_strain, 2)
    F_subpb2 = _restore_at_nodes(F_subpb2, F_subpb1, mark_hole_strain, 4)

    return U_subpb2, F_subpb2


def _compute_cumulative_displacements(
    result_disp: list[FrameResult | None],
    result_fe_mesh: list[DICMesh | None],
    n_frames: int,
    reference_mode: str,
) -> list[FrameResult | None]:
    """Transform incremental displacements to cumulative (if incremental mode).

    For incremental mode: interpolates each frame's displacement onto
    the running coordinate system to accumulate total displacement from
    frame 1.

    For accumulative mode: U is already relative to frame 1, so
    U_accum = U directly.
    """
    if reference_mode == "accumulative":
        for i in range(n_frames - 1):
            if result_disp[i] is not None:
                result_disp[i] = replace(
                    result_disp[i], U_accum=result_disp[i].U.copy(),
                )
        return result_disp

    # Incremental mode: cumulative interpolation
    if result_fe_mesh[0] is None or result_disp[0] is None:
        return result_disp

    ref_coords = result_fe_mesh[0].coordinates_fem
    coord_curr = ref_coords.copy()

    for i in range(n_frames - 1):
        if result_disp[i] is None or result_fe_mesh[i] is None:
            break

        frame_coords = result_fe_mesh[i].coordinates_fem
        frame_U = result_disp[i].U

        u_inc = frame_U[0::2]
        v_inc = frame_U[1::2]

        # Interpolate incremental displacement to current coordinates
        disp_x = scattered_interpolant(frame_coords, u_inc, coord_curr)
        disp_y = scattered_interpolant(frame_coords, v_inc, coord_curr)

        coord_curr = coord_curr + np.column_stack([disp_x, disp_y])

        # Cumulative displacement = current - reference
        delta = coord_curr - ref_coords
        U_accum = np.empty(2 * len(ref_coords), dtype=np.float64)
        U_accum[0::2] = delta[:, 0]
        U_accum[1::2] = delta[:, 1]

        result_disp[i] = replace(result_disp[i], U_accum=U_accum)

    return result_disp


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_aldic(
    para: DICPara,
    images: list[NDArray[np.float64]],
    masks: list[NDArray[np.float64]],
    progress_fn: Callable[[float, str], None] | None = None,
    stop_fn: Callable[[], bool] | None = None,
    compute_strain: bool = True,
    mesh: DICMesh | None = None,
    U0: NDArray[np.float64] | None = None,
) -> PipelineResult:
    """Execute the full AL-DIC pipeline.

    Processes a sequence of image pairs through the augmented Lagrangian
    DIC algorithm: FFT initial guess -> local IC-GN -> ADMM global/local
    alternation -> strain computation.

    Args:
        para: Validated DIC parameters (from ``dicpara_default``).
        images: List of grayscale images as float64 arrays (H, W),
            normalized to [0, 1].  ``images[0]`` is the reference;
            ``images[1:]`` are deformed frames.
        masks: List of binary mask images (H, W), float64.
            ``masks[0]`` is the reference mask; ``masks[1:]`` are
            deformed frame masks.  Use ``np.ones((H, W))`` for no
            masking.
        progress_fn: Optional callback ``(fraction, message)`` for
            progress updates.  ``fraction`` is in [0.0, 1.0].
        stop_fn: Optional callback returning ``True`` to abort.
            Checked between major pipeline steps.
        compute_strain: If ``False``, skip Section 8 (strain computation)
            and return ``PipelineResult`` with empty ``result_strain``.
        mesh: Pre-built DICMesh.  If ``None``, a mesh is generated
            automatically from the FFT search grid.
        U0: Initial displacement guess (2*n_nodes,).  If ``None``,
            computed via FFT cross-correlation (``integer_search``).

    Returns:
        PipelineResult containing per-frame displacements, deformation
        gradients, strains, and mesh snapshots.

    Raises:
        ValueError: If fewer than 2 images are provided.
        RuntimeError: If ``stop_fn`` returns True (user abort) or if
            mesh/U0 are needed but not provided.
    """
    # =====================================================================
    # Validation & defaults
    # =====================================================================
    if len(images) < 2:
        raise ValueError(
            f"At least 2 images required (got {len(images)})"
        )
    if len(masks) != len(images):
        raise ValueError(
            f"masks length ({len(masks)}) != images length ({len(images)})"
        )

    progress = progress_fn or _default_progress
    should_stop = stop_fn or _default_stop

    # =====================================================================
    # Section 2b: Normalize images and initialize storage
    # =====================================================================
    progress(0.0, "Section 2b: Normalizing images...")
    logger.info("--- Section 2b Start ---")

    img_normalized, clamped_roi = normalize_images(images, para.gridxy_roi_range)
    img_h, img_w = images[0].shape
    para = replace(para, gridxy_roi_range=clamped_roi, img_size=(img_h, img_w))

    # Auto-scale FFT search region
    img_min_dim = min(img_h, img_w)
    max_safe = max(10, img_min_dim // 4 - para.winsize)
    if para.size_of_fft_search_region > max_safe:
        old_val = para.size_of_fft_search_region
        new_val = max(10, max_safe)
        para = replace(para, size_of_fft_search_region=new_val)
        msg = (
            f"Auto-scaled FFT search region: {old_val} -> {new_val} "
            f"(image {img_h}x{img_w})"
        )
        warnings.warn(msg, stacklevel=2)
        progress(0.0, msg)

    n_frames = len(img_normalized)
    result_disp: list[FrameResult | None] = [None] * (n_frames - 1)
    result_def_grad: list[FrameResult | None] = [None] * (n_frames - 1)
    result_strain: list[StrainResult | None] = [None] * (n_frames - 1)
    result_fe_mesh: list[DICMesh | None] = [None] * (n_frames - 1)

    dic_mesh: DICMesh | None = mesh
    current_U0: NDArray[np.float64] | None = U0.copy() if U0 is not None else None

    logger.info("--- Section 2b Done ---")

    # =====================================================================
    # Main frame loop (Sections 3-6)
    # =====================================================================
    for frame_idx in range(1, n_frames):
        # --- Stop check ---
        if should_stop():
            logger.info("User abort at frame %d/%d", frame_idx + 1, n_frames)
            raise RuntimeError(
                f"Pipeline aborted by user at frame {frame_idx + 1}"
            )

        frac = (frame_idx - 1) / max(1, n_frames - 2) * 0.6
        progress(frac, f"Processing frame {frame_idx + 1}/{n_frames}")
        logger.info("=== Frame %d/%d ===", frame_idx + 1, n_frames)

        # --- Load reference and deformed images ---
        if para.reference_mode == "accumulative":
            f_mask = masks[0].astype(np.float64)
            f_img = img_normalized[0] * f_mask
        else:  # incremental
            f_mask = masks[frame_idx - 1].astype(np.float64)
            f_img = img_normalized[frame_idx - 1] * f_mask

        g_mask = masks[frame_idx].astype(np.float64)
        g_img = img_normalized[frame_idx] * g_mask
        para = replace(para, img_ref_mask=f_mask)

        Df = compute_image_gradient(f_img, f_mask)

        # =================================================================
        # Section 3: Initial guess / mesh
        # =================================================================
        logger.info("--- Section 3 Start ---")

        if frame_idx == 1 or dic_mesh is None:
            if dic_mesh is None or current_U0 is None:
                # No pre-built mesh/U0: use FFT integer search
                use_pyramid = para.init_fft_search_method >= 2
                if use_pyramid:
                    logger.info("Running pyramid NCC search for initial guess...")
                    x0, y0, u_grid, v_grid, fft_info = integer_search_pyramid(
                        f_img, g_img, para,
                    )
                else:
                    logger.info("Running FFT integer search for initial guess...")
                    x0, y0, u_grid, v_grid, fft_info = integer_search(
                        f_img, g_img, para,
                    )
                current_U0 = init_disp(
                    u_grid, v_grid, fft_info["cc_max"], x0, y0,
                )

                # Build mesh from the FFT grid if not provided
                if dic_mesh is None:
                    dic_mesh = mesh_setup(x0, y0, para)

            # Apply mask: NaN for nodes in masked-out regions
            n_nodes = dic_mesh.coordinates_fem.shape[0]
            node_x = np.clip(
                np.round(dic_mesh.coordinates_fem[:, 0]).astype(int), 0, img_w - 1,
            )
            node_y = np.clip(
                np.round(dic_mesh.coordinates_fem[:, 1]).astype(int), 0, img_h - 1,
            )
            mask_vals = f_mask[node_y, node_x]
            nan_nodes = np.where(mask_vals < 1.0)[0]
            if len(nan_nodes) > 0:
                current_U0[2 * nan_nodes] = np.nan
                current_U0[2 * nan_nodes + 1] = np.nan
        else:
            # Subsequent frames: use previous result as initial guess
            prev = result_disp[frame_idx - 2]
            n_nodes = dic_mesh.coordinates_fem.shape[0]
            if prev is not None:
                current_U0 = prev.U.copy()
            else:
                current_U0 = np.zeros(2 * n_nodes, dtype=np.float64)

        # Snapshot mesh for this frame
        result_fe_mesh[frame_idx - 1] = DICMesh(
            coordinates_fem=dic_mesh.coordinates_fem.copy(),
            elements_fem=dic_mesh.elements_fem.copy(),
            mark_coord_hole_edge=dic_mesh.mark_coord_hole_edge.copy(),
        )

        # Precompute node-to-region mapping for smoothing
        n_nodes = dic_mesh.coordinates_fem.shape[0]
        node_region_map = precompute_node_regions(
            dic_mesh.coordinates_fem, f_mask, (img_h, img_w),
        )

        # Validate
        assert n_nodes > 0, "Section 3: mesh is empty"
        assert len(current_U0) == 2 * n_nodes, (
            f"Section 3: U0 length ({len(current_U0)}) != 2*nNodes ({2 * n_nodes})"
        )

        progress(frac, f"Frame {frame_idx + 1}: S3 done ({n_nodes} nodes)")
        logger.info("--- Section 3 Done ---")

        # =================================================================
        # Section 4: Local ICGN (6-DOF, initial subproblem 1)
        # =================================================================
        logger.info("--- Section 4 Start ---")

        tol = para.tol
        (
            U_subpb1, F_subpb1, local_time, conv_iter_s4,
            bad_pt_num_s4, mark_hole_strain,
        ) = local_icgn(
            current_U0, dic_mesh.coordinates_fem, Df, f_img, g_img, para, tol,
        )

        assert not np.all(np.isnan(U_subpb1)), "Section 4: USubpb1 is entirely NaN"
        assert len(F_subpb1) == 4 * n_nodes, (
            f"Section 4: FSubpb1 length mismatch: {len(F_subpb1)} != {4 * n_nodes}"
        )

        progress(
            frac + 0.1,
            f"Frame {frame_idx + 1}: S4 done (local ICGN, {local_time:.1f}s)",
        )
        logger.info("--- Section 4 Done (%.1fs) ---", local_time)

        if para.use_global_step:
            # =============================================================
            # Section 5: Subproblem 2 initial solve
            # =============================================================
            logger.info("--- Section 5 Start ---")
            t5_start = time.perf_counter()

            # Optional smoothing of initial ICGN results
            if para.disp_smoothness > 1e-6:
                U_subpb1 = _smooth_disp(
                    U_subpb1, dic_mesh.coordinates_fem, para, node_region_map,
                )
            if para.strain_smoothness > 1e-6:
                F_subpb1 = _smooth_strain_field(
                    F_subpb1, dic_mesh.coordinates_fem, para, node_region_map,
                )

            # Initialize ADMM dual variables (all zero at start)
            mu_val = para.mu
            grad_dual = np.zeros(4 * n_nodes, dtype=np.float64)  # MATLAB: udual
            disp_dual = np.zeros(2 * n_nodes, dtype=np.float64)  # MATLAB: vdual
            alpha = para.alpha

            # Beta auto-tuning (first frame only)
            if frame_idx == 1:
                beta_val = _auto_tune_beta(
                    dic_mesh, para, mu_val, U_subpb1, F_subpb1,
                )
            else:
                if para.beta is not None:
                    beta_val = para.beta
                else:
                    beta_val = 1e-3 * para.winstepsize ** 2 * mu_val

            # Solve subpb2
            U_subpb2 = subpb2_solver(
                dic_mesh, para.gauss_pt_order, beta_val, mu_val,
                U_subpb1, F_subpb1, grad_dual, disp_dual,
                alpha, para.winstepsize,
            )
            F_subpb2 = global_nodal_strain_fem(dic_mesh, para, U_subpb2)

            # Post-solve corrections: smoothing + edge/hole restoration
            U_subpb2, F_subpb2 = _apply_post_solve_corrections(
                U_subpb2, F_subpb2, U_subpb1, F_subpb1,
                dic_mesh, para, node_region_map, mark_hole_strain,
            )

            # Update dual variables
            grad_dual = F_subpb2 - F_subpb1
            disp_dual = U_subpb2 - U_subpb1

            t5_elapsed = time.perf_counter() - t5_start
            assert not np.all(np.isnan(U_subpb2)), (
                "Section 5: USubpb2 is entirely NaN"
            )
            progress(
                frac + 0.2,
                f"Frame {frame_idx + 1}: S5 done ({t5_elapsed:.1f}s)",
            )
            logger.info("--- Section 5 Done (%.1fs) ---", t5_elapsed)

            # =============================================================
            # Section 6: ADMM iterations
            # =============================================================
            logger.info("--- Section 6 Start ---")
            tol2 = para.admm_tol

            # Track previous step for convergence check
            U_subpb2_prev = U_subpb2.copy()
            U_subpb1_prev = U_subpb1.copy()

            admm_step = 1
            for admm_iter in range(2, para.admm_max_iter + 1):
                admm_step = admm_iter
                logger.info("ADMM step %d/%d", admm_step, para.admm_max_iter)

                # Per-node winsize (uniform)
                winsize_list = np.full(
                    (n_nodes, 2), para.winsize, dtype=np.float64,
                )
                para = replace(para, winsize_list=winsize_list)

                # --- Subproblem 1 (2-DOF IC-GN) ---
                # Python subpb1_solver expects: (U, F, disp_dual, grad_dual, ...)
                # because its 3rd param "udual" is actually the displacement dual
                U_subpb1, sub1_time, conv_iter_admm, bad_pt_admm = subpb1_solver(
                    U_subpb2, F_subpb2,
                    disp_dual,   # subpb1's "udual" = displacement dual (2*n)
                    grad_dual,   # subpb1's "vdual" = gradient dual (unused)
                    dic_mesh.coordinates_fem,
                    Df, f_img, g_img, mu_val, beta_val, para, tol,
                )
                F_subpb1 = F_subpb2.copy()  # F carries over from subpb2

                assert not np.all(np.isnan(U_subpb1)), (
                    f"ADMM step {admm_step}: USubpb1 is entirely NaN"
                )

                # --- Subproblem 2 (global FEM) ---
                U_subpb2 = subpb2_solver(
                    dic_mesh, para.gauss_pt_order, beta_val, mu_val,
                    U_subpb1, F_subpb1, grad_dual, disp_dual,
                    alpha, para.winstepsize,
                )
                F_subpb2 = global_nodal_strain_fem(dic_mesh, para, U_subpb2)

                # Post-solve corrections
                U_subpb2, F_subpb2 = _apply_post_solve_corrections(
                    U_subpb2, F_subpb2, U_subpb1, F_subpb1,
                    dic_mesh, para, node_region_map, mark_hole_strain,
                )

                assert not np.all(np.isnan(U_subpb2)), (
                    f"ADMM step {admm_step}: USubpb2 is entirely NaN"
                )

                # Convergence check
                update_global = np.linalg.norm(
                    U_subpb2_prev - U_subpb2
                ) / np.sqrt(len(U_subpb2))
                update_local = np.linalg.norm(
                    U_subpb1_prev - U_subpb1
                ) / np.sqrt(len(U_subpb1))
                logger.info(
                    "ADMM step %d: global=%.4e, local=%.4e",
                    admm_step, update_global, update_local,
                )

                # Update dual variables
                grad_dual = F_subpb2 - F_subpb1
                disp_dual = U_subpb2 - U_subpb1

                # Save for next convergence check
                U_subpb2_prev = U_subpb2.copy()
                U_subpb1_prev = U_subpb1.copy()

                if update_global < tol2 or update_local < tol2:
                    logger.info("ADMM converged at step %d", admm_step)
                    break

            progress(
                frac + 0.3,
                f"Frame {frame_idx + 1}: S6 done (ADMM {admm_step} steps)",
            )
            logger.info("--- Section 6 Done (%d steps) ---", admm_step)

            # Use subpb2 result
            U_final = U_subpb2
            F_final = F_subpb2
        else:
            # No global step: local ICGN result is final
            U_final = U_subpb1
            F_final = F_subpb1

        # Store frame results
        result_disp[frame_idx - 1] = FrameResult(U=U_final)
        result_def_grad[frame_idx - 1] = FrameResult(U=U_final, F=F_final)

    # =====================================================================
    # Cumulative displacement transform
    # =====================================================================
    progress(0.6, "Computing cumulative displacements...")
    logger.info("--- Cumulative displacement transform ---")

    result_disp = _compute_cumulative_displacements(
        result_disp, result_fe_mesh, n_frames, para.reference_mode,
    )

    # Validate cumulative result
    if result_disp[0] is not None and result_fe_mesh[0] is not None:
        expected_len = 2 * result_fe_mesh[0].coordinates_fem.shape[0]
        actual_len = (
            len(result_disp[-1].U_accum)
            if result_disp[-1] is not None and result_disp[-1].U_accum is not None
            else 0
        )
        if actual_len > 0:
            assert actual_len == expected_len, (
                f"Cumulative disp length {actual_len} != expected {expected_len}"
            )

    # =====================================================================
    # Section 8: Compute strains (optional)
    # =====================================================================
    if compute_strain and result_fe_mesh[0] is not None:
        progress(0.7, "Section 8: Computing strains...")
        logger.info("--- Section 8 Start ---")

        # Use frame 1 mesh for strain computation (cumulative coords)
        strain_mesh = DICMesh(
            coordinates_fem=result_fe_mesh[0].coordinates_fem,
            elements_fem=result_fe_mesh[0].elements_fem,
            mark_coord_hole_edge=result_fe_mesh[0].mark_coord_hole_edge,
        )

        # Region map on frame 1 mask
        strain_mask = masks[0].astype(np.float64)
        strain_region_map = precompute_node_regions(
            strain_mesh.coordinates_fem, strain_mask, (img_h, img_w),
        )
        para_s8 = replace(para, img_ref_mask=strain_mask)

        for i in range(n_frames - 1):
            if result_disp[i] is None:
                continue

            U_accum = result_disp[i].U_accum
            if U_accum is None:
                U_accum = result_disp[i].U

            # Extra displacement smoothing (Section 8 feature)
            U_local = U_accum.copy()
            if para.smoothness > 0:
                for _ in range(3):
                    U_local = _smooth_disp(
                        U_local, strain_mesh.coordinates_fem,
                        para, strain_region_map,
                    )

            # Compute strain via the strain module
            result_strain[i] = _compute_strain_fn(
                strain_mesh, para_s8, U_local, strain_region_map,
            )

            s8_frac = 0.7 + 0.2 * (i + 1) / (n_frames - 1)
            progress(s8_frac, f"S8: Frame {i + 2}/{n_frames}")

        logger.info("--- Section 8 Done ---")

    # =====================================================================
    # Assemble output
    # =====================================================================
    progress(0.95, "Assembling results...")

    # Filter None entries (from interrupted runs)
    valid_disp = [r for r in result_disp if r is not None]
    valid_grad = [r for r in result_def_grad if r is not None]
    valid_strain = [r for r in result_strain if r is not None]
    valid_mesh = [r for r in result_fe_mesh if r is not None]

    pipeline_result = PipelineResult(
        dic_para=para,
        dic_mesh=dic_mesh
        or DICMesh(
            coordinates_fem=np.empty((0, 2)),
            elements_fem=np.empty((0, 8), dtype=np.int64),
        ),
        result_disp=valid_disp,
        result_def_grad=valid_grad,
        result_strain=valid_strain,
        result_fe_mesh_each_frame=valid_mesh,
    )

    progress(1.0, "Pipeline complete.")
    return pipeline_result
