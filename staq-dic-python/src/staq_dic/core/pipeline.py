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
    FrameSchedule,
    PipelineResult,
    StrainResult,
)
from ..io.image_ops import compute_image_gradient, normalize_images
from ..mesh.mesh_setup import mesh_setup
from ..solver.init_disp import init_disp
from ..solver.integer_search import integer_search, integer_search_pyramid
from ..solver.local_icgn import local_icgn
from ..solver.subpb1_solver import precompute_subpb1, subpb1_solver
from ..solver.subpb2_solver import precompute_subpb2, subpb2_solver
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


def _compute_cumulative_displacements_tree(
    result_disp: list[FrameResult | None],
    result_fe_mesh: list[DICMesh | None],
    n_frames: int,
    schedule: FrameSchedule,
) -> list[FrameResult | None]:
    """Transform per-pair displacements to cumulative via tree-based composition.

    For each deformed frame, traces the reference chain back to frame 0
    and composes displacements along the path.  Intermediate coordinates
    are cached so that shared path segments are computed only once.

    Composition formula for chain A -> B -> C::

        coords_B = coords_A + u_AB(coords_A)
        coords_C = coords_B + u_BC(coords_B)
        u_AC     = coords_C - coords_A

    When a frame's reference is frame 0 directly (accumulative-style),
    U_accum = U (no composition needed).

    Args:
        result_disp: Per-frame displacement results (length n_frames-1).
        result_fe_mesh: Per-frame mesh snapshots (length n_frames-1).
        n_frames: Total number of frames.
        schedule: FrameSchedule defining the reference tree.

    Returns:
        Updated result_disp with U_accum set for each frame.
    """
    if result_fe_mesh[0] is None or result_disp[0] is None:
        return result_disp

    ref_coords = result_fe_mesh[0].coordinates_fem

    # Cache: cum_coords[frame_idx] = absolute coordinates after tracking
    # from frame 0 to frame_idx.  frame 0 = ref_coords.
    cum_coords_cache: dict[int, NDArray[np.float64]] = {0: ref_coords.copy()}

    for i in range(n_frames - 1):
        frame = i + 1  # 1-based deformed frame index
        if result_disp[i] is None or result_fe_mesh[i] is None:
            continue

        # Trace path from this frame to root and find deepest cached ancestor
        path = schedule.path_to_root(frame)  # [frame, ..., 0]

        # Find deepest ancestor already in cache (always at least frame 0)
        cached_idx = 0
        for j, ancestor in enumerate(path):
            if ancestor in cum_coords_cache:
                cached_idx = j
                break

        # Compose from cached ancestor down to this frame
        # path[cached_idx] is the deepest cached, path[0] is frame itself
        # We need to walk path[cached_idx] -> path[cached_idx-1] -> ... -> path[0]
        coord_curr = cum_coords_cache[path[cached_idx]].copy()

        for step in range(cached_idx - 1, -1, -1):
            child_frame = path[step]
            # result_disp[child_frame - 1].U is the displacement for
            # the pair (parent -> child_frame)
            child_result = result_disp[child_frame - 1]
            child_mesh = result_fe_mesh[child_frame - 1]
            if child_result is None or child_mesh is None:
                break

            child_U = child_result.U
            u_inc = child_U[0::2]
            v_inc = child_U[1::2]

            # Interpolate incremental displacement at current coordinates
            disp_x = scattered_interpolant(
                child_mesh.coordinates_fem, u_inc, coord_curr,
            )
            disp_y = scattered_interpolant(
                child_mesh.coordinates_fem, v_inc, coord_curr,
            )
            coord_curr = coord_curr + np.column_stack([disp_x, disp_y])

            # Cache intermediate result for reuse by descendant frames
            if child_frame not in cum_coords_cache:
                cum_coords_cache[child_frame] = coord_curr.copy()

        # Cumulative displacement = final coordinates - reference
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

    # --- Resolve frame schedule ---
    schedule = para.frame_schedule
    if schedule is not None:
        if len(schedule) != n_frames - 1:
            raise ValueError(
                f"frame_schedule length ({len(schedule)}) != "
                f"n_frames-1 ({n_frames - 1})"
            )
    else:
        schedule = FrameSchedule.from_mode(para.reference_mode, n_frames)

    logger.info(
        "Frame schedule: %s (n_frames=%d)", schedule.ref_indices, n_frames,
    )
    logger.info("--- Section 2b Done ---")

    # =====================================================================
    # Caches for reference image precomputation
    # =====================================================================
    # ref_cache[ref_idx] = (f_img, f_img_raw, f_mask, Df)
    ref_cache: dict[int, tuple[
        NDArray[np.float64], NDArray[np.float64], object,
    ]] = {}

    # subpb1_cache[ref_idx] = precomputed subpb1 data (depends on ref image)
    subpb1_precompute_cache: dict[int, object] = {}

    # Track which ref_idx was used for beta tuning
    beta_tuned_for_ref: int | None = None
    beta_val: float = 0.0

    # subpb2 precompute cache (depends on mesh/beta/mu, not ref image)
    subpb2_cache_obj: object = None
    subpb2_cache_beta: float | None = None

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

        # --- Determine reference frame via schedule ---
        ref_idx = schedule.parent(frame_idx)
        logger.info(
            "=== Frame %d/%d (ref=%d) ===", frame_idx + 1, n_frames, ref_idx,
        )

        # --- Load reference image (with cache) ---
        if ref_idx in ref_cache:
            f_img, f_img_raw, f_mask, Df = ref_cache[ref_idx]
        else:
            f_mask = masks[ref_idx].astype(np.float64)
            f_img_raw = img_normalized[ref_idx].copy()
            f_img = f_img_raw * f_mask  # masked version for integer_search
            Df = compute_image_gradient(f_img, f_mask, img_raw=f_img_raw)
            ref_cache[ref_idx] = (f_img, f_img_raw, f_mask, Df)

        # --- Load deformed image ---
        g_mask = masks[frame_idx].astype(np.float64)
        g_img = img_normalized[frame_idx] * g_mask
        para = replace(para, img_ref_mask=f_mask)

        # =================================================================
        # Section 3: Initial guess / mesh
        # =================================================================
        logger.info("--- Section 3 Start ---")

        if dic_mesh is None or current_U0 is None:
            # First frame or no pre-built mesh/U0: use FFT integer search
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
            # Subsequent frames: try sibling reuse, then fall back to FFT
            n_nodes = dic_mesh.coordinates_fem.shape[0]

            # Sibling reuse: find a completed frame with the same reference
            sibling_U = None
            for prev_i in range(frame_idx - 1, 0, -1):
                if (
                    result_disp[prev_i - 1] is not None
                    and schedule.parent(prev_i) == ref_idx
                ):
                    sibling_U = result_disp[prev_i - 1].U
                    logger.info(
                        "Sibling reuse: frame %d -> frame %d (same ref=%d)",
                        prev_i + 1, frame_idx + 1, ref_idx,
                    )
                    break

            if sibling_U is not None:
                current_U0 = sibling_U.copy()
            else:
                # Fall back to previous frame result (may have different ref)
                prev = result_disp[frame_idx - 2]
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
            current_U0, dic_mesh.coordinates_fem, Df, f_img_raw, g_img, para, tol,
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

            # Beta auto-tuning: re-tune when reference frame changes
            if beta_tuned_for_ref != ref_idx:
                beta_val = _auto_tune_beta(
                    dic_mesh, para, mu_val, U_subpb1, F_subpb1,
                )
                beta_tuned_for_ref = ref_idx
            elif para.beta is not None:
                beta_val = para.beta
            # else: keep beta_val from previous tuning for same ref

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

            # Per-node winsize (uniform) — set once before loop
            winsize_list = np.full(
                (n_nodes, 2), para.winsize, dtype=np.float64,
            )
            para = replace(para, winsize_list=winsize_list)

            # Pre-compute reference subsets (cached per ref image)
            if ref_idx not in subpb1_precompute_cache:
                subpb1_precompute_cache[ref_idx] = precompute_subpb1(
                    dic_mesh.coordinates_fem, Df, f_img_raw, para,
                )
            subpb1_pre = subpb1_precompute_cache[ref_idx]

            # Pre-compute and factorize subpb2 stiffness matrix
            # (depends on mesh/beta/mu, not ref image — cache when beta unchanged)
            if subpb2_cache_beta != beta_val or subpb2_cache_obj is None:
                subpb2_cache_obj = precompute_subpb2(
                    dic_mesh, para.gauss_pt_order, beta_val, mu_val, alpha,
                )
                subpb2_cache_beta = beta_val

            for admm_iter in range(2, para.admm_max_iter + 1):
                admm_step = admm_iter
                logger.info("ADMM step %d/%d", admm_step, para.admm_max_iter)

                # --- Subproblem 1 (2-DOF IC-GN) ---
                U_subpb1, sub1_time, conv_iter_admm, bad_pt_admm = subpb1_solver(
                    U_subpb2, F_subpb2,
                    disp_dual,
                    grad_dual,
                    dic_mesh.coordinates_fem,
                    Df, f_img_raw, g_img, mu_val, beta_val, para, tol,
                    precomputed=subpb1_pre,
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
                    precomputed=subpb2_cache_obj,
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

        # Store frame results (with ref_frame metadata)
        result_disp[frame_idx - 1] = FrameResult(
            U=U_final, ref_frame=ref_idx,
        )
        result_def_grad[frame_idx - 1] = FrameResult(
            U=U_final, F=F_final, ref_frame=ref_idx,
        )

    # =====================================================================
    # Cumulative displacement transform (tree-based)
    # =====================================================================
    progress(0.6, "Computing cumulative displacements...")
    logger.info("--- Cumulative displacement transform ---")

    result_disp = _compute_cumulative_displacements_tree(
        result_disp, result_fe_mesh, n_frames, schedule,
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
        frame_schedule=schedule,
    )

    progress(1.0, "Pipeline complete.")
    return pipeline_result
