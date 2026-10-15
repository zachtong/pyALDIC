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
from collections import OrderedDict
from dataclasses import replace
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ._brush_warp import warp_brush_mask_to_ref
from ._grid_extent import ReferenceGrids, log_grid_plan, plan_reference_grids
from .data_structures import (
    DICMesh,
    DICPara,
    FrameProvider,
    FrameResult,
    FrameSchedule,
    PipelineResult,
    StrainResult,
)
from ..io.image_ops import ListFrameProvider, compute_image_gradient
from ..mesh.criteria.brush_region import BrushRegionCriterion
from ..mesh.element_interp import (
    ElementInterpolator,
    GapSuspector,
    majority_masked,
)
from ..mesh.mark_bridging import trimmed_keep_indices
from ..mesh.mesh_setup import mesh_setup
from ..mesh.rasterize import crack_mask_from_deformed
from ..mesh.refinement import RefinementContext, RefinementPolicy, refine_mesh
from ..solver.init_disp import init_disp
from ..solver.integer_search import integer_search, integer_search_pyramid
from ..solver.local_icgn import local_icgn
from ..solver.seed_prop_pipeline import (
    SeedPropagationState,
    build_grid_for_roi,
    capture_for_next_frame,
    compute_seed_prop_init_guess,
)
from ..solver.seed_propagation import (
    PartialResultError,
    SeedPropagationError,
)
from ..solver.subpb1_solver import precompute_subpb1, subpb1_solver
from ..solver.subpb2_solver import precompute_subpb2, subpb2_solver
from ..strain.compute_strain import compute_strain as _compute_strain_fn
from ..strain.nodal_strain_fem import global_nodal_strain_fem
from ..strain.smooth_field import compute_node_local_spacing, smooth_field_sparse
from ..utils.interpolation import scattered_interpolant_uv
from ..utils.region_analysis import NodeRegionMap, precompute_node_regions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _default_progress(frac: float, msg: str) -> None:
    """Print progress to console (fallback)."""
    logger.info("[%3.0f%%] %s", frac * 100, msg)


def _maybe_warp_brush_for_ref(
    criteria: list,
    *,
    ref_idx: int,
    schedule: "FrameSchedule",
    result_disp: list,
    result_fe_mesh: list,
    img_shape: tuple[int, int],
) -> list:
    """Replace the BrushRegionCriterion with a warped version for ref_idx>0.

    The user-painted brush mask lives in frame-0 image coordinates.  When
    AL-DIC switches to a non-zero reference frame K (incremental mode),
    the same material points have moved, so the mask must be warped using
    the cumulative displacement field 0 -> K already computed within the
    same Run.

    V1 only handles the simple case where ``schedule.parent(ref_idx) == 0``,
    i.e. K was tracked directly against frame 0 and ``result_disp[K-1].U``
    is already the frame-0 -> frame-K displacement on K's mesh.  Deeper
    parent chains fall back to the raw mask with a warning.

    Parameters
    ----------
    criteria : list of refinement criteria
        Original pre-solve criteria from the policy (will be returned
        unchanged when no brush criterion is present).
    ref_idx : int
        Current reference frame index (0-based).  Caller guarantees > 0.
    schedule : FrameSchedule
        For walking the parent chain.
    result_disp : list of FrameResult or None
        Per-frame displacement results computed so far.  Index ``i``
        corresponds to frame ``i+1`` (1-based naming).
    result_fe_mesh : list of DICMesh or None
        Per-frame meshes computed so far.  Same indexing as result_disp.
    img_shape : (H, W)
        Image dimensions in pixels.

    Returns
    -------
    new_criteria : list
        Same as input when no brush criterion exists or warping fails;
        otherwise the original list with the BrushRegionCriterion
        replaced by one whose mask has been warped to ref_idx.
    """
    brush_idx = next(
        (i for i, c in enumerate(criteria) if isinstance(c, BrushRegionCriterion)),
        None,
    )
    if brush_idx is None:
        return criteria

    parent_of_ref = schedule.parent(ref_idx)
    if parent_of_ref != 0:
        logger.warning(
            "Brush warp not yet supported for ref frame %d (parent=%d != 0); "
            "using raw frame-0 mask",
            ref_idx, parent_of_ref,
        )
        return criteria

    # result_disp[K-1] is the result for frame K (1-based naming).
    # Since parent(K) == 0, its U is already the frame-0 -> frame-K
    # displacement on the K-th mesh.
    disp_for_ref = result_disp[ref_idx - 1] if ref_idx - 1 < len(result_disp) else None
    mesh_for_ref = (
        result_fe_mesh[ref_idx - 1] if ref_idx - 1 < len(result_fe_mesh) else None
    )
    if disp_for_ref is None or mesh_for_ref is None:
        logger.warning(
            "Brush warp skipped: missing prior result for ref frame %d",
            ref_idx,
        )
        return criteria

    brush_orig = criteria[brush_idx]
    try:
        warped_bool = warp_brush_mask_to_ref(
            brush_orig.refinement_mask.astype(bool),
            disp_for_ref.U,
            mesh_for_ref.coordinates_fem,
            img_shape,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "Brush warp failed for ref frame %d: %s — using raw frame-0 mask",
            ref_idx, exc,
        )
        return criteria

    new_criteria = list(criteria)
    new_criteria[brush_idx] = BrushRegionCriterion(
        refinement_mask=warped_bool.astype(np.float64),
        min_element_size=brush_orig.min_element_size,
    )
    logger.info(
        "Brush mask warped to ref frame %d (%d -> %d painted px)",
        ref_idx,
        int(brush_orig.refinement_mask.sum()),
        int(warped_bool.sum()),
    )
    return new_criteria


def _default_stop() -> bool:
    """Never stop (fallback)."""
    return False


def _smooth_disp(
    U: NDArray[np.float64],
    coords: NDArray[np.float64],
    para: DICPara,
    region_map: NodeRegionMap,
    mesh: DICMesh | None = None,
) -> NDArray[np.float64]:
    """Smooth displacement field via sparse Gaussian kernel.

    When *mesh* is provided the sigma is adaptive: proportional to
    each node's local element spacing (prevents over-smoothing on
    refined meshes).  Otherwise falls back to a uniform sigma based
    on ``para.winstepsize``.
    """
    factor = 500.0 * para.disp_smoothness
    if mesh is not None:
        sigma = compute_node_local_spacing(mesh.coordinates_fem, mesh.elements_fem) * factor
    else:
        sigma = para.winstepsize * factor
    return smooth_field_sparse(U, coords, sigma, region_map, n_components=2)


def _smooth_strain_field(
    F: NDArray[np.float64],
    coords: NDArray[np.float64],
    para: DICPara,
    region_map: NodeRegionMap,
    mesh: DICMesh | None = None,
) -> NDArray[np.float64]:
    """Smooth deformation gradient field via sparse Gaussian kernel.

    When *mesh* is provided the sigma is adaptive (see ``_smooth_disp``).
    """
    factor = 500.0 * para.strain_smoothness
    if mesh is not None:
        sigma = compute_node_local_spacing(mesh.coordinates_fem, mesh.elements_fem) * factor
    else:
        sigma = para.winstepsize * factor
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
    mark_hole_strain: NDArray[np.int64] | None = None,
) -> float:
    """Auto-tune ADMM penalty beta via grid search + quadratic refinement.

    Sweeps ``para.beta_range * winstepsize^2 * mu``, solves subpb2 for each
    candidate, and finds the beta minimizing the normalized error sum.

    Boundary nodes (``mark_hole_strain``) are excluded from the error metric
    because their IC-GN subsets may overlap mask edges, producing noisier
    displacement/gradient estimates that would bias beta selection toward
    over-regularization.

    Args:
        mesh: DIC FE mesh.
        para: DIC parameters (uses beta_range, winstepsize, gauss_pt_order).
        mu: ADMM image-matching weight.
        U_subpb1: Displacement from subproblem 1 (2*n_nodes,).
        F_subpb1: Deformation gradient from subproblem 1 (4*n_nodes,).
        mark_hole_strain: Node indices near mask boundary to exclude from
            error metric.  ``None`` or empty means use all nodes.

    Returns:
        Optimal beta value.
    """
    beta_list = np.array(para.beta_range) * para.winstepsize ** 2 * mu
    n_beta = len(beta_list)
    n_nodes = mesh.coordinates_fem.shape[0]

    # Build interior-node mask (exclude boundary nodes from error metric)
    interior_u = np.ones(2 * n_nodes, dtype=bool)
    interior_f = np.ones(4 * n_nodes, dtype=bool)
    if mark_hole_strain is not None and len(mark_hole_strain) > 0:
        for idx in mark_hole_strain:
            interior_u[2 * idx] = False
            interior_u[2 * idx + 1] = False
            interior_f[4 * idx] = False
            interior_f[4 * idx + 1] = False
            interior_f[4 * idx + 2] = False
            interior_f[4 * idx + 3] = False
        logger.info(
            "Beta tuning: excluding %d boundary nodes (%d interior)",
            len(mark_hole_strain), int(np.sum(interior_u[::2])),
        )

    # Build trimmed mesh for F computation (matches main pipeline path)
    trimmed_mesh = mesh
    if mark_hole_strain is not None and len(mark_hole_strain) > 0:
        mhs_set = set(int(i) for i in mark_hole_strain)
        trimmed_elems = mesh.elements_fem.copy()
        for e in range(trimmed_elems.shape[0]):
            for j in range(trimmed_elems.shape[1]):
                if trimmed_elems[e, j] >= 0 and trimmed_elems[e, j] in mhs_set:
                    trimmed_elems[e, :] = -1
                    break
        trimmed_mesh = DICMesh(
            coordinates_fem=mesh.coordinates_fem,
            elements_fem=trimmed_elems,
            mark_coord_hole_edge=mesh.mark_coord_hole_edge,
        )

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
        # Use trimmed mesh for F (consistent with main ADMM loop)
        F_trial = global_nodal_strain_fem(trimmed_mesh, para, U_trial)
        err1[k] = np.linalg.norm((U_subpb1 - U_trial)[interior_u])
        err2[k] = np.linalg.norm((F_subpb1 - F_trial)[interior_f])

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

    # Smooth displacement (adaptive sigma when mesh is non-uniform)
    if para.disp_smoothness > 1e-6:
        U_subpb2 = _smooth_disp(U_subpb2, coords, para, region_map, mesh=mesh)

    # Restore F at hole/edge nodes
    F_subpb2 = _restore_at_nodes(
        F_subpb2, F_subpb1, mesh.mark_coord_hole_edge, 4,
    )

    # Blend + smooth strain
    if para.strain_smoothness > 1e-6:
        F_blend = 0.1 * F_subpb2 + 0.9 * F_subpb1
        F_subpb2 = _smooth_strain_field(F_blend, coords, para, region_map, mesh=mesh)

    # Restore at strain-hole nodes
    U_subpb2 = _restore_at_nodes(U_subpb2, U_subpb1, mark_hole_strain, 2)
    F_subpb2 = _restore_at_nodes(F_subpb2, F_subpb1, mark_hole_strain, 4)

    return U_subpb2, F_subpb2


_TRI_CACHE_SIZE = 4  # max cached Delaunay triangulations in the composition.
# Per-reference grids give each reference's pairs their own lattice, so an
# unbounded cache would pin one triangulation per reference frame. Oldest
# first eviction is value-neutral: Delaunay is deterministic, so a miss
# rebuilds the identical triangulation.


def _compute_cumulative_displacements_tree(
    result_disp: list[FrameResult | None],
    result_fe_mesh: list[DICMesh | None],
    n_frames: int,
    schedule: FrameSchedule,
    progress_fn: Callable[[float, str], None] | None = None,
    progress_start: float = 0.90,
    masks: list[NDArray[np.float64]] | None = None,
    crack_radius: float | None = None,
) -> list[FrameResult | None]:
    """Transform per-pair displacements to cumulative via tree-based composition.

    For each deformed frame, traces the reference chain back to frame 0
    and composes displacements along the path.  Intermediate coordinates
    are cached so that shared path segments are computed only once.

    Composition formula for chain A -> B -> C::

        coords_B = coords_A + u_AB(coords_A)
        coords_C = coords_B + u_BC(coords_B)
        u_AC     = coords_C - coords_B + ... (see loop)

    When a frame's reference is frame 0 directly (accumulative-style),
    the single interpolation step queries exactly at the mesh nodes, so no
    cross-gap mixing can occur.

    Crack awareness (``masks`` + ``crack_radius`` provided): the Delaunay
    interpolant connects the two faces of an open crack straight across the
    gap, so composed steps near a crack mix both faces' motions and smear
    ``U_accum`` along the crack wake.  For query points within
    ``crack_radius`` of a masked-out pixel of the pair's REFERENCE mask, the
    increment is re-evaluated bilinearly INSIDE the pair mesh's own
    (crack-cut) elements -- a kept element never spans the gap, so a face
    point only sees same-face nodes.  Points that resolve to no element and
    have no material within a small margin (deep inside the open gap, i.e.
    destroyed material) become NaN and stay NaN for all descendant frames.
    Nodes whose frame-0 reference pixel is masked (they never were material,
    e.g. the initial crack band) are NaN from the start.  Everything farther
    than ``crack_radius`` from a gap keeps the byte-identical Delaunay path.

    Args:
        result_disp: Per-frame displacement results (length n_frames-1).
        result_fe_mesh: Per-frame mesh snapshots (length n_frames-1).
        n_frames: Total number of frames.
        schedule: FrameSchedule defining the reference tree.
        masks: Optional per-frame ROI masks (length n_frames); enables the
            crack-aware near-gap override.  ``None`` preserves the legacy
            behaviour exactly.
        crack_radius: Suspicion radius in px around masked pixels (typically
            ~2 mesh steps).  ``None`` disables the override.

    Returns:
        Updated result_disp with U_accum set for each frame.  With crack
        awareness enabled, U_accum is NaN at destroyed (in-crack) nodes.
    """
    if result_fe_mesh[0] is None or result_disp[0] is None:
        return result_disp

    ref_coords = result_fe_mesh[0].coordinates_fem

    crack_aware = (
        masks is not None
        and crack_radius is not None
        and crack_radius > 0
        and len(masks) >= n_frames
    )
    # Per-frame gap suspectors, built lazily.  Bounded: along an incremental
    # chain each frame index is needed for at most two consecutive steps
    # (child-side check of step k, parent-side of step k+1), while in
    # accumulative mode frame 0 recurs for every step -- a tiny LRU covers
    # both without accumulating one block map per frame across hundreds of
    # frames.  ElementInterpolators are deliberately NOT cached: each pair
    # mesh is interpolated in exactly one composition step (cum_coords_cache
    # walks one new step per frame), so a cache would never hit but would
    # pin every frame's mesh data in memory.
    gap_cache: OrderedDict[int, GapSuspector] = OrderedDict()
    _GAP_CACHE_SIZE = 4

    def _gap_suspector(frame_idx: int) -> GapSuspector:
        gs = gap_cache.get(frame_idx)
        if gs is None:
            gs = GapSuspector(masks[frame_idx], crack_radius)
            gap_cache[frame_idx] = gs
            if len(gap_cache) > _GAP_CACHE_SIZE:
                gap_cache.popitem(last=False)
        else:
            gap_cache.move_to_end(frame_idx)
        return gs

    coord0 = ref_coords.copy()
    if crack_aware:
        # Nodes whose frame-0 pixel is masked never were material (initial
        # crack band / outside-ROI skirt near a gap): dead from the start.
        h0, w0 = masks[0].shape
        px0 = np.clip(np.round(ref_coords[:, 0]).astype(int), 0, w0 - 1)
        py0 = np.clip(np.round(ref_coords[:, 1]).astype(int), 0, h0 - 1)
        born_dead = (
            (masks[0][py0, px0] <= 0.5) & _gap_suspector(0).flag(ref_coords)
        )
        coord0[born_dead] = np.nan

    # Cache: cum_coords[frame_idx] = absolute coordinates after tracking
    # from frame 0 to frame_idx.  frame 0 = ref_coords.
    cum_coords_cache: dict[int, NDArray[np.float64]] = {0: coord0}
    # Reuse the mesh triangulation across frames: every frame interpolates on
    # the same mesh, and the Delaunay build (~98% of the transform) would
    # otherwise be repeated per frame. Keyed by (points, valid mask) -- one
    # build per uniform-mesh run instead of one per frame, byte-identical
    # away from any crack-aware override region. Bounded to _TRI_CACHE_SIZE
    # entries (oldest evicted after each step).
    tri_cache: dict = {}

    for i in range(n_frames - 1):
        frame = i + 1  # 1-based deformed frame index
        if progress_fn is not None and n_frames > 1:
            # The main frame loop fills 0.0-progress_start (see _loop_end);
            # map this cumulative phase to [progress_start, 1.0]. Now that the
            # transform is fast (cross-frame Delaunay cache), progress_start is
            # near 1.0 so this final sliver is imperceptible rather than a
            # jarring 90%->100% jump on a slow-looking tail.
            progress_fn(
                progress_start
                + (1.0 - progress_start) * (i + 1) / (n_frames - 1),
                "Composing cumulative displacements...",
            )
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

            # Interpolate incremental displacement at current coordinates.
            # u and v are interpolated through ONE shared triangulation
            # (built once instead of twice) -- byte-identical to two
            # scattered_interpolant calls when u/v share the same NaN nodes.
            # Rows already dead (NaN from a crack) stay dead: query only the
            # finite rows so NaN coordinates never reach the interpolators.
            finite_q = np.isfinite(coord_curr).all(axis=1)
            if finite_q.all():
                disp_x, disp_y = scattered_interpolant_uv(
                    child_mesh.coordinates_fem, u_inc, v_inc, coord_curr,
                    tri_cache=tri_cache,
                )
            else:
                disp_x = np.full(len(coord_curr), np.nan)
                disp_y = np.full(len(coord_curr), np.nan)
                if finite_q.any():
                    dx_f, dy_f = scattered_interpolant_uv(
                        child_mesh.coordinates_fem, u_inc, v_inc,
                        coord_curr[finite_q], tri_cache=tri_cache,
                    )
                    disp_x[finite_q] = dx_f
                    disp_y[finite_q] = dy_f

            if crack_aware:
                # Near-gap override: within crack_radius of a masked pixel of
                # the pair's reference mask, re-evaluate the increment inside
                # the pair's own crack-cut elements so it never mixes the two
                # crack faces.  parent = path[step + 1] is the pair reference.
                parent_frame = path[step + 1]
                suspects = (
                    _gap_suspector(parent_frame).flag(coord_curr) & finite_q
                )
                if suspects.any():
                    qi = np.where(suspects)[0]
                    # Death first: a point whose 3x3 neighbourhood is
                    # majority-masked sits in (or is smeared onto) the open
                    # gap -- destroyed material, dead from this frame on.
                    # This must precede the element lookup: an element can
                    # survive the >50% hole-trim rule while a sub-half band
                    # of its bbox is gap, so "geometrically inside a kept
                    # element" does NOT prove the point is material.  A live
                    # node on a straight boundary sees a material majority,
                    # so it cannot be killed here.
                    dead = majority_masked(
                        coord_curr[qi], masks[parent_frame],
                    )
                    kill = qi[dead]
                    disp_x[kill] = np.nan
                    disp_y[kill] = np.nan
                    # Survivors near the gap: re-evaluate inside the pair's
                    # crack-cut elements so the increment never blends the
                    # two crack faces.  Unresolved survivors (boundary
                    # jitter, non-rect elements) keep the Delaunay value.
                    alive = qi[~dead]
                    if alive.size:
                        # Built per step, not cached: each pair mesh is
                        # composed through exactly once (see gap_cache note).
                        ei = ElementInterpolator(
                            child_mesh.coordinates_fem,
                            child_mesh.elements_fem,
                        )
                        u_b, v_b, resolved = ei.interpolate_uv(
                            u_inc, v_inc, coord_curr[alive],
                        )
                        sel = alive[resolved]
                        disp_x[sel] = u_b[resolved]
                        disp_y[sel] = v_b[resolved]

            coord_curr = coord_curr + np.column_stack([disp_x, disp_y])
            while len(tri_cache) > _TRI_CACHE_SIZE:
                tri_cache.pop(next(iter(tri_cache)))  # oldest first

            if crack_aware and child_frame < len(masks):
                # Child-side death check: material consumed BETWEEN parent and
                # child (the freshly-grown crack segment) is invisible in the
                # parent mask -- test the landed position against the CHILD
                # frame's mask so such nodes die immediately instead of
                # surviving one transition frame with a garbage increment.
                sus_c = _gap_suspector(child_frame).flag(coord_curr)
                if sus_c.any():
                    ci = np.where(sus_c)[0]
                    dead_c = majority_masked(
                        coord_curr[ci], masks[child_frame],
                    )
                    coord_curr[ci[dead_c]] = np.nan

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


_REF_BUNDLE_CACHE_SIZE = 2  # max cached reference bundles (bounded LRU).
# Must be >= 1 so the single reference in accumulative mode (ref_idx == 0)
# never evicts (zero regression). 2 optimally serves incremental mode
# (ref = frame-1). Eviction is correctness-safe: a miss deterministically
# recomputes the identical bundle from the frame provider.

_PRECOMPUTE_CACHE_SIZE = 2  # max cached solver precompute contexts (LRU):
# icgn_ctx_cache (6-DOF, keyed by ref+mesh) and subpb1_precompute_cache
# (2-DOF, keyed by ref). Each entry is the heavy ~3 GB (N, Sy, Sx) subset
# cube set; left unbounded they leak O(N_refs) in incremental mode. Bounded
# like the ref bundles; recompute on a miss is byte-identical. Must be >= 1
# so a just-inserted entry survives its own post-insert evict check.


def run_aldic(
    para: DICPara,
    images: list[NDArray[np.float64]] | FrameProvider,
    masks: list[NDArray[np.float64]],
    progress_fn: Callable[[float, str], None] | None = None,
    stop_fn: Callable[[], bool] | None = None,
    compute_strain: bool = True,
    mesh: DICMesh | None = None,
    U0: NDArray[np.float64] | None = None,
    refinement_policy: RefinementPolicy | None = None,
) -> PipelineResult:
    """Execute the full AL-DIC pipeline.

    Processes a sequence of image pairs through the augmented Lagrangian
    DIC algorithm: FFT initial guess -> local IC-GN -> ADMM global/local
    alternation -> strain computation.

    Args:
        para: Validated DIC parameters (from ``dicpara_default``).
            ``para.gridxy_roi_range`` is the frame-0 grid: pairs whose
            reference is frame 0, the canonical mesh and ``U_accum`` use
            it, and the result reports it. A reference frame K != 0 whose
            mask has material outside it (incremental mode, a moving
            specimen) gets its own grid instead: same step and lattice
            phase, covering that material plus at least one ring of
            nodes, capped so every subset fits in the image. A reference
            mask that is empty, equal to ``masks[0]``, has no masked-out
            pixel, or touches all four image edges while ``masks[0]``
            does too keeps the frame-0 grid.
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
            automatically from the FFT search grid.  A given mesh is used
            for every pair; no reference frame gets its own grid.
        U0: Initial displacement guess (2*n_nodes,).  If ``None``,
            computed via FFT cross-correlation (``integer_search``).
        refinement_policy: Optional ``RefinementPolicy`` containing
            pre-solve and/or post-solve refinement criteria.  When
            provided, the mesh is adaptively refined before the solver
            runs each frame.  ``None`` (default) uses a uniform mesh.

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
    # Validate up front, before any eager normalization, so an invalid call
    # raises immediately without allocating the normalized frame stack.
    # len(images) works for both a raw list and a streaming provider.
    if len(images) < 2:
        raise ValueError(
            f"At least 2 images required (got {len(images)})"
        )
    if len(masks) != len(images):
        raise ValueError(
            f"masks length ({len(masks)}) != images length ({len(images)})"
        )

    # Resolve the frame provider. A raw list is wrapped in ListFrameProvider
    # (eager-normalized -- byte-identical to the previous normalize_images
    # path); an object already exposing get_normalized() is used as-is. From
    # here on run_aldic touches only the FrameProvider interface, so a
    # streaming provider can later supply frames lazily with no core change.
    if hasattr(images, "get_normalized"):
        provider: FrameProvider = images  # type: ignore[assignment]
    else:
        provider = ListFrameProvider(images, para.gridxy_roi_range)

    progress = progress_fn or _default_progress
    should_stop = stop_fn or _default_stop

    # =====================================================================
    # Section 2b: Normalize images and initialize storage
    # =====================================================================
    progress(0.0, "Section 2b: Normalizing images...")
    logger.info("--- Section 2b Start ---")

    clamped_roi = provider.clamped_roi
    img_h, img_w = provider.shape
    para = replace(para, gridxy_roi_range=clamped_roi, img_size=(img_h, img_w))

    # Auto-scale FFT search region. Constraint is FFT-grid-specific:
    # the batch NCC kernel needs (2*search + winsize) <= image_dim / 2.
    # Seed propagation uses per-seed single-point NCC whose bound is
    # per-seed (based on seed position vs image edge), not a global
    # image-fraction limit, so skip this clamp for that mode.
    if para.init_guess_mode != "seed_propagation":
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

    n_frames = len(provider)
    result_disp: list[FrameResult | None] = [None] * (n_frames - 1)
    result_def_grad: list[FrameResult | None] = [None] * (n_frames - 1)
    result_strain: list[StrainResult | None] = [None] * (n_frames - 1)
    result_fe_mesh: list[DICMesh | None] = [None] * (n_frames - 1)

    dic_mesh: DICMesh | None = mesh
    mesh_is_external = mesh is not None
    current_U0: NDArray[np.float64] | None = U0.copy() if U0 is not None else None

    # Untrimmed mesh stored for re-trimming when mask changes across frames
    # (e.g., ref-switch in incremental mode with per-frame ROIs/masks).
    base_mesh: DICMesh | None = None
    prev_mask_hash: int | None = None

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

    # --- Plan one grid per reference frame ---
    # gridxy_roi_range is the frame-0 grid. A reference frame whose mask
    # leaves it gets its own phase-aligned grid; every other reference
    # keeps this very object, so a run where nothing leaves it is unchanged.
    # An external mesh is never replaced, so nothing is planned for it.
    grid_roi_ref0 = para.gridxy_roi_range
    if mesh_is_external:
        grids = ReferenceGrids(roi0=grid_roi_ref0)
    else:
        grids = plan_reference_grids(
            grid_roi_ref0, masks, schedule.ref_frame_set, (img_h, img_w),
            para.winsize, para.winstepsize,
        )
        log_grid_plan(grids, (img_h, img_w), para.winsize, para.winstepsize)
    assert grids.roi0 is grid_roi_ref0
    active_roi = grid_roi_ref0  # lattice of the current dic_mesh

    # --- Resolve init guess mode ---
    # "auto" / "fft_ref_update" → treated as "previous" in the loop;
    # ref-switch detection (lines below) already forces FFT when the
    # reference frame changes, making "fft_ref_update" semantically
    # correct without any extra branch.
    init_guess_mode = para.init_guess_mode
    if init_guess_mode in ("auto", "fft_ref_update"):
        init_guess_mode = "previous"
    # Track previous ref to detect ref switches (force FFT on switch)
    prev_ref_idx: int | None = None
    # Collect displacement-frame indices that triggered a ref switch.
    # Surfaced via ``PipelineResult.ref_switch_frames`` so consumers
    # (e.g. the GUI frame navigator) can render markers.
    ref_switch_frames: list[int] = []
    # Partial-results tracking: if a seed-propagation frame fails even after
    # auto-place (a region with no viable seed), stop the loop and keep the
    # frames already computed rather than discarding the whole run.
    stopped_early = False
    stopped_at_frame: int | None = None
    stop_reason = ""
    # Distinguishes a user Cancel (deliver the partial results like a success,
    # no error dialog) from a seed-propagation failure (raise
    # PartialResultError so the GUI shows a remediation dialog). Both KEEP the
    # frames already computed; they differ only in how the final block returns.
    user_cancelled = False

    # Seed-propagation per-run state (None when mode is anything else).
    # V1 restriction: seed_propagation is incompatible with refinement
    # policies that have a pre_solve hook — the pre_solve mesh rebuild
    # invalidates the seed's node_idx mid-frame. Reject loudly.
    use_seed_prop = init_guess_mode == "seed_propagation"
    seed_prop_state: SeedPropagationState | None = None
    if use_seed_prop:
        if refinement_policy is not None and refinement_policy.has_pre_solve:
            raise ValueError(
                "init_guess_mode='seed_propagation' is incompatible with "
                "refinement policies that have pre_solve=True in V1. "
                "Disable pre_solve refinement or use a different mode."
            )
        seed_prop_state = SeedPropagationState.from_para(para)

    logger.info(
        "Frame schedule: %s (n_frames=%d), init_guess=%s",
        schedule.ref_indices, n_frames, init_guess_mode,
    )
    logger.info("--- Section 2b Done ---")

    # =====================================================================
    # Caches for reference image precomputation
    # =====================================================================
    # ref_cache[ref_idx] = (f_img, f_img_raw, f_mask, Df)
    # Bounded LRU (OrderedDict): in incremental mode ref_idx changes every
    # frame, so an unbounded dict accumulates every reference's full bundle
    # (O(N) memory leak). _REF_BUNDLE_CACHE_SIZE keeps only the most-recently
    # -used references; an evicted reference is recomputed identically on the
    # next miss, so DIC results are unchanged.
    ref_cache: OrderedDict[int, tuple[
        NDArray[np.float64], NDArray[np.float64],
        NDArray[np.float64], object,
    ]] = OrderedDict()

    # subpb1_cache[ref_idx] = precomputed subpb1 data (depends on ref image).
    # Bounded LRU: unbounded this leaks ~3 GB per reference in incremental.
    subpb1_precompute_cache: OrderedDict[int, object] = OrderedDict()

    # local_icgn 6-DOF context cache, keyed by ref_idx + mesh hash. The
    # 6-DOF precompute (ref_all, gx_all, gy_all, ... at every node) is
    # heavy — at 4096² with step=16 it's ~3 GB. Without this cache,
    # Section 4 rebuilds it every frame, and on frames 2+ the new 6-DOF
    # cache stacks on top of the still-alive 2-DOF subpb1 cache,
    # doubling the per-frame peak. With this cache, frames sharing a
    # reference reuse the same arrays. Invalidated on ref switch +
    # whenever the mesh changes (mid-run refinement).
    # Bounded LRU: unbounded this leaks ~3 GB per reference in incremental.
    icgn_ctx_cache: OrderedDict[tuple[int, int], object] = OrderedDict()

    # Cross-frame FFT search-radius memo: once auto-expand grows the
    # search radius to cover a frame's actual displacement, subsequent
    # frames on the SAME reference reuse that radius as their starting
    # point instead of restarting from para.size_of_fft_search_region
    # and rediscovering the expansion every frame. Cleared per-ref.
    fft_search_learned: dict[int, int] = {}

    # Track which ref_idx was used for beta tuning
    beta_tuned_for_ref: int | None = None
    beta_val: float = 0.0

    # subpb2 precompute cache (depends on mesh/beta/mu, not ref image)
    subpb2_cache_obj: object = None
    subpb2_cache_beta: float | None = None
    subpb2_cache_trimmed_mesh: DICMesh | None = None
    subpb2_cache_mhs_key: frozenset | None = None

    # =====================================================================
    # Main frame loop (Sections 3-6)
    # =====================================================================
    # Progress budget: frame loop (incl. per-frame strain, Section 8) fills
    # 0–99%; the cumulative transform is the final phase and, now that it is
    # fast (cross-frame Delaunay cache), only needs the last 1%. This keeps
    # the bar advancing smoothly to completion instead of jumping 90->100.
    _loop_end = 0.99
    _n_pairs = max(1, n_frames - 2)
    _frame_budget = _loop_end / _n_pairs

    for frame_idx in range(1, n_frames):
        # --- Stop check ---
        if should_stop():
            # User pressed Cancel. Keep the frames already computed rather than
            # discarding the whole run (honours the Cancel button's promise).
            # Reuse the same partial-result path as a seed-propagation early
            # stop, but flag it as a user cancel so the final block RETURNS the
            # result normally instead of raising PartialResultError — a cancel
            # is not an error and must not pop an error dialog.
            logger.info(
                "User cancel at frame %d/%d — keeping %d completed frame(s)",
                frame_idx + 1, n_frames, frame_idx - 1,
            )
            stopped_early = True
            stopped_at_frame = frame_idx + 1
            stop_reason = "Computation canceled by user."
            user_cancelled = True
            break

        frac = (frame_idx - 1) * _frame_budget
        progress(frac, f"Frame {frame_idx}/{n_frames - 1}")

        # --- Determine reference frame via schedule ---
        ref_idx = schedule.parent(frame_idx)
        logger.info(
            "=== Frame %d/%d (ref=%d) ===", frame_idx + 1, n_frames, ref_idx,
        )

        # Detect ref switch (used both by FFT path and seed-prop warp).
        ref_switched_this_frame = (
            prev_ref_idx is not None and ref_idx != prev_ref_idx
        )
        # Force FFT when reference frame changes — the previous frame's
        # displacement is relative to a different reference, so it cannot
        # serve as a valid initial guess for the new reference.
        if ref_switched_this_frame:
            logger.info(
                "Ref switch %d -> %d: forcing FFT for initial guess",
                prev_ref_idx, ref_idx,
            )
            current_U0 = None
            ref_switch_frames.append(frame_idx)
        prev_ref_idx = ref_idx

        # Solve the pair on its reference frame's grid. A lattice switch
        # only comes with a ref switch, which already reset the initial
        # guess above. The mask hash is reset too: two refs can share one
        # mask but not one grid object, and the fresh untrimmed mesh must
        # be trimmed again.
        if not mesh_is_external:
            pair_roi = grids.roi_for(ref_idx)
            if pair_roi is not active_roi:
                para = replace(para, gridxy_roi_range=pair_roi)
                dic_mesh = None
                base_mesh = None
                prev_mask_hash = None
                active_roi = pair_roi

        # --- Load reference image (with cache) ---
        if ref_idx in ref_cache:
            ref_cache.move_to_end(ref_idx)  # mark most-recently-used
            f_img, f_img_raw, f_mask, Df = ref_cache[ref_idx]
        else:
            f_mask = masks[ref_idx].astype(np.float64)
            f_img_raw = provider.get_normalized(ref_idx).copy()
            f_img = f_img_raw * f_mask  # masked version for gradient computation
            Df = compute_image_gradient(f_img, f_mask, img_raw=f_img_raw)
            ref_cache[ref_idx] = (f_img, f_img_raw, f_mask, Df)
            if len(ref_cache) > _REF_BUNDLE_CACHE_SIZE:
                ref_cache.popitem(last=False)  # evict least-recently-used

        # --- Load deformed image ---
        g_mask = masks[frame_idx].astype(np.float64)
        # FFT search and IC-GN both want the unmasked deformed frame
        # (masking zeros corrupts NCC near ROI boundaries). They both
        # treat the input as read-only — verified: integer_search casts
        # to float32 internally, IC-GN samples via map_coordinates.
        # Share a single copy instead of allocating two identical
        # buffers per frame (saves ~134 MB per frame at 4096²).
        g_img_fft = provider.get_normalized(frame_idx).copy()
        g_img_icgn = g_img_fft
        para = replace(para, img_ref_mask=f_mask)

        # Per-frame mesh independence when refinement policy is active.
        # Don't set dic_mesh = None — that would force a full mesh_setup
        # rebuild.  Instead, force re-trim from base_mesh via mask hash
        # invalidation; pre-solve refinement (below) then refines further.
        # NOTE: do *not* clear current_U0 here.  Sibling reuse (the
        # `else` branch around line 815) re-fetches the previous
        # frame's converged U from result_disp and interpolates it
        # onto the freshly re-trimmed mesh, avoiding an FFT search per
        # frame.  Clearing current_U0 here would force need_fft=True
        # and defeat that warm-start path.
        if refinement_policy is not None and refinement_policy.has_pre_solve:
            prev_mask_hash = None  # Force re-trim from base_mesh
            # Invalidate subpb1 precompute cache since mesh will change
            subpb1_precompute_cache.pop(ref_idx, None)
            # Drop any matching 6-DOF entries: the mesh hash changes
            # after re-trim/refine, so stale entries become unreachable
            # but still hold memory. Rebuild the cache from scratch.
            for _k in [k for k in icgn_ctx_cache if k[0] == ref_idx]:
                icgn_ctx_cache.pop(_k, None)

        # Force FFT for every frame when init_guess_mode == "fft"
        # (keep the mesh, only reset the initial guess).
        # Skip when mesh is externally provided — its grid may differ
        # from the FFT grid, causing size mismatch.
        if (
            init_guess_mode == "fft"
            and dic_mesh is not None
            and not mesh_is_external
        ):
            current_U0 = None

        # Periodic FFT reset: limits warm-start error propagation in
        # accumulative mode.  frame_idx=0 is always FFT (dic_mesh is None),
        # so skip it here to avoid double-reset on frame 0.
        if (
            para.fft_reset_interval > 0
            and frame_idx > 0
            and frame_idx % para.fft_reset_interval == 0
            and dic_mesh is not None
            and not mesh_is_external
        ):
            logger.info(
                "Periodic FFT reset at frame %d (interval=%d)",
                frame_idx + 1, para.fft_reset_interval,
            )
            current_U0 = None

        # =================================================================
        # Section 3: Initial guess / mesh
        # =================================================================
        logger.info("--- Section 3 Start ---")

        need_fft = dic_mesh is None or current_U0 is None

        # For seed_propagation: pre-build the mesh from ROI (first frame)
        # so the standard trim/refine flow below has something to work
        # with without needing FFT's x0/y0 output.
        if use_seed_prop and dic_mesh is None:
            img_h_sp, img_w_sp = f_img.shape[:2]
            x0, y0 = build_grid_for_roi(para, img_h_sp, img_w_sp)

        if need_fft and not use_seed_prop:
            # Start FFT from the last search radius that successfully
            # covered this reference's displacement (if any). This
            # shortcut matters for accumulative / incremental-with-
            # stable-ref segments where the per-frame displacement
            # grows monotonically: once frame N needed search=80,
            # frame N+1 is very likely to need >= 80 too, so
            # restarting from 20 and climbing 20 -> 40 -> 80 again
            # is pure waste.
            learned = fft_search_learned.get(ref_idx)
            if learned is not None and learned != para.size_of_fft_search_region:
                para = replace(para, size_of_fft_search_region=learned)
            # FFT integer search for initial guess.
            # Use raw (unmasked) images for both ref and deformed:
            # benchmarked 3 strategies on bubble 30→150→30 (25 frames,
            # 4 configs: ACC/INC × ref-mask/GT-masks):
            #   both raw:   ACC+GT 0.26, INC+GT 138
            #   both mask:  ACC+GT 0.33, INC+GT 181
            #   mask-f only: ACC+GT 0.26, INC+GT 158
            # Raw-raw wins across all configs.  para.img_ref_mask
            # handles node-level ROI filtering afterward.
            use_pyramid = para.init_fft_search_method >= 2
            if use_pyramid:
                logger.info("Running pyramid NCC search for initial guess...")
                x0, y0, u_grid, v_grid, fft_info = integer_search_pyramid(
                    f_img_raw, g_img_fft, para,
                )
            else:
                logger.info("Running FFT integer search for initial guess...")
                x0, y0, u_grid, v_grid, fft_info = integer_search(
                    f_img_raw, g_img_fft, para,
                )

            # Auto-retry loop: keep enlarging search region until peaks
            # are no longer clipped (or hard cap reached). Each retry
            # grows by at least 1.5x so the loop terminates quickly.
            # Only applies to direct FFT; pyramid has its own failure
            # mode on masked ROI (see docs/plans/...) and is not retried.
            if not use_pyramid and para.fft_auto_expand_search:
                img_h, img_w = f_img_raw.shape[:2]
                max_search_cap = max(32, min(img_h, img_w) // 2)
                current_search = para.size_of_fft_search_region
                for retry_idx in range(6):
                    if not fft_info.get("peak_clipped", False):
                        break
                    max_disp = fft_info["max_abs_disp"]
                    needed = int(np.ceil(max_disp * 2.0)) + 2
                    grown = current_search * 2
                    new_search = max(needed, grown)
                    if new_search >= max_search_cap:
                        new_search = max_search_cap
                        logger.warning(
                            "FFT peaks still clipped (%d nodes, "
                            "max_disp=%.1f); search_region hit hard cap "
                            "%d (image half-size). Accepting result.",
                            fft_info["n_clipped"], max_disp, new_search,
                        )
                        if new_search == current_search:
                            break
                    else:
                        logger.warning(
                            "FFT peaks clipped at search boundary "
                            "(%d nodes, max_disp=%.1f). Retry %d: "
                            "search_region %d -> %d.",
                            fft_info["n_clipped"], max_disp,
                            retry_idx + 1, current_search, new_search,
                        )
                    para_retry = replace(
                        para, size_of_fft_search_region=new_search,
                    )
                    x0, y0, u_grid, v_grid, fft_info = integer_search(
                        f_img_raw, g_img_fft, para_retry,
                    )
                    current_search = new_search
                    if new_search >= max_search_cap:
                        break
                # Remember the final radius for this reference so the
                # next frame on the same ref doesn't re-climb from the
                # user's configured floor.
                fft_search_learned[ref_idx] = current_search
            elif not use_pyramid:
                # auto_expand disabled: still remember whatever the
                # initial call used, so a user who toggles auto_expand
                # mid-run doesn't lose the info on the following frame.
                fft_search_learned[ref_idx] = para.size_of_fft_search_region

            current_U0 = init_disp(
                u_grid, v_grid, fft_info["cc_max"], x0, y0,
            )

        # Build mesh from the FFT grid if needed (first frame only).
        # Store the untrimmed mesh as base_mesh for later re-trimming
        # when the mask changes (e.g., ref-switch in incremental mode).
        if dic_mesh is None:
            dic_mesh = mesh_setup(x0, y0, para)
            base_mesh = DICMesh(
                coordinates_fem=dic_mesh.coordinates_fem.copy(),
                elements_fem=dic_mesh.elements_fem.copy(),
                irregular=dic_mesh.irregular,
                mark_coord_hole_edge=dic_mesh.mark_coord_hole_edge.copy(),
                x0=dic_mesh.x0,
                y0=dic_mesh.y0,
                element_min_size=dic_mesh.element_min_size,
            )

        # Re-trim mesh when mask changes (handles ref-switch in
        # incremental mode where different frames have different masks).
        mask_hash = hash(f_mask.tobytes())
        if mask_hash != prev_mask_hash:
            # Start from untrimmed base mesh
            trim_source = base_mesh if base_mesh is not None else dic_mesh
            # Keep = not-in-a-hole AND not bridging a thin continuous barrier
            # (a crack narrower than an element still cuts the mesh here).
            outside_idx = trimmed_keep_indices(
                trim_source.coordinates_fem,
                trim_source.elements_fem,
                f_mask,
            )
            if len(outside_idx) < trim_source.elements_fem.shape[0]:
                trimmed_elems = trim_source.elements_fem[outside_idx]
            else:
                trimmed_elems = trim_source.elements_fem.copy()
            dic_mesh = DICMesh(
                coordinates_fem=trim_source.coordinates_fem.copy(),
                elements_fem=trimmed_elems,
                irregular=trim_source.irregular,
                mark_coord_hole_edge=trim_source.mark_coord_hole_edge.copy(),
                x0=trim_source.x0,
                y0=trim_source.y0,
                element_min_size=trim_source.element_min_size,
            )
            logger.info(
                "Trimmed mesh to mask (hash=%s): %d -> %d elements",
                mask_hash,
                trim_source.elements_fem.shape[0],
                trimmed_elems.shape[0],
            )
            # Invalidate subpb2 cache since mesh topology changed
            subpb2_cache_obj = None
            subpb2_cache_mhs_key = None
            prev_mask_hash = mask_hash

        if use_seed_prop:
            # Seed propagation init-guess path (replaces FFT init for
            # this mode). Mesh is already built/trimmed above. Compute
            # U0 via BFS propagation from the seed set, warping across
            # ref switches when needed.
            try:
                current_U0 = compute_seed_prop_init_guess(
                    seed_prop_state,
                    dic_mesh,
                    f_img_raw,
                    g_img_icgn,
                    f_mask,
                    Df,
                    para,
                    tol=para.tol,
                    ref_switched=ref_switched_this_frame,
                    frame_idx=frame_idx,
                    ref_idx=ref_idx,
                )
            except SeedPropagationError as exc:
                # A region could not be seeded even after auto-place. Keep the
                # frames already computed instead of aborting the whole run.
                logger.error(
                    "Seed propagation failed at frame %d/%d: %s",
                    frame_idx + 1, n_frames, exc,
                )
                stopped_early = True
                stopped_at_frame = frame_idx + 1
                stop_reason = str(exc)
                break

        if need_fft and not use_seed_prop:
            # When re-running FFT for a subsequent frame (incremental mode
            # with init_guess_mode="fft"), the FFT grid may differ from the
            # existing mesh (e.g. auto-retry with enlarged search region on
            # frame 1 produced a smaller grid, or refinement_policy reset
            # the mesh to base_mesh).  Interpolate the cleaned U0 onto the
            # current mesh coordinates.
            n_mesh = dic_mesh.coordinates_fem.shape[0]
            if len(current_U0) != 2 * n_mesh:
                from scipy.interpolate import RegularGridInterpolator

                ny_fft, nx_fft = len(y0), len(x0)
                # Reverse init_disp assembly: U0[0::2] = u.T.ravel()
                u_2d = current_U0[0::2].reshape(nx_fft, ny_fft).T
                v_2d = current_U0[1::2].reshape(nx_fft, ny_fft).T

                interp_u = RegularGridInterpolator(
                    (y0, x0), u_2d, method="linear",
                    bounds_error=False, fill_value=np.nan,
                )
                interp_v = RegularGridInterpolator(
                    (y0, x0), v_2d, method="linear",
                    bounds_error=False, fill_value=np.nan,
                )

                mesh_xy = dic_mesh.coordinates_fem  # (n, 2): [x, y]
                query = np.column_stack([mesh_xy[:, 1], mesh_xy[:, 0]])
                current_U0 = np.zeros(2 * n_mesh, dtype=np.float64)
                current_U0[0::2] = interp_u(query)
                current_U0[1::2] = interp_v(query)

                logger.info(
                    "Interpolated FFT U0 (%d nodes) to mesh (%d nodes)",
                    nx_fft * ny_fft, n_mesh,
                )

        if need_fft or use_seed_prop:
            # Apply mask: NaN for nodes in masked-out regions.
            # Note: low-coverage boundary nodes are already NaN from
            # integer_search (coverage pre-filter skips NCC for them).
            # init_disp fills those NaN from reliable neighbours, giving
            # IC-GN a smooth initial guess — no re-NaN here.
            # Seed-propagation path also runs the mask NaN pass so
            # out-of-mask nodes in the propagated U0 get NaN'd and
            # filled by the downstream IDW (same semantics as FFT path).
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
            # Subsequent frames (FFT modes only): try sibling reuse,
            # then fall back to FFT. Must skip this block when
            # use_seed_prop so sibling_U doesn't clobber the
            # seed-propagation result we just computed above.
            n_nodes = dic_mesh.coordinates_fem.shape[0]

            # Sibling reuse: find a completed frame with the same reference
            sibling_U = None
            sibling_coords = None
            for prev_i in range(frame_idx - 1, 0, -1):
                if (
                    result_disp[prev_i - 1] is not None
                    and schedule.parent(prev_i) == ref_idx
                ):
                    sibling_U = result_disp[prev_i - 1].U
                    sibling_mesh = result_fe_mesh[prev_i - 1]
                    if sibling_mesh is not None:
                        sibling_coords = sibling_mesh.coordinates_fem
                    logger.info(
                        "Sibling reuse: frame %d -> frame %d (same ref=%d)",
                        prev_i + 1, frame_idx + 1, ref_idx,
                    )
                    break

            if sibling_U is not None:
                # When the sibling's mesh has the same node count as the
                # current (post-trim) mesh, the node ordering matches and
                # we can copy directly.  When refinement + per-frame masks
                # cause the meshes to differ, interpolate via the
                # connected-component aware ``_interpolate_u0`` helper.
                if (
                    sibling_coords is not None
                    and len(sibling_U) != 2 * n_nodes
                ):
                    from ..mesh.generate_mesh import _interpolate_u0
                    current_U0 = _interpolate_u0(
                        sibling_coords,
                        dic_mesh.coordinates_fem,
                        sibling_U,
                        f_mask,
                        (img_h, img_w),
                    )
                    logger.info(
                        "Sibling reuse: interpolated U0 from %d -> %d nodes",
                        sibling_coords.shape[0], n_nodes,
                    )
                else:
                    current_U0 = sibling_U.copy()
            elif frame_idx >= 2:
                # Fall back to previous frame result (may have different ref)
                prev = result_disp[frame_idx - 2]
                if prev is not None:
                    prev_mesh = result_fe_mesh[frame_idx - 2]
                    if (
                        prev_mesh is not None
                        and len(prev.U) != 2 * n_nodes
                    ):
                        from ..mesh.generate_mesh import _interpolate_u0
                        current_U0 = _interpolate_u0(
                            prev_mesh.coordinates_fem,
                            dic_mesh.coordinates_fem,
                            prev.U,
                            f_mask,
                            (img_h, img_w),
                        )
                    else:
                        current_U0 = prev.U.copy()
                else:
                    current_U0 = np.zeros(2 * n_nodes, dtype=np.float64)
            else:
                # frame_idx == 1: no previous results.  If a stale
                # current_U0 from a re-trimmed mesh persists, drop it
                # so the assertion below doesn't trip.
                if current_U0 is None or len(current_U0) != 2 * n_nodes:
                    current_U0 = np.zeros(2 * n_nodes, dtype=np.float64)

        # --- Pre-solve refinement ---
        if refinement_policy is not None and refinement_policy.has_pre_solve:
            logger.info(
                "Applying pre-solve refinement (%d criteria)...",
                len(refinement_policy.pre_solve),
            )
            # Material-point auto-warp for the brush refinement criterion.
            # The user paints the brush mask in frame-0 image coordinates.
            # When the current ref frame is non-zero (incremental tracking),
            # the painted region must follow the same material points, so
            # we warp the mask using the cumulative displacement that has
            # already been computed for that ref frame within this Run.
            pre_solve_criteria = list(refinement_policy.pre_solve)
            if ref_idx > 0:
                pre_solve_criteria = _maybe_warp_brush_for_ref(
                    pre_solve_criteria,
                    ref_idx=ref_idx,
                    schedule=schedule,
                    result_disp=result_disp,
                    result_fe_mesh=result_fe_mesh,
                    img_shape=(img_h, img_w),
                )
            ref_ctx = RefinementContext(
                mesh=dic_mesh, mask=f_mask,
            )
            dic_mesh, current_U0 = refine_mesh(
                dic_mesh, pre_solve_criteria, ref_ctx, current_U0,
                mask=f_mask, img_size=(img_h, img_w),
            )
            n_nodes = dic_mesh.coordinates_fem.shape[0]
            progress(frac, f"Frame {frame_idx}/{n_frames - 1}: refined to {n_nodes} nodes")

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

        progress(frac + _frame_budget * 0.15, f"Frame {frame_idx}/{n_frames - 1}: S3 done ({n_nodes} nodes)")
        logger.info("--- Section 3 Done ---")

        # =================================================================
        # Section 4: Local ICGN (6-DOF, initial subproblem 1)
        # =================================================================
        logger.info("--- Section 4 Start ---")

        tol = para.tol
        # Look up cached 6-DOF context (matches the 2-DOF subpb1 cache
        # pattern). Key includes a stable mesh fingerprint so mid-run
        # refinement / re-trim invalidates the entry automatically.
        from ..solver.local_icgn import local_icgn_precompute
        _mesh_key = (
            int(dic_mesh.coordinates_fem.shape[0]),
            int(dic_mesh.elements_fem.shape[0]),
        )
        _icgn_key = (ref_idx, hash(_mesh_key))
        icgn_ctx = icgn_ctx_cache.get(_icgn_key)
        if icgn_ctx is None:
            icgn_ctx = local_icgn_precompute(
                dic_mesh.coordinates_fem, Df, f_img_raw, para,
            )
            icgn_ctx_cache[_icgn_key] = icgn_ctx
            if len(icgn_ctx_cache) > _PRECOMPUTE_CACHE_SIZE:
                icgn_ctx_cache.popitem(last=False)  # evict LRU
        else:
            icgn_ctx_cache.move_to_end(_icgn_key)
        (
            U_subpb1, F_subpb1, local_time, conv_iter_s4,
            bad_pt_num_s4, mark_hole_strain,
        ) = local_icgn(
            current_U0, dic_mesh.coordinates_fem, Df, f_img_raw, g_img_icgn, para, tol,
            ctx=icgn_ctx,
        )

        assert not np.all(np.isnan(U_subpb1)), "Section 4: USubpb1 is entirely NaN"
        assert len(F_subpb1) == 4 * n_nodes, (
            f"Section 4: FSubpb1 length mismatch: {len(F_subpb1)} != {4 * n_nodes}"
        )

        progress(
            frac + _frame_budget * 0.45,
            f"Frame {frame_idx}/{n_frames - 1}: S4 done (local ICGN, {local_time:.1f}s)",
        )
        logger.info("--- Section 4 Done (%.1fs) ---", local_time)

        if para.use_global_step:
            # =============================================================
            # Section 5: Subproblem 2 initial solve
            # =============================================================
            logger.info("--- Section 5 Start ---")
            t5_start = time.perf_counter()

            # Optional smoothing of initial ICGN results (adaptive sigma)
            if para.disp_smoothness > 1e-6:
                U_subpb1 = _smooth_disp(
                    U_subpb1, dic_mesh.coordinates_fem, para, node_region_map,
                    mesh=dic_mesh,
                )
            if para.strain_smoothness > 1e-6:
                F_subpb1 = _smooth_strain_field(
                    F_subpb1, dic_mesh.coordinates_fem, para, node_region_map,
                    mesh=dic_mesh,
                )

            # Initialize ADMM dual variables (all zero at start)
            mu_val = para.mu
            grad_dual = np.zeros(4 * n_nodes, dtype=np.float64)  # MATLAB: udual
            disp_dual = np.zeros(2 * n_nodes, dtype=np.float64)  # MATLAB: vdual
            alpha = para.alpha

            # Beta: use manual value if provided, else auto-tune per ref frame
            if para.beta is not None:
                beta_val = para.beta
            elif beta_tuned_for_ref != ref_idx:
                beta_val = _auto_tune_beta(
                    dic_mesh, para, mu_val, U_subpb1, F_subpb1,
                    mark_hole_strain=mark_hole_strain,
                )
                beta_tuned_for_ref = ref_idx
            # else: keep beta_val from previous tuning for same ref

            # Pre-compute subpb2 stiffness matrix (reused by S5 and S6 ADMM)
            # Trim elements touching mark_hole_strain nodes from the FEM
            # assembly.  These boundary nodes have unreliable IC-GN results
            # (subset overlaps mask edge) and their error propagates to
            # interior nodes through element connectivity during SubPb2.
            mhs_key = frozenset(mark_hole_strain.tolist())
            if (subpb2_cache_beta != beta_val
                    or subpb2_cache_mhs_key != mhs_key
                    or subpb2_cache_obj is None):
                mesh_for_subpb2 = dic_mesh
                if len(mark_hole_strain) > 0:
                    mhs_set = set(mark_hole_strain.tolist())
                    trimmed_elems = dic_mesh.elements_fem.copy()
                    for e in range(trimmed_elems.shape[0]):
                        for j in range(trimmed_elems.shape[1]):
                            if trimmed_elems[e, j] >= 0 and trimmed_elems[e, j] in mhs_set:
                                trimmed_elems[e, :] = -1
                                break
                    mesh_for_subpb2 = DICMesh(
                        coordinates_fem=dic_mesh.coordinates_fem,
                        elements_fem=trimmed_elems,
                        mark_coord_hole_edge=dic_mesh.mark_coord_hole_edge,
                    )
                    logger.info(
                        "SubPb2: trimmed %d boundary elements (%d remain)",
                        int(np.sum(np.all(trimmed_elems == -1, axis=1)))
                        - int(np.sum(np.all(dic_mesh.elements_fem == -1, axis=1))),
                        int(np.sum(np.any(trimmed_elems >= 0, axis=1))),
                    )
                subpb2_cache_obj = precompute_subpb2(
                    mesh_for_subpb2, para.gauss_pt_order, beta_val, mu_val, alpha,
                )
                subpb2_cache_beta = beta_val
                subpb2_cache_mhs_key = mhs_key
                subpb2_cache_trimmed_mesh = mesh_for_subpb2

            # Solve subpb2
            U_subpb2 = subpb2_solver(
                dic_mesh, para.gauss_pt_order, beta_val, mu_val,
                U_subpb1, F_subpb1, grad_dual, disp_dual,
                alpha, para.winstepsize,
                precomputed=subpb2_cache_obj,
            )
            # Use trimmed mesh for F computation to avoid contamination
            # from garbage U at mark_hole_strain nodes (their elements
            # were removed from SubPb2, so their U is unconstrained).
            F_subpb2 = global_nodal_strain_fem(
                subpb2_cache_trimmed_mesh, para, U_subpb2,
            )

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
                frac + _frame_budget * 0.65,
                f"Frame {frame_idx}/{n_frames - 1}: S5 done ({t5_elapsed:.1f}s)",
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
                if len(subpb1_precompute_cache) > _PRECOMPUTE_CACHE_SIZE:
                    subpb1_precompute_cache.popitem(last=False)  # evict LRU
            else:
                subpb1_precompute_cache.move_to_end(ref_idx)
            subpb1_pre = subpb1_precompute_cache[ref_idx]

            # subpb2 cache already created before S5 solve above
            for admm_iter in range(2, para.admm_max_iter + 1):
                admm_step = admm_iter
                logger.info("ADMM step %d/%d", admm_step, para.admm_max_iter)

                # --- Subproblem 1 (2-DOF IC-GN) ---
                U_subpb1, sub1_time, conv_iter_admm, bad_pt_admm = subpb1_solver(
                    U_subpb2, F_subpb2,
                    disp_dual,
                    grad_dual,
                    dic_mesh.coordinates_fem,
                    Df, f_img_raw, g_img_icgn, mu_val, beta_val, para, tol,
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
                F_subpb2 = global_nodal_strain_fem(
                    subpb2_cache_trimmed_mesh, para, U_subpb2,
                )

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
                frac + _frame_budget * 0.95,
                f"Frame {frame_idx}/{n_frames - 1}: S6 done (ADMM {admm_step} steps)",
            )
            logger.info("--- Section 6 Done (%d steps) ---", admm_step)

            # Use subpb2 result
            U_final = U_subpb2
            F_final = F_subpb2
        else:
            # No global step: local ICGN result is final
            U_final = U_subpb1
            F_final = F_subpb1

        # Update mesh snapshot for this frame
        result_fe_mesh[frame_idx - 1] = DICMesh(
            coordinates_fem=dic_mesh.coordinates_fem.copy(),
            elements_fem=dic_mesh.elements_fem.copy(),
            mark_coord_hole_edge=dic_mesh.mark_coord_hole_edge.copy(),
        )

        # Store frame results (with ref_frame metadata)
        result_disp[frame_idx - 1] = FrameResult(
            U=U_final, ref_frame=ref_idx,
        )
        result_def_grad[frame_idx - 1] = FrameResult(
            U=U_final, F=F_final, ref_frame=ref_idx,
        )

        # Seed-propagation mode: quality-gate the frame's U at the seed
        # nodes (NaN at a seed means its IC-GN result drove unreliable
        # BFS propagation), and cache coords+U for the next frame's
        # ref-switch warp if/when one happens.
        if use_seed_prop and seed_prop_state is not None:
            try:
                capture_for_next_frame(seed_prop_state, dic_mesh, U_final)
            except SeedPropagationError as exc:
                # Seed became a bad point this frame; the next frame's warp
                # would be unreliable. Keep the frames computed so far.
                logger.error(
                    "Seed quality gate failed at frame %d/%d: %s",
                    frame_idx + 1, n_frames, exc,
                )
                stopped_early = True
                stopped_at_frame = frame_idx + 1
                stop_reason = str(exc)
                break

    # Restore the frame-0 grid range, whatever lattice the last pair used:
    # the result reports it (only the solver loop above reads it).
    if para.gridxy_roi_range is not grid_roi_ref0:
        para = replace(para, gridxy_roi_range=grid_roi_ref0)

    # =====================================================================
    # Cumulative displacement transform (tree-based)
    # =====================================================================
    progress(_loop_end, "Computing cumulative displacements...")
    logger.info("--- Cumulative displacement transform ---")

    result_disp = _compute_cumulative_displacements_tree(
        result_disp, result_fe_mesh, n_frames, schedule,
        progress_fn=progress, progress_start=_loop_end,
        # Crack-aware near-gap override: keeps composed U_accum from mixing
        # the two faces of an open crack (smeared wake strain otherwise).
        masks=masks,
        crack_radius=(
            2.0 * para.winstepsize if para.winstepsize > 0 else None
        ),
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
        progress(0.65, "Section 8: Computing strains...")
        logger.info("--- Section 8 Start ---")

        # Strain is total Lagrangian: coordinates + U_accum both live in the
        # frame-0 reference configuration, so the strain mesh keeps frame-0
        # coords.  The crack, however, is known only in each frame's DEFORMED
        # image (masks[i+1]); warp it back to frame-0 coords per frame so
        # edge-trim + crack-aware plane fitting follow the current crack at the
        # reference position (thin, symmetric) rather than freezing on the
        # frame-0 mask for every frame.
        ref_mesh_s8 = result_fe_mesh[0]
        ref_coords_s8 = ref_mesh_s8.coordinates_fem
        ref_elems_s8 = ref_mesh_s8.elements_fem
        fallback_mask_s8 = masks[0].astype(np.float64)

        # Plane-fit strain (method 2) uses these frame-0 coordinates for EVERY
        # frame (total-Lagrangian), so the geometric neighbour search is
        # frame-invariant: build the neighbour cache once and reuse it, skipping
        # the per-frame KDTree build + query_ball_point.  Only the per-frame
        # crack mask / displacements change.  (FEM nodal strain does not use it.)
        platefit_neighbors = None
        if para.method_to_compute_strain == 2 and ref_coords_s8.shape[0] > 0:
            from ..strain.platefit_kernel import build_neighbor_cache
            platefit_neighbors = build_neighbor_cache(
                ref_coords_s8, para.strain_plane_fit_rad,
            )

        for i in range(n_frames - 1):
            if result_disp[i] is None:
                continue

            U_accum = result_disp[i].U_accum
            if U_accum is None:
                U_accum = result_disp[i].U

            deformed_mask_i = (
                masks[i + 1].astype(np.float64) if (i + 1) < len(masks) else None
            )
            if (
                deformed_mask_i is not None
                and len(U_accum) == 2 * len(ref_coords_s8)
            ):
                mask_i, elems_i = crack_mask_from_deformed(
                    ref_coords_s8, ref_elems_s8, U_accum,
                    deformed_mask_i, (img_h, img_w),
                )
                if not np.any(mask_i):
                    mask_i, elems_i = fallback_mask_s8, ref_elems_s8
            else:
                mask_i, elems_i = fallback_mask_s8, ref_elems_s8

            region_map_i = precompute_node_regions(
                ref_coords_s8, mask_i, (img_h, img_w),
            )
            strain_mesh = DICMesh(
                coordinates_fem=ref_coords_s8,
                elements_fem=elems_i,
                mark_coord_hole_edge=ref_mesh_s8.mark_coord_hole_edge,
            )
            para_s8 = replace(para, img_ref_mask=mask_i)

            # Extra displacement smoothing (Section 8 feature)
            U_local = U_accum.copy()
            if para.smoothness > 0:
                for _ in range(3):
                    U_local = _smooth_disp(
                        U_local, ref_coords_s8,
                        para, region_map_i,
                        mesh=strain_mesh,
                    )

            # Compute strain via the strain module
            result_strain[i] = _compute_strain_fn(
                strain_mesh, para_s8, U_local, region_map_i,
                neighbors=platefit_neighbors,
            )

            s8_frac = 0.65 + 0.25 * (i + 1) / (n_frames - 1)
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

    # The canonical mesh exposed via ``PipelineResult.dic_mesh`` is the
    # frame-0 reference mesh — that is the mesh on which
    # ``_compute_cumulative_displacements_tree`` defines ``U_accum`` for
    # every frame.  Pre-fix this was set to the loop variable
    # ``dic_mesh`` (the *last* iteration's mesh), which broke GUI
    # overlay rendering whenever incremental + per-frame refinement
    # produced differently sized meshes per frame.
    canonical_mesh = (
        result_fe_mesh[0]
        if result_fe_mesh and result_fe_mesh[0] is not None
        else DICMesh(
            coordinates_fem=np.empty((0, 2)),
            elements_fem=np.empty((0, 8), dtype=np.int64),
        )
    )
    reseed_events = tuple(
        seed_prop_state.reseed_events
    ) if seed_prop_state is not None else ()
    pipeline_result = PipelineResult(
        dic_para=para,
        dic_mesh=canonical_mesh,
        result_disp=valid_disp,
        result_def_grad=valid_grad,
        result_strain=valid_strain,
        result_fe_mesh_each_frame=valid_mesh,
        frame_schedule=schedule,
        ref_switch_frames=tuple(ref_switch_frames),
        reseed_events=reseed_events,
        stopped_early=stopped_early,
        stopped_at_frame=stopped_at_frame,
        stop_reason=stop_reason,
    )

    progress(1.0, "Pipeline complete.")
    if stopped_early and not user_cancelled:
        # Seed-propagation failure: surface the failure (the original message
        # keeps its actionable guidance) but carry the completed frames so the
        # caller keeps them instead of losing the whole run.
        n_done = len(valid_disp)
        raise PartialResultError(
            f"{stop_reason} Kept partial results: {n_done} of "
            f"{n_frames - 1} frame(s) completed before stopping at frame "
            f"{stopped_at_frame}.",
            partial_result=pipeline_result,
            stopped_at_frame=stopped_at_frame,
        )
    # Normal completion OR user cancel: deliver the (possibly partial) result.
    # For a user cancel, stopped_early / stopped_at_frame / stop_reason are set
    # on the result so the caller can label it as stopped early, but it is
    # returned like a success (no exception, no error dialog).
    return pipeline_result
