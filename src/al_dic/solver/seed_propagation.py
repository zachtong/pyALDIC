"""Seed propagation: F-aware reliability-based spatial init for IC-GN.

Replaces full-grid FFT initialization for large-displacement and
discontinuous-field scenarios. BFS starts from user-placed (or
auto-placed) seed nodes and propagates both displacement U and
deformation gradient F to unsolved neighbours. IC-GN runs on each
frontier node with the F-aware initial guess.

Design rationale lives in project memory
'project_seed_propagation_design.md'. Implementation plan lives at
~/.claude/plans/phase-1-4-curried-wind.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

from ..utils.region_analysis import NodeRegionMap
from .local_icgn import LocalICGNContext, local_icgn_solve_subset


# --- Exceptions ---------------------------------------------------------


class SeedPropagationError(Exception):
    """Base class for seed-propagation failures."""


class SeedNCCBelowThreshold(SeedPropagationError):
    """A seed's single-point NCC peak fell below the configured threshold.

    Raised loudly (not silently degraded) so the user can adjust the
    seed location or the threshold rather than receive misleading
    initial guesses.
    """


class MissingSeedForRegion(SeedPropagationError):
    """At least one connected region has no seed assigned.

    BFS cannot propagate across disconnected components; every region
    produced by ``precompute_node_regions`` needs at least one seed.
    """


class SeedICGNDiverged(SeedPropagationError):
    """A seed node's own IC-GN solve did not converge.

    Unlike frontier divergence (which is retried from another neighbour),
    a seed failure is terminal — the whole region cannot start.
    """


class SeedQualityError(SeedPropagationError):
    """A seed converged but was post-hoc flagged as a statistical outlier.

    Distinct from SeedICGNDiverged: IC-GN reported convergence within
    max_iter, but detect_bad_points (outlier check on conv_iter) flagged
    this node as abnormal. Because the seed's F was propagated to the
    whole BFS tree, the entire frame's result is suspect — raise rather
    than silently IDW-filling the seed and continuing.
    """


class SeedWarpFailure(SeedPropagationError):
    """Unable to warp a seed to a new reference frame's coordinate system.

    Caused by the warped position landing outside any tracked region in
    the new mesh, or by displacement at the seed being NaN (which
    should never happen if SeedQualityError is enforced on the producing
    frame, but guards against the case defensively).
    """


# --- Data structures ----------------------------------------------------


@dataclass(frozen=True)
class Seed:
    """A single propagation starting point.

    Attributes:
        node_idx: Index into coordinates_fem where this seed starts.
        region_id: Connected-component region this seed belongs to, as
            produced by ``utils.region_analysis.precompute_node_regions``.
            Used by the BFS dispatcher to enforce at-least-one-seed per
            region.
        user_hint_uv: Optional (u, v) prior guess for the seed's
            single-point NCC search window. None means center the
            search on the seed's reference-frame coordinate.
    """

    node_idx: int
    region_id: int
    user_hint_uv: tuple[float, float] | None = None


@dataclass(frozen=True)
class SeedSet:
    """Collection of seeds plus propagation tuning knobs.

    Attributes:
        seeds: Tuple of Seed records.
        ncc_threshold: Minimum NCC peak value accepted for a seed's
            bootstrap single-point search. Below this triggers
            ``SeedNCCBelowThreshold`` rather than a silent degrade.
        max_bfs_depth: Hard cap on BFS layers (0 = unlimited). Safety
            guard against pathological meshes.
    """

    seeds: tuple[Seed, ...]
    ncc_threshold: float = 0.70
    max_bfs_depth: int = 0


# --- Adjacency graph ----------------------------------------------------


def build_node_adjacency(
    elements_fem: NDArray[np.int64],
    n_nodes: int,
) -> list[set[int]]:
    """Build undirected node adjacency graph from Q8 connectivity.

    Two nodes are neighbours iff they share at least one element.
    Placeholder entries (-1, used when Q4 elements reuse the Q8 slot)
    are skipped. Hanging midside nodes on refined edges naturally
    become neighbours of both the corners they split and the other
    nodes of the fine element they belong to.

    Args:
        elements_fem: (n_elements, 8) Q8 node indices, 0-based; -1 = unused.
        n_nodes: Total node count. Entries >= n_nodes are skipped
            defensively (stale/corrupted meshes).

    Returns:
        ``adjacency[i]`` is the set of node indices connected to ``i``
        (does not include ``i`` itself).
    """
    adjacency: list[set[int]] = [set() for _ in range(n_nodes)]
    if elements_fem.size == 0:
        return adjacency

    for elem in elements_fem:
        nodes = [int(n) for n in elem if 0 <= n < n_nodes]
        n_valid = len(nodes)
        for i in range(n_valid):
            a = nodes[i]
            for j in range(i + 1, n_valid):
                b = nodes[j]
                if a == b:
                    continue
                adjacency[a].add(b)
                adjacency[b].add(a)
    return adjacency


# --- Single-point NCC bootstrap ----------------------------------------


class SeedFFTResult(NamedTuple):
    """Outcome of ``seed_single_point_fft``.

    Attributes:
        valid: True iff the reference patch and search window lie fully
            inside the image. If False, the other fields carry 0/NaN
            sentinels.
        du: Integer x-displacement estimate.
        dv: Integer y-displacement estimate.
        ncc_peak: Normalized cross-correlation peak value in [-1, 1]
            (TM_CCOEFF_NORMED). 1.0 = perfect match.
        peak_clipped: True iff the peak was found on the search-window
            boundary, signalling that the true displacement may lie
            outside ``search_radius``. Caller should expand and retry.
    """

    valid: bool
    du: int
    dv: int
    ncc_peak: float
    peak_clipped: bool


def seed_single_point_fft(
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    seed_xy: tuple[float, float],
    winsize: int,
    search_radius: int,
    hint_uv: tuple[float, float] | None = None,
) -> SeedFFTResult:
    """Normalized cross-correlation at one seed with a given search radius.

    Far cheaper than the full-grid FFT: single reference patch, single
    search window, cost scales linearly in ``search_radius`` rather than
    quadratically with the full grid. At search=120 on a 512x512 image
    this costs ~3 ms vs ~1.5 s for the full grid.

    Args:
        f_img: Reference image (H, W), float64.
        g_img: Deformed image (H, W), float64.
        seed_xy: (x, y) seed coordinate in the reference image, pixels.
            Sub-pixel values are rounded to integer pixel center.
        winsize: Template window size (pixels). Must be even.
        search_radius: Half-width of the search window in pixels.
        hint_uv: Optional prior (u, v) guess to center the search
            window on ``seed_xy + hint_uv``. Halves the effective search
            radius required when accumulated displacement is known.

    Returns:
        SeedFFTResult with integer (du, dv), peak NCC, and clipping flag.
    """
    sx, sy = int(round(seed_xy[0])), int(round(seed_xy[1]))
    hu, hv = (0, 0) if hint_uv is None else (
        int(round(hint_uv[0])), int(round(hint_uv[1])),
    )
    half_w = winsize // 2

    h, w = f_img.shape

    # Reference template window
    tpl_lo_x = sx - half_w
    tpl_hi_x = sx + half_w + 1
    tpl_lo_y = sy - half_w
    tpl_hi_y = sy + half_w + 1

    # Search window on deformed image, centered at seed + hint
    search_cx = sx + hu
    search_cy = sy + hv
    search_lo_x = search_cx - half_w - search_radius
    search_hi_x = search_cx + half_w + search_radius + 1
    search_lo_y = search_cy - half_w - search_radius
    search_hi_y = search_cy + half_w + search_radius + 1

    if (
        tpl_lo_x < 0 or tpl_hi_x > w or tpl_lo_y < 0 or tpl_hi_y > h
        or search_lo_x < 0 or search_hi_x > w
        or search_lo_y < 0 or search_hi_y > h
    ):
        return SeedFFTResult(
            valid=False, du=0, dv=0, ncc_peak=float("nan"), peak_clipped=False,
        )

    template = f_img[tpl_lo_y:tpl_hi_y, tpl_lo_x:tpl_hi_x].astype(np.float32)
    search_img = g_img[search_lo_y:search_hi_y, search_lo_x:search_hi_x].astype(np.float32)

    ncc_map = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(ncc_map)
    peak_x, peak_y = int(max_loc[0]), int(max_loc[1])

    du = peak_x - search_radius + hu
    dv = peak_y - search_radius + hv

    ncc_h, ncc_w = ncc_map.shape
    peak_clipped = (
        peak_x == 0 or peak_x == ncc_w - 1
        or peak_y == 0 or peak_y == ncc_h - 1
    )

    return SeedFFTResult(
        valid=True,
        du=du,
        dv=dv,
        ncc_peak=float(max_val),
        peak_clipped=peak_clipped,
    )


def _bootstrap_seed_fft(
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    seed_xy: tuple[float, float],
    winsize: int,
    search_radius: int,
    hint_uv: tuple[float, float] | None,
    ncc_threshold: float,
    max_retries: int = 10,
) -> SeedFFTResult:
    """Auto-expanding wrapper around seed_single_point_fft.

    Expands the search radius by 2x each retry while the result looks
    unreliable, up to ``max_retries`` or the image half-size cap.
    'Unreliable' means either:
      - the NCC peak hit the search-window boundary (``peak_clipped``),
        the classic "true displacement exceeds search range" signal; OR
      - the NCC peak value is below ``ncc_threshold``. A low peak inside
        the window usually means the true peak lies outside — argmax
        then selects a noise-level local maximum that happens not to
        touch the window edge, so ``peak_clipped`` alone misses this
        case. Expanding lets us find the real peak before raising.

    If the window leaves the image before we converge, or if NCC
    remains below threshold after the final expansion, raises a
    typed exception.
    """
    h, w = f_img.shape
    max_radius = min(h, w) // 2
    current = search_radius
    result = None

    for _ in range(max_retries):
        result = seed_single_point_fft(
            f_img, g_img, seed_xy, winsize, current, hint_uv,
        )
        if not result.valid:
            break
        if not result.peak_clipped and result.ncc_peak >= ncc_threshold:
            break
        new_radius = min(max_radius, current * 2)
        if new_radius == current:
            break
        current = new_radius

    if result is None or not result.valid:
        raise SeedPropagationError(
            f"Seed FFT search window out of image bounds at {seed_xy}."
        )
    if result.ncc_peak < ncc_threshold:
        raise SeedNCCBelowThreshold(
            f"Seed at {seed_xy}: NCC peak {result.ncc_peak:.3f} "
            f"below threshold {ncc_threshold:.3f}. Try moving the seed "
            f"to a more textured region or lowering ncc_threshold."
        )
    return result


# --- F-aware BFS propagation ------------------------------------------


@dataclass(frozen=True)
class PropagationResult:
    """Full-mesh outcome of propagate_from_seeds.

    Attributes:
        U_2d: (n_nodes, 2) displacements. NaN rows = unsolved.
        F_2d: (n_nodes, 4) deformation gradients, [dudx, dvdx, dudy, dvdy].
        conv_iter: (n_nodes,) IC-GN iteration counts; > max_iter = diverged.
        n_seeds: Number of seeds used.
        max_bfs_depth_reached: Deepest BFS layer processed.
        seed_ncc_min: Lowest bootstrap NCC across all seeds (sanity metric).
        n_solve_calls: Count of local_icgn_solve_subset invocations
            (seed bootstraps + BFS layers). For profiling.
        unsolved_nodes: Indices of nodes that never converged.
    """

    U_2d: NDArray[np.float64]
    F_2d: NDArray[np.float64]
    conv_iter: NDArray[np.int64]
    n_seeds: int
    max_bfs_depth_reached: int
    seed_ncc_min: float
    n_solve_calls: int
    unsolved_nodes: NDArray[np.int64]


def _solve_single_node(
    ctx: LocalICGNContext,
    node_idx: int,
    u0: float,
    v0: float,
    g_img: NDArray[np.float64],
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Run IC-GN on one node. Returns (U (2,), F (4,), conv_iter)."""
    idx = np.array([node_idx], dtype=np.int64)
    u0_2d = np.array([[u0, v0]], dtype=np.float64)
    U_sub, F_sub, conv_sub = local_icgn_solve_subset(
        ctx, idx, u0_2d, g_img, tol,
    )
    return U_sub[0], F_sub[0], int(conv_sub[0])


def warp_seeds_to_new_ref(
    old_seed_set: SeedSet,
    old_coordinates_fem: NDArray[np.float64],
    old_U_2d: NDArray[np.float64],
    new_coordinates_fem: NDArray[np.float64],
    new_region_map: NodeRegionMap,
    max_snap_distance: float | None = None,
) -> SeedSet:
    """Warp each seed from old reference frame to new reference frame.

    When a ref switch happens (accumulative single-ref → new ref at frame
    M, or custom FrameSchedule), the new reference is the deformed state
    of the old reference at some frame M. Each old seed's material point
    moves by U_old[seed.node_idx], so in the new reference's coordinate
    system the same material point sits at old_xy + U_old.

    Steps per seed:
      1. Get x_old = old_coordinates_fem[seed.node_idx].
      2. Get u_old = old_U_2d[seed.node_idx]. NaN → SeedWarpFailure
         (means the seed became a bad_point last frame; that frame
         should already have raised SeedQualityError).
      3. x_new = x_old + u_old.
      4. Nearest new node: argmin Euclidean distance in new_coordinates_fem.
      5. If max_snap_distance is given and best distance exceeds it:
         SeedWarpFailure (material point moved outside the new ROI).
      6. Look up region_id from new_region_map; not found → SeedWarpFailure.
      7. Build new Seed(node_idx=nearest, region_id, user_hint_uv=None).

    Multiple old seeds may collapse to the same new node (two close
    seeds, small mesh). Duplicates are removed — first occurrence wins
    — so the resulting SeedSet has unique node_idx values.

    Args:
        old_seed_set: Seeds in the old ref's coordinate and node system.
        old_coordinates_fem: (n_old, 2) old mesh node coords.
        old_U_2d: (n_old, 2) converged displacements at old mesh nodes
            (should be NaN-free if SeedQualityError is enforced).
        new_coordinates_fem: (n_new, 2) new mesh node coords.
        new_region_map: NodeRegionMap for the new ref's mask.
        max_snap_distance: Optional cap on the distance between warped
            position and nearest new node. Default None (no cap).

    Returns:
        A new SeedSet with warped seeds; thresholds preserved.

    Raises:
        SeedWarpFailure: any seed fails to warp cleanly.
    """
    if old_coordinates_fem.shape[0] != old_U_2d.shape[0]:
        raise ValueError(
            f"old_coordinates_fem ({old_coordinates_fem.shape[0]} nodes) "
            f"and old_U_2d ({old_U_2d.shape[0]} rows) must match."
        )

    # Node -> region lookup in new mesh
    n_new = new_coordinates_fem.shape[0]
    new_node_to_region = np.full(n_new, -1, dtype=np.int64)
    for region_idx, nodes in enumerate(new_region_map.region_node_lists):
        new_node_to_region[nodes] = region_idx

    new_seeds: list[Seed] = []
    seen_node_indices: set[int] = set()

    for seed in old_seed_set.seeds:
        if seed.node_idx >= old_coordinates_fem.shape[0]:
            raise SeedWarpFailure(
                f"Old seed node_idx {seed.node_idx} out of range "
                f"for old mesh with {old_coordinates_fem.shape[0]} nodes."
            )
        x_old = old_coordinates_fem[seed.node_idx]
        u_old = old_U_2d[seed.node_idx]
        if np.any(np.isnan(u_old)):
            raise SeedWarpFailure(
                f"Seed at old node {seed.node_idx} has NaN displacement — "
                f"the producing frame's SeedQualityError gate was bypassed "
                f"or the seed was silently IDW-filled. Cannot warp."
            )

        x_new = x_old + u_old
        dists = np.linalg.norm(new_coordinates_fem - x_new, axis=1)
        nearest = int(np.argmin(dists))
        best_dist = float(dists[nearest])

        if max_snap_distance is not None and best_dist > max_snap_distance:
            raise SeedWarpFailure(
                f"Seed at old node {seed.node_idx} warped to "
                f"{tuple(x_new)}; nearest new node is {best_dist:.2f} px "
                f"away, exceeding max_snap_distance={max_snap_distance:.2f}. "
                f"Material point likely moved outside the new ROI."
            )

        new_region = int(new_node_to_region[nearest])
        if new_region < 0:
            raise SeedWarpFailure(
                f"Seed at old node {seed.node_idx} warped to "
                f"nearest new node {nearest} which is not in any tracked "
                f"region of the new mesh. Adjust the new ROI or seed."
            )

        if nearest in seen_node_indices:
            # Silent dedupe — two old seeds collapsed to same new node
            continue
        seen_node_indices.add(nearest)

        new_seeds.append(
            Seed(
                node_idx=nearest,
                region_id=new_region,
                user_hint_uv=None,
            ),
        )

    if not new_seeds:
        raise SeedWarpFailure(
            "All seeds failed to warp to the new reference; no usable "
            "seeds remain. Re-place seeds manually in the new mesh."
        )

    return SeedSet(
        seeds=tuple(new_seeds),
        ncc_threshold=old_seed_set.ncc_threshold,
        max_bfs_depth=old_seed_set.max_bfs_depth,
    )


def _validate_multi_region_seeds(
    seed_set: SeedSet,
    node_region_map: NodeRegionMap,
    n_nodes: int,
) -> None:
    """Enforce one-seed-per-region and seed.region_id consistency.

    Two checks, both fatal:
      1. Every region in node_region_map has at least one seed.
         BFS cannot cross region boundaries (adjacency only contains
         edges within an element; cracks/holes break elements), so
         a region without a seed will never be solved.
      2. Each seed's declared region_id matches the actual region of
         its node as mapped by node_region_map. A mismatch would
         propagate using the wrong region's neighbours.
    """
    # Build node -> region_id lookup from NodeRegionMap
    node_to_region = np.full(n_nodes, -1, dtype=np.int64)
    for region_idx, node_list in enumerate(node_region_map.region_node_lists):
        node_to_region[node_list] = region_idx

    # Which regions have at least one seed?
    seeded_regions: set[int] = set()
    for seed in seed_set.seeds:
        actual_region = int(node_to_region[seed.node_idx])
        if actual_region < 0:
            raise MissingSeedForRegion(
                f"Seed at node {seed.node_idx} lies outside any tracked "
                f"region (mask hole or below min_area). Move the seed "
                f"to a node inside a valid region."
            )
        if actual_region != seed.region_id:
            raise SeedPropagationError(
                f"Seed at node {seed.node_idx}: declared region_id="
                f"{seed.region_id} but node actually belongs to region "
                f"{actual_region}. Re-resolve seeds against the current "
                f"NodeRegionMap."
            )
        seeded_regions.add(actual_region)

    missing = set(range(node_region_map.n_regions)) - seeded_regions
    if missing:
        raise MissingSeedForRegion(
            f"{len(missing)} region(s) have no seed: indices {sorted(missing)}. "
            f"Each connected region needs at least one seed — BFS cannot "
            f"cross region boundaries."
        )


def propagate_from_seeds(
    ctx: LocalICGNContext,
    seed_set: SeedSet,
    adjacency: list[set[int]],
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    search_radius: int,
    tol: float,
    node_region_map: NodeRegionMap | None = None,
) -> PropagationResult:
    """Eager first-converged BFS from seeds, with F-aware init per layer.

    Per the plan's design decisions:
      - Propagate both U and F (F comes from IC-GN for free).
      - Hanging nodes treated symmetrically via adjacency sharing any element.
      - Eager first-converged: each frontier node uses the first already-
        solved neighbour found as its propagation parent.
      - No weighted multi-neighbour fusion.
      - Layer-sync BFS: whole frontier batched per local_icgn_solve_subset
        call, exposing Numba prange parallelism.

    Args:
        ctx: Precomputed reference-side state for the full mesh.
        seed_set: Seeds + thresholds (ncc_threshold, max_bfs_depth).
        adjacency: Output of build_node_adjacency for the same mesh.
        f_img: Reference image.
        g_img: Deformed image.
        search_radius: Initial single-point NCC search radius for seeds.
            Auto-expands on clipped peaks.
        tol: IC-GN convergence tolerance (same as local_icgn).
        node_region_map: Optional connected-component map. When provided,
            every region must have at least one seed (BFS cannot cross
            region boundaries), and each seed's declared region_id must
            match its node's actual region. See _validate_multi_region_seeds.

    Returns:
        PropagationResult. Unsolved nodes carry NaN in U_2d and whatever
        diverged IC-GN wrote in F_2d; caller's postprocess handles NaN.

    Raises:
        SeedNCCBelowThreshold: A seed's bootstrap NCC < threshold.
        SeedICGNDiverged: A seed's own IC-GN solve failed to converge.
        MissingSeedForRegion: A region has no seed, or a seed's region_id
            disagrees with node_region_map.
        SeedPropagationError: Seed FFT window out of image bounds, or
            seed lies outside any tracked region.
    """
    n = ctx.n_nodes
    max_iter = ctx.max_iter
    winsize = ctx.winsize

    if node_region_map is not None:
        _validate_multi_region_seeds(seed_set, node_region_map, n)

    U_2d = np.full((n, 2), np.nan, dtype=np.float64)
    F_2d = np.zeros((n, 4), dtype=np.float64)
    conv_iter = np.full(n, max_iter + 2, dtype=np.int64)
    solved = np.zeros(n, dtype=bool)

    seed_ncc_min = float("inf")
    n_solve_calls = 0

    # --- Seed bootstrap: single-point NCC + IC-GN per seed --------------
    for seed in seed_set.seeds:
        seed_xy = (
            float(ctx.coordinates_fem[seed.node_idx, 0]),
            float(ctx.coordinates_fem[seed.node_idx, 1]),
        )
        fft_result = _bootstrap_seed_fft(
            f_img, g_img, seed_xy, winsize, search_radius,
            seed.user_hint_uv, seed_set.ncc_threshold,
        )
        seed_ncc_min = min(seed_ncc_min, fft_result.ncc_peak)

        U_seed, F_seed, iter_seed = _solve_single_node(
            ctx, seed.node_idx,
            float(fft_result.du), float(fft_result.dv),
            g_img, tol,
        )
        n_solve_calls += 1
        if iter_seed > max_iter:
            raise SeedICGNDiverged(
                f"Seed at node {seed.node_idx} "
                f"(xy={seed_xy}) IC-GN did not converge "
                f"(iterations={iter_seed}, max={max_iter})."
            )
        U_2d[seed.node_idx] = U_seed
        F_2d[seed.node_idx] = F_seed
        conv_iter[seed.node_idx] = iter_seed
        solved[seed.node_idx] = True

    if seed_ncc_min == float("inf"):
        seed_ncc_min = float("nan")

    # --- Layer-sync BFS -----------------------------------------------
    depth = 0
    max_depth = (
        seed_set.max_bfs_depth if seed_set.max_bfs_depth > 0 else n + 1
    )
    coords = ctx.coordinates_fem

    while True:
        frontier: list[int] = []
        parent: list[int] = []
        for i in range(n):
            if solved[i]:
                continue
            for nb in adjacency[i]:
                if solved[nb]:
                    frontier.append(i)
                    parent.append(nb)
                    break

        if not frontier:
            break
        if depth >= max_depth:
            break

        m = len(frontier)
        U0_subset = np.empty((m, 2), dtype=np.float64)
        for k in range(m):
            i = frontier[k]
            nb = parent[k]
            dx = coords[i, 0] - coords[nb, 0]
            dy = coords[i, 1] - coords[nb, 1]
            U0_subset[k, 0] = (
                U_2d[nb, 0] + F_2d[nb, 0] * dx + F_2d[nb, 2] * dy
            )
            U0_subset[k, 1] = (
                U_2d[nb, 1] + F_2d[nb, 1] * dx + F_2d[nb, 3] * dy
            )

        idx_arr = np.asarray(frontier, dtype=np.int64)
        U_sub, F_sub, conv_sub = local_icgn_solve_subset(
            ctx, idx_arr, U0_subset, g_img, tol,
        )
        n_solve_calls += 1

        solved_before = int(solved.sum())
        for k in range(m):
            i = frontier[k]
            U_2d[i] = U_sub[k]
            F_2d[i] = F_sub[k]
            conv_iter[i] = conv_sub[k]
            if conv_sub[k] <= max_iter:
                solved[i] = True

        # Progress check: if no node converged this layer, stop (rest
        # of the mesh is unreachable or consistently diverges from
        # these parents). Caller's postprocess IDW-fills NaN.
        if int(solved.sum()) == solved_before:
            break

        depth += 1

    unsolved = np.where(~solved)[0].astype(np.int64)
    return PropagationResult(
        U_2d=U_2d,
        F_2d=F_2d,
        conv_iter=conv_iter,
        n_seeds=len(seed_set.seeds),
        max_bfs_depth_reached=depth,
        seed_ncc_min=seed_ncc_min,
        n_solve_calls=n_solve_calls,
        unsolved_nodes=unsolved,
    )
