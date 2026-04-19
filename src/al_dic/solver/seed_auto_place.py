"""Pure solver-layer 3-tier auto-placement for seed propagation.

Extracted from ``SeedController.auto_place_seeds`` so both the GUI
path (initial placement triggered by the user) and the pipeline
ref-switch fallback (when ``warp_seeds_to_new_ref`` raises) can share
one implementation. No Qt imports, no AppState; pure numpy + scipy.

Selection proceeds in three tiers, applied per region:

  Tier 1 - Quality: keep nodes with single-point NCC >= high_quality_ncc
           (default 0.85). If none qualify, relax to the
           ``ncc_threshold`` accept/reject floor and record a warning.
           If still none qualify, skip the region (the caller-side
           pipeline bootstrap will raise a dedicated error).

  Tier 2 - Edge distance: of Tier-1 survivors, keep the top
           ``edge_dist_top_pct`` (default 30%) by EDT distance to the
           nearest mask-boundary pixel.

  Tier 3 - BFS topology: among Tier-2 survivors, pick the node whose
           in-region BFS propagation tree has the lowest max depth.
           NCC is the tiebreaker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, distance_transform_edt

from ..utils.region_analysis import NodeRegionMap
from .seed_propagation import (
    Seed,
    SeedSet,
    build_node_adjacency,
    seed_single_point_fft,
)


@dataclass(frozen=True)
class AutoPlaceConfig:
    """Thresholds and knobs for the 3-tier auto-placement.

    Attributes:
        ncc_threshold: Hard accept/reject floor (matches
            ``AppState.seed_ncc_threshold`` / ``SeedSet.ncc_threshold``;
            default 0.70). If no node in a region meets this floor,
            the region is skipped.
        high_quality_ncc: Strict bar for "this is clearly a good match"
            (default 0.85). Tier 1 prefers these; falls back to
            ``ncc_threshold`` with a warning.
        edge_dist_top_pct: Fraction of Tier-1 survivors kept by EDT
            distance to the mask edge (default 0.30).
        stride: Evaluate every Nth interior node (perf knob, default 3).
    """

    ncc_threshold: float
    high_quality_ncc: float = 0.85
    edge_dist_top_pct: float = 0.30
    stride: int = 3


@dataclass(frozen=True)
class AutoPlaceResult:
    """Outcome of a pure auto-placement run.

    Attributes:
        seed_set: Freshly built ``SeedSet`` (may be empty if no region
            had a viable candidate). ``ncc_threshold`` is copied from
            the provided config; ``max_bfs_depth`` is left at its
            SeedSet default (0, meaning unlimited).
        seed_xy: Parallel tuple of (x, y) canvas coordinates for each
            seed. Useful for GUI rendering. Same length as
            ``seed_set.seeds``.
        seed_ncc: Parallel tuple of NCC peak values for each seed,
            matching ``seed_set.seeds`` order. GUI copies this into
            ``SeedRecord.ncc_peak`` for the seed legend.
        warnings: Human-readable messages that a caller may surface
            through a UI toast or Python ``logger.warning`` call.
            Typical entries describe regions that relaxed to the NCC
            floor or had no viable candidate.
        n_regions_skipped: Regions that produced zero seeds because
            every candidate fell below ``ncc_threshold``.
    """

    seed_set: SeedSet
    seed_xy: tuple[tuple[float, float], ...]
    seed_ncc: tuple[float, ...]
    warnings: tuple[str, ...]
    n_regions_skipped: int


def _bfs_max_depth(
    start: int,
    allowed: set[int],
    adjacency: list[set[int]],
) -> int:
    """BFS from ``start``, staying within ``allowed``; returns max layer.

    Duplicated from ``SeedController`` so this module carries no GUI
    dependency. The two copies are structurally identical and should
    stay in sync.
    """
    if start not in allowed:
        return 0
    visited = {start}
    frontier = {start}
    depth = 0
    while frontier:
        nxt: set[int] = set()
        for node in frontier:
            for nb in adjacency[node]:
                if nb in allowed and nb not in visited:
                    visited.add(nb)
                    nxt.add(nb)
        if not nxt:
            break
        depth += 1
        frontier = nxt
    return depth


def auto_place_seeds_on_mesh(
    coordinates_fem: NDArray[np.float64],
    elements_fem: NDArray[np.int64],
    node_region_map: NodeRegionMap,
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    mask: NDArray[np.bool_ | np.float64],
    winsize: int,
    search_radius: int,
    config: AutoPlaceConfig,
    skip_region_ids: frozenset[int] | None = None,
    adjacency: list[set[int]] | None = None,
) -> AutoPlaceResult:
    """Place one seed per region via the 3-tier selection.

    Args:
        coordinates_fem: (n_nodes, 2) FEM node coordinates (x, y).
        elements_fem: (n_elements, 8) Q8 connectivity, -1 for unused
            slots. Consumed by ``build_node_adjacency`` if
            ``adjacency`` is not provided.
        node_region_map: Region partition for the current mesh + mask.
        f_img: Reference image (float64, H x W).
        g_img: Deformed image (float64, H x W).
        mask: Boolean-like mask (H x W). Truthy = inside ROI.
        winsize: Template window = subset size.
        search_radius: Single-point NCC search half-width.
        config: Thresholds and stride.
        skip_region_ids: Region ids to leave untouched (e.g. those
            already seeded). None = place on every region.
        adjacency: Optional pre-built adjacency graph. If None, built
            internally from ``elements_fem``.

    Returns:
        ``AutoPlaceResult`` with the fresh seed set and any warnings
        the caller may wish to surface.
    """
    n_nodes = coordinates_fem.shape[0]
    if adjacency is None:
        adjacency = build_node_adjacency(elements_fem, n_nodes)

    mask_bool = np.asarray(mask).astype(bool)
    h, w = mask_bool.shape

    # Interior mask: a node whose full subset window fits inside ``mask``.
    half_w = max(1, winsize // 2)
    struct = np.ones((2 * half_w + 1, 2 * half_w + 1), dtype=bool)
    interior_mask = binary_erosion(mask_bool, structure=struct)

    # EDT of the mask: per-pixel distance to the nearest outside-mask
    # pixel. Higher = deeper into the region.
    edge_dist = distance_transform_edt(mask_bool)

    skip_set: frozenset[int] = skip_region_ids or frozenset()
    warnings: list[str] = []
    placed_seeds: list[Seed] = []
    placed_xy: list[tuple[float, float]] = []
    placed_ncc: list[float] = []
    n_skipped = 0

    for region_id, nodes in enumerate(node_region_map.region_node_lists):
        if region_id in skip_set:
            continue

        # Prefer interior nodes; fall back to all nodes when the region
        # is small and no strict-interior candidates exist.
        interior: list[int] = []
        for node_idx in nodes:
            n_idx = int(node_idx)
            nx_i = int(np.clip(round(coordinates_fem[n_idx, 0]), 0, w - 1))
            ny_i = int(np.clip(round(coordinates_fem[n_idx, 1]), 0, h - 1))
            if interior_mask[ny_i, nx_i]:
                interior.append(n_idx)
        candidates = interior if interior else [int(n) for n in nodes]
        if config.stride > 1:
            candidates = candidates[:: config.stride]
        if not candidates:
            n_skipped += 1
            continue

        # Evaluate NCC + edge distance for every candidate.
        cand_data: list[dict] = []
        for n_idx in candidates:
            nx = float(coordinates_fem[n_idx, 0])
            ny = float(coordinates_fem[n_idx, 1])
            r = seed_single_point_fft(
                f_img, g_img, (nx, ny), winsize, search_radius,
            )
            if not r.valid:
                continue
            xi = int(np.clip(round(nx), 0, w - 1))
            yi = int(np.clip(round(ny), 0, h - 1))
            cand_data.append({
                "node": n_idx,
                "xy": (nx, ny),
                "ncc": float(r.ncc_peak),
                "edge_dist": float(edge_dist[yi, xi]),
            })
        if not cand_data:
            n_skipped += 1
            continue

        # --- Tier 1: quality ---
        tier1 = [c for c in cand_data if c["ncc"] >= config.high_quality_ncc]
        if not tier1:
            tier1 = [c for c in cand_data if c["ncc"] >= config.ncc_threshold]
            if not tier1:
                n_skipped += 1
                continue
            best_ncc_here = max(c["ncc"] for c in tier1)
            warnings.append(
                f"Auto-place region {region_id}: no high-quality "
                f"candidate (NCC >= {config.high_quality_ncc:.2f}); "
                f"using best available NCC={best_ncc_here:.3f}. "
                f"Consider using a more textured ROI.",
            )

        # --- Tier 2: edge distance ---
        tier1.sort(key=lambda c: -c["edge_dist"])
        keep_n = max(3, int(round(len(tier1) * config.edge_dist_top_pct)))
        tier2 = tier1[:keep_n]

        # --- Tier 3: BFS max depth within the region ---
        region_node_set = set(int(n) for n in nodes)
        for c in tier2:
            c["max_depth"] = _bfs_max_depth(
                c["node"], region_node_set, adjacency,
            )
        tier2.sort(key=lambda c: (c["max_depth"], -c["ncc"]))
        best = tier2[0]

        placed_seeds.append(
            Seed(
                node_idx=int(best["node"]),
                region_id=region_id,
                user_hint_uv=None,
            ),
        )
        placed_xy.append(best["xy"])
        placed_ncc.append(float(best["ncc"]))

    seed_set = SeedSet(
        seeds=tuple(placed_seeds),
        ncc_threshold=config.ncc_threshold,
    )
    return AutoPlaceResult(
        seed_set=seed_set,
        seed_xy=tuple(placed_xy),
        seed_ncc=tuple(placed_ncc),
        warnings=tuple(warnings),
        n_regions_skipped=n_skipped,
    )
