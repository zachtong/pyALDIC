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
