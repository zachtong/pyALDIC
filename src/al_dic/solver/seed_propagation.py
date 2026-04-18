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
