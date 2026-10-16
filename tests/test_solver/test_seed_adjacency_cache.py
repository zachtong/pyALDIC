"""Seed propagation's node-adjacency cache must follow the mesh content.

Incremental runs give each reference frame its own grid, so meshes come and go
and a new element array can land at the address of a freed one.  The cache was
keyed by id(elements_fem): a recycled address then returned the adjacency of
an older, smaller mesh (IndexError in propagate_from_seeds, seen on CI).
"""
from __future__ import annotations

import numpy as np

from al_dic.solver.seed_prop_pipeline import SeedPropagationState, node_adjacency_for
from al_dic.solver.seed_propagation import build_node_adjacency


def _quads(nx, ny):
    idx = np.arange(nx * ny).reshape(ny, nx)
    els = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            els.append([idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i], -1, -1, -1, -1])
    return np.array(els, dtype=np.int64)


def test_same_array_object_with_new_content_is_rebuilt():
    state = SeedPropagationState(current_seeds=None)
    small = _quads(3, 3)                      # 9 nodes, 4 elements
    elements = small.copy()
    first = node_adjacency_for(state, elements, 9)
    assert len(first) == 9
    # Same ndarray object (same id), now describing a bigger mesh.
    bigger = _quads(5, 2)                     # 10 nodes, 4 elements -> same shape
    elements[:] = bigger
    second = node_adjacency_for(state, elements, 10)
    assert len(second) == 10
    assert second == build_node_adjacency(bigger, 10)


def test_identical_content_is_served_from_the_cache():
    state = SeedPropagationState(current_seeds=None)
    a = _quads(4, 4)
    first = node_adjacency_for(state, a, 16)
    assert node_adjacency_for(state, a.copy(), 16) is first


def test_cache_keeps_only_the_current_mesh():
    state = SeedPropagationState(current_seeds=None)
    for n in range(2, 7):
        node_adjacency_for(state, _quads(n, n), n * n)
    assert len(state.adjacency_cache) == 1
