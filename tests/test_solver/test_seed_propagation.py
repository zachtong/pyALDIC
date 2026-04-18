"""Tests for seed propagation module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as ndimage_shift

from al_dic.solver.seed_propagation import (
    MissingSeedForRegion,
    Seed,
    SeedFFTResult,
    SeedICGNDiverged,
    SeedNCCBelowThreshold,
    SeedPropagationError,
    SeedSet,
    build_node_adjacency,
    seed_single_point_fft,
)


def _speckle(size=256, seed=42):
    rng = np.random.RandomState(seed)
    img = rng.rand(size, size).astype(np.float64)
    img = gaussian_filter(img, sigma=3.0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(SeedNCCBelowThreshold, SeedPropagationError)
        assert issubclass(MissingSeedForRegion, SeedPropagationError)
        assert issubclass(SeedICGNDiverged, SeedPropagationError)


class TestDataStructures:
    def test_seed_is_frozen(self):
        s = Seed(node_idx=5, region_id=0)
        with pytest.raises((AttributeError, Exception)):
            s.node_idx = 7  # type: ignore[misc]

    def test_seedset_defaults(self):
        ss = SeedSet(seeds=(Seed(0, 0),))
        assert ss.ncc_threshold == 0.70
        assert ss.max_bfs_depth == 0
        assert len(ss.seeds) == 1


class TestBuildNodeAdjacency:
    def test_empty_mesh(self):
        adj = build_node_adjacency(np.empty((0, 8), dtype=np.int64), n_nodes=4)
        assert len(adj) == 4
        assert all(s == set() for s in adj)

    def test_single_q4_element(self):
        # One Q4 element with nodes 0,1,2,3 (plus -1 placeholders)
        elems = np.array([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=np.int64)
        adj = build_node_adjacency(elems, n_nodes=4)
        # Every node connects to the other three
        for i in range(4):
            assert adj[i] == {0, 1, 2, 3} - {i}

    def test_two_adjacent_elements_share_edge(self):
        # Element A: nodes 0,1,2,3 ; Element B: nodes 1,4,5,2 (shares 1-2)
        elems = np.array([
            [0, 1, 2, 3, -1, -1, -1, -1],
            [1, 4, 5, 2, -1, -1, -1, -1],
        ], dtype=np.int64)
        adj = build_node_adjacency(elems, n_nodes=6)
        # Node 0 only in element A
        assert adj[0] == {1, 2, 3}
        # Nodes 1 and 2 are in both elements — neighbours of all
        assert adj[1] == {0, 2, 3, 4, 5}
        assert adj[2] == {0, 1, 3, 4, 5}
        # Nodes 4 and 5 only in element B
        assert adj[4] == {1, 2, 5}
        assert adj[5] == {1, 2, 4}

    def test_negative_placeholders_skipped(self):
        # Malformed element with only 2 valid nodes
        elems = np.array([[0, 1, -1, -1, -1, -1, -1, -1]], dtype=np.int64)
        adj = build_node_adjacency(elems, n_nodes=3)
        assert adj[0] == {1}
        assert adj[1] == {0}
        assert adj[2] == set()

    def test_out_of_range_indices_skipped(self):
        # Index 99 exceeds n_nodes=3; must not crash
        elems = np.array([[0, 1, 99, 2, -1, -1, -1, -1]], dtype=np.int64)
        adj = build_node_adjacency(elems, n_nodes=3)
        # 0,1,2 remain valid; index 99 ignored
        assert adj[0] == {1, 2}
        assert adj[1] == {0, 2}
        assert adj[2] == {0, 1}

    def test_hanging_midside_node(self):
        # Q8 element: corners 0,1,2,3, midside 4 (on edge 0-1)
        # Second element reuses corner 1 and midside 4 as its own corner
        elems = np.array([
            [0, 1, 2, 3, 4, -1, -1, -1],  # Q8 with one midside (hanging)
            [4, 1, 5, 6, -1, -1, -1, -1],  # fine element uses midside as corner
        ], dtype=np.int64)
        adj = build_node_adjacency(elems, n_nodes=7)
        # Midside 4 connects to corners 0,1,2,3 via element 1
        # AND to 1,5,6 via element 2
        assert adj[4] == {0, 1, 2, 3, 5, 6}


class TestSeedSinglePointFFT:
    def test_recovers_integer_translation(self):
        ref = _speckle(size=256, seed=42)
        dx, dy = 8, -5
        deformed = ndimage_shift(ref, [dy, dx], order=3, mode="reflect")

        result = seed_single_point_fft(
            ref, deformed, seed_xy=(128.0, 128.0),
            winsize=40, search_radius=20,
        )

        assert result.valid is True
        assert result.du == dx
        assert result.dv == dy
        assert result.ncc_peak > 0.95
        assert result.peak_clipped is False

    def test_recovers_large_translation(self):
        ref = _speckle(size=512, seed=7)
        dx, dy = 80, 0
        deformed = ndimage_shift(ref, [dy, dx], order=3, mode="reflect")

        result = seed_single_point_fft(
            ref, deformed, seed_xy=(200.0, 256.0),
            winsize=32, search_radius=100,
        )

        assert result.valid is True
        assert result.du == dx
        assert result.dv == dy

    def test_out_of_bounds_returns_invalid(self):
        ref = _speckle(size=128, seed=1)
        deformed = ref.copy()

        result = seed_single_point_fft(
            ref, deformed, seed_xy=(5.0, 5.0),
            winsize=20, search_radius=50,
        )
        assert result.valid is False
        assert np.isnan(result.ncc_peak)

    def test_peak_clipped_when_search_too_small(self):
        ref = _speckle(size=256, seed=3)
        # True displacement (15, 0) but we only search ±5
        deformed = ndimage_shift(ref, [0, 15], order=3, mode="reflect")

        result = seed_single_point_fft(
            ref, deformed, seed_xy=(128.0, 128.0),
            winsize=30, search_radius=5,
        )
        assert result.valid is True
        assert result.peak_clipped is True

    def test_hint_uv_shifts_search_center(self):
        ref = _speckle(size=512, seed=9)
        dx, dy = 60, 40
        deformed = ndimage_shift(ref, [dy, dx], order=3, mode="reflect")

        # With hint (60, 40), small search radius suffices
        result = seed_single_point_fft(
            ref, deformed, seed_xy=(256.0, 256.0),
            winsize=30, search_radius=5, hint_uv=(60.0, 40.0),
        )
        assert result.valid is True
        assert result.du == dx
        assert result.dv == dy
        assert result.peak_clipped is False
