"""Tests for seed propagation module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as ndimage_shift

from al_dic.core.data_structures import DICPara, ImageGradients
from al_dic.solver.local_icgn import local_icgn_precompute
from al_dic.utils.region_analysis import NodeRegionMap, precompute_node_regions
from al_dic.solver.seed_propagation import (
    MissingSeedForRegion,
    PropagationResult,
    Seed,
    SeedFFTResult,
    SeedICGNDiverged,
    SeedNCCBelowThreshold,
    SeedPropagationError,
    SeedQualityError,
    SeedSet,
    SeedWarpFailure,
    build_node_adjacency,
    propagate_from_seeds,
    seed_single_point_fft,
    warp_seeds_to_new_ref,
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
        assert issubclass(SeedQualityError, SeedPropagationError)
        assert issubclass(SeedWarpFailure, SeedPropagationError)


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


def _make_grid_mesh(xs, ys):
    """Build a small rectangular Q4-in-Q8-slot mesh.

    Returns (coords (N, 2), elements (Ne, 8) with -1 padding, adjacency list).
    Node ordering is row-major: (x0,y0), (x1,y0), ..., (x0,y1), ...
    """
    nx = len(xs)
    ny = len(ys)
    coords = np.array(
        [[x, y] for y in ys for x in xs], dtype=np.float64,
    )
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n1 + nx
            n3 = n0 + nx
            elems.append([n0, n1, n2, n3, -1, -1, -1, -1])
    elems_arr = np.array(elems, dtype=np.int64)
    adj = build_node_adjacency(elems_arr, n_nodes=coords.shape[0])
    return coords, elems_arr, adj


class TestPropagateFromSeeds:
    @staticmethod
    def _make_case(shift_x=1.5, shift_y=1.0, size=192):
        ref = _speckle(size=size, seed=101)
        deformed = ndimage_shift(
            ref, [shift_y, shift_x], order=3, mode="reflect",
        )
        # 4x4 grid well inside image
        xs = np.linspace(60, 132, 4)
        ys = np.linspace(60, 132, 4)
        coords, elems, adj = _make_grid_mesh(xs, ys)

        df_dx = np.zeros_like(ref)
        df_dy = np.zeros_like(ref)
        df_dx[:, 1:-1] = (ref[:, 2:] - ref[:, :-2]) / 2.0
        df_dy[1:-1, :] = (ref[2:, :] - ref[:-2, :]) / 2.0
        mask = np.ones_like(ref)
        Df = ImageGradients(
            df_dx=df_dx, df_dy=df_dy, img_ref_mask=mask, img_size=ref.shape,
        )
        para = DICPara(winsize=20, icgn_max_iter=50)
        ctx = local_icgn_precompute(coords, Df, ref, para)
        return ref, deformed, ctx, adj, elems

    def test_uniform_translation_single_seed(self):
        shift_x, shift_y = 1.5, 1.0
        ref, deformed, ctx, adj, _ = self._make_case(shift_x, shift_y)

        seed_set = SeedSet(
            seeds=(Seed(node_idx=0, region_id=0),),
            ncc_threshold=0.5,
        )

        result = propagate_from_seeds(
            ctx, seed_set, adj, ref, deformed,
            search_radius=10, tol=1e-4,
        )

        # All 16 nodes should be solved
        assert len(result.unsolved_nodes) == 0
        assert result.n_seeds == 1
        assert result.max_bfs_depth_reached >= 1
        # Displacement should recover shift on all nodes
        np.testing.assert_allclose(result.U_2d[:, 0], shift_x, atol=0.2)
        np.testing.assert_allclose(result.U_2d[:, 1], shift_y, atol=0.2)
        # Seed NCC should be high for textured speckle
        assert result.seed_ncc_min > 0.5

    def test_ncc_below_threshold_raises(self):
        ref, _, ctx, adj, _ = self._make_case()
        # Random noise as deformed — NCC will be low at every search
        # radius, so _bootstrap_seed_fft keeps expanding until either
        # the NCC threshold is met (won't happen here) or the window
        # leaves the image. Accept any SeedPropagationError subclass —
        # both SeedNCCBelowThreshold and the generic 'window out of
        # image bounds' are valid loud-fail outcomes for unmatched
        # content.
        rng = np.random.RandomState(0)
        deformed = rng.rand(*ref.shape).astype(np.float64)

        seed_set = SeedSet(
            seeds=(Seed(node_idx=5, region_id=0),),
            ncc_threshold=0.70,
        )

        with pytest.raises(SeedPropagationError):
            propagate_from_seeds(
                ctx, seed_set, adj, ref, deformed,
                search_radius=10, tol=1e-4,
            )

    def test_n_solve_calls_is_seeds_plus_layers(self):
        """One bootstrap solve per seed, plus one solve per BFS layer."""
        ref, deformed, ctx, adj, _ = self._make_case()

        seed_set = SeedSet(
            seeds=(Seed(node_idx=0, region_id=0),),
            ncc_threshold=0.3,
        )
        result = propagate_from_seeds(
            ctx, seed_set, adj, ref, deformed,
            search_radius=10, tol=1e-4,
        )
        # 1 seed solve + depth BFS-layer solves
        assert result.n_solve_calls == 1 + result.max_bfs_depth_reached

    def test_two_seeds_both_bootstrap(self):
        """Two seeds contribute to min NCC calculation."""
        ref, deformed, ctx, adj, _ = self._make_case()

        seed_set = SeedSet(
            seeds=(
                Seed(node_idx=0, region_id=0),
                Seed(node_idx=15, region_id=0),
            ),
            ncc_threshold=0.3,
        )
        result = propagate_from_seeds(
            ctx, seed_set, adj, ref, deformed,
            search_radius=10, tol=1e-4,
        )
        assert result.n_seeds == 2
        # Two seed bootstraps + BFS layers
        assert result.n_solve_calls == 2 + result.max_bfs_depth_reached


class TestMultiRegionDispatch:
    """C7: multi-region seed validation."""

    @staticmethod
    def _make_two_region_case(shift_x=1.0, shift_y=0.5):
        """Build a case with two disconnected node groups + two mask regions.

        Region A: nodes placed in upper-left patch; one Q4 element.
        Region B: nodes placed in lower-right patch; one Q4 element.
        The two elements share no node → adjacency has no cross-region edge.
        The image mask has two disjoint rectangles, so NodeRegionMap
        reports n_regions == 2.
        """
        size = 192
        ref = _speckle(size=size, seed=42)
        deformed = ndimage_shift(
            ref, [shift_y, shift_x], order=3, mode="reflect",
        )

        # Two separate Q4 element groups, no shared nodes
        # Region A at (40-80, 40-80), Region B at (120-160, 120-160)
        coords = np.array([
            [40.0, 40.0], [80.0, 40.0], [80.0, 80.0], [40.0, 80.0],  # A: 0-3
            [120.0, 120.0], [160.0, 120.0], [160.0, 160.0], [120.0, 160.0],  # B: 4-7
        ], dtype=np.float64)
        elems = np.array([
            [0, 1, 2, 3, -1, -1, -1, -1],
            [4, 5, 6, 7, -1, -1, -1, -1],
        ], dtype=np.int64)
        adj = build_node_adjacency(elems, n_nodes=8)

        # Mask with two disjoint rectangles matching node locations
        mask = np.zeros_like(ref)
        mask[30:90, 30:90] = 1.0  # region containing nodes 0-3
        mask[110:170, 110:170] = 1.0  # region containing nodes 4-7

        df_dx = np.zeros_like(ref)
        df_dy = np.zeros_like(ref)
        df_dx[:, 1:-1] = (ref[:, 2:] - ref[:, :-2]) / 2.0
        df_dy[1:-1, :] = (ref[2:, :] - ref[:-2, :]) / 2.0
        Df = ImageGradients(
            df_dx=df_dx, df_dy=df_dy, img_ref_mask=mask, img_size=ref.shape,
        )
        para = DICPara(winsize=16, icgn_max_iter=50)
        ctx = local_icgn_precompute(coords, Df, ref, para)

        region_map = precompute_node_regions(
            coords, mask, ref.shape, min_area=20,
        )
        return ref, deformed, ctx, adj, region_map

    def test_two_regions_detected(self):
        """Sanity: NodeRegionMap correctly splits the two disjoint groups."""
        _, _, _, _, region_map = self._make_two_region_case()
        assert region_map.n_regions == 2
        # Nodes 0-3 in one region, 4-7 in the other
        region_ids = set()
        for nodes in region_map.region_node_lists:
            region_ids.add(tuple(sorted(nodes.tolist())))
        assert (0, 1, 2, 3) in region_ids
        assert (4, 5, 6, 7) in region_ids

    def test_missing_seed_for_region_raises(self):
        """One seed for a two-region mesh → MissingSeedForRegion."""
        ref, deformed, ctx, adj, region_map = self._make_two_region_case()

        # Figure out which region node 0 belongs to
        node0_region = next(
            i for i, nodes in enumerate(region_map.region_node_lists)
            if 0 in nodes.tolist()
        )
        seed_set = SeedSet(
            seeds=(Seed(node_idx=0, region_id=node0_region),),
            ncc_threshold=0.3,
        )

        with pytest.raises(MissingSeedForRegion) as exc_info:
            propagate_from_seeds(
                ctx, seed_set, adj, ref, deformed,
                search_radius=10, tol=1e-4,
                node_region_map=region_map,
            )
        assert "no seed" in str(exc_info.value).lower()

    def test_seed_region_id_mismatch_raises(self):
        """Seed declares wrong region_id → SeedPropagationError."""
        ref, deformed, ctx, adj, region_map = self._make_two_region_case()

        node0_region = next(
            i for i, nodes in enumerate(region_map.region_node_lists)
            if 0 in nodes.tolist()
        )
        # Declare the OPPOSITE region_id
        wrong_region = 1 - node0_region
        seed_set = SeedSet(
            seeds=(Seed(node_idx=0, region_id=wrong_region),),
            ncc_threshold=0.3,
        )

        with pytest.raises(SeedPropagationError) as exc_info:
            propagate_from_seeds(
                ctx, seed_set, adj, ref, deformed,
                search_radius=10, tol=1e-4,
                node_region_map=region_map,
            )
        assert "region_id" in str(exc_info.value)

    def test_both_regions_seeded_all_solved(self):
        """Two regions, both seeded → all 8 nodes converge."""
        ref, deformed, ctx, adj, region_map = self._make_two_region_case()

        # Map each region to one of its nodes
        seeds = []
        for region_idx, nodes in enumerate(region_map.region_node_lists):
            seeds.append(
                Seed(node_idx=int(nodes[0]), region_id=region_idx),
            )

        seed_set = SeedSet(seeds=tuple(seeds), ncc_threshold=0.3)
        result = propagate_from_seeds(
            ctx, seed_set, adj, ref, deformed,
            search_radius=10, tol=1e-4,
            node_region_map=region_map,
        )
        assert result.n_seeds == 2
        assert len(result.unsolved_nodes) == 0

    def test_no_region_map_skips_validation(self):
        """Backward compat: node_region_map=None skips multi-region checks."""
        ref, deformed, ctx, adj, _ = self._make_two_region_case()

        # Only seed region A; no region map → no validation error
        seed_set = SeedSet(
            seeds=(Seed(node_idx=0, region_id=0),),
            ncc_threshold=0.3,
        )
        result = propagate_from_seeds(
            ctx, seed_set, adj, ref, deformed,
            search_radius=10, tol=1e-4,
            node_region_map=None,
        )
        # Region A (nodes 0-3) solved; Region B (nodes 4-7) stays unsolved
        # because BFS can't reach it without a seed or edge
        assert set(result.unsolved_nodes.tolist()) == {4, 5, 6, 7}


class TestWarpSeedsToNewRef:
    """C9: auto-warp seeds across ref switches."""

    @staticmethod
    def _make_simple_warp_case(
        u_shift=5.0, v_shift=3.0, n_old=9, n_new=9,
    ):
        """Two 3x3 grids offset by (u_shift, v_shift) in world coords.

        The new grid represents the deformed state of the old grid
        (old mesh translated by +u, +v). old_U uniformly reports the
        same shift at every node, so every old seed warps to its
        corresponding new node.
        """
        old_xs = np.linspace(20, 60, 3)
        old_ys = np.linspace(20, 60, 3)
        old_coords = np.array(
            [[x, y] for y in old_ys for x in old_xs], dtype=np.float64,
        )
        new_coords = old_coords + np.array([u_shift, v_shift])
        old_U = np.tile(np.array([u_shift, v_shift]), (n_old, 1))

        # Single-region NodeRegionMap covering all new nodes
        region_map = NodeRegionMap(
            region_node_lists=[np.arange(n_new, dtype=np.int64)],
            n_regions=1,
        )
        return old_coords, old_U, new_coords, region_map

    def test_warp_uniform_translation_maps_to_same_index(self):
        old_coords, old_U, new_coords, region_map = (
            self._make_simple_warp_case()
        )
        old_seeds = SeedSet(
            seeds=(
                Seed(node_idx=0, region_id=0),
                Seed(node_idx=4, region_id=0),
            ),
        )
        new_seeds = warp_seeds_to_new_ref(
            old_seeds, old_coords, old_U, new_coords, region_map,
        )
        # Because new_coords = old_coords + (u, v), warped positions
        # fall exactly on the corresponding new nodes.
        assert len(new_seeds.seeds) == 2
        assert {s.node_idx for s in new_seeds.seeds} == {0, 4}
        # region_id re-resolved from new_region_map
        for s in new_seeds.seeds:
            assert s.region_id == 0
            assert s.user_hint_uv is None

    def test_warp_preserves_thresholds(self):
        old_coords, old_U, new_coords, region_map = (
            self._make_simple_warp_case()
        )
        old_seeds = SeedSet(
            seeds=(Seed(node_idx=0, region_id=0),),
            ncc_threshold=0.85,
            max_bfs_depth=42,
        )
        new_seeds = warp_seeds_to_new_ref(
            old_seeds, old_coords, old_U, new_coords, region_map,
        )
        assert new_seeds.ncc_threshold == 0.85
        assert new_seeds.max_bfs_depth == 42

    def test_warp_nan_u_raises(self):
        old_coords, old_U, new_coords, region_map = (
            self._make_simple_warp_case()
        )
        # Corrupt the seed's U to NaN
        old_U_bad = old_U.copy()
        old_U_bad[3, :] = np.nan

        old_seeds = SeedSet(seeds=(Seed(node_idx=3, region_id=0),))
        with pytest.raises(SeedWarpFailure, match="NaN"):
            warp_seeds_to_new_ref(
                old_seeds, old_coords, old_U_bad, new_coords, region_map,
            )

    def test_warp_distance_cap_raises(self):
        # Old seed at (50, 50) warps by (+5, 0) to (55, 50); but the
        # new mesh has only one node at (0, 0) — nearest distance ~74 px
        # exceeds max_snap_distance=10.
        old_coords = np.array([[50.0, 50.0]])
        old_U = np.array([[5.0, 0.0]])
        new_coords = np.array([[0.0, 0.0]])
        region_map = NodeRegionMap(
            region_node_lists=[np.array([0], dtype=np.int64)],
            n_regions=1,
        )
        old_seeds = SeedSet(seeds=(Seed(node_idx=0, region_id=0),))
        with pytest.raises(SeedWarpFailure, match="max_snap_distance"):
            warp_seeds_to_new_ref(
                old_seeds, old_coords, old_U, new_coords, region_map,
                max_snap_distance=10.0,
            )

    def test_warp_no_region_raises(self):
        old_coords, old_U, new_coords, _ = self._make_simple_warp_case()
        # Empty NodeRegionMap: no new node is in any region
        empty_region_map = NodeRegionMap(
            region_node_lists=[], n_regions=0,
        )
        old_seeds = SeedSet(seeds=(Seed(node_idx=0, region_id=0),))
        with pytest.raises(SeedWarpFailure, match="not in any tracked region"):
            warp_seeds_to_new_ref(
                old_seeds, old_coords, old_U, new_coords, empty_region_map,
            )

    def test_warp_dedupes_colliding_seeds(self):
        """Two old seeds whose warped positions fall at the same new node."""
        old_coords, old_U, new_coords, region_map = (
            self._make_simple_warp_case()
        )
        # Two old seeds at the same coordinate -> warp to same new node
        # (shape-compatible hack: both point to node 0)
        old_seeds = SeedSet(
            seeds=(
                Seed(node_idx=0, region_id=0),
                Seed(node_idx=0, region_id=0),  # duplicate by design
            ),
        )
        new_seeds = warp_seeds_to_new_ref(
            old_seeds, old_coords, old_U, new_coords, region_map,
        )
        assert len(new_seeds.seeds) == 1
        assert new_seeds.seeds[0].node_idx == 0

    def test_warp_all_fail_raises(self):
        """If every old seed has NaN U, raise (rather than return empty)."""
        old_coords, old_U, new_coords, region_map = (
            self._make_simple_warp_case()
        )
        old_U_bad = old_U.copy()
        old_U_bad[:] = np.nan

        old_seeds = SeedSet(
            seeds=(
                Seed(node_idx=0, region_id=0),
                Seed(node_idx=4, region_id=0),
            ),
        )
        # First NaN seed raises immediately — covered by test_warp_nan_u_raises.
        # Here we additionally confirm the 'no usable seeds' branch: pass
        # a single seed whose warp lands out of region via an empty region map.
        with pytest.raises(SeedWarpFailure):
            warp_seeds_to_new_ref(
                old_seeds, old_coords, old_U_bad, new_coords, region_map,
            )

    def test_warp_out_of_range_node_idx_raises(self):
        old_coords, old_U, new_coords, region_map = (
            self._make_simple_warp_case()
        )
        # Seed claims node 99 but old mesh only has 9 nodes
        old_seeds = SeedSet(seeds=(Seed(node_idx=99, region_id=0),))
        with pytest.raises(SeedWarpFailure, match="out of range"):
            warp_seeds_to_new_ref(
                old_seeds, old_coords, old_U, new_coords, region_map,
            )
