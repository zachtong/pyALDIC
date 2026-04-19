"""Tests for the solver-layer 3-tier auto_place_seeds_on_mesh."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from al_dic.solver.seed_auto_place import (
    AutoPlaceConfig,
    AutoPlaceResult,
    auto_place_seeds_on_mesh,
)
from al_dic.utils.region_analysis import NodeRegionMap, precompute_node_regions


def _speckle(size: int = 192, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.rand(size, size).astype(np.float64)
    img = gaussian_filter(img, sigma=3.0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def _uniform_mesh(
    roi: tuple[int, int, int, int],  # x0, y0, x1, y1
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a tiny uniform Q8 mesh inside the ROI at the given step."""
    x0, y0, x1, y1 = roi
    xs = np.arange(x0, x1 + 1, step)
    ys = np.arange(y0, y1 + 1, step)
    nx, ny = len(xs), len(ys)
    coords = np.array(
        [[xs[i], ys[j]] for j in range(ny) for i in range(nx)],
        dtype=np.float64,
    )
    # Q8 elements: 4 corners + 4 midside slots (-1 for uniform mesh)
    elems: list[list[int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = n00 + 1
            n11 = n00 + nx + 1
            n01 = n00 + nx
            elems.append([n00, n10, n11, n01, -1, -1, -1, -1])
    return coords, np.array(elems, dtype=np.int64)


def _full_mask(img_size: tuple[int, int]) -> np.ndarray:
    return np.ones(img_size, dtype=bool)


def _build_region_map(
    coords: np.ndarray, mask: np.ndarray,
) -> NodeRegionMap:
    return precompute_node_regions(coords, mask.astype(np.float64), mask.shape)


class TestBasicPlacement:
    def test_single_region_one_seed(self):
        img = _speckle(192, seed=11)
        coords, elems = _uniform_mesh((40, 40, 150, 150), step=8)
        mask = _full_mask(img.shape)
        region_map = _build_region_map(coords, mask)
        result = auto_place_seeds_on_mesh(
            coordinates_fem=coords,
            elements_fem=elems,
            node_region_map=region_map,
            f_img=img, g_img=img,
            mask=mask,
            winsize=32,
            search_radius=16,
            config=AutoPlaceConfig(ncc_threshold=0.70),
        )
        assert isinstance(result, AutoPlaceResult)
        assert len(result.seed_set.seeds) == 1
        assert len(result.seed_xy) == 1
        assert len(result.seed_ncc) == 1
        assert result.seed_ncc[0] > 0.95  # zero-disp on same image
        assert result.n_regions_skipped == 0
        assert result.warnings == ()

    def test_seed_is_inside_interior(self):
        """Tier-2 edge-distance should push seed away from ROI corners."""
        img = _speckle(192, seed=11)
        coords, elems = _uniform_mesh((40, 40, 150, 150), step=8)
        mask = _full_mask(img.shape)
        region_map = _build_region_map(coords, mask)
        result = auto_place_seeds_on_mesh(
            coordinates_fem=coords,
            elements_fem=elems,
            node_region_map=region_map,
            f_img=img, g_img=img,
            mask=mask,
            winsize=32, search_radius=16,
            config=AutoPlaceConfig(ncc_threshold=0.70),
        )
        seed = result.seed_set.seeds[0]
        x, y = coords[seed.node_idx]
        # Interior = not at the ROI corners
        assert 60 < x < 140
        assert 60 < y < 140


class TestTierFallback:
    def test_no_high_quality_falls_back_to_floor_with_warning(self):
        """If NCC never reaches high_quality, relax to floor + warn."""
        img = _speckle(192, seed=22)
        coords, elems = _uniform_mesh((40, 40, 150, 150), step=8)
        mask = _full_mask(img.shape)
        region_map = _build_region_map(coords, mask)
        # Set high_quality_ncc > 1.0 — impossible to satisfy even on
        # identical images where NCC ~= 0.99999.
        config = AutoPlaceConfig(
            ncc_threshold=0.30,
            high_quality_ncc=1.01,
        )
        result = auto_place_seeds_on_mesh(
            coordinates_fem=coords,
            elements_fem=elems,
            node_region_map=region_map,
            f_img=img, g_img=img,
            mask=mask,
            winsize=32, search_radius=16,
            config=config,
        )
        # Seed still placed (relaxed to floor)
        assert len(result.seed_set.seeds) == 1
        # Warning captures the fallback
        assert len(result.warnings) == 1
        assert "no high-quality" in result.warnings[0]

    def test_all_below_floor_skips_region(self):
        """If all NCC below threshold, produce no seed + no warning."""
        img = _speckle(192, seed=33)
        # g_img with random content → near-zero NCC everywhere.
        rng = np.random.RandomState(99)
        g_img = rng.rand(192, 192).astype(np.float64)
        coords, elems = _uniform_mesh((40, 40, 150, 150), step=8)
        mask = _full_mask(img.shape)
        region_map = _build_region_map(coords, mask)
        config = AutoPlaceConfig(ncc_threshold=0.95)  # impossibly high
        result = auto_place_seeds_on_mesh(
            coordinates_fem=coords,
            elements_fem=elems,
            node_region_map=region_map,
            f_img=img, g_img=g_img,
            mask=mask,
            winsize=32, search_radius=16,
            config=config,
        )
        assert len(result.seed_set.seeds) == 0
        assert result.n_regions_skipped == region_map.n_regions


class TestSkipRegions:
    def test_skip_region_ids_leaves_them_alone(self):
        """Regions listed in skip_region_ids must produce no seed."""
        # Two-blob mask → 2 regions
        img = _speckle(256, seed=44)
        mask = np.zeros((256, 256), dtype=bool)
        mask[20:100, 20:100] = True
        mask[160:240, 160:240] = True
        # Mesh covers both
        coords, elems = _uniform_mesh((0, 0, 255, 255), step=10)
        region_map = _build_region_map(coords, mask)
        assert region_map.n_regions == 2

        config = AutoPlaceConfig(ncc_threshold=0.70)
        all_regions = auto_place_seeds_on_mesh(
            coordinates_fem=coords, elements_fem=elems,
            node_region_map=region_map,
            f_img=img, g_img=img, mask=mask,
            winsize=24, search_radius=12,
            config=config,
        )
        assert len(all_regions.seed_set.seeds) == 2

        only_r1 = auto_place_seeds_on_mesh(
            coordinates_fem=coords, elements_fem=elems,
            node_region_map=region_map,
            f_img=img, g_img=img, mask=mask,
            winsize=24, search_radius=12,
            config=config,
            skip_region_ids=frozenset({0}),
        )
        assert len(only_r1.seed_set.seeds) == 1
        assert only_r1.seed_set.seeds[0].region_id == 1


class TestAdjacencyReuse:
    def test_accepts_prebuilt_adjacency(self):
        """Passing adjacency should produce the same result as building it."""
        from al_dic.solver.seed_propagation import build_node_adjacency

        img = _speckle(192, seed=55)
        coords, elems = _uniform_mesh((40, 40, 150, 150), step=8)
        mask = _full_mask(img.shape)
        region_map = _build_region_map(coords, mask)
        config = AutoPlaceConfig(ncc_threshold=0.70)

        without = auto_place_seeds_on_mesh(
            coordinates_fem=coords, elements_fem=elems,
            node_region_map=region_map,
            f_img=img, g_img=img, mask=mask,
            winsize=32, search_radius=16,
            config=config,
        )
        adjacency = build_node_adjacency(elems, coords.shape[0])
        with_prebuilt = auto_place_seeds_on_mesh(
            coordinates_fem=coords, elements_fem=elems,
            node_region_map=region_map,
            f_img=img, g_img=img, mask=mask,
            winsize=32, search_radius=16,
            config=config,
            adjacency=adjacency,
        )
        # Same seed(s) picked
        assert len(without.seed_set.seeds) == len(with_prebuilt.seed_set.seeds)
        for a, b in zip(without.seed_set.seeds, with_prebuilt.seed_set.seeds):
            assert a.node_idx == b.node_idx


class TestEmptyInputs:
    def test_no_regions_returns_empty(self):
        """precompute_node_regions on a small mask can yield zero regions."""
        img = _speckle(64, seed=66)
        mask = np.zeros((64, 64), dtype=bool)  # empty mask
        coords, elems = _uniform_mesh((10, 10, 50, 50), step=8)
        region_map = _build_region_map(coords, mask)
        assert region_map.n_regions == 0
        result = auto_place_seeds_on_mesh(
            coordinates_fem=coords, elements_fem=elems,
            node_region_map=region_map,
            f_img=img, g_img=img, mask=mask,
            winsize=24, search_radius=12,
            config=AutoPlaceConfig(ncc_threshold=0.70),
        )
        assert result.seed_set.seeds == ()
        assert result.seed_xy == ()
        assert result.seed_ncc == ()
        assert result.n_regions_skipped == 0


class TestConfigThreshold:
    def test_seedset_inherits_ncc_threshold_from_config(self):
        img = _speckle(192, seed=77)
        coords, elems = _uniform_mesh((40, 40, 150, 150), step=8)
        mask = _full_mask(img.shape)
        region_map = _build_region_map(coords, mask)
        config = AutoPlaceConfig(ncc_threshold=0.42)
        result = auto_place_seeds_on_mesh(
            coordinates_fem=coords, elements_fem=elems,
            node_region_map=region_map,
            f_img=img, g_img=img, mask=mask,
            winsize=32, search_radius=16,
            config=config,
        )
        assert result.seed_set.ncc_threshold == 0.42
