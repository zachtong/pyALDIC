"""Tests for SeedController (Phase 5.2a).

Exercises the data-layer controller without any GUI widgets:
  - add/remove/clear on AppState.seeds
  - preview mesh cache invalidation
  - re-snap on ROI / winsize / step changes (Q3-B)
  - regions_status / all_regions_seeded for UI consumers
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from al_dic.gui.app_state import AppState, SeedRecord
from al_dic.gui.controllers.seed_controller import SeedController


def _safe_disconnect(signal) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            signal.disconnect()
        except (RuntimeError, TypeError):
            pass


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _reset_singleton():
    state = AppState.instance()
    for sig in (
        state.roi_changed, state.params_changed, state.seeds_changed,
        state.images_changed,
    ):
        _safe_disconnect(sig)
    state.reset()
    yield


def _set_two_region_mask(state: AppState, h: int = 128, w: int = 128) -> None:
    """Install a mask with two disjoint rectangular regions on frame 0."""
    mask = np.zeros((h, w), dtype=bool)
    mask[20:50, 20:50] = True   # Region A (upper-left)
    mask[80:110, 80:110] = True  # Region B (lower-right)
    state.per_frame_rois[0] = mask
    state.subset_size = 16
    state.subset_step = 4


def _set_one_region_mask(state: AppState, h: int = 128, w: int = 128) -> None:
    mask = np.zeros((h, w), dtype=bool)
    mask[20:100, 20:100] = True
    state.per_frame_rois[0] = mask
    state.subset_size = 16
    state.subset_step = 4


class TestSeedControllerMutation:
    def test_add_seed_inside_mask_succeeds(self, qapp):
        state = AppState.instance()
        _set_one_region_mask(state)

        ctrl = SeedController()
        assert ctrl.add_seed_at_xy(60.0, 60.0) is True
        assert len(state.seeds) == 1
        s = state.seeds[0]
        assert s.region_id == 0
        assert s.is_warped is False
        # xy_canvas is the snapped node position, not the click position,
        # so the seed marker renders exactly on the mesh node.
        assert abs(s.xy_canvas[0] - 60.0) <= 4  # within subset_step
        assert abs(s.xy_canvas[1] - 60.0) <= 4
        # node_idx should be the nearest mesh node — not necessarily 0
        assert s.node_idx >= 0

    def test_add_seed_outside_mask_fails(self, qapp):
        state = AppState.instance()
        _set_one_region_mask(state)

        ctrl = SeedController()
        # (5, 5) is outside the 20:100 mask
        assert ctrl.add_seed_at_xy(5.0, 5.0) is False
        assert len(state.seeds) == 0

    def test_add_seed_no_mask_fails(self, qapp):
        state = AppState.instance()
        ctrl = SeedController()
        assert ctrl.add_seed_at_xy(50.0, 50.0) is False
        assert len(state.seeds) == 0

    def test_add_seeds_in_two_regions_different_region_ids(self, qapp):
        state = AppState.instance()
        _set_two_region_mask(state)

        ctrl = SeedController()
        assert ctrl.add_seed_at_xy(35.0, 35.0) is True  # Region A
        assert ctrl.add_seed_at_xy(95.0, 95.0) is True  # Region B
        assert len(state.seeds) == 2
        r0 = state.seeds[0].region_id
        r1 = state.seeds[1].region_id
        assert r0 != r1

    def test_remove_seed_near_finds_closest(self, qapp):
        state = AppState.instance()
        _set_two_region_mask(state)

        ctrl = SeedController()
        ctrl.add_seed_at_xy(35.0, 35.0)
        ctrl.add_seed_at_xy(95.0, 95.0)
        assert ctrl.remove_seed_near(34.0, 34.0, radius=10.0) is True
        assert len(state.seeds) == 1
        remaining_xy = state.seeds[0].xy_canvas
        # After fix B, xy_canvas is the snapped node position, not the
        # original click, so 95.0 may round to the nearest grid node.
        assert abs(remaining_xy[0] - 95.0) <= 4
        assert abs(remaining_xy[1] - 95.0) <= 4

    def test_remove_seed_out_of_range(self, qapp):
        state = AppState.instance()
        _set_two_region_mask(state)

        ctrl = SeedController()
        ctrl.add_seed_at_xy(35.0, 35.0)
        assert ctrl.remove_seed_near(95.0, 95.0, radius=5.0) is False
        assert len(state.seeds) == 1

    def test_clear_seeds(self, qapp):
        state = AppState.instance()
        _set_one_region_mask(state)
        ctrl = SeedController()
        ctrl.add_seed_at_xy(50.0, 50.0)
        ctrl.add_seed_at_xy(60.0, 60.0)
        ctrl.clear_seeds()
        assert state.seeds == []

    def test_seeds_changed_signal_fires(self, qapp):
        state = AppState.instance()
        _set_one_region_mask(state)
        ctrl = SeedController()
        received = []
        state.seeds_changed.connect(lambda: received.append(True))
        ctrl.add_seed_at_xy(50.0, 50.0)
        ctrl.remove_seed_near(50.0, 50.0, radius=20.0)
        assert len(received) == 2


class TestSeedControllerQueries:
    def test_is_xy_in_mask(self, qapp):
        state = AppState.instance()
        _set_one_region_mask(state)
        ctrl = SeedController()
        assert ctrl.is_xy_in_mask(50.0, 50.0) is True
        assert ctrl.is_xy_in_mask(5.0, 5.0) is False
        assert ctrl.is_xy_in_mask(-1.0, 50.0) is False
        assert ctrl.is_xy_in_mask(50.0, 999.0) is False

    def test_nearest_node_preview_returns_coords(self, qapp):
        state = AppState.instance()
        _set_one_region_mask(state)
        ctrl = SeedController()
        result = ctrl.nearest_node_preview(50.0, 50.0)
        assert result is not None
        node_idx, node_x, node_y = result
        # snapped distance should be within step (4) of click
        assert abs(node_x - 50.0) <= 4
        assert abs(node_y - 50.0) <= 4

    def test_regions_status_two_regions(self, qapp):
        state = AppState.instance()
        _set_two_region_mask(state)
        ctrl = SeedController()
        status = ctrl.regions_status()
        assert len(status) == 2
        assert all(has is False for _, has, _ in status)

        ctrl.add_seed_at_xy(35.0, 35.0)
        status = ctrl.regions_status()
        # exactly one region now seeded
        assert sum(has for _, has, _ in status) == 1

    def test_all_regions_seeded(self, qapp):
        state = AppState.instance()
        _set_two_region_mask(state)
        ctrl = SeedController()
        assert ctrl.all_regions_seeded() is False
        ctrl.add_seed_at_xy(35.0, 35.0)
        assert ctrl.all_regions_seeded() is False  # only 1 / 2
        ctrl.add_seed_at_xy(95.0, 95.0)
        assert ctrl.all_regions_seeded() is True


class TestSeedControllerReSnap:
    def test_winsize_change_rebuilds_mesh_and_re_snaps(self, qapp):
        state = AppState.instance()
        _set_one_region_mask(state)
        ctrl = SeedController()
        ctrl.add_seed_at_xy(50.0, 50.0)
        seed_before = state.seeds[0]
        node_before = seed_before.node_idx

        # Change winsize -> mesh rebuild
        state.subset_size = 24
        state.params_changed.emit()

        # xy_canvas must be preserved, but node_idx may have shifted
        seed_after = state.seeds[0]
        assert seed_after.xy_canvas == seed_before.xy_canvas
        assert seed_after.node_idx >= 0
        # ncc_peak is invalidated on re-snap
        assert seed_after.ncc_peak is None

    def test_mask_shrink_drops_orphan_seed(self, qapp):
        state = AppState.instance()
        _set_two_region_mask(state)
        ctrl = SeedController()
        ctrl.add_seed_at_xy(35.0, 35.0)  # Region A
        ctrl.add_seed_at_xy(95.0, 95.0)  # Region B
        assert len(state.seeds) == 2

        # Shrink mask: remove Region A entirely
        mask = state.per_frame_rois[0].copy()
        mask[20:50, 20:50] = False
        state.per_frame_rois[0] = mask
        state.roi_changed.emit()

        # Seed in Region A (xy=(35,35)) is now outside any mask pixel -> dropped
        assert len(state.seeds) == 1
        # xy_canvas stores the snapped node position (Fix B), which may
        # differ from the original click by up to subset_step.
        remaining_xy = state.seeds[0].xy_canvas
        assert abs(remaining_xy[0] - 95.0) <= 4
        assert abs(remaining_xy[1] - 95.0) <= 4

    def test_new_region_remains_unseeded(self, qapp):
        state = AppState.instance()
        _set_one_region_mask(state)  # single 80x80 region
        ctrl = SeedController()
        ctrl.add_seed_at_xy(50.0, 50.0)

        # Grow the mask to include a second disjoint region
        mask = state.per_frame_rois[0].copy()
        mask[105:125, 105:125] = True  # new region, far from first
        state.per_frame_rois[0] = mask
        state.roi_changed.emit()

        status = ctrl.regions_status()
        # Two regions total; original seed should still seed one of them,
        # the other is unseeded (will be yellow in UI)
        seeded_count = sum(has for _, has, _ in status)
        assert seeded_count == 1
        assert len(status) == 2

    def test_auto_place_seeds_picks_one_per_region(self, qapp):
        """auto_place_seeds drops existing seeds and places one per region."""
        state = AppState.instance()
        _set_two_region_mask(state)

        # Speckle-like textured images so NCC is computable
        rng = np.random.RandomState(123)
        ref_img = rng.rand(128, 128).astype(np.float64)
        from scipy.ndimage import gaussian_filter
        ref_img = gaussian_filter(ref_img, sigma=2.0)
        def_img = ref_img.copy()  # zero displacement

        ctrl = SeedController()
        # Existing manual seed in region A should be replaced
        ctrl.add_seed_at_xy(35.0, 35.0)
        assert len(state.seeds) == 1

        placed = ctrl.auto_place_seeds(
            ref_img, def_img,
            winsize=state.subset_size,
            search_radius=10,
        )
        assert placed == 2  # one per region
        # Exactly one seed per region
        region_ids = {s.region_id for s in state.seeds}
        assert region_ids == {0, 1}
        # ncc_peak populated
        for s in state.seeds:
            assert s.ncc_peak is not None
            assert s.ncc_peak > 0.9  # zero-disp speckle → near-perfect NCC

    def test_auto_place_prefers_interior_over_corner(self, qapp):
        """Tier-2 edge-distance filter should push seed inward, not to
        a corner, even when corner NCC is equally high on zero-disp
        speckle.
        """
        state = AppState.instance()
        # One large region so many candidates exist.
        mask = np.zeros((128, 128), dtype=bool)
        mask[20:108, 20:108] = True  # 88x88 region
        state.per_frame_rois[0] = mask
        state.subset_size = 16
        state.subset_step = 4

        rng = np.random.RandomState(7)
        ref = rng.rand(128, 128).astype(np.float64)
        from scipy.ndimage import gaussian_filter
        ref = gaussian_filter(ref, sigma=2.0)
        def_img = ref.copy()

        ctrl = SeedController()
        ctrl.auto_place_seeds(
            ref, def_img,
            winsize=state.subset_size,
            search_radius=8,
            stride=1,  # evaluate every node
        )
        assert len(state.seeds) == 1
        seed = state.seeds[0]
        # Seed should be well inside the region, not at a corner.
        # Mask interior spans [20, 107]; a non-corner seed means
        # both x and y are at least ~20 px away from any edge.
        x, y = seed.xy_canvas
        dist_to_edge = min(x - 20, 108 - x, y - 20, 108 - y)
        assert dist_to_edge >= 18, (
            f"Seed at ({x}, {y}) too close to region boundary "
            f"(dist_to_edge={dist_to_edge})"
        )

    def test_auto_place_warn_emitted_on_fallback(self, qapp):
        """When no candidate meets HIGH_QUALITY_NCC but some beat the
        absolute threshold, a warning is logged and the best-of-the-worst
        is placed.
        """
        state = AppState.instance()
        _set_one_region_mask(state)

        rng = np.random.RandomState(0)
        ref = rng.rand(128, 128).astype(np.float64)
        # Deformed image unrelated to ref → NCC will be low across the
        # board, below 0.85 high-quality bar but possibly above 0.70 for
        # a handful of candidates. Lower the floor to be sure.
        state.seed_ncc_threshold = -1.0  # everything passes tier 1b
        def_img = rng.rand(128, 128).astype(np.float64)

        ctrl = SeedController()
        warns: list[str] = []
        state.log_message.connect(
            lambda msg, lvl: warns.append((msg, lvl)),
        )
        ctrl.auto_place_seeds(
            ref, def_img,
            winsize=state.subset_size,
            search_radius=8,
            stride=2,
        )
        # Either placed something with a warning OR placed nothing at all.
        # Both are legal outcomes on pure-noise mismatch. We only verify
        # that IF a seed was placed, a warning was emitted about quality.
        if state.seeds:
            warn_msgs = [m for m, l in warns if l == "warn"]
            assert any("high-quality" in m for m in warn_msgs), (
                f"No quality warning in {warn_msgs}"
            )

    def test_unrelated_param_change_does_not_rebuild(self, qapp):
        """Changing a non-mesh param (e.g. tracking_mode) shouldn't re-snap."""
        state = AppState.instance()
        _set_one_region_mask(state)
        ctrl = SeedController()
        ctrl.add_seed_at_xy(50.0, 50.0)
        ncc_before = state.seeds[0].ncc_peak
        state.seeds[0] = SeedRecord(
            node_idx=state.seeds[0].node_idx,
            region_id=state.seeds[0].region_id,
            is_warped=False,
            ncc_peak=0.91,
            xy_canvas=state.seeds[0].xy_canvas,
        )

        # Non-mesh param change: mesh cache key unchanged -> no re-snap
        state.tracking_mode = "accumulative"
        state.params_changed.emit()

        # ncc_peak retained (re-snap would have nulled it)
        assert state.seeds[0].ncc_peak == 0.91
