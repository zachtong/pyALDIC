"""Tests for AppState per-frame ROI data model."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from staq_dic.gui.app_state import AppState


@pytest.fixture
def state():
    s = AppState()
    return s


class TestPerFrameROI:
    def test_initial_per_frame_rois_empty(self, state):
        assert state.per_frame_rois == {}

    def test_roi_mask_property_returns_frame0(self, state):
        mask = np.ones((100, 100), dtype=bool)
        state.per_frame_rois[0] = mask
        assert state.roi_mask is not None
        assert np.array_equal(state.roi_mask, mask)

    def test_roi_mask_property_none_when_no_frame0(self, state):
        assert state.roi_mask is None

    def test_set_roi_mask_sets_frame0(self, state):
        mask = np.ones((100, 100), dtype=bool)
        state.set_roi_mask(mask)
        assert 0 in state.per_frame_rois
        assert np.array_equal(state.per_frame_rois[0], mask)

    def test_set_roi_mask_none_clears_frame0(self, state):
        state.per_frame_rois[0] = np.ones((10, 10), dtype=bool)
        state.set_roi_mask(None)
        assert 0 not in state.per_frame_rois

    def test_set_roi_mask_emits_signal(self, state):
        received = []
        state.roi_changed.connect(lambda: received.append(True))
        state.set_roi_mask(np.ones((10, 10), dtype=bool))
        assert len(received) == 1

    def test_get_effective_roi_ref_frame_inherits(self, state):
        mask0 = np.ones((50, 50), dtype=bool)
        state.per_frame_rois[0] = mask0
        roi = state.get_effective_roi(3, is_ref_frame=True)
        assert np.array_equal(roi, mask0)

    def test_get_effective_roi_ref_with_own(self, state):
        mask0 = np.ones((50, 50), dtype=bool)
        mask3 = np.zeros((50, 50), dtype=bool)
        mask3[10:40, 10:40] = True
        state.per_frame_rois[0] = mask0
        state.per_frame_rois[3] = mask3
        roi = state.get_effective_roi(3, is_ref_frame=True)
        assert np.array_equal(roi, mask3)

    def test_get_effective_roi_nonref_no_mask(self, state):
        roi = state.get_effective_roi(5, is_ref_frame=False)
        assert roi is None

    def test_get_effective_roi_nonref_with_own(self, state):
        mask5 = np.ones((30, 30), dtype=bool)
        state.per_frame_rois[5] = mask5
        roi = state.get_effective_roi(5, is_ref_frame=False)
        assert np.array_equal(roi, mask5)

    def test_get_effective_roi_frame0_returns_directly(self, state):
        mask0 = np.ones((50, 50), dtype=bool)
        state.per_frame_rois[0] = mask0
        roi = state.get_effective_roi(0, is_ref_frame=True)
        assert np.array_equal(roi, mask0)

    def test_inc_ref_mode_default(self, state):
        assert state.inc_ref_mode == "every_frame"

    def test_inc_ref_interval_default(self, state):
        assert state.inc_ref_interval == 5

    def test_inc_custom_refs_default(self, state):
        assert state.inc_custom_refs == []

    def test_reset_clears_per_frame(self, state):
        state.per_frame_rois[0] = np.ones((10, 10), dtype=bool)
        state.per_frame_rois[3] = np.ones((10, 10), dtype=bool)
        state.inc_ref_mode = "interval"
        state.inc_ref_interval = 10
        state.inc_custom_refs = [1, 3, 5]
        state.reset()
        assert state.per_frame_rois == {}
        assert state.inc_ref_mode == "every_frame"
        assert state.inc_ref_interval == 5
        assert state.inc_custom_refs == []

    def test_set_frame_roi(self, state):
        mask = np.ones((30, 30), dtype=bool)
        state.set_frame_roi(5, mask)
        assert 5 in state.per_frame_rois
        assert np.array_equal(state.per_frame_rois[5], mask)

    def test_set_frame_roi_none_removes(self, state):
        state.per_frame_rois[5] = np.ones((10, 10), dtype=bool)
        state.set_frame_roi(5, None)
        assert 5 not in state.per_frame_rois

    def test_set_frame_roi_emits_signal(self, state):
        received = []
        state.roi_changed.connect(lambda: received.append(True))
        state.set_frame_roi(3, np.ones((10, 10), dtype=bool))
        assert len(received) == 1

    def test_deformed_masks_unchanged(self, state):
        """Verify deformed_masks field still exists and works as before."""
        assert state.deformed_masks is None
        state.deformed_masks = {1: np.ones((10, 10), dtype=bool)}
        assert 1 in state.deformed_masks
