"""Tests for _build_masks helper in pipeline_controller."""

import numpy as np
import pytest

from al_dic.gui.controllers.pipeline_controller import _build_masks


class TestBuildMasks:
    """Unit tests for the per-frame mask construction logic."""

    def test_no_per_frame_rois_except_0(self):
        """Non-ref frames without ROI get all-ones masks."""
        per_frame = {0: np.zeros((100, 100), dtype=bool)}
        per_frame[0][10:90, 10:90] = True
        masks = _build_masks(per_frame, 5, (100, 100), {0})
        # Frame 0 uses its own ROI
        assert masks[0].sum() == per_frame[0].sum()
        assert masks[0].dtype == np.float64
        # Frames 1-4 are non-ref, no own ROI -> all-ones
        for i in range(1, 5):
            assert masks[i].sum() == 100 * 100

    def test_per_frame_roi_used(self):
        """Frame with its own ROI uses that ROI, not frame 0's."""
        per_frame = {
            0: np.ones((50, 50), dtype=bool),
            2: np.zeros((50, 50), dtype=bool),
        }
        per_frame[2][5:10, 5:10] = True
        masks = _build_masks(per_frame, 4, (50, 50), {0, 2})
        # Frame 2 has its own ROI
        assert masks[2].sum() == per_frame[2].sum()
        assert masks[2].dtype == np.float64

    def test_ref_frame_inherits_from_0(self):
        """Ref frame without own ROI inherits from frame 0."""
        mask0 = np.zeros((50, 50), dtype=bool)
        mask0[10:40, 10:40] = True
        per_frame = {0: mask0}
        masks = _build_masks(per_frame, 6, (50, 50), {0, 3})
        # Frame 3 is ref but has no own ROI -> inherit from 0
        assert np.array_equal(masks[3], mask0.astype(np.float64))

    def test_nonref_no_roi_all_ones(self):
        """Non-ref frame without own ROI gets all-ones mask."""
        per_frame = {0: np.ones((20, 20), dtype=bool)}
        masks = _build_masks(per_frame, 3, (20, 20), {0})
        assert masks[1].sum() == 20 * 20
        assert masks[2].sum() == 20 * 20

    def test_output_dtype_is_float64(self):
        """All masks should be float64 regardless of input dtype."""
        per_frame = {0: np.ones((30, 30), dtype=bool)}
        masks = _build_masks(per_frame, 2, (30, 30), {0})
        for m in masks:
            assert m.dtype == np.float64

    def test_output_length_matches_n_frames(self):
        """Output list length should match n_frames."""
        per_frame = {0: np.ones((10, 10), dtype=bool)}
        masks = _build_masks(per_frame, 7, (10, 10), {0})
        assert len(masks) == 7

    def test_empty_ref_frame_set(self):
        """If ref_frame_set is empty, only own ROIs are used."""
        per_frame = {0: np.zeros((20, 20), dtype=bool)}
        per_frame[0][5:15, 5:15] = True
        masks = _build_masks(per_frame, 3, (20, 20), set())
        # Frame 0 still has its own ROI
        assert masks[0].sum() == per_frame[0].sum()
        # Frames 1,2: not ref, no own ROI -> all-ones
        assert masks[1].sum() == 20 * 20
        assert masks[2].sum() == 20 * 20

    def test_multiple_custom_rois(self):
        """Multiple frames with custom ROIs use their own masks."""
        mask0 = np.ones((40, 40), dtype=bool)
        mask2 = np.zeros((40, 40), dtype=bool)
        mask2[0:10, 0:10] = True
        mask4 = np.zeros((40, 40), dtype=bool)
        mask4[20:30, 20:30] = True
        per_frame = {0: mask0, 2: mask2, 4: mask4}
        masks = _build_masks(per_frame, 6, (40, 40), {0, 2, 4})
        assert masks[0].sum() == mask0.sum()
        assert masks[2].sum() == mask2.sum()
        assert masks[4].sum() == mask4.sum()
        # Frame 1, 3, 5: non-ref (not in ref_frame_set), no own ROI
        assert masks[1].sum() == 40 * 40
        assert masks[3].sum() == 40 * 40
        assert masks[5].sum() == 40 * 40
