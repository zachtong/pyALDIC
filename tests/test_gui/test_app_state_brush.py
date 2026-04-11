"""Tests for AppState.refine_brush_mask field (brush refinement)."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.app_state import AppState


@pytest.fixture
def state() -> AppState:
    return AppState()


def test_refine_brush_mask_default_none(state: AppState) -> None:
    assert state.refine_brush_mask is None


def test_set_refine_brush_mask_emits_roi_changed(state: AppState) -> None:
    received: list[bool] = []
    state.roi_changed.connect(lambda: received.append(True))
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    state.set_refine_brush_mask(mask)
    assert state.refine_brush_mask is mask
    assert received == [True]


def test_clear_refine_brush_mask(state: AppState) -> None:
    state.set_refine_brush_mask(np.ones((4, 4), dtype=bool))
    state.set_refine_brush_mask(None)
    assert state.refine_brush_mask is None


def test_reset_clears_refine_brush_mask(state: AppState) -> None:
    state.set_refine_brush_mask(np.ones((4, 4), dtype=bool))
    state.reset()
    assert state.refine_brush_mask is None


def test_set_image_files_clears_refine_brush_mask(
    state: AppState, tmp_path
) -> None:
    state.set_refine_brush_mask(np.ones((4, 4), dtype=bool))
    state.set_image_files([str(tmp_path / "fake.bmp")])
    assert state.refine_brush_mask is None
