"""Tests for pipeline_controller forwarding refine_brush_mask to the
RefinementPolicy factory.

We don't actually run the worker — instead we monkey-patch
``build_refinement_policy`` to capture its kwargs and abort the run
right after with an unrelated error path so the rest of start() is
exercised but no DIC is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from al_dic.gui.app_state import AppState
from al_dic.gui.controllers import pipeline_controller as pc_module
from al_dic.gui.controllers.image_controller import ImageController
from al_dic.gui.controllers.pipeline_controller import PipelineController


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_state_with_brush(tmp_path: Path) -> AppState:
    """Build an AppState that should reach the policy-build line."""
    # Create two dummy 64x64 BMP files so image loading succeeds.
    import cv2
    img = np.zeros((64, 64), dtype=np.uint8)
    img[20:40, 20:40] = 200
    p1 = tmp_path / "f0.bmp"
    p2 = tmp_path / "f1.bmp"
    cv2.imwrite(str(p1), img)
    cv2.imwrite(str(p2), img)

    state = AppState()
    state.image_files = [str(p1), str(p2)]
    state.current_frame = 0
    # Define a frame-0 ROI sufficient to pass the bbox check
    roi = np.zeros((64, 64), dtype=bool)
    roi[8:56, 8:56] = True
    state.per_frame_rois[0] = roi
    state.subset_size = 16
    state.subset_step = 8
    state.search_range = 8
    state.tracking_mode = "accumulative"
    # Test exercises brush-refinement wiring, not seed path.
    state.init_guess_mode = "previous"
    # Brush mask painted in frame 0 coordinates
    brush = np.zeros((64, 64), dtype=bool)
    brush[20:30, 20:30] = True
    state.refine_brush_mask = brush
    return state


def test_brush_mask_forwarded_to_policy(qapp, tmp_path, monkeypatch) -> None:
    state = _make_state_with_brush(tmp_path)
    image_ctrl = ImageController(state)
    ctrl = PipelineController(state, image_ctrl)

    captured: dict = {}

    def fake_build_policy(**kwargs):
        captured.update(kwargs)
        # Force start() to short-circuit before launching the worker.
        raise RuntimeError("test sentinel")

    monkeypatch.setattr(pc_module, "build_refinement_policy", fake_build_policy)

    # start() catches the RuntimeError via its outer try/except path.
    try:
        ctrl.start()
    except Exception:
        pass

    assert "refinement_mask" in captured
    refmask = captured["refinement_mask"]
    assert refmask is not None
    assert refmask.dtype == np.float64
    assert refmask.shape == (64, 64)
    # Painted region equals 1.0 inside the box
    assert refmask[25, 25] == 1.0
    assert refmask[5, 5] == 0.0


def test_no_brush_mask_passes_none(qapp, tmp_path, monkeypatch) -> None:
    state = _make_state_with_brush(tmp_path)
    state.refine_brush_mask = None
    image_ctrl = ImageController(state)
    ctrl = PipelineController(state, image_ctrl)

    captured: dict = {}

    def fake_build_policy(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("test sentinel")

    monkeypatch.setattr(pc_module, "build_refinement_policy", fake_build_policy)
    try:
        ctrl.start()
    except Exception:
        pass

    assert captured.get("refinement_mask") is None
