"""Tests for the incomplete-ROI confirmation dialog in incremental mode.

When the user picks incremental mode but only frame 0 has a per-frame ROI
mask, the controller currently inherits frame 0's mask for every later
reference frame.  That is geometrically *not* strictly correct (the
material moves between frames), so we want a confirmation popup before
running so the user can either go back and define more masks or
acknowledge that the inherited masks are good enough for their case.

These tests pin the contracts:

1. Incremental mode + every-frame refs + only frame 0 has an ROI mask
   → dialog must be shown.  If user clicks ``No``, the worker is
   *not* started.
2. Same setup but user clicks ``Yes`` → worker *is* started.
3. Accumulative mode + same incomplete ROIs → dialog must *not* be
   shown (only frame 0 is a ref in accumulative mode).
4. Incremental mode + complete per-frame ROIs → dialog must *not* be
   shown.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from al_dic.gui.app_state import AppState
from al_dic.gui.controllers import pipeline_controller as pc_module
from al_dic.gui.controllers.pipeline_controller import PipelineController


def _safe_disconnect(signal) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            signal.disconnect()
        except (RuntimeError, TypeError):
            pass


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _reset_singleton():
    state = AppState.instance()
    for sig in (
        state.images_changed,
        state.current_frame_changed,
        state.roi_changed,
        state.params_changed,
        state.run_state_changed,
        state.progress_updated,
        state.results_changed,
        state.display_changed,
        state.log_message,
    ):
        _safe_disconnect(sig)
    state.reset()
    yield


class _CapturingWorker:
    """Stand-in for PipelineWorker that records ctor args and never runs."""

    instances: list["_CapturingWorker"] = []

    class _DummySignal:
        def connect(self, *_args, **_kwargs):
            pass

    def __init__(self, para, images, masks, refinement_policy=None):
        self.para = para
        self.images = images
        self.masks = masks
        self.refinement_policy = refinement_policy
        self.progress = self._DummySignal()
        self.log = self._DummySignal()
        self.finished_result = self._DummySignal()
        self.started = False
        type(self).instances.append(self)

    def start(self):
        self.started = True


class _StubImageController:
    def __init__(self, shape: tuple[int, int]) -> None:
        self._img = np.linspace(
            0, 1, shape[0] * shape[1], dtype=np.float64
        ).reshape(shape)

    def read_image(self, _idx: int) -> np.ndarray:
        return self._img.copy()


def _setup_state(
    state: AppState,
    *,
    n_frames: int,
    shape: tuple[int, int],
    tracking_mode: str,
    inc_ref_mode: str = "every_frame",
) -> np.ndarray:
    """Populate AppState minimally for PipelineController.start.

    Returns the frame-0 mask so the test can also seed extra frames.
    """
    state.image_files = [f"/fake/img{i}.tif" for i in range(n_frames)]
    state.subset_size = 40           # display 41
    state.subset_step = 16
    state.search_range = 10
    state.tracking_mode = tracking_mode
    state.inc_ref_mode = inc_ref_mode
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    state.per_frame_rois = {0: mask}
    return mask


class _DialogSpy:
    """Records calls to QMessageBox.question and returns a canned answer."""

    def __init__(self, answer: QMessageBox.StandardButton) -> None:
        self._answer = answer
        self.calls: list[dict] = []

    def __call__(
        self, parent, title, text, buttons=None, default_button=None
    ) -> QMessageBox.StandardButton:
        self.calls.append(
            {
                "parent": parent,
                "title": title,
                "text": text,
                "buttons": buttons,
                "default_button": default_button,
            }
        )
        return self._answer


class TestIncompleteROIDialog:
    def test_incremental_missing_refs_dialog_no_aborts(self, qapp, monkeypatch):
        """Incremental + missing refs + user clicks No → no worker started."""
        monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
        _CapturingWorker.instances.clear()

        spy = _DialogSpy(QMessageBox.StandardButton.No)
        monkeypatch.setattr(QMessageBox, "question", spy)

        state = AppState.instance()
        # 4 frames in every-frame incremental → refs = {0, 1, 2}
        # Only frame 0 has an ROI → missing refs = [1, 2]
        _setup_state(
            state, n_frames=4, shape=(128, 128),
            tracking_mode="incremental",
        )

        ctrl = PipelineController(state, _StubImageController((128, 128)))
        ctrl.start()

        assert len(spy.calls) == 1, "dialog should be shown exactly once"
        # Dialog text should mention the missing frames (1-based: 2, 3)
        dialog_text = spy.calls[0]["text"]
        assert "2" in dialog_text and "3" in dialog_text, (
            f"dialog should list missing frames in 1-based form: {dialog_text}"
        )
        # No worker started → run aborted
        assert len(_CapturingWorker.instances) == 0

    def test_incremental_missing_refs_dialog_yes_starts_worker(
        self, qapp, monkeypatch
    ):
        """Incremental + missing refs + user clicks Yes → worker started."""
        monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
        _CapturingWorker.instances.clear()

        spy = _DialogSpy(QMessageBox.StandardButton.Yes)
        monkeypatch.setattr(QMessageBox, "question", spy)

        state = AppState.instance()
        _setup_state(
            state, n_frames=4, shape=(128, 128),
            tracking_mode="incremental",
        )

        ctrl = PipelineController(state, _StubImageController((128, 128)))
        ctrl.start()

        assert len(spy.calls) == 1
        assert len(_CapturingWorker.instances) == 1
        assert _CapturingWorker.instances[0].started is True

    def test_accumulative_with_missing_refs_no_dialog(self, qapp, monkeypatch):
        """Accumulative mode → frame 0 is the only ref → no dialog."""
        monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
        _CapturingWorker.instances.clear()

        spy = _DialogSpy(QMessageBox.StandardButton.No)
        monkeypatch.setattr(QMessageBox, "question", spy)

        state = AppState.instance()
        _setup_state(
            state, n_frames=4, shape=(128, 128),
            tracking_mode="accumulative",
        )

        ctrl = PipelineController(state, _StubImageController((128, 128)))
        ctrl.start()

        assert len(spy.calls) == 0, "no dialog in accumulative mode"
        assert len(_CapturingWorker.instances) == 1

    def test_incremental_complete_rois_no_dialog(self, qapp, monkeypatch):
        """Every reference frame has its own ROI mask → no dialog."""
        monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
        _CapturingWorker.instances.clear()

        spy = _DialogSpy(QMessageBox.StandardButton.No)
        monkeypatch.setattr(QMessageBox, "question", spy)

        state = AppState.instance()
        # 4 frames every-frame inc → refs = {0, 1, 2}
        mask0 = _setup_state(
            state, n_frames=4, shape=(128, 128),
            tracking_mode="incremental",
        )
        # Provide masks for refs 1 and 2 as well
        state.per_frame_rois = {0: mask0, 1: mask0.copy(), 2: mask0.copy()}

        ctrl = PipelineController(state, _StubImageController((128, 128)))
        ctrl.start()

        assert len(spy.calls) == 0, "no dialog when every ref has an ROI"
        assert len(_CapturingWorker.instances) == 1

    def test_incremental_every_n_partial_refs(self, qapp, monkeypatch):
        """Custom every-N=2 schedule with 5 frames → refs = {0, 2}.

        Frame 2 has no ROI → dialog should fire and list frame 3 (1-based).
        """
        monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
        _CapturingWorker.instances.clear()

        spy = _DialogSpy(QMessageBox.StandardButton.No)
        monkeypatch.setattr(QMessageBox, "question", spy)

        state = AppState.instance()
        _setup_state(
            state, n_frames=5, shape=(128, 128),
            tracking_mode="incremental",
            inc_ref_mode="every_n",
        )
        state.inc_ref_interval = 2

        ctrl = PipelineController(state, _StubImageController((128, 128)))
        ctrl.start()

        assert len(spy.calls) == 1
        # Frame 2 (0-based) → "3" in dialog text
        assert "3" in spy.calls[0]["text"]
        assert len(_CapturingWorker.instances) == 0
