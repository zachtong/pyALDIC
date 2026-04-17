"""Test that PipelineController preserves state.show_deformed across runs.

Regression for the bug where re-running the pipeline silently reset the
"show on deformed frame" toggle, even though the checkbox stayed checked.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

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
        self.refinement_policy = refinement_policy
        self.progress = self._DummySignal()
        self.log = self._DummySignal()
        self.finished_result = self._DummySignal()
        self.fatal_error = self._DummySignal()
        self.started = False
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def isRunning(self) -> bool:
        return False

    def wait(self, _timeout: int = 0) -> bool:
        return True

    def request_stop(self) -> None:
        pass


class _StubImageController:
    def __init__(self, shape: tuple[int, int]) -> None:
        self._img = np.linspace(
            0, 1, shape[0] * shape[1], dtype=np.float64
        ).reshape(shape)

    def read_image(self, _idx: int) -> np.ndarray:
        return self._img.copy()


def _setup_state_for_run(state: AppState, shape: tuple[int, int]) -> None:
    state.image_files = [f"/fake/img{i}.tif" for i in range(2)]
    state.subset_size = 40
    state.subset_step = 16
    state.search_range = 10
    state.tracking_mode = "accumulative"
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    state.per_frame_rois = {0: mask}


def test_show_deformed_preserved_across_reruns(qapp, monkeypatch):
    """If user enabled show_deformed before re-running, it must stay on."""
    monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
    _CapturingWorker.instances.clear()

    state = AppState.instance()
    _setup_state_for_run(state, shape=(128, 128))

    # Simulate first run completing with show_deformed flipped on by user
    state.show_deformed = True

    # Second run starts -- the controller must NOT silently reset the flag.
    ctrl = PipelineController(state, _StubImageController((128, 128)))
    ctrl.start()

    assert state.show_deformed is True, (
        "PipelineController.start() should preserve user's show_deformed "
        "preference across runs"
    )
