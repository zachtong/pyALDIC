"""GUI path of the incremental grid fix (plan step 2.2).

The core tests call ``run_aldic`` directly. Here the real
``PipelineController.start()`` builds the run from ``AppState`` -- the
frame-0 range from the bbox of the frame-1 ROI, the uint8 per-frame masks,
the streaming frame provider over PNG files -- and launches a worker. The
worker class is swapped for one that only records its arguments; the real
``PipelineWorker`` then runs synchronously on them and its result goes
through ``_on_finished``. Accuracy thresholds are those of the core drift
test (``tests/test_integration/_grid_drift.py``).
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from al_dic.gui.app_state import AppState, RunState
from al_dic.gui.controllers import pipeline_controller as pc_module
from al_dic.gui.controllers.pipeline_controller import (
    PipelineController,
    PipelineWorker,
)

from tests.test_integration._grid_drift import (
    N_DEF,
    SEARCH,
    STEP,
    WIN,
    assert_frame_accuracy,
    drift_data,
    range_of,
)


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
        state.run_warning,
    ):
        _safe_disconnect(sig)
    state.reset()
    yield
    state.reset()


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
        self.fatal_error = self._DummySignal()
        type(self).instances.append(self)

    def start(self):
        pass


def _no_dialog(*_args, **_kwargs):
    raise AssertionError("every reference frame has an ROI: no dialog expected")


def _write_frames(folder, imgs) -> list[str]:
    paths = []
    for k, img in enumerate(imgs):
        path = folder / f"drift_{k:02d}.png"
        cv2.imwrite(str(path), np.clip(np.round(img), 0, 255).astype(np.uint8))
        paths.append(str(path))
    return paths


def _answer_yes(*_args, **_kwargs):
    return QMessageBox.StandardButton.Yes


def _capture_run(
    state: AppState, monkeypatch, question=_no_dialog,
) -> tuple[PipelineController, _CapturingWorker]:
    monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
    monkeypatch.setattr(QMessageBox, "question", question)
    _CapturingWorker.instances.clear()
    ctrl = PipelineController(state, None)
    ctrl.start()
    assert len(_CapturingWorker.instances) == 1, "start() did not launch"
    return ctrl, _CapturingWorker.instances[0]


def _run_captured(captured: _CapturingWorker):
    """Run the real worker synchronously on the captured arguments."""
    worker = PipelineWorker(
        captured.para, captured.images, captured.masks,
        refinement_policy=captured.refinement_policy,
    )
    delivered: list = []
    worker.finished_result.connect(delivered.append)
    worker.run()
    assert len(delivered) == 1 and delivered[0] is not None, "run failed"
    return delivered[0]


def _drift_state(state: AppState, folder, n_frames: int, rois) -> None:
    imgs, _ = drift_data()
    state.image_files = _write_frames(folder, imgs[:n_frames])
    state.per_frame_rois = rois
    state.subset_size = WIN
    state.subset_step = STEP
    state.search_range = SEARCH
    state.tracking_mode = "incremental"
    state.inc_ref_mode = "every_frame"
    state.init_guess_mode = "fft_every"


def test_gui_path_drift(qapp, tmp_path, monkeypatch):
    _, masks = drift_data()
    state = AppState.instance()
    _drift_state(
        state, tmp_path, N_DEF + 1, {k: m > 0.5 for k, m in enumerate(masks)},
    )
    warnings: list = []
    state.run_warning.connect(lambda *args: warnings.append(args))

    ctrl, captured = _capture_run(state, monkeypatch)
    result = _run_captured(captured)
    ctrl._on_finished(result)

    assert state.run_state == RunState.DONE
    assert state.results is result
    assert result.dic_para.gridxy_roi_range == range_of(masks[0])
    for k in range(1, N_DEF + 1):
        assert_frame_accuracy(result, k)
    assert warnings == [], "a covered moving specimen raised a warning"


def test_gui_path_inherited_roi_warns(qapp, tmp_path, monkeypatch):
    """Scenario E: only frame 1 has a Region of Interest.

    The user confirms the incomplete-ROI question, so reference frames reuse
    frame 1's region unwarped (layer 1 cannot help them). The run finishes,
    and the material it lost is reported: a translated warn line in the
    console and one request for the warning dialog.
    """
    _, masks = drift_data()
    state = AppState.instance()
    _drift_state(state, tmp_path, 3, {0: masks[0] > 0.5})
    logs: list = []
    state.log_message.connect(lambda msg, lvl: logs.append((msg, lvl)))
    warnings: list = []
    state.run_warning.connect(lambda *args: warnings.append(args))

    ctrl, captured = _capture_run(state, monkeypatch, question=_answer_yes)
    result = _run_captured(captured)
    ctrl._on_finished(result)

    assert state.run_state == RunState.DONE
    assert state.results is result
    warn_lines = [msg for msg, lvl in logs if lvl == "warn"]
    assert warn_lines, logs
    assert any("Region of Interest" in msg for msg in warn_lines), warn_lines
    assert len(warnings) == 1
    title, text, informative, details = warnings[0]
    assert "Region of Interest" in informative  # the advice
