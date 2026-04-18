"""Tests that PipelineController forwards refinement settings to the worker.

We monkeypatch ``PipelineWorker`` so the worker thread never actually starts —
the goal is purely to assert that the controller builds the right
``RefinementPolicy`` from ``AppState`` and passes it through.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from al_dic.gui.app_state import AppState
from al_dic.gui.controllers import pipeline_controller as pc_module
from al_dic.gui.controllers.pipeline_controller import PipelineController
from al_dic.mesh.criteria.mask_boundary import MaskBoundaryCriterion
from al_dic.mesh.criteria.roi_edge import ROIEdgeCriterion
from al_dic.mesh.refinement import RefinementPolicy


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

    # Match PipelineWorker public signal interface enough for the
    # controller's connect() calls to succeed.
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
        self.started = False
        type(self).instances.append(self)

    def start(self):
        self.started = True


class _StubImageController:
    """Returns a fixed dummy image for any frame index."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self._img = np.linspace(0, 1, shape[0] * shape[1], dtype=np.float64).reshape(
            shape
        )

    def read_image(self, _idx: int) -> np.ndarray:
        return self._img.copy()


def _setup_state_for_run(state: AppState, shape: tuple[int, int]) -> None:
    """Populate AppState with the minimum needed for PipelineController.start."""
    state.image_files = [f"/fake/img{i}.tif" for i in range(2)]
    state.subset_size = 40           # display 41
    state.subset_step = 16
    state.search_range = 10
    state.tracking_mode = "accumulative"
    # These refinement-wiring tests exercise the non-seed init path;
    # override the default (now seed_propagation) explicitly so the
    # pipeline doesn't bail out on 'no seeds' before the worker runs.
    state.init_guess_mode = "previous"
    # Build a centered rectangular ROI big enough to satisfy
    # 2*subset_step minimum-size guard (=32 px).
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    state.per_frame_rois = {0: mask}


class TestPipelineRefinementWiring:
    def test_no_refinement_policy_when_both_off(self, qapp, monkeypatch):
        monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
        _CapturingWorker.instances.clear()

        state = AppState.instance()
        _setup_state_for_run(state, shape=(128, 128))
        state.refine_inner = False
        state.refine_outer = False

        ctrl = PipelineController(state, _StubImageController((128, 128)))
        ctrl.start()

        assert len(_CapturingWorker.instances) == 1
        worker = _CapturingWorker.instances[0]
        assert worker.refinement_policy is None
        assert worker.started is True

    def test_inner_only_builds_mask_boundary_criterion(self, qapp, monkeypatch):
        monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
        _CapturingWorker.instances.clear()

        state = AppState.instance()
        _setup_state_for_run(state, shape=(128, 128))
        state.refine_inner = True
        state.refine_outer = False
        state.refinement_level = 2  # step=16 -> min_size=4

        ctrl = PipelineController(state, _StubImageController((128, 128)))
        ctrl.start()

        worker = _CapturingWorker.instances[0]
        policy = worker.refinement_policy
        assert isinstance(policy, RefinementPolicy)
        criteria = policy.pre_solve
        assert len(criteria) == 1
        assert isinstance(criteria[0], MaskBoundaryCriterion)
        # Min size derived from level + step
        assert criteria[0].min_element_size == 4

    def test_outer_only_builds_roi_edge_criterion(self, qapp, monkeypatch):
        monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
        _CapturingWorker.instances.clear()

        state = AppState.instance()
        _setup_state_for_run(state, shape=(128, 128))
        state.refine_inner = False
        state.refine_outer = True
        state.refinement_level = 1  # step=16 -> min_size=8

        ctrl = PipelineController(state, _StubImageController((128, 128)))
        ctrl.start()

        worker = _CapturingWorker.instances[0]
        policy = worker.refinement_policy
        assert isinstance(policy, RefinementPolicy)
        criteria = policy.pre_solve
        assert len(criteria) == 1
        assert isinstance(criteria[0], ROIEdgeCriterion)
        assert criteria[0].min_element_size == 8
        # half_win = subset_size // 2 = 40 // 2 = 20
        assert criteria[0].half_win == 20

    def test_both_inner_and_outer_build_two_criteria(self, qapp, monkeypatch):
        monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
        _CapturingWorker.instances.clear()

        state = AppState.instance()
        _setup_state_for_run(state, shape=(128, 128))
        state.refine_inner = True
        state.refine_outer = True
        state.refinement_level = 1

        ctrl = PipelineController(state, _StubImageController((128, 128)))
        ctrl.start()

        worker = _CapturingWorker.instances[0]
        policy = worker.refinement_policy
        assert isinstance(policy, RefinementPolicy)
        types = {type(c) for c in policy.pre_solve}
        assert MaskBoundaryCriterion in types
        assert ROIEdgeCriterion in types

    def test_min_size_floor_2_for_aggressive_settings(self, qapp, monkeypatch):
        """Level pushed to the maximum should yield min_size = 2."""
        monkeypatch.setattr(pc_module, "PipelineWorker", _CapturingWorker)
        _CapturingWorker.instances.clear()

        state = AppState.instance()
        _setup_state_for_run(state, shape=(256, 256))
        state.subset_size = 80   # display 81 -> max_level = 4 with step=16
        state.subset_step = 16
        state.refine_inner = True
        state.set_refinement_level(4)  # min_size = 16/16 = 1, clamped to 2
        # Build a bigger ROI that satisfies 2*step
        mask = np.zeros((256, 256), dtype=bool)
        mask[64:192, 64:192] = True
        state.per_frame_rois = {0: mask}

        ctrl = PipelineController(state, _StubImageController((256, 256)))
        ctrl.start()

        worker = _CapturingWorker.instances[0]
        criteria = worker.refinement_policy.pre_solve
        assert criteria[0].min_element_size == 2
