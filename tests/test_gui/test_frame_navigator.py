"""Tests for FrameNavigator's ref-switch / auto-reseed markers."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.app_state import AppState
from al_dic.gui.widgets.frame_navigator import (
    FrameNavigator,
    _RESEED_COLOR,
    _REF_SWITCH_COLOR,
)
from al_dic.solver.seed_prop_pipeline import ReseedEvent
from tests.test_gui._helpers import make_synthetic_pipeline_result


@pytest.fixture
def nav_with_state():
    """Clean AppState + a FrameNavigator bound to it, plus N loaded frames."""
    state = AppState.instance()
    # Reset
    state.results = None
    state.image_files = ["f0.png", "f1.png", "f2.png", "f3.png", "f4.png"]
    state.current_frame = 0
    state.images_changed.emit()
    nav = FrameNavigator()
    return nav, state


def test_no_results_leaves_markers_empty(nav_with_state):
    nav, _state = nav_with_state
    assert nav._slider._markers == {}


def test_ref_switch_frames_render_blue_markers(nav_with_state):
    nav, state = nav_with_state
    result, _mask = make_synthetic_pipeline_result(n_frames=5)
    # Inject ref_switch_frames into the frozen-ish dataclass via
    # ``dataclasses.replace`` to avoid mutating the original fixture.
    from dataclasses import replace
    result = replace(result, ref_switch_frames=(2, 4), reseed_events=())
    state.set_results(result)

    markers = nav._slider._markers
    assert set(markers.keys()) == {2, 4}
    for m in markers.values():
        assert m.color.name() == _REF_SWITCH_COLOR.name()
        assert "Ref-switch" in m.tooltip


def test_reseed_events_override_with_orange(nav_with_state):
    nav, state = nav_with_state
    result, _mask = make_synthetic_pipeline_result(n_frames=5)
    from dataclasses import replace

    # Frame 3 appears in BOTH ref-switches AND reseed_events; the reseed
    # marker wins (more informative, orange > blue in precedence).
    result = replace(
        result,
        ref_switch_frames=(2, 3),
        reseed_events=(
            ReseedEvent(
                frame_idx=3, ref_idx=2,
                reason="demo", n_new_seeds=2,
            ),
        ),
    )
    state.set_results(result)
    markers = nav._slider._markers
    assert set(markers.keys()) == {2, 3}
    # Frame 2 stays blue (plain ref-switch)
    assert markers[2].color.name() == _REF_SWITCH_COLOR.name()
    # Frame 3 is orange (reseed overrode)
    assert markers[3].color.name() == _RESEED_COLOR.name()
    assert "auto-placed 2 new seed" in markers[3].tooltip
    assert "demo" in markers[3].tooltip


def test_clearing_results_clears_markers(nav_with_state):
    nav, state = nav_with_state
    result, _mask = make_synthetic_pipeline_result(n_frames=5)
    from dataclasses import replace
    result = replace(result, ref_switch_frames=(2,))
    state.set_results(result)
    assert len(nav._slider._markers) == 1

    # Clearing results (equivalent to loading new images) drops markers
    state.set_results(None)
    # set_results(None) raises (it stores None but results_changed still
    # fires). Our handler sees results is None and clears.
    assert nav._slider._markers == {}


def test_new_images_clears_markers(nav_with_state):
    nav, state = nav_with_state
    result, _mask = make_synthetic_pipeline_result(n_frames=5)
    from dataclasses import replace
    result = replace(result, ref_switch_frames=(1, 2))
    state.set_results(result)
    assert len(nav._slider._markers) == 2

    # Simulate loading a fresh image set
    state.image_files = ["a.png", "b.png", "c.png"]
    state.results = None
    state.images_changed.emit()
    assert nav._slider._markers == {}
