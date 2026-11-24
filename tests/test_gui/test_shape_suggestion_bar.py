"""The post-run suggestion bar and its Apply button."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.core.data_structures import ShapeAdvice
from al_dic.gui.app_state import AppState, RunState
from al_dic.gui.widgets.param_panel import ParamPanel
from al_dic.gui.widgets.shape_suggestion_bar import ShapeSuggestionBar
from al_dic.gui.widgets.workflow_type_panel import WorkflowTypePanel


@pytest.fixture
def state():
    AppState._instance = None
    s = AppState.instance()
    yield s
    AppState._instance = None


def _results(advice, step=16):
    return SimpleNamespace(shape_advice=advice, dic_para=SimpleNamespace(winstepsize=step))


def _suggestion(**overrides):
    fields = dict(status="suggest", current_winsize=40, current_order="first",
                  suggested_winsize=60, suggested_order="second", predicted_gain=0.32,
                  n_nodes=900)
    return ShapeAdvice(**{**fields, **overrides})


def test_bar_is_hidden_without_a_suggestion(state):
    bar = ShapeSuggestionBar()
    assert bar.isHidden()

    for advice in (None, ShapeAdvice(status="keep", current_winsize=40, suggested_winsize=40),
                   ShapeAdvice(status="unavailable", reason="too_few_nodes")):
        state.set_results(_results(advice))
        assert bar.isHidden()


def test_bar_shows_the_suggestion_in_gui_units(state):
    bar = ShapeSuggestionBar()
    state.set_results(_results(_suggestion()))

    assert not bar.isHidden()
    text = bar._label.text()
    assert "61 px" in text                 # the GUI shows winsize + 1
    assert "2nd order" in text
    assert "30 %" in text                  # 32 % rounded to the nearest 5 %


def test_apply_sets_both_parameters_and_the_panels_follow(state):
    bar, params, workflow = ShapeSuggestionBar(), ParamPanel(), WorkflowTypePanel()
    logs: list = []
    state.log_message.connect(lambda msg, lvl: logs.append((msg, lvl)))
    state.set_results(_results(_suggestion()))

    bar._apply_btn.click()

    assert state.subset_size == 60
    assert state.shape_function_order == "second"
    assert params._subset_size.value() == 61
    assert workflow._shape_order.currentData() == "second"
    assert bar.isHidden()                  # nothing left to apply
    assert len(logs) == 1 and "61 px" in logs[0][0] and logs[0][1] == "info"


def test_bar_only_applies_to_the_parameters_of_that_run(state):
    bar = ShapeSuggestionBar()
    state.set_results(_results(_suggestion()))
    assert not bar.isHidden()

    state.set_param("subset_size", 30)                 # changed by hand
    assert bar.isHidden()
    state.set_param("subset_size", 40)                 # back to the run's value
    assert not bar.isHidden()
    state.set_param("subset_step", 8)                  # the run used step 16
    assert bar.isHidden()
    state.set_param("subset_step", 16)
    state.set_param("use_admm", True)                  # the suggestion is for Local DIC
    assert bar.isHidden()
    state.set_param("use_admm", False)
    state.set_run_state(RunState.RUNNING)
    assert bar.isHidden()


def test_apply_does_nothing_once_the_suggestion_no_longer_applies(state):
    bar = ShapeSuggestionBar()
    state.set_results(_results(_suggestion()))
    state.set_param("shape_function_order", "second")  # now differs from the run

    bar._on_apply()

    assert state.subset_size == 40
