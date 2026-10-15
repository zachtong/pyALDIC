"""The GUI reports poor tracking quality after a run (plan 4.1, D18/D19).

``PipelineController._on_finished`` turns the run's tracking diagnostics into
translated console lines and, for WARNING findings of a complete run, one
``AppState.run_warning`` signal. ``MainWindow`` shows that as a modeless
message box (``show()``: it blocks no window and runs no nested event loop)
that deletes itself when closed, and that a newer warning, a new run or
closing the main window closes. The report is advisory: if building it
fails, the finished run stands.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
import shiboken6
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from al_dic.gui.app_state import AppState, RunState
from al_dic.gui.controllers import pipeline_controller as pc_module
from al_dic.gui.controllers import tracking_messages as tm_module
from al_dic.gui.controllers.pipeline_controller import PipelineController
from al_dic.gui.controllers.tracking_messages import build_tracking_messages
from al_dic.utils.locale_format import format_number

from ._helpers import (
    TRACKING_FRAME0_GRID,
    TRACKING_LARGEST_GRID,
    TRACKING_N_NODES,
    make_synthetic_pipeline_result,
    make_tracking_diagnostics,
    make_tracking_para,
)

CONTROLLER_LOGGER = "al_dic.gui.controllers.pipeline_controller"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def state(qapp):
    """A fresh AppState singleton per test (connections do not leak)."""
    AppState._instance = None
    st = AppState.instance()
    st.image_files = ["a.tif", "b.tif", "c.tif"]
    yield st
    AppState._instance = None


class _Recorder:
    """Every log line, run_warning and frame change, in emission order."""

    def __init__(self, state: AppState) -> None:
        self.events: list[tuple] = []
        state.log_message.connect(
            lambda msg, lvl: self.events.append(("log", msg, lvl)))
        state.run_warning.connect(
            lambda *args: self.events.append(("run_warning", *args)))
        state.current_frame_changed.connect(
            lambda idx: self.events.append(("frame", idx)))
        state.fatal_error.connect(
            lambda *args: self.events.append(("fatal", *args)))

    def logs(self, level: str) -> list[str]:
        return [e[1] for e in self.events if e[0] == "log" and e[2] == level]

    def warnings(self) -> list[tuple]:
        return [e[1:] for e in self.events if e[0] == "run_warning"]

    def index(self, kind: str) -> int:
        return next(i for i, e in enumerate(self.events) if e[0] == kind)


def _result(kind: str | None, **changes):
    result, _ = make_synthetic_pipeline_result(n_frames=3)
    diag = None if kind is None else make_tracking_diagnostics(kind)
    return replace(result, tracking_diagnostics=diag, **changes)


def _expected(result):
    return build_tracking_messages(result.tracking_diagnostics, result.dic_para)


def _finish(state: AppState, result) -> _Recorder:
    rec = _Recorder(state)
    PipelineController(state, None)._on_finished(result)
    return rec


def _has_number(number: str, text: str) -> bool:
    return re.search(rf"(?<![\d.,]){re.escape(number)}(?![\d])", text) is not None


# ---------------------------------------------------------------------------
# PipelineController._on_finished
# ---------------------------------------------------------------------------


def test_flagged_run_logs_the_findings_and_asks_for_one_dialog(state):
    result = _result("warning")
    expected = _expected(result)
    rec = _finish(state, result)

    assert state.run_state == RunState.DONE
    assert state.results is result
    tracking_logs = [e[1:] for e in rec.events
                     if e[0] == "log" and e[2] in ("warn", "info")]
    assert tracking_logs == list(expected.log_lines)
    assert len(rec.logs("warn")) == 2  # grid_miss, material_loss
    assert rec.warnings() == [(
        expected.title, expected.text, expected.informative, expected.details,
    )]
    # The dialog comes last: after the run is DONE and frame 1 is shown.
    assert rec.index("frame") < rec.index("run_warning")
    assert rec.events[-1][0] == "run_warning"


def test_burst_finding_also_asks_for_a_dialog(state):
    rec = _finish(state, _result("burst"))
    assert len(rec.logs("warn")) == 1
    assert len(rec.warnings()) == 1


@pytest.mark.parametrize("kind", ["healthy", "notice"])
def test_run_without_warning_findings_neither_warns_nor_asks(state, kind):
    result = _result(kind)
    rec = _finish(state, result)
    assert state.run_state == RunState.DONE
    assert rec.logs("warn") == []
    assert rec.warnings() == []
    expected = _expected(result)
    assert rec.logs("info") == [text for text, _ in expected.log_lines]


def test_clean_run_reports_nothing(state):
    """No finding and no grid of its own: not a single tracking line."""
    result = _result("clean")
    assert _expected(result) is None
    rec = _finish(state, result)
    assert state.run_state == RunState.DONE
    assert rec.logs("warn") == [] and rec.logs("info") == []
    assert rec.logs("success")  # "Results received: ..."
    assert rec.warnings() == []


def test_partial_run_is_logged_without_a_dialog(state):
    result = _result(
        "warning", stopped_early=True, stopped_at_frame=3,
        stop_reason="Analysis stopped by user.",
    )
    rec = _finish(state, result)
    assert state.run_state == RunState.DONE
    assert len(rec.logs("warn")) == 2
    assert rec.warnings() == []


def test_unknown_diagnostics_report_nothing(state):
    rec = _finish(state, _result(None))
    assert state.run_state == RunState.DONE
    assert rec.logs("warn") == [] and rec.logs("info") == []
    assert rec.logs("success")  # "Results received: ..."
    assert rec.warnings() == []


def test_result_without_a_diagnostics_field_reports_nothing(state, caplog):
    """A result object from before diagnostics existed is not an error."""
    rec = _Recorder(state)
    with caplog.at_level(logging.WARNING, logger=CONTROLLER_LOGGER):
        PipelineController(state, None)._report_tracking_quality(
            SimpleNamespace(result_disp=[], stopped_early=False))
    assert rec.events == []
    assert [r for r in caplog.records if r.name == CONTROLLER_LOGGER] == []


def test_a_failing_report_keeps_the_finished_run(state, monkeypatch, caplog):
    def boom(*_args):
        raise RuntimeError("forced")

    monkeypatch.setattr(pc_module, "build_tracking_messages", boom)
    result = _result("warning")
    with caplog.at_level(logging.ERROR, logger=CONTROLLER_LOGGER):
        rec = _finish(state, result)

    assert state.run_state == RunState.DONE
    assert state.results is result
    assert state.current_frame == 1
    assert rec.warnings() == []
    assert not [e for e in rec.events if e[0] == "fatal"]
    assert rec.logs("error") == []  # not "Failed to process results"
    errors = [r for r in caplog.records
              if r.name == CONTROLLER_LOGGER and r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].exc_info is not None


def test_failed_run_reports_nothing(state):
    rec = _finish(state, None)
    assert state.run_state == RunState.IDLE
    assert rec.warnings() == [] and rec.logs("warn") == []


# ---------------------------------------------------------------------------
# build_tracking_messages
# ---------------------------------------------------------------------------


def test_grid_note_compares_frame_1_lattice_with_the_reference_grids():
    """Grids with grids: frame 1's lattice, never its (refined) mesh.

    The reference grids are counted as lattices, and the frame-0 mesh can
    hold refinement nodes its lattice does not (1000 against 49 here).
    """
    messages = build_tracking_messages(
        make_tracking_diagnostics("healthy"), make_tracking_para())
    ((note, level),) = messages.log_lines
    assert level == "info"
    assert _has_number(str(TRACKING_FRAME0_GRID), note), note
    assert format_number(TRACKING_LARGEST_GRID) in note, note
    assert str(TRACKING_N_NODES) not in note, note
    assert format_number(TRACKING_N_NODES) not in note, note


def test_unknown_finding_is_skipped_and_logged(monkeypatch, caplog):
    """A finding key the GUI has no sentence for is not shown as another."""
    diag, para = make_tracking_diagnostics("notice"), make_tracking_para()
    expected = build_tracking_messages(diag, para)
    known = tm_module.assess_tracking(diag)
    mystery = replace(known[0], key="mystery", severity="warning")
    monkeypatch.setattr(
        tm_module, "assess_tracking", lambda _diag: (mystery, *known))

    with caplog.at_level(logging.WARNING, logger=tm_module.__name__):
        messages = build_tracking_messages(diag, para)

    assert messages == expected
    assert not messages.has_warning
    logged = [r.getMessage() for r in caplog.records
              if r.name == tm_module.__name__]
    assert len(logged) == 1 and "mystery" in logged[0], logged


def test_console_shows_the_advice_line_by_line(qapp):
    """The console renders HTML: its line breaks must survive."""
    from al_dic.gui.widgets.console_log import ConsoleLog

    messages = build_tracking_messages(
        make_tracking_diagnostics("warning"), make_tracking_para())
    advice, level = messages.log_lines[-1]
    advice_lines = advice.split("\n")
    assert level == "info" and len(advice_lines) == 4, advice

    console = ConsoleLog()
    console.append_log(advice, level)
    shown = console.toPlainText().split("\n")
    assert shown[-4].endswith(advice_lines[0]), shown
    assert shown[-3:] == advice_lines[1:], shown


# ---------------------------------------------------------------------------
# MainWindow: the dialog
# ---------------------------------------------------------------------------


class _Visible:
    def __init__(self, visible: bool) -> None:
        self._visible = visible

    def isVisible(self) -> bool:
        return self._visible


def test_warning_parent_prefers_a_visible_strain_window():
    from al_dic.gui.app import _warning_parent

    main = object()
    assert _warning_parent(None, main) is main
    assert _warning_parent(_Visible(False), main) is main
    strain = _Visible(True)
    assert _warning_parent(strain, main) is strain


def _no_modal(*_args, **_kwargs):
    raise AssertionError("the tracking warning must not be modal")


@pytest.fixture
def window(qapp, state, monkeypatch):
    """A MainWindow whose message boxes can never go modal."""
    from al_dic.gui.app import MainWindow

    monkeypatch.setattr(QMessageBox, "open", _no_modal)
    monkeypatch.setattr(QMessageBox, "exec", _no_modal)
    win = MainWindow()
    assert win._strain_window is None
    return win


def _flush_deletes(qapp) -> None:
    qapp.processEvents()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _warning_boxes(parent: QWidget) -> list[QMessageBox]:
    """The run warnings (titled "Run ...") still alive under ``parent``."""
    return [box for box in parent.findChildren(QMessageBox)
            if box.windowTitle().startswith("Run ")]


def _warn(state: AppState, title: str) -> None:
    state.run_warning.emit(title, "Text", "Informative", "Details")


def test_run_warning_shows_a_modeless_box_that_deletes_itself(
    qapp, state, window,
):
    _warn(state, "Run 1")

    (box,) = _warning_boxes(window)
    assert box.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    assert box.windowModality() == Qt.WindowModality.NonModal
    assert QApplication.activeModalWidget() is None
    assert box.icon() == QMessageBox.Icon.Warning
    assert box.windowTitle() == "Run 1"
    assert box.text() == "Text"
    assert box.informativeText() == "Informative"
    assert box.detailedText() == "Details"
    assert box.parent() is window
    assert box.isVisible()

    box.button(QMessageBox.StandardButton.Ok).click()  # the user's OK
    _flush_deletes(qapp)
    assert not shiboken6.isValid(box)


def test_a_new_warning_replaces_the_one_still_open(qapp, state, window):
    _warn(state, "Run 1")
    (first,) = _warning_boxes(window)

    _warn(state, "Run 2")
    _flush_deletes(qapp)

    assert not shiboken6.isValid(first)
    (second,) = _warning_boxes(window)
    assert second.windowTitle() == "Run 2" and second.isVisible()


def test_starting_a_run_closes_the_previous_warning(qapp, state, window):
    _warn(state, "Run 1")
    (box,) = _warning_boxes(window)

    state.set_run_state(RunState.RUNNING)
    _flush_deletes(qapp)

    assert not shiboken6.isValid(box)
    assert _warning_boxes(window) == []


def test_closing_the_main_window_closes_the_warning(qapp, state, window):
    """Also when the box sits over the strain window, which closing the
    main window only hides (its children would stay on screen)."""
    strain = QWidget()
    strain.show()
    window._strain_window = strain
    _warn(state, "Run 1")
    (box,) = _warning_boxes(strain)

    window.close()
    _flush_deletes(qapp)

    assert not shiboken6.isValid(box)


def test_session_reopen_shows_no_tracking_warning(qapp, state, tmp_path):
    """Opening a saved session never repeats the run's report."""
    from al_dic.gui.session import apply_session, load_session, save_session

    result = _result("warning")
    state.image_folder = tmp_path
    state.results = result
    out = tmp_path / "s.aldic"
    save_session(out, state, include_results=True)
    session = load_session(out)
    assert session.results.tracking_diagnostics == result.tracking_diagnostics

    AppState._instance = None
    fresh = AppState.instance()
    rec = _Recorder(fresh)

    class _StubImageCtrl:
        def load_folder(self, _folder):
            pass

    apply_session(session, fresh, _StubImageCtrl())
    assert fresh.run_state == RunState.DONE
    assert rec.warnings() == [] and rec.logs("warn") == []
