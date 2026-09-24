"""The Analysis tab: placement, the probe list, and what the chart shows."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from al_dic.analysis.probes import AreaGeom, LineGeom, PointGeom
from al_dic.gui.app_state import AppState
from al_dic.gui.panels.analysis_tab import AnalysisTab
from al_dic.gui.panels.probe_canvas import ProbeCanvas


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(["pyALDIC-tests"])


@pytest.fixture
def state(qapp):
    s = AppState.instance()
    s.reset()
    s.probes.clear()
    return s


@pytest.fixture
def tab(state):
    return AnalysisTab(state)


# --- the tab exists in the strain window ---------------------------------

def test_strain_window_has_two_tabs(state):
    from al_dic.gui.strain_window import StrainWindow

    win = StrainWindow(state)
    assert win._tabs.count() == 2
    assert win._tabs.tabText(1) == "Analysis"


# --- placement -----------------------------------------------------------

def test_canvas_starts_with_no_tool(qapp):
    assert ProbeCanvas().tool == "none"


def test_a_point_needs_one_click(qapp):
    canvas = ProbeCanvas()
    seen = []
    canvas.probe_requested.connect(lambda k, g: seen.append((k, g)))
    canvas.set_tool("point")
    canvas._pending.append(QPointF(10.0, 20.0))
    canvas._commit_if_complete()
    assert seen == [("point", PointGeom(10.0, 20.0))]
    assert canvas.tool == "none", "the tool disarms after placing"


def test_a_line_needs_two_clicks(qapp):
    canvas = ProbeCanvas()
    seen = []
    canvas.probe_requested.connect(lambda k, g: seen.append((k, g)))
    canvas.set_tool("line")
    canvas._pending.append(QPointF(0.0, 0.0))
    canvas._commit_if_complete()
    assert seen == [], "one click is not a line"
    canvas._pending.append(QPointF(30.0, 40.0))
    canvas._commit_if_complete()
    assert seen[0][1] == LineGeom(0.0, 0.0, 30.0, 40.0)


def test_a_degenerate_shape_is_discarded_not_stored(qapp):
    """A double-click is not an instruction to make a zero-length gauge."""
    canvas = ProbeCanvas()
    seen = []
    canvas.probe_requested.connect(lambda k, g: seen.append((k, g)))
    canvas.set_tool("line")
    canvas._pending.extend([QPointF(5.0, 5.0), QPointF(5.0, 5.0)])
    canvas._commit_if_complete()
    assert seen == []


def test_a_polygon_closes_on_double_click(qapp):
    canvas = ProbeCanvas()
    seen = []
    canvas.probe_requested.connect(lambda k, g: seen.append((k, g)))
    canvas.set_tool("area_polygon")
    for pt in ((0.0, 0.0), (10.0, 0.0), (5.0, 9.0)):
        canvas._pending.append(QPointF(*pt))
    canvas._emit_polygon()
    assert seen[0][1] == AreaGeom.polygon([(0, 0), (10, 0), (5, 9)])


def test_escape_cancels_placement(qapp):
    canvas = ProbeCanvas()
    cancelled = []
    canvas.placement_cancelled.connect(lambda: cancelled.append(True))
    canvas.set_tool("area_rect")
    canvas.cancel_placement()
    assert canvas.tool == "none"
    assert cancelled == [True]


# --- probe list ----------------------------------------------------------

def test_placing_a_probe_adds_a_row(tab, state):
    tab._on_probe_placed("point", PointGeom(5.0, 5.0))
    assert tab._table.rowCount() == 1
    assert len(state.probes) == 1


def test_a_probe_can_be_renamed(tab, state):
    tab._on_probe_placed("point", PointGeom(5.0, 5.0))
    item = tab._table.item(0, 1)
    item.setText("crack tip")
    assert state.probes.get(1).label == "crack tip"


def test_a_probe_can_be_hidden(tab, state):
    """The reference offers no visibility control at all."""
    tab._on_probe_placed("point", PointGeom(5.0, 5.0))
    tab._table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert state.probes.get(1).visible is False


def test_a_selected_probe_can_be_deleted(tab, state):
    """Not 'delete the last one', which is the only removal the reference has."""
    tab._on_probe_placed("point", PointGeom(1.0, 1.0))
    tab._on_probe_placed("point", PointGeom(2.0, 2.0))
    tab._selected_id = 1
    tab._on_delete()
    assert [p.id for p in state.probes] == [2]


# --- chart ---------------------------------------------------------------

def test_chart_says_what_to_do_before_a_run(tab):
    """An empty chart with no explanation reads as a broken chart."""
    texts = [t.get_text() for t in tab._chart.figure.axes[0].texts]
    assert any("Run a DIC analysis" in t for t in texts)


def test_reductions_offered_follow_the_probe_kind(tab):
    tab._kind_box.setCurrentIndex(tab._kind_box.findData("point"))
    tab._populate_reduction_box()
    point_reductions = {
        tab._reduction_box.itemData(i)
        for i in range(tab._reduction_box.count())
    }
    # A point is a one-sample region: its value stands in for a mean, a
    # median or an extreme, but not for a spread.
    assert point_reductions == {"value", "mean", "median", "max", "min"}

    tab._kind_box.setCurrentIndex(tab._kind_box.findData("line"))
    tab._populate_reduction_box()
    line_reductions = {
        tab._reduction_box.itemData(i)
        for i in range(tab._reduction_box.count())
    }
    assert {"strain", "cod", "median", "valid_fraction"} <= line_reductions

    tab._kind_box.setCurrentIndex(tab._kind_box.findData("area"))
    tab._populate_reduction_box()
    area_reductions = {
        tab._reduction_box.itemData(i)
        for i in range(tab._reduction_box.count())
    }
    assert not ({"strain", "cod"} & area_reductions)


def test_every_canonical_field_is_offered(tab):
    from al_dic.core.fields import ALL_FIELDS

    offered = {
        tab._field_box.itemData(i) for i in range(tab._field_box.count())
    }
    assert offered == set(ALL_FIELDS)


def test_chart_plots_a_probe_once_results_exist(tab, state):
    from al_dic.core.data_structures import (
        DICMesh, FrameResult, PipelineResult,
    )

    xs = np.arange(0.0, 41.0, 4.0)
    gx, gy = np.meshgrid(xs, xs)
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    n = len(nodes)
    disp = []
    for f in (1, 2, 3):
        U = np.zeros(2 * n)
        U[0::2] = 0.01 * f * nodes[:, 0]
        disp.append(FrameResult(U=U, U_accum=U.copy()))
    state.results = PipelineResult(
        dic_para=type("P", (), {"winstepsize": 4.0})(),
        dic_mesh=DICMesh(coordinates_fem=nodes,
                         elements_fem=np.zeros((0, 4), np.int64)),
        result_disp=disp, result_def_grad=[], result_strain=[],
        result_fe_mesh_each_frame=[],
    )

    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    tab._field_box.setCurrentIndex(tab._field_box.findData("disp_u"))
    tab._refresh_chart()

    lines = tab._chart.figure.axes[0].get_lines()
    assert lines, "the probe should have produced a curve"
    ydata = lines[0].get_ydata()
    np.testing.assert_allclose(ydata, [0.0, 0.2, 0.4, 0.6], atol=1e-9)


def test_export_without_data_does_not_write(tab, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(
        "al_dic.gui.panels.analysis_tab.QMessageBox.information",
        lambda *a, **k: shown.append(a),
    )
    tab._last_entries = []
    tab._on_export_csv()
    assert shown, "the user is told there is nothing to export"
    assert not list(tmp_path.iterdir())
