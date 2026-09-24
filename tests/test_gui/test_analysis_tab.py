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

def _run(state, *, strain: bool = False, n_frames: int = 3):
    """u = 0.01 * f * x on an 11 x 11 grid; optional uniform strain."""
    from types import SimpleNamespace

    from al_dic.core.data_structures import (
        DICMesh, FrameResult, PipelineResult, StrainResult,
    )

    xs = np.arange(0.0, 41.0, 4.0)
    gx, gy = np.meshgrid(xs, xs)
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    n = len(nodes)
    disp, strains = [], []
    for f in range(1, n_frames + 1):
        U = np.zeros(2 * n)
        U[0::2] = 0.01 * f * nodes[:, 0]
        disp.append(FrameResult(U=U, U_accum=U.copy()))
        e = np.full(n, 0.01 * f)
        strains.append(StrainResult(
            disp_u=U[0::2], disp_v=U[1::2], strain_exx=e, strain_eyy=e,
            strain_exy=e, strain_principal_max=e, strain_principal_min=e,
            strain_maxshear=e, strain_von_mises=e, strain_rotation=e,
        ))
    state.results = PipelineResult(
        dic_para=SimpleNamespace(winstepsize=4.0, img_size=(44, 44)),
        dic_mesh=DICMesh(coordinates_fem=nodes,
                         elements_fem=np.zeros((0, 4), np.int64)),
        result_disp=disp, result_def_grad=[],
        result_strain=strains if strain else [],
        result_fe_mesh_each_frame=[],
    )


def _plot(tab, name: str, *, gauge: bool = False, statistic: str = "mean"):
    from al_dic.gui.panels.analysis_tab import _Quantity

    tab._quantity_box.setCurrentIndex(
        tab._quantity_box.findData(
            _Quantity("gauge" if gauge else "field", name).key))
    tab._statistic_box.setCurrentIndex(tab._statistic_box.findData(statistic))
    tab._refresh()


def _labels(tab) -> list[str]:
    legend = tab._chart.figure.axes[0].get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend else []


def test_chart_says_what_to_do_before_a_run(tab):
    """An empty chart with no explanation reads as a broken chart."""
    tab._refresh()
    texts = [t.get_text() for t in tab._chart.figure.axes[0].texts]
    assert any("Run a DIC analysis" in t for t in texts)
    assert not tab._export_btn.isEnabled()


def test_every_probe_field_is_offered_in_the_field_tabs_order(tab):
    from al_dic.core.fields import ALL_FIELDS
    from al_dic.gui.panels.analysis_tab import _FIELDS

    from al_dic.gui.panels.analysis_tab import _Quantity

    offered = [
        _Quantity.from_key(tab._quantity_box.itemData(i))
        for i in range(tab._quantity_box.count())
    ]
    fields = [q.name for q in offered if q is not None and not q.is_gauge]
    gauges = [q.name for q in offered if q is not None and q.is_gauge]
    assert fields == list(_FIELDS) and set(fields) == set(ALL_FIELDS)
    assert "strain" in gauges and "cod_sliding" in gauges


def test_the_extensometer_is_found_by_name(tab):
    """The word appeared nowhere in the first version's interface."""
    names = [tab._quantity_box.itemText(i)
             for i in range(tab._quantity_box.count())]
    assert any("Extensometer" in n for n in names)


def test_a_point_and_a_region_share_a_chart_of_the_same_quantity(tab, state):
    """Kind-only comparison forbade this; the y-axis is what must match."""
    _run(state)
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    tab._on_probe_placed("area", AreaGeom.rect(4.0, 4.0, 36.0, 36.0))
    _plot(tab, "disp_u")
    assert len(_labels(tab)) == 2


def test_gauges_plot_lines_only_and_say_why_others_are_absent(tab, state):
    _run(state)
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    tab._on_probe_placed("line", LineGeom(4.0, 20.0, 36.0, 20.0))
    _plot(tab, "strain", gauge=True)
    assert len(_labels(tab)) == 1
    assert tab._statistic_box.isHidden(), "a gauge reads no field, so no statistic"
    note = tab._table.item(0, 4).text()
    assert "line" in note, "the point's row says why it is not plotted"


def test_the_extensometer_curve_is_the_applied_stretch(tab, state):
    _run(state)
    tab._on_probe_placed("line", LineGeom(4.0, 20.0, 36.0, 20.0))
    _plot(tab, "strain", gauge=True)
    y = tab._chart.figure.axes[0].get_lines()[0].get_ydata()
    np.testing.assert_allclose(y, [0.0, 0.01, 0.02, 0.03], atol=1e-9)


def test_strain_can_be_shown_in_percent(tab, state):
    _run(state)
    tab._on_probe_placed("line", LineGeom(4.0, 20.0, 36.0, 20.0))
    tab._unit_box.setCurrentIndex(tab._unit_box.findData("percent"))
    _plot(tab, "strain", gauge=True)
    y = tab._chart.figure.axes[0].get_lines()[0].get_ydata()
    np.testing.assert_allclose(y, [0.0, 1.0, 2.0, 3.0], atol=1e-9)
    assert "(%)" in tab._chart.figure.axes[0].get_ylabel()


def test_a_point_axis_does_not_say_value(tab, state):
    """'εyy — Value' was the first version's label for a point."""
    _run(state)
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    _plot(tab, "disp_u")
    assert "—" not in tab._chart.figure.axes[0].get_ylabel()


def test_strain_that_was_not_computed_asks_for_it(tab, state):
    _run(state, strain=False)
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    _plot(tab, "strain_eyy")
    texts = [t.get_text() for t in tab._chart.figure.axes[0].texts]
    assert any("not been computed" in t for t in texts)


# --- frame ---------------------------------------------------------------

def test_the_chart_cursor_follows_the_strain_windows_frame(tab, state):
    _run(state)
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    _plot(tab, "disp_u")
    tab.set_frame(2)
    cursor = tab._chart._cursor
    assert cursor is not None and cursor.get_xdata()[0] == 3.0   # 1-based
    assert tab._nav._current == 2


def test_a_click_on_the_chart_jumps_to_that_frame(tab, state):
    _run(state)
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    _plot(tab, "disp_u")
    asked = []
    tab.frame_requested.connect(asked.append)
    tab._on_chart_clicked(3.2)          # nearest frame label is 3
    assert asked == [2]


def test_the_two_tabs_share_one_frame(state):
    from al_dic.gui.strain_window import StrainWindow

    _run(state)
    win = StrainWindow(state)
    win.set_strain_frame(2)
    assert win._analysis_tab._frame == 2
    win._analysis_tab.frame_requested.emit(1)
    assert win._strain_current_frame == 1


# --- doing no more work than needed --------------------------------------

def test_nothing_is_read_while_the_tab_is_hidden(tab, state, monkeypatch):
    """The strain window is never destroyed; it used to extract while closed."""
    from al_dic.analysis.engine import AnalysisEngine

    calls = []
    real = AnalysisEngine.series
    monkeypatch.setattr(AnalysisEngine, "series",
                        lambda self, *a, **k: calls.append(1) or real(self, *a, **k))
    state.probes.add("point", PointGeom(20.0, 20.0))
    _run(state)
    state.results_changed.emit()
    assert not tab.isVisible() and calls == []
    assert tab._dirty


def test_renaming_redraws_without_reading_the_run(tab, state, monkeypatch):
    from al_dic.analysis.engine import AnalysisEngine

    _run(state)
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    _plot(tab, "disp_u")
    calls = []
    real = AnalysisEngine.series
    monkeypatch.setattr(AnalysisEngine, "series",
                        lambda self, *a, **k: calls.append(1) or real(self, *a, **k))
    tab._table.item(0, 1).setText("renamed")
    assert state.probes.get(1).label == "renamed"
    assert calls == [], "a label change is not a new measurement"


def test_editing_a_probe_keeps_the_users_zoom(tab, state):
    _run(state)
    tab._canvas.set_image(np.zeros((44, 44, 3), dtype=np.uint8))
    tab._background_loaded = True
    tab._canvas.scale(4.0, 4.0)
    before = tab._canvas.transform().m11()
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    tab._table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert tab._canvas.transform().m11() == before


# --- export --------------------------------------------------------------

def test_export_is_unavailable_when_nothing_is_plotted(tab, state):
    _run(state)
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    _plot(tab, "disp_u")
    assert tab._export_btn.isEnabled()
    tab._selected_id = 1
    tab._on_delete()
    assert not tab._export_btn.isEnabled(), "a deleted probe cannot be exported"


def test_export_writes_what_is_plotted_now(tab, state, tmp_path, monkeypatch):
    """It used to write the last successful plot -- deleted probes included."""
    import csv

    _run(state)
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    tab._on_probe_placed("point", PointGeom(8.0, 8.0))
    _plot(tab, "disp_u")
    tab._selected_id = 1
    tab._on_delete()
    out = tmp_path / "probes.csv"
    monkeypatch.setattr(
        "al_dic.gui.panels.analysis_tab.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), ""),
    )
    tab._on_export_csv()
    with open(out, encoding="utf-8-sig") as fh:
        header = next(csv.reader(ln for ln in fh if not ln.startswith("#")))
    assert any(h.startswith("P2_") for h in header)
    assert not any(h.startswith("P1_") for h in header)
