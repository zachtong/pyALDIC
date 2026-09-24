"""The Analysis tab's line views: a field along a line, and its kymograph.

The run is u = 0.01 * f * x on an 11 x 11 grid (nodes 0..40, step 4), so a
horizontal line from x = 4 to x = 36 reads u = 0.01 * f * (4 + d) at arc
length d: every value below can be checked by hand.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from al_dic.analysis.probes import LineGeom, PointGeom
from al_dic.gui.app_state import AppState
from al_dic.gui.panels.analysis import AnalysisTab
from tests.test_gui.test_analysis_tab import _plot, _run


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
    _run(state, strain=True)
    t = AnalysisTab(state)
    t._on_probe_placed("line", LineGeom(4.0, 20.0, 36.0, 20.0))
    return t


def _view(tab, name: str) -> None:
    tab._view_bar.setCurrentIndex(tab._view_bar_index(name))
    tab._refresh()


def _at_frame(tab, frame: int) -> None:
    """Hidden, the tab only notes the frame; draw as it would when shown."""
    tab.set_frame(frame)
    tab._refresh()


def _ax(tab):
    return tab._chart.figure.axes[0]


def _lines(tab, gid: str):
    return [ln for ln in _ax(tab).get_lines() if ln.get_gid() == gid]


def _placeholder(tab) -> str:
    return " ".join(t.get_text() for t in _ax(tab).texts)


# --- the views -----------------------------------------------------------------

def test_the_views_are_offered_in_order(tab):
    keys = [tab._view_bar.tabData(i) for i in range(tab._view_bar.count())]
    assert keys == ["time", "profile", "kymograph", "stress_strain"]
    assert tab._view_bar.currentIndex() == 0, "over time stays the default"


def test_line_views_hide_what_does_not_apply(tab):
    _plot(tab, "strain_eyy")
    _view(tab, "profile")
    assert tab._statistic_box.isHidden(), "a profile is every sample, not a statistic"
    assert tab._threshold.isHidden()
    assert not tab._unit_box.isHidden(), "strain still reads in the chosen unit"


# --- profile -----------------------------------------------------------------------

def test_the_profile_is_the_field_along_the_line_at_the_current_frame(tab):
    _plot(tab, "disp_u")
    _view(tab, "profile")
    _at_frame(tab, 2)
    (current,) = _lines(tab, "current")
    d, u = current.get_xdata(), current.get_ydata()
    assert d[0] == pytest.approx(0.0) and d[-1] == pytest.approx(32.0)
    np.testing.assert_allclose(u, 0.02 * (4.0 + d), atol=1e-9)


def test_the_profile_follows_the_frame(tab):
    _plot(tab, "disp_u")
    _view(tab, "profile")
    _at_frame(tab, 3)
    (current,) = _lines(tab, "current")
    np.testing.assert_allclose(
        current.get_ydata(), 0.03 * (4.0 + current.get_xdata()), atol=1e-9)


def test_a_shown_profile_redraws_when_the_frame_moves(tab):
    tab.show()
    _plot(tab, "disp_u")
    _view(tab, "profile")
    tab.set_frame(3)
    (current,) = _lines(tab, "current")
    np.testing.assert_allclose(
        current.get_ydata(), 0.03 * (4.0 + current.get_xdata()), atol=1e-9)
    tab.close()


def test_other_frames_are_drawn_faintly_and_can_be_hidden(tab):
    _plot(tab, "disp_u")
    _view(tab, "profile")
    _at_frame(tab, 1)
    assert len(_lines(tab, "other")) == 3, "frames 0, 2 and 3 beside frame 1"
    tab._other_frames_box.setChecked(False)
    tab._refresh()
    assert _lines(tab, "other") == []


def test_the_profile_reads_the_selected_line(tab, state):
    tab._on_probe_placed("line", LineGeom(20.0, 8.0, 36.0, 8.0))
    tab._on_canvas_selected(1)
    _plot(tab, "disp_u")
    _view(tab, "profile")
    _at_frame(tab, 3)
    assert _lines(tab, "current")[0].get_ydata()[0] == pytest.approx(0.03 * 4.0)
    tab._on_canvas_selected(2)
    assert _lines(tab, "current")[0].get_ydata()[0] == pytest.approx(0.03 * 20.0)


def test_a_profile_needs_a_line(state):
    _run(state)
    tab = AnalysisTab(state)
    tab._on_probe_placed("point", PointGeom(20.0, 20.0))
    _plot(tab, "disp_u")
    _view(tab, "profile")
    assert "line" in _placeholder(tab)
    assert not tab._export_btn.isEnabled()


def test_a_gauge_reading_has_no_profile(tab):
    _plot(tab, "strain", gauge=True)
    _view(tab, "profile")
    assert "field" in _placeholder(tab)


def test_strain_along_the_line_takes_the_display_unit(tab):
    tab._unit_box.setCurrentIndex(tab._unit_box.findData("percent"))
    _plot(tab, "strain_exx")
    _view(tab, "profile")
    _at_frame(tab, 2)
    np.testing.assert_allclose(_lines(tab, "current")[0].get_ydata(), 2.0, atol=1e-9)
    assert "(%)" in _ax(tab).get_ylabel()


# --- kymograph -----------------------------------------------------------------

def test_the_kymograph_is_distance_by_frame(tab):
    _plot(tab, "disp_u")
    _view(tab, "kymograph")
    image = next(im for im in _ax(tab).get_images() if im.get_gid() == "values")
    data = np.asarray(image.get_array())
    assert data.shape[1] == 4, "one column per frame, reference included"
    assert data[-1, -1] == pytest.approx(0.03 * 36.0), "last sample, last frame"
    assert data[0, 0] == pytest.approx(0.0), "the reference frame is at rest"


def test_the_kymograph_marks_the_current_frame(tab):
    _plot(tab, "disp_u")
    _view(tab, "kymograph")
    tab.set_frame(2)
    cursor = tab._chart._cursor
    assert cursor is not None and cursor.get_xdata()[0] == 3.0   # 1-based


def test_a_click_on_the_kymograph_jumps_to_that_frame(tab):
    _plot(tab, "disp_u")
    _view(tab, "kymograph")
    asked = []
    tab.frame_requested.connect(asked.append)
    tab._on_chart_clicked(2.4)
    assert asked == [1]


def test_consumed_material_shows_as_its_own_band(state):
    """A crack's consumed samples are neither zero nor missing data."""
    _run(state)
    result = state.results
    nodes = result.dic_mesh.coordinates_fem
    dead = np.flatnonzero(np.isclose(nodes[:, 0], 20.0))
    for fr in result.result_disp[1:]:                     # frames 2 and 3
        fr.U_accum[2 * dead] = np.nan
        fr.U_accum[2 * dead + 1] = np.nan
    tab = AnalysisTab(state)
    tab._on_probe_placed("line", LineGeom(4.0, 20.0, 36.0, 20.0))
    _plot(tab, "disp_u")
    _view(tab, "kymograph")
    band = next(im for im in _ax(tab).get_images() if im.get_gid() == "consumed")
    shown = ~np.ma.getmaskarray(band.get_array())
    assert shown[:, 2:].any(), "consumed on frames 2 and 3"
    assert not shown[:, :2].any(), "and not before"


# --- export --------------------------------------------------------------------

def _read(path):
    with open(path, encoding="utf-8-sig") as fh:
        comments = [ln for ln in fh if ln.startswith("#")]
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.reader(ln for ln in fh if not ln.startswith("#")))
    return comments, rows[0], rows[1:]


def test_line_data_is_one_row_per_sample_and_a_column_per_frame(
        tab, tmp_path, monkeypatch):
    _plot(tab, "disp_u")
    _view(tab, "profile")
    out = tmp_path / "line.csv"
    monkeypatch.setattr(
        "al_dic.gui.panels.analysis.tab.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), ""))
    tab._on_export_line_csv()
    comments, header, rows = _read(out)
    assert header == ["distance_px", "x_px", "y_px",
                      "frame_1", "frame_2", "frame_3", "frame_4"]
    last = rows[-1]
    assert float(last[0]) == pytest.approx(32.0)
    assert (float(last[1]), float(last[2])) == pytest.approx((36.0, 20.0))
    assert float(last[-1]) == pytest.approx(0.03 * 36.0)
    assert any("frame_N matches" in c for c in comments)


def test_line_data_writes_v_in_world_axes(tab, state, tmp_path, monkeypatch):
    """As the node export does: v positive up."""
    for fr in state.results.result_disp:
        fr.U_accum[1::2] = 0.5                         # 0.5 px down on screen
    _plot(tab, "disp_v")
    _view(tab, "profile")
    out = tmp_path / "line_v.csv"
    monkeypatch.setattr(
        "al_dic.gui.panels.analysis.tab.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), ""))
    tab._on_export_line_csv()
    comments, _, rows = _read(out)
    assert float(rows[0][-1]) == pytest.approx(-0.5)
    assert any("world axes" in c for c in comments)


# --- a line with nothing to show says why ------------------------------------

def test_a_line_off_the_specimen_says_so(state):
    """An empty chart with -0.04..0.04 on both axes read as a broken view."""
    _run(state)
    tab = AnalysisTab(state)
    tab._on_probe_placed("line", LineGeom(60.0, 20.0, 80.0, 20.0))  # mesh: 0..40
    _plot(tab, "disp_u")
    _view(tab, "profile")
    assert "off the measured area" in _placeholder(tab)
    assert not tab._export_btn.isEnabled()


def test_a_line_a_crack_consumed_says_so(state):
    _run(state)
    result = state.results
    nodes = result.dic_mesh.coordinates_fem
    row = np.flatnonzero(np.isclose(nodes[:, 1], 20.0))
    for fr in result.result_disp:
        fr.U_accum[2 * row] = np.nan
        fr.U_accum[2 * row + 1] = np.nan
    tab = AnalysisTab(state)
    tab._on_probe_placed("line", LineGeom(4.0, 20.0, 36.0, 20.0))
    _plot(tab, "disp_u")
    _view(tab, "kymograph")
    assert "crack" in _placeholder(tab)


def test_a_profile_spans_its_line_even_where_it_has_gaps(tab):
    _plot(tab, "disp_u")
    _view(tab, "profile")
    assert _ax(tab).get_xlim() == pytest.approx((0.0, 32.0))


def test_trimmed_strain_is_named_before_what_cannot_be_helped(state):
    """A line over a hole and a trimmed ligament: most of it is off the
    material, but the trim is what the user can change -- so it is named."""
    from dataclasses import replace as dc_replace

    _run(state, strain=True)
    result = state.results
    n = result.dic_mesh.coordinates_fem.shape[0]
    result.result_strain[:] = [dc_replace(sr, strain_valid=np.zeros(n, bool))
                               for sr in result.result_strain]
    tab = AnalysisTab(state)
    tab._on_probe_placed("line", LineGeom(30.0, 20.0, 70.0, 20.0))  # 3/4 off
    _plot(tab, "strain_eyy")
    _view(tab, "profile")
    assert "trimmed" in _placeholder(tab)
