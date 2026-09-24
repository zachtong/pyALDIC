"""The machine's load in the Analysis tab: axes, export, the import button.

The run is u = 0.01 * f * x (frames 0..3) and the machine logged loads 0, 10,
20 and 30 N at frames 1..4, so every value below can be checked by hand.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog

from al_dic.analysis.load_data import LoadData, LoadSync, LoadTable
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
    _run(state)
    t = AnalysisTab(state)
    t._on_probe_placed("point", PointGeom(20.0, 20.0))
    _plot(t, "disp_u")
    return t


def _machine(area=None) -> LoadData:
    table = LoadTable(("frame", "F"), (np.array([1.0, 2.0, 3.0, 4.0]),
                                       np.array([0.0, 10.0, 20.0, 30.0])), "run7.csv")
    return LoadData(table, LoadSync(mode="frame", load_column="F", frame_column="frame"),
                    area_mm2=area)


def _axes(tab) -> list[str]:
    return [tab._x_box.itemData(i) for i in range(tab._x_box.count())]


def _against(tab, key: str) -> None:
    tab._x_box.setCurrentIndex(tab._x_box.findData(key))
    tab._refresh()


def _ax(tab):
    return tab._chart.figure.axes[0]


# --- axes -------------------------------------------------------------------------

def test_load_is_offered_as_an_axis_once_there_is_a_record(tab, state):
    assert "load" not in _axes(tab)
    state.set_load_data(_machine())
    assert "load" in _axes(tab) and "stress" not in _axes(tab), "no area, no stress"
    state.set_load_data(_machine(area=2.0))
    assert "stress" in _axes(tab)
    state.set_load_data(None)
    assert "load" not in _axes(tab)


def test_a_curve_against_load(tab, state):
    state.set_load_data(_machine())
    _against(tab, "load")
    line = _ax(tab).get_lines()[0]
    np.testing.assert_allclose(line.get_xdata(), [0.0, 10.0, 20.0, 30.0])
    assert "Load (N)" in _ax(tab).get_xlabel()


def test_a_curve_against_engineering_stress(tab, state):
    state.set_load_data(_machine(area=2.0))
    _against(tab, "stress")
    np.testing.assert_allclose(_ax(tab).get_lines()[0].get_xdata(), [0.0, 5.0, 10.0, 15.0])
    assert "Stress (MPa)" in _ax(tab).get_xlabel()


def test_a_click_on_a_load_axis_goes_to_the_nearest_frame(tab, state):
    state.set_load_data(_machine())
    _against(tab, "load")
    asked = []
    tab.frame_requested.connect(asked.append)
    tab._on_chart_clicked(19.0)
    assert asked == [2]


def test_a_kymograph_keeps_frames_on_its_x_axis(tab, state):
    """Loads are not evenly spaced, and a kymograph's cells must be."""
    tab._on_probe_placed("line", LineGeom(4.0, 20.0, 36.0, 20.0))
    state.set_load_data(_machine())
    _against(tab, "load")
    tab._view_bar.setCurrentIndex(tab._view_bar_index("kymograph"))
    tab._refresh()
    image = next(im for im in _ax(tab).get_images() if im.get_gid() == "values")
    assert image.get_extent()[:2] == pytest.approx([0.5, 4.5])
    index = tab._x_box.findData("load")
    assert not tab._x_box.model().item(index).isEnabled()


def test_the_axis_falls_back_to_frames_when_the_record_goes(tab, state):
    state.set_load_data(_machine())
    _against(tab, "load")
    state.set_load_data(None)
    tab._refresh()
    assert tab._x_box.currentData() == "frame"


# --- export ---------------------------------------------------------------------

def test_the_probe_csv_carries_load_and_stress(tab, state, tmp_path, monkeypatch):
    state.set_load_data(_machine(area=2.0))
    tab._refresh()
    out = tmp_path / "probes.csv"
    monkeypatch.setattr("al_dic.gui.panels.analysis.tab.QFileDialog.getSaveFileName",
                        lambda *a, **k: (str(out), ""))
    tab._on_export_csv()
    with open(out, encoding="utf-8-sig") as fh:
        comments = [ln for ln in fh if ln.startswith("#")]
    with open(out, encoding="utf-8-sig") as fh:
        rows = list(csv.reader(ln for ln in fh if not ln.startswith("#")))
    header = rows[0]
    assert "load_N" in header and "stress_MPa" in header
    load = [float(r[header.index("load_N")]) for r in rows[1:]]
    stress = [float(r[header.index("stress_MPa")]) for r in rows[1:]]
    assert load == [0.0, 10.0, 20.0, 30.0] and stress == [0.0, 5.0, 10.0, 15.0]
    assert any("run7.csv" in c for c in comments)


# --- the button --------------------------------------------------------------------

def test_the_button_stores_what_the_dialog_returns(tab, state, monkeypatch):
    record = _machine(area=3.0)

    class FakeDialog:
        def __init__(self, current, n_frames, frame_rate, parent):
            self.args = (current, n_frames, frame_rate)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def load_data(self):
            return record

    monkeypatch.setattr("al_dic.gui.panels.analysis.tab.LoadDataDialog", FakeDialog)
    tab._on_load_data()
    assert state.load_data is record


def test_cancelling_the_dialog_keeps_the_record(tab, state, monkeypatch):
    state.set_load_data(_machine())
    before = state.load_data

    class FakeDialog:
        def __init__(self, *args):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr("al_dic.gui.panels.analysis.tab.LoadDataDialog", FakeDialog)
    tab._on_load_data()
    assert state.load_data is before


# --- the stress-strain view --------------------------------------------------------

@pytest.fixture
def gauge_tab(state):
    """An extensometer reading 0, 1, 2, 3 % at frames 0..3."""
    _run(state)
    t = AnalysisTab(state)
    t._on_probe_placed("line", LineGeom(4.0, 20.0, 36.0, 20.0))
    t._unit_box.setCurrentIndex(t._unit_box.findData("percent"))
    _plot(t, "strain", gauge=True)
    t._view_bar.setCurrentIndex(t._view_bar_index("stress_strain"))
    t._refresh()
    return t


def _curve(tab):
    return next(ln for ln in _ax(tab).get_lines() if ln.get_gid() != "frame_mark")


def test_stress_strain_asks_for_the_machine_record_first(gauge_tab):
    texts = " ".join(t.get_text() for t in _ax(gauge_tab).texts)
    assert "Load data" in texts


def test_stress_against_extensometer_strain(gauge_tab, state):
    state.set_load_data(_machine(area=2.0))
    gauge_tab._refresh()                    # hidden, it only noted the change
    line = _curve(gauge_tab)
    np.testing.assert_allclose(line.get_xdata(), [0.0, 1.0, 2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(line.get_ydata(), [0.0, 5.0, 10.0, 15.0])
    assert "Stress (MPa)" in _ax(gauge_tab).get_ylabel()
    assert "(%)" in _ax(gauge_tab).get_xlabel()


def test_without_an_area_it_is_load_against_strain(gauge_tab, state):
    state.set_load_data(_machine())
    gauge_tab._refresh()
    np.testing.assert_allclose(_curve(gauge_tab).get_ydata(), [0.0, 10.0, 20.0, 30.0])
    assert "Load (N)" in _ax(gauge_tab).get_ylabel()


def test_the_current_frame_is_marked_on_the_curve(gauge_tab, state):
    state.set_load_data(_machine(area=2.0))
    gauge_tab.set_frame(2)
    gauge_tab._refresh()
    (mark,) = [ln for ln in _ax(gauge_tab).get_lines() if ln.get_gid() == "frame_mark"]
    assert (mark.get_xdata()[0], mark.get_ydata()[0]) == pytest.approx((2.0, 10.0))


def test_stress_strain_takes_no_x_axis_choice_and_no_frame_clicks(gauge_tab, state):
    state.set_load_data(_machine(area=2.0))
    assert gauge_tab._x_box.isHidden()
    asked = []
    gauge_tab.frame_requested.connect(asked.append)
    gauge_tab._on_chart_clicked(2.0)
    assert asked == [], "x is a strain there, not a frame"


def test_stress_strain_data_exports_with_the_load(gauge_tab, state):
    state.set_load_data(_machine(area=2.0))
    gauge_tab._refresh()
    assert gauge_tab._export_btn.isEnabled()


def test_the_axis_box_grows_to_fit_axes_added_later(tab, state):
    """German "Spannung (MPa)" was cut to "Spannu..." in a box sized for "Bild"."""
    from PySide6.QtWidgets import QComboBox

    assert tab._x_box.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToContents
    before = tab._x_box.sizeHint().width()
    state.set_load_data(_machine(area=2.0))
    assert tab._x_box.sizeHint().width() >= before
