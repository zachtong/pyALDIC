"""The embedded chart's line views: what they draw, not how it looks."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from al_dic.gui.widgets.mpl_chart import MplChart


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(["pyALDIC-tests"])


@pytest.fixture
def chart(qapp):
    return MplChart()


def _ax(chart):
    return chart.figure.axes[0]


def test_a_profile_keeps_its_gaps(chart):
    x = np.arange(5.0)
    y = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    chart.plot_profile(x, y, colour="#ff0000", label="P1", x_label="d", y_label="u")
    (line,) = [ln for ln in _ax(chart).get_lines() if ln.get_gid() == "current"]
    assert np.isnan(line.get_ydata()[2]), "a gap is drawn as a gap, not bridged"
    assert chart.has_data


def test_a_profile_shades_consumed_material_once_per_stretch(chart):
    x = np.arange(6.0)
    consumed = np.array([False, True, True, False, True, False])
    chart.plot_profile(x, np.zeros(6), colour="#ff0000", label="P1",
                       x_label="d", y_label="u", consumed=consumed,
                       consumed_label="crack")
    spans = [p for p in _ax(chart).patches]
    assert len(spans) == 2
    labels = [t.get_text() for t in _ax(chart).get_legend().get_texts()]
    assert labels.count("crack") == 1


def test_a_profile_draws_the_other_frames_behind(chart):
    x = np.arange(3.0)
    chart.plot_profile(x, np.ones(3), colour="#ff0000", label="P1",
                       x_label="d", y_label="u",
                       others=[np.zeros(3), 2 * np.ones(3)])
    others = [ln for ln in _ax(chart).get_lines() if ln.get_gid() == "other"]
    current = [ln for ln in _ax(chart).get_lines() if ln.get_gid() == "current"]
    assert len(others) == 2
    assert all(o.get_zorder() < current[0].get_zorder() for o in others)


def test_a_kymograph_places_frames_and_samples_on_their_cells(chart):
    values = np.arange(12.0).reshape(3, 4)             # 3 samples x 4 frames
    chart.plot_kymograph(values, x=np.arange(1.0, 5.0), distance=np.array([0.0, 5.0, 10.0]),
                         x_label="Frame", y_label="d", value_label="u")
    image = next(im for im in _ax(chart).get_images() if im.get_gid() == "values")
    assert image.get_extent() == pytest.approx([0.5, 4.5, -2.5, 12.5])
    assert np.asarray(image.get_array())[2, 3] == 11.0


def test_a_kymograph_fixed_range_is_honoured(chart):
    chart.plot_kymograph(np.zeros((2, 2)), x=np.arange(1.0, 3.0),
                         distance=np.array([0.0, 1.0]), x_label="f", y_label="d",
                         value_label="u", vmin=-2.0, vmax=3.0)
    image = next(im for im in _ax(chart).get_images() if im.get_gid() == "values")
    assert image.get_clim() == (-2.0, 3.0)


def test_a_kymograph_without_cracks_has_no_band(chart):
    chart.plot_kymograph(np.zeros((2, 2)), x=np.arange(1.0, 3.0),
                         distance=np.array([0.0, 1.0]), x_label="f", y_label="d",
                         value_label="u", consumed=np.zeros((2, 2), bool))
    assert [im.get_gid() for im in _ax(chart).get_images()] == ["values"]


def test_the_frame_cursor_moves_on_a_kymograph(chart):
    chart.plot_kymograph(np.zeros((2, 3)), x=np.arange(1.0, 4.0),
                         distance=np.array([0.0, 1.0]), x_label="f", y_label="d",
                         value_label="u", cursor_x=2.0)
    assert chart._cursor.get_xdata()[0] == 2.0
    chart.set_cursor(3.0)
    assert chart._cursor.get_xdata()[0] == 3.0
