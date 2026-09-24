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


# --- publication output ----------------------------------------------------------

def _curves(chart):
    from al_dic.analysis.series import FrameStatus
    from al_dic.gui.widgets.mpl_chart import Curve

    x = np.arange(1.0, 6.0)
    chart.plot_curves(
        [Curve(label="P1", colour="#ef4444", x=x, y=x * 0.1,
               status=[FrameStatus.OK] * 5)],
        x_label="Frame", y_label="u (px)", cursor_x=3.0)


def test_an_exported_chart_is_light_whatever_the_screen_shows(chart, tmp_path):
    """The UI is dark; a figure for a paper is not."""
    from PySide6.QtGui import QImage

    _curves(chart)
    out = tmp_path / "chart.png"
    chart.export_figure(out)
    image = QImage(str(out))
    assert image.pixelColor(2, 2).name() == "#ffffff"
    assert chart.figure.get_facecolor()[:3] != (1.0, 1.0, 1.0), "the screen stays dark"


def test_a_raster_export_is_300_dpi_at_a_papers_width(chart, tmp_path):
    from PySide6.QtGui import QImage

    _curves(chart)
    out = tmp_path / "chart.png"
    chart.export_figure(out)
    image = QImage(str(out))
    assert abs(image.width() - round(16 / 2.54 * 300)) <= 2
    assert abs(image.height() - round(10 / 2.54 * 300)) <= 2


def test_vector_exports_keep_their_text_editable(chart, tmp_path):
    _curves(chart)
    svg = tmp_path / "chart.svg"
    chart.export_figure(svg)
    content = svg.read_text(encoding="utf-8")
    assert "<text" in content and ">P1<" in content.replace("\n", "")
    pdf = tmp_path / "chart.pdf"
    chart.export_figure(pdf)
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_the_frame_cursor_is_left_out_of_an_export(chart, tmp_path):
    _curves(chart)
    figure = chart.publication_figure()
    ax = figure.axes[0]
    assert not [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]


def test_a_kymograph_exports_with_its_colorbar(chart, tmp_path):
    chart.plot_kymograph(np.arange(6.0).reshape(2, 3), x=np.arange(1.0, 4.0),
                         distance=np.array([0.0, 1.0]), x_label="f", y_label="d",
                         value_label="u (px)")
    figure = chart.publication_figure()
    assert len(figure.axes) == 2, "plot and colorbar"
    assert figure.axes[1].get_ylabel() == "u (px)"


def test_nothing_to_export_before_anything_is_drawn(chart, tmp_path):
    chart.clear("nothing yet")
    with pytest.raises(ValueError):
        chart.export_figure(tmp_path / "empty.png")


def test_the_chart_can_be_copied_as_an_image(chart):
    _curves(chart)
    image = chart.publication_image()
    assert not image.isNull()
    assert image.pixelColor(2, 2).name() == "#ffffff"


# --- the plotted numbers -------------------------------------------------------

def test_the_plotted_curves_come_out_as_a_table(chart):
    from al_dic.analysis.series import FrameStatus
    from al_dic.gui.widgets.mpl_chart import Curve

    x = np.array([1.0, 2.0, 3.0])
    chart.plot_curves(
        [Curve(label="P1 · gaps", name="P1", colour="#ef4444", x=x,
               y=np.array([0.0, np.nan, 0.25]), status=[FrameStatus.OK] * 3),
         Curve(label="P2", colour="#22c55e", x=x, y=np.array([1.0, 2.0, 3.0]),
               status=[FrameStatus.OK] * 3)],
        x_label="Frame", y_label="u (px)")
    rows = chart.plotted_table()
    assert rows[0] == ["Frame", "P1 — u (px)", "P2 — u (px)"]
    assert rows[1:] == [["1", "0", "1"], ["2", "", "2"], ["3", "0.25", "3"]]


def test_a_profile_table_is_distance_and_the_current_frame(chart):
    chart.plot_profile(np.array([0.0, 5.0]), np.array([1.0, 2.0]), colour="#fff",
                       label="P1, frame 3", x_label="d (px)", y_label="u (px)",
                       others=[np.zeros(2)])
    assert chart.plotted_table() == [
        ["d (px)", "P1, frame 3 — u (px)"], ["0", "1"], ["5", "2"]]


def test_a_kymograph_table_is_a_row_per_sample_and_a_column_per_frame(chart):
    chart.plot_kymograph(np.array([[1.0, 2.0], [3.0, np.nan]]), x=np.array([1.0, 2.0]),
                         distance=np.array([0.0, 4.0]), x_label="Frame",
                         y_label="d (px)", value_label="u (px)")
    assert chart.plotted_table() == [
        ["d (px) \\ Frame", "1", "2"], ["0", "1", "2"], ["4", "3", ""]]


def test_there_is_no_table_without_a_plot(chart):
    chart.clear("nothing yet")
    with pytest.raises(ValueError):
        chart.plotted_table()


def test_a_selected_probe_is_not_emphasised_on_the_page(chart):
    """Emphasis marks the selection on screen; a paper has no selection."""
    from al_dic.analysis.series import FrameStatus
    from al_dic.gui.widgets.mpl_chart import Curve

    x = np.arange(1.0, 4.0)
    chart.plot_curves(
        [Curve(label="P1", colour="#ef4444", x=x, y=x, status=[FrameStatus.OK] * 3,
               emphasised=True),
         Curve(label="P2", colour="#22c55e", x=x, y=2 * x, status=[FrameStatus.OK] * 3)],
        x_label="Frame", y_label="u")
    screen = {ln.get_label(): ln.get_linewidth() for ln in chart.figure.axes[0].get_lines()}
    assert screen["P1"] > screen["P2"]
    page = {ln.get_label(): ln.get_linewidth()
            for ln in chart.publication_figure().axes[0].get_lines()}
    assert page["P1"] == page["P2"]


def test_curves_on_their_own_x_copy_as_column_pairs(chart):
    """Two probes in a stress-strain chart do not share their strains."""
    from al_dic.analysis.series import FrameStatus
    from al_dic.gui.widgets.mpl_chart import Curve

    chart.plot_curves(
        [Curve(label="E1", colour="#ef4444", x=np.array([0.0, 1.0]),
               y=np.array([0.0, 5.0]), status=[FrameStatus.OK] * 2),
         Curve(label="E2", colour="#22c55e", x=np.array([0.0, 2.0, 3.0]),
               y=np.array([0.0, 5.0, 7.5]), status=[FrameStatus.OK] * 3)],
        x_label="ε (%)", y_label="Stress (MPa)", integer_x=False)
    assert chart.plotted_table() == [
        ["E1 — ε (%)", "E1 — Stress (MPa)", "E2 — ε (%)", "E2 — Stress (MPa)"],
        ["0", "0", "0", "0"], ["1", "5", "2", "5"], ["", "", "3", "7.5"]]


def test_the_frame_ring_stays_off_the_page(chart):
    from al_dic.analysis.series import FrameStatus
    from al_dic.gui.widgets.mpl_chart import Curve

    chart.plot_curves([Curve(label="E1", colour="#ef4444", x=np.array([0.0, 1.0]),
                             y=np.array([0.0, 5.0]), status=[FrameStatus.OK] * 2,
                             mark=1)], x_label="ε", y_label="σ", integer_x=False)
    assert [ln for ln in chart.figure.axes[0].get_lines() if ln.get_gid() == "frame_mark"]
    page = chart.publication_figure().axes[0]
    assert not [ln for ln in page.get_lines() if ln.get_gid() == "frame_mark"]


def test_a_kymograph_legend_has_a_backing_to_read_over_the_image(chart):
    """Dark type straight on a dark colormap could not be read on a page."""
    chart.plot_kymograph(np.zeros((2, 2)), x=np.arange(1.0, 3.0),
                         distance=np.array([0.0, 1.0]), x_label="f", y_label="d",
                         value_label="u", consumed=np.ones((2, 2), bool),
                         consumed_label="crack")
    frame = chart.publication_figure().axes[0].get_legend().get_frame()
    assert frame.get_alpha() >= 0.5
