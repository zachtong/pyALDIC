"""Selecting, moving and reshaping probes on the Analysis canvas.

Driven through real mouse events on the viewport, because what breaks in
canvas code is the mapping between screen and image -- which calling the
handlers with scene coordinates would skip.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from al_dic.analysis.probes import AreaGeom, LineGeom, PointGeom, ProbeSet, replace
from al_dic.gui.panels.probe_canvas import ProbeCanvas, ProbeLabel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(["pyALDIC-tests"])


@pytest.fixture
def canvas(qapp):
    c = ProbeCanvas()
    c.resize(420, 420)
    c.set_image(np.zeros((200, 200, 3), dtype=np.uint8))
    c.show()
    QApplication.processEvents()
    c.resetTransform()                      # one screen pixel per image pixel
    c.centerOn(100.0, 100.0)
    yield c
    c.close()


def _probes(*specs) -> ProbeSet:
    ps = ProbeSet()
    for kind, geom in specs:
        ps.add(kind, geom)
    return ps


def _at(canvas, x: float, y: float) -> QPoint:
    return canvas.mapFromScene(QPointF(x, y))


def _press(canvas, x, y):
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, _at(canvas, x, y))


def _move(canvas, x, y):
    QTest.mouseMove(canvas.viewport(), _at(canvas, x, y))


def _release(canvas, x, y):
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, _at(canvas, x, y))


def _drag(canvas, start, end, steps=4):
    _press(canvas, *start)
    for i in range(1, steps + 1):
        t = i / steps
        _move(canvas, start[0] + t * (end[0] - start[0]),
              start[1] + t * (end[1] - start[1]))
    _release(canvas, *end)


def _record(signal) -> list:
    seen: list = []
    signal.connect(lambda *args: seen.append(args if len(args) > 1 else args[0]))
    return seen


def _labels(canvas) -> list[str]:
    return sorted(i.text() for i in canvas.scene().items() if isinstance(i, ProbeLabel))


# --- selection -------------------------------------------------------------

def test_clicking_a_probe_selects_it(canvas):
    canvas.set_probes(_probes(("point", PointGeom(50.0, 50.0))), None)
    selected = _record(canvas.probe_selected)
    edited = _record(canvas.probe_edited)
    _press(canvas, 51, 50)
    _release(canvas, 51, 50)
    assert selected == [1]
    assert edited == [], "a click is not an edit"


def test_clicking_empty_space_clears_the_selection(canvas):
    canvas.set_probes(_probes(("point", PointGeom(50.0, 50.0))), 1)
    selected = _record(canvas.probe_selected)
    _press(canvas, 150, 150)
    _release(canvas, 150, 150)
    assert selected == [None]


def test_hidden_probes_cannot_be_selected(canvas):
    ps = _probes(("point", PointGeom(50.0, 50.0)))
    ps.replace(replace(ps.get(1), visible=False))
    canvas.set_probes(ps, None)
    selected = _record(canvas.probe_selected)
    _press(canvas, 50, 50)
    _release(canvas, 50, 50)
    assert 1 not in selected


def test_an_armed_tool_places_rather_than_selects(canvas):
    canvas.set_probes(_probes(("point", PointGeom(50.0, 50.0))), None)
    selected = _record(canvas.probe_selected)
    placed = _record(canvas.probe_requested)
    canvas.set_tool("point")
    _press(canvas, 50, 50)
    _release(canvas, 50, 50)
    assert selected == []
    assert placed == [("point", PointGeom(50.0, 50.0))]


def test_double_clicking_a_probe_asks_to_rename_it(canvas):
    canvas.set_probes(_probes(("point", PointGeom(50.0, 50.0))), None)
    activated = _record(canvas.probe_activated)
    QTest.mouseDClick(canvas.viewport(), Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier, _at(canvas, 50, 50))
    assert activated == [1]


# --- moving ------------------------------------------------------------------

def test_dragging_a_probe_moves_it(canvas):
    canvas.set_probes(_probes(("point", PointGeom(50.0, 50.0))), None)
    edited = _record(canvas.probe_edited)
    _drag(canvas, (50, 50), (70, 60))
    assert len(edited) == 1
    probe_id, geom = edited[0]
    assert probe_id == 1
    assert (geom.x, geom.y) == pytest.approx((70.0, 60.0))


def test_a_region_moves_by_its_interior(canvas):
    canvas.set_probes(_probes(("area", AreaGeom.rect(40.0, 40.0, 80.0, 70.0))), None)
    edited = _record(canvas.probe_edited)
    _drag(canvas, (60, 55), (70, 50))
    (_, geom), = edited
    assert geom.data == pytest.approx((50.0, 35.0, 90.0, 65.0))


def test_a_small_jitter_is_a_click_not_a_drag(canvas):
    """Moving a probe by one pixel because the hand shook is a silent edit."""
    canvas.set_probes(_probes(("point", PointGeom(50.0, 50.0))), None)
    edited = _record(canvas.probe_edited)
    _drag(canvas, (50, 50), (51, 50), steps=1)
    assert edited == []


def test_escape_abandons_a_drag(canvas):
    canvas.set_probes(_probes(("point", PointGeom(50.0, 50.0))), None)
    edited = _record(canvas.probe_edited)
    _press(canvas, 50, 50)
    _move(canvas, 60, 60)
    _move(canvas, 70, 70)
    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    _release(canvas, 70, 70)
    assert edited == []


# --- reshaping -----------------------------------------------------------------

def test_dragging_a_selected_lines_end_reshapes_it(canvas):
    canvas.set_probes(_probes(("line", LineGeom(20.0, 100.0, 120.0, 100.0))), 1)
    edited = _record(canvas.probe_edited)
    _drag(canvas, (120, 100), (140, 120))
    (_, geom), = edited
    assert (geom.x0, geom.y0) == pytest.approx((20.0, 100.0)), "the other end stays"
    assert (geom.x1, geom.y1) == pytest.approx((140.0, 120.0))


def test_an_unselected_line_moves_whole_from_its_end(canvas):
    """Handles belong to the selected probe; elsewhere a press grabs the body."""
    canvas.set_probes(_probes(("line", LineGeom(20.0, 100.0, 120.0, 100.0))), None)
    edited = _record(canvas.probe_edited)
    _drag(canvas, (120, 100), (130, 100))
    (_, geom), = edited
    assert (geom.x0, geom.x1) == pytest.approx((30.0, 130.0))


def test_a_reshape_that_would_collapse_the_probe_keeps_the_last_good_shape(canvas):
    canvas.set_probes(_probes(("line", LineGeom(20.0, 100.0, 120.0, 100.0))), 1)
    edited = _record(canvas.probe_edited)
    _press(canvas, 120, 100)
    _move(canvas, 60, 100)
    _move(canvas, 20, 100)              # onto the other end: refused
    _release(canvas, 20, 100)
    (_, geom), = edited
    assert (geom.x1, geom.y1) == pytest.approx((60.0, 100.0))


# --- drawing -------------------------------------------------------------------

def _device_width(canvas, item) -> float:
    return item.deviceTransform(canvas.viewportTransform()).mapRect(
        item.boundingRect()).width()


def test_markers_and_labels_keep_their_size_when_zooming(canvas):
    canvas.set_probes(_probes(("point", PointGeom(50.0, 50.0))), None)
    label = next(i for i in canvas.scene().items() if isinstance(i, ProbeLabel))
    before = _device_width(canvas, label)
    canvas.scale(4.0, 4.0)
    assert _device_width(canvas, label) == pytest.approx(before)


def test_a_line_is_labelled_with_its_gauge_length(canvas):
    canvas.set_length_format(lambda px: f"{px:.1f} px")
    canvas.set_probes(_probes(("line", LineGeom(0.0, 0.0, 30.0, 40.0))), None)
    assert _labels(canvas) == ["P1 · 50.0 px"]


def test_the_length_label_follows_a_reshape_live(canvas):
    canvas.set_length_format(lambda px: f"{px:.0f} px")
    canvas.set_probes(_probes(("line", LineGeom(20.0, 100.0, 120.0, 100.0))), 1)
    _press(canvas, 120, 100)
    _move(canvas, 130, 100)
    _move(canvas, 140, 100)
    assert _labels(canvas) == ["P1 · 120 px"], "updated before the release"
    _release(canvas, 140, 100)


def test_the_extensometer_and_crack_gauge_tools_draw_lines(canvas):
    placed = _record(canvas.probe_requested)
    for tool in ("extensometer", "crack_gauge"):
        canvas.set_tool(tool)
        canvas._pending.extend([QPointF(10.0, 10.0), QPointF(10.0, 60.0)])
        canvas._commit_if_complete()
    assert [kind for kind, _ in placed] == ["line", "line"]
    assert placed[0][1] == LineGeom(10.0, 10.0, 10.0, 60.0)


# --- navigation ----------------------------------------------------------------

def test_a_left_drag_on_empty_space_pans(qapp):
    c = ProbeCanvas()
    c.resize(300, 300)
    c.set_image(np.zeros((2000, 2000, 3), dtype=np.uint8))
    c.show()
    QApplication.processEvents()
    c.resetTransform()
    c.centerOn(1000.0, 1000.0)
    before = c.horizontalScrollBar().value()
    start = QPoint(150, 150)
    QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, start)
    for dx in (10, 20, 40):
        QTest.mouseMove(c.viewport(), start + QPoint(dx, 0))
    QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, start + QPoint(40, 0))
    assert c.horizontalScrollBar().value() == before - 40
    c.close()
