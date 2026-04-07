"""Tests for ROIToolbar — Refine brush dropdown signals.

Only tests the public Qt-signal contract; the popup widgets themselves
are exercised by the manual GUI workflow.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from staq_dic.gui.widgets.roi_toolbar import ROIToolbar


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_brush_requested_signal(qapp) -> None:
    tb = ROIToolbar()
    received: list[tuple[str, int]] = []
    tb.brush_requested.connect(lambda mode, radius: received.append((mode, radius)))
    tb._on_brush_selected("paint", 12)
    assert received == [("paint", 12)]


def test_brush_clear_signal(qapp) -> None:
    tb = ROIToolbar()
    received: list[bool] = []
    tb.brush_clear_requested.connect(lambda: received.append(True))
    tb._brush_clear_btn.click()
    assert received == [True]


def test_brush_button_exists(qapp) -> None:
    tb = ROIToolbar()
    # Public attribute used by MainWindow when wiring signals
    assert hasattr(tb, "_btn_refine")


def test_brush_radius_default(qapp) -> None:
    tb = ROIToolbar()
    assert tb._brush_radius_spin.value() > 0


def test_deactivate_resets_brush_highlight(qapp) -> None:
    tb = ROIToolbar()
    tb._on_brush_selected("paint", 8)
    assert tb._active_mode == "brush_paint"
    tb.deactivate()
    assert tb._active_mode is None
