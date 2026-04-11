"""Tests for ROIToolbar — Refine brush dropdown signals.

Only tests the public Qt-signal contract; the popup widgets themselves
are exercised by the manual GUI workflow.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from al_dic.gui.widgets.roi_toolbar import ROIToolbar


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


def test_brush_radius_max_is_500(qapp) -> None:
    """Allow large brushes for big images / coarse refinement zones."""
    tb = ROIToolbar()
    assert tb._brush_radius_spin.maximum() == 500


def test_brush_radius_changed_signal_live(qapp) -> None:
    """Spinbox value change must emit brush_radius_changed live so the
    canvas radius updates without re-clicking Paint/Erase."""
    tb = ROIToolbar()
    received: list[int] = []
    tb.brush_radius_changed.connect(lambda r: received.append(r))
    tb._brush_radius_spin.setValue(42)
    assert received == [42]


def test_erase_passes_current_radius(qapp) -> None:
    """Erase must pass the current spinbox value, just like Paint."""
    tb = ROIToolbar()
    received: list[tuple[str, int]] = []
    tb.brush_requested.connect(lambda m, r: received.append((m, r)))
    tb._brush_radius_spin.setValue(33)
    tb._on_brush_selected("erase", tb._brush_radius_spin.value())
    assert received == [("erase", 33)]
