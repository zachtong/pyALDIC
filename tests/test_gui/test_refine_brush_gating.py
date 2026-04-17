"""Tests for the Refine-brush gating (frame-1-only).

Refine strokes are warped forward from frame 1, so painting on any
later frame produces results in the wrong coordinate system. The
button is gated to frame 1 via ``LeftSidebar.current_frame_changed``
-> ``ROIToolbar.set_refine_brush_enabled``. These tests pin the
visual gating down so future stylesheet refactors do not silently
erase the disabled appearance (which used to happen when a custom
``QPushButton { ... }`` rule shadowed Qt's default gray-out).
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.widgets.roi_toolbar import ROIToolbar


@pytest.fixture
def toolbar():
    return ROIToolbar()


def test_refine_button_enabled_by_default(toolbar):
    assert toolbar._btn_refine.isEnabled() is True


def test_refine_disabled_state_toggle(toolbar):
    toolbar.set_refine_brush_enabled(False)
    assert toolbar._btn_refine.isEnabled() is False
    toolbar.set_refine_brush_enabled(True)
    assert toolbar._btn_refine.isEnabled() is True


def test_refine_tooltip_switches_on_disable(toolbar):
    """The disabled tooltip must tell the user how to re-enable it."""
    toolbar.set_refine_brush_enabled(False)
    tip = toolbar._btn_refine.toolTip()
    assert "frame 1" in tip.lower()
    assert "only" in tip.lower() or "switch" in tip.lower()


def test_base_style_has_disabled_rule():
    """Regression guard: the custom QPushButton style must include a
    :disabled rule so setEnabled(False) produces a visible gray-out.

    Without this rule, the custom ``QPushButton { ... }`` selector
    shadows Qt's default disabled palette and the button looks
    identical whether or not it is clickable.
    """
    style = ROIToolbar._BASE_STYLE
    assert "QPushButton:disabled" in style, (
        "ROIToolbar._BASE_STYLE must define a :disabled rule so that "
        "gated buttons (e.g. Refine brush on non-frame-1) are visually "
        "distinguishable from enabled ones."
    )
