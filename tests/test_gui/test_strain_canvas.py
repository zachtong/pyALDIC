"""Tests for StrainCanvas -- read-only QGraphicsView with overlay layer."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QPainter, QPixmap, QColor
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.panels.strain_canvas import StrainCanvas


@pytest.fixture
def canvas():
    return StrainCanvas()


def _make_red_pixmap(w: int = 32, h: int = 32) -> QPixmap:
    pm = QPixmap(w, h)
    pm.fill(QColor(255, 0, 0, 255))
    return pm


def test_set_image_updates_pixmap_and_scene_rect(canvas):
    """A grayscale->RGB image becomes the background and sets the scene rect."""
    h, w = 64, 96
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # red channel
    canvas.set_image(rgb)

    bg_pm = canvas._bg_item.pixmap()
    assert not bg_pm.isNull()
    assert bg_pm.width() == w
    assert bg_pm.height() == h

    rect = canvas._scene.sceneRect()
    assert rect == QRectF(0, 0, w, h)


def test_set_overlay_pixmap_updates_overlay_layer(canvas):
    """set_overlay_pixmap installs the pixmap on the overlay item."""
    pm = _make_red_pixmap(40, 30)
    canvas.set_overlay_pixmap(pm)
    assert not canvas._overlay_item.pixmap().isNull()
    assert canvas._overlay_item.pixmap().width() == 40
    assert canvas._overlay_item.pixmap().height() == 30


def test_set_overlay_alpha_applied(canvas):
    """The overlay opacity tracks set_overlay_alpha."""
    canvas.set_overlay_alpha(0.25)
    assert canvas._overlay_item.opacity() == pytest.approx(0.25)
    canvas.set_overlay_alpha(0.9)
    assert canvas._overlay_item.opacity() == pytest.approx(0.9)


def test_set_overlay_alpha_clamped(canvas):
    """Alpha is clamped to [0, 1] so callers cannot push out-of-range values."""
    canvas.set_overlay_alpha(-0.5)
    assert canvas._overlay_item.opacity() == pytest.approx(0.0)
    canvas.set_overlay_alpha(2.0)
    assert canvas._overlay_item.opacity() == pytest.approx(1.0)


def test_set_overlay_pos(canvas):
    """The overlay can be offset from (0, 0) -- needed when the field
    grid origin does not coincide with the image origin."""
    canvas.set_overlay_pos(12.5, -3.0)
    pos = canvas._overlay_item.pos()
    assert pos.x() == pytest.approx(12.5)
    assert pos.y() == pytest.approx(-3.0)


def test_clear_overlay_resets_pixmap(canvas):
    """clear_overlay drops the overlay pixmap (back to a null pixmap)."""
    canvas.set_overlay_pixmap(_make_red_pixmap())
    canvas.clear_overlay()
    assert canvas._overlay_item.pixmap().isNull()


def test_no_brush_or_roi_api(canvas):
    """StrainCanvas is read-only -- it must NOT expose any brush, ROI,
    or drawing tool APIs from ImageCanvas."""
    forbidden = (
        "set_brush_controller",
        "set_brush_radius",
        "set_brush_mode",
        "set_roi_controller",
        "set_drawing_mode",
        "set_tool",
        "update_roi_overlay",
    )
    for name in forbidden:
        assert not hasattr(canvas, name), (
            f"StrainCanvas must not expose {name!r} -- it is read-only"
        )


def test_zvalues_layered_correctly(canvas):
    """Background must sit below the overlay so the field paints on top."""
    assert canvas._bg_item.zValue() < canvas._overlay_item.zValue()
