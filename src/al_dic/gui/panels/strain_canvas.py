"""Read-only canvas for the strain post-processing window.

Stripped-down :class:`QGraphicsView` with two layers:

    z=0  Background image  (``_bg_item``)
    z=5  Field overlay     (``_overlay_item``)

Supports wheel zoom and middle-button pan ONLY. All ROI editing,
brush refinement, shape drawing, and tool switching from
:class:`canvas_area.ImageCanvas` are intentionally absent -- the strain
window never edits its inputs, it only views them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from al_dic.gui.theme import COLORS

# Match canvas_area zoom defaults so the two views feel identical.
_ZOOM_FACTOR = 1.15
_ZOOM_MIN = 0.10
_ZOOM_MAX = 20.0


class StrainCanvas(QGraphicsView):
    """Read-only zoomable image canvas with a single overlay layer."""

    view_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # --- Layer 0: background image ---
        self._bg_item = QGraphicsPixmapItem()
        self._scene.addItem(self._bg_item)
        self._bg_item.setZValue(0)

        # --- Layer 5: field overlay (semi-transparent colormap) ---
        self._overlay_item = QGraphicsPixmapItem()
        self._scene.addItem(self._overlay_item)
        self._overlay_item.setZValue(5)
        self._overlay_item.setOpacity(0.7)

        # --- Render settings (mirror ImageCanvas) ---
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setBackgroundBrush(QBrush(QColor(COLORS.BG_CANVAS)))

        # --- Pan state (middle button only) ---
        self._panning = False
        self._pan_start = QPointF()
        self._zoom_level: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, rgb: NDArray[np.uint8]) -> None:
        """Display *rgb* (H, W, 3) uint8 array as the background image."""
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        # .copy() so QImage owns the data after numpy array may be freed
        self._bg_item.setPixmap(QPixmap.fromImage(qimg.copy()))
        self._scene.setSceneRect(QRectF(0, 0, w, h))

    def set_overlay_pixmap(self, pixmap: QPixmap) -> None:
        """Install *pixmap* on the overlay layer."""
        self._overlay_item.setPixmap(pixmap)

    def set_overlay_alpha(self, alpha: float) -> None:
        """Set the overlay opacity. *alpha* is clamped to ``[0, 1]``."""
        a = max(0.0, min(1.0, float(alpha)))
        self._overlay_item.setOpacity(a)

    def set_overlay_pos(self, x: float, y: float) -> None:
        """Offset the overlay item from the scene origin."""
        self._overlay_item.setPos(float(x), float(y))

    def clear_overlay(self) -> None:
        """Drop the overlay pixmap (back to a null pixmap)."""
        self._overlay_item.setPixmap(QPixmap())

    def zoom_in(self) -> None:
        self._apply_zoom(_ZOOM_FACTOR)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / _ZOOM_FACTOR)

    def zoom_to_100(self) -> None:
        """Reset zoom to 100% (1:1 pixels)."""
        self.resetTransform()
        self._zoom_level = 1.0
        self.view_changed.emit()

    def fit_to_view(self) -> None:
        """Reset the view to fit the current scene rect."""
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = self.transform().m11()
        self.view_changed.emit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_zoom(self, factor: float) -> None:
        new_level = self._zoom_level * factor
        if new_level < _ZOOM_MIN or new_level > _ZOOM_MAX:
            return
        self.scale(factor, factor)
        self._zoom_level = new_level
        self.view_changed.emit()

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta > 0:
            self._apply_zoom(_ZOOM_FACTOR)
        elif delta < 0:
            self._apply_zoom(1.0 / _ZOOM_FACTOR)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            self._panning
            and event.button() == Qt.MouseButton.MiddleButton
        ):
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802
        super().scrollContentsBy(dx, dy)
        self.view_changed.emit()
