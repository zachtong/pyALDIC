"""Center canvas panel — zoomable/pannable QGraphicsView with layered items.

Layers:
    0: Background image (QGraphicsPixmapItem)
    1: Field overlay — semi-transparent colormapped result
    2: ROI mask overlay — blue semi-transparent boolean mask

Supports:
    - Wheel zoom (factor 1.15, 10%–2000%) anchored under mouse
    - Middle-button pan (always) + left-button pan when tool == "pan"
    - Shape tool forwarding: rect / polygon / circle drawing with add/cut modes
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
    QMouseEvent,
    QKeyEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from staq_dic.gui.app_state import AppState
from staq_dic.gui.theme import COLORS
from staq_dic.gui.widgets.frame_navigator import FrameNavigator

from staq_dic.core.data_structures import split_uv
from staq_dic.gui.controllers.viz_controller import VizController

if TYPE_CHECKING:
    from staq_dic.gui.controllers.image_controller import ImageController
    from staq_dic.gui.controllers.roi_controller import ROIController


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ZOOM_FACTOR = 1.15
_ZOOM_MIN = 0.10
_ZOOM_MAX = 20.0

_PEN_ADD = QPen(QColor(59, 130, 246, 200), 2)  # blue
_PEN_CUT = QPen(QColor(239, 68, 68, 200), 2)   # red

_ROI_COLOR_ADD = QColor(59, 130, 246, 80)       # blue semi-transparent
_ROI_COLOR_CUT = QColor(239, 68, 68, 80)        # red semi-transparent


# ---------------------------------------------------------------------------
# ImageCanvas — QGraphicsView
# ---------------------------------------------------------------------------
class ImageCanvas(QGraphicsView):
    """Zoomable, pannable image canvas with layered QGraphicsItems."""

    # Signal emitted when a point is clicked in image coordinates
    point_clicked = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # --- Layer 0: background image ---
        from PySide6.QtWidgets import QGraphicsPixmapItem

        self._bg_item = QGraphicsPixmapItem()
        self._scene.addItem(self._bg_item)
        self._bg_item.setZValue(0)

        # --- Layer 1: field overlay (semi-transparent) ---
        self._overlay_item = QGraphicsPixmapItem()
        self._scene.addItem(self._overlay_item)
        self._overlay_item.setZValue(1)
        self._overlay_item.setOpacity(0.7)

        # --- Layer 2: ROI mask overlay ---
        self._roi_item = QGraphicsPixmapItem()
        self._scene.addItem(self._roi_item)
        self._roi_item.setZValue(2)

        # --- Render settings ---
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

        # --- Interaction state ---
        self._current_tool: str = "select"   # select, pan, rect, polygon, circle
        self._drawing_mode: str = "add"      # add or cut
        self._draw_state: dict | None = None  # in-progress drawing data
        self._preview_items: list = []        # temp graphics items while drawing
        self._roi_ctrl: ROIController | None = None

        # Pan state (middle-button or left-button when tool == "pan")
        self._panning = False
        self._pan_start = QPointF()

        # Current zoom level (1.0 = 100%)
        self._zoom_level: float = 1.0

    # ----- public API -----

    def set_tool(self, tool: str) -> None:
        """Set the active tool: select, pan, rect, polygon, circle."""
        self._current_tool = tool
        self._cancel_drawing()
        if tool == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif tool in ("rect", "polygon", "circle"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_drawing_mode(self, mode: str) -> None:
        """Set drawing mode: 'add' or 'cut'."""
        self._drawing_mode = mode

    def set_roi_controller(self, ctrl: ROIController) -> None:
        """Attach the ROI controller for mask operations."""
        self._roi_ctrl = ctrl

    def set_image(self, rgb: NDArray[np.uint8]) -> None:
        """Display a numpy RGB uint8 array as the background image."""
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        # .copy() so QImage owns the data after numpy array may be freed
        self._bg_item.setPixmap(QPixmap.fromImage(qimg.copy()))
        self._scene.setSceneRect(QRectF(0, 0, w, h))

    def update_roi_overlay(self) -> None:
        """Refresh the ROI overlay from the current ROI controller mask."""
        if self._roi_ctrl is None:
            self._roi_item.setPixmap(QPixmap())
            return
        mask = self._roi_ctrl.mask
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask, :] = [59, 130, 246, 80]  # blue semi-transparent
        bytes_per_line = w * 4
        qimg = QImage(
            rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
        )
        self._roi_item.setPixmap(QPixmap.fromImage(qimg.copy()))

    def fit_to_view(self) -> None:
        """Fit the entire scene into the viewport."""
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = self.transform().m11()

    def zoom_to_100(self) -> None:
        """Reset zoom to 100% (1:1 pixels)."""
        self.resetTransform()
        self._zoom_level = 1.0

    def zoom_in(self) -> None:
        """Zoom in by one step."""
        self._apply_zoom(_ZOOM_FACTOR)

    def zoom_out(self) -> None:
        """Zoom out by one step."""
        self._apply_zoom(1.0 / _ZOOM_FACTOR)

    # ----- zoom helpers -----

    def _apply_zoom(self, factor: float) -> None:
        new_level = self._zoom_level * factor
        if new_level < _ZOOM_MIN or new_level > _ZOOM_MAX:
            return
        self.scale(factor, factor)
        self._zoom_level = new_level

    # ----- wheel zoom -----

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta > 0:
            self._apply_zoom(_ZOOM_FACTOR)
        elif delta < 0:
            self._apply_zoom(1.0 / _ZOOM_FACTOR)

    # ----- mouse events -----

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        # Middle-button pan (always available)
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # Left-button pan when tool == "pan"
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._current_tool == "pan"
        ):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # Shape drawing tools
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._current_tool in ("rect", "polygon", "circle")
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            self._handle_draw_press(scene_pos)
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

        # Drawing preview
        if self._draw_state is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._handle_draw_move(scene_pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._panning = False
            # Restore cursor for current tool
            self.set_tool(self._current_tool)
            event.accept()
            return

        # Shape finalize (rect, circle — release-to-finish)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._draw_state is not None
            and self._current_tool in ("rect", "circle")
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            self._handle_draw_release(scene_pos)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        # Polygon: double-click closes the polygon
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._current_tool == "polygon"
            and self._draw_state is not None
        ):
            self._finalize_polygon()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        # Escape cancels in-progress drawing
        if event.key() == Qt.Key.Key_Escape and self._draw_state is not None:
            self._cancel_drawing()
            event.accept()
            return
        super().keyPressEvent(event)

    # ----- drawing logic -----

    def _handle_draw_press(self, pos: QPointF) -> None:
        pen = _PEN_CUT if self._drawing_mode == "cut" else _PEN_ADD

        if self._current_tool == "rect":
            self._draw_state = {"start": pos}
            rect_item = QGraphicsRectItem(QRectF(pos, pos))
            rect_item.setPen(pen)
            rect_item.setZValue(10)
            self._scene.addItem(rect_item)
            self._preview_items = [rect_item]

        elif self._current_tool == "circle":
            self._draw_state = {"center": pos}
            ellipse = QGraphicsEllipseItem(QRectF(pos, pos))
            ellipse.setPen(pen)
            ellipse.setZValue(10)
            self._scene.addItem(ellipse)
            self._preview_items = [ellipse]

        elif self._current_tool == "polygon":
            if self._draw_state is None:
                # First point
                self._draw_state = {"points": [pos]}
                self._preview_items = []
            else:
                self._draw_state["points"].append(pos)

            # Add a new line segment preview
            if len(self._draw_state["points"]) >= 2:
                pts = self._draw_state["points"]
                line = QGraphicsLineItem(
                    pts[-2].x(), pts[-2].y(), pts[-1].x(), pts[-1].y()
                )
                line.setPen(pen)
                line.setZValue(10)
                self._scene.addItem(line)
                self._preview_items.append(line)

    def _handle_draw_move(self, pos: QPointF) -> None:
        if self._current_tool == "rect" and self._preview_items:
            start = self._draw_state["start"]
            rect = QRectF(start, pos).normalized()
            self._preview_items[0].setRect(rect)

        elif self._current_tool == "circle" and self._preview_items:
            center = self._draw_state["center"]
            dx = pos.x() - center.x()
            dy = pos.y() - center.y()
            radius = math.sqrt(dx * dx + dy * dy)
            self._preview_items[0].setRect(QRectF(
                center.x() - radius, center.y() - radius,
                2 * radius, 2 * radius,
            ))

    def _handle_draw_release(self, pos: QPointF) -> None:
        if self._roi_ctrl is None:
            self._cancel_drawing()
            return

        if self._current_tool == "rect":
            start = self._draw_state["start"]
            x1 = int(min(start.x(), pos.x()))
            y1 = int(min(start.y(), pos.y()))
            x2 = int(max(start.x(), pos.x()))
            y2 = int(max(start.y(), pos.y()))
            self._roi_ctrl.add_rectangle(x1, y1, x2, y2, self._drawing_mode)
            self._finish_drawing()

        elif self._current_tool == "circle":
            center = self._draw_state["center"]
            dx = pos.x() - center.x()
            dy = pos.y() - center.y()
            radius = int(math.sqrt(dx * dx + dy * dy))
            self._roi_ctrl.add_circle(
                int(center.x()), int(center.y()), radius, self._drawing_mode
            )
            self._finish_drawing()

    def _finalize_polygon(self) -> None:
        if self._roi_ctrl is None or self._draw_state is None:
            self._cancel_drawing()
            return
        pts = self._draw_state.get("points", [])
        if len(pts) < 3:
            self._cancel_drawing()
            return
        int_pts = [(int(p.x()), int(p.y())) for p in pts]
        self._roi_ctrl.add_polygon(int_pts, self._drawing_mode)
        self._finish_drawing()

    def _finish_drawing(self) -> None:
        """Clean up preview items and update ROI overlay."""
        self._remove_preview_items()
        self._draw_state = None
        self.update_roi_overlay()
        # Update global state
        if self._roi_ctrl is not None:
            AppState.instance().set_roi_mask(self._roi_ctrl.mask.copy())

    def _cancel_drawing(self) -> None:
        """Cancel in-progress drawing and remove preview items."""
        self._remove_preview_items()
        self._draw_state = None

    def _remove_preview_items(self) -> None:
        for item in self._preview_items:
            self._scene.removeItem(item)
        self._preview_items = []


# ---------------------------------------------------------------------------
# CanvasArea — wrapper widget
# ---------------------------------------------------------------------------
class CanvasArea(QWidget):
    """Canvas wrapper with top toolbar and the ImageCanvas."""

    def __init__(
        self,
        image_ctrl: ImageController,
        viz_ctrl: VizController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_ctrl = image_ctrl
        self._viz_ctrl = viz_ctrl or VizController()
        self._state = AppState.instance()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Top toolbar ---
        toolbar = QWidget()
        toolbar.setFixedHeight(36)
        toolbar.setStyleSheet(
            f"background: {COLORS.BG_PANEL}; "
            f"border-bottom: 1px solid {COLORS.BORDER};"
        )
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 2, 8, 2)
        tb_layout.setSpacing(4)

        btn_fit = QPushButton("Fit")
        btn_fit.setToolTip("Fit image to viewport")
        btn_fit.setFixedWidth(40)

        btn_100 = QPushButton("100%")
        btn_100.setToolTip("Zoom to 100% (1:1)")
        btn_100.setFixedWidth(48)

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setToolTip("Zoom in")
        btn_zoom_in.setFixedWidth(28)

        btn_zoom_out = QPushButton("\u2013")  # en-dash as minus
        btn_zoom_out.setToolTip("Zoom out")
        btn_zoom_out.setFixedWidth(28)

        tb_layout.addWidget(btn_fit)
        tb_layout.addWidget(btn_100)
        tb_layout.addWidget(btn_zoom_in)
        tb_layout.addWidget(btn_zoom_out)
        tb_layout.addStretch()

        layout.addWidget(toolbar)

        # --- Canvas ---
        self._canvas = ImageCanvas()
        layout.addWidget(self._canvas, stretch=1)

        # --- Bottom frame navigator ---
        self._frame_nav = FrameNavigator()
        layout.addWidget(self._frame_nav)

        # --- Connections ---
        btn_fit.clicked.connect(self._canvas.fit_to_view)
        btn_100.clicked.connect(self._canvas.zoom_to_100)
        btn_zoom_in.clicked.connect(self._canvas.zoom_in)
        btn_zoom_out.clicked.connect(self._canvas.zoom_out)

        self._state.images_changed.connect(self._on_images_changed)
        self._state.current_frame_changed.connect(self._on_frame_changed)
        self._state.results_changed.connect(self._refresh_overlay)
        self._state.display_changed.connect(self._refresh_overlay)

    @property
    def canvas(self) -> ImageCanvas:
        """Access the underlying ImageCanvas."""
        return self._canvas

    def _on_images_changed(self) -> None:
        """Load first image when images are set."""
        if self._state.image_files:
            self._load_frame(0)

    def _on_frame_changed(self, idx: int) -> None:
        """Update displayed image and overlay when frame changes."""
        self._load_frame(idx)
        self._refresh_overlay()

    def _load_frame(self, idx: int) -> None:
        """Load and display an image frame by index."""
        try:
            rgb = self._image_ctrl.read_image_rgb(idx)
            self._canvas.set_image(rgb)
        except (IndexError, FileNotFoundError, ValueError):
            pass

    def _refresh_overlay(self, *_args: object) -> None:
        """Update the field overlay based on current results and display settings."""
        state = self._state
        if state.results is None:
            self._canvas._overlay_item.setPixmap(QPixmap())
            return

        result = state.results
        # Frame 0 is reference; results start at index 0 for frame pair 0->1
        frame = state.current_frame - 1
        if frame < 0 or frame >= len(result.result_disp):
            self._canvas._overlay_item.setPixmap(QPixmap())
            return

        nodes = result.dic_mesh.coordinates_fem
        u_accum = result.result_disp[frame].U_accum
        if u_accum is None:
            self._canvas._overlay_item.setPixmap(QPixmap())
            return

        u, v = split_uv(u_accum)
        values = u if state.display_field == "disp_u" else v

        vmin, vmax = state.color_min, state.color_max
        if state.color_auto:
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                pct = np.percentile(valid, [2, 98])
                vmin, vmax = float(pct[0]), float(pct[1])
            else:
                vmin, vmax = 0.0, 1.0
            state.color_min = vmin
            state.color_max = vmax

        try:
            pixmap, xg, yg = self._viz_ctrl.render_field(
                frame,
                state.display_field,
                nodes,
                values,
                img_shape=result.dic_para.img_size,
                mesh_step=result.dic_para.winstepsize,
                vmin=vmin,
                vmax=vmax,
            )
        except Exception:
            # Interpolation can fail with degenerate meshes; clear overlay
            self._canvas._overlay_item.setPixmap(QPixmap())
            return

        # Position the overlay at the correct image coordinates
        self._canvas._overlay_item.setPixmap(pixmap)
        if xg is not None and yg is not None:
            self._canvas._overlay_item.setPos(float(xg.min()), float(yg.min()))
