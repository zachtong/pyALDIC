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
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer, QEvent
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
    QCheckBox,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from staq_dic.gui.app_state import AppState
from staq_dic.gui.theme import COLORS
from staq_dic.gui.widgets.colorbar_overlay import ColorbarOverlay
from staq_dic.gui.widgets.frame_navigator import FrameNavigator
from staq_dic.gui.widgets.mesh_overlay import MeshOverlay

from staq_dic.core.data_structures import split_uv
from staq_dic.gui.controllers.viz_controller import VizController

try:
    from staq_dic.gui.icons import icon_maximize, icon_zoom_in, icon_zoom_out
    _HAS_ICONS = True
except ImportError:  # pragma: no cover
    _HAS_ICONS = False


def resolve_color_range(
    state: AppState, values: NDArray[np.float64]
) -> tuple[float, float]:
    """Return the effective ``(vmin, vmax)`` for colormap rendering.

    Two modes:

    * **Auto** (``state.color_auto`` True): take the 2-98 percentile of
      the non-NaN entries in *values* (the *current frame's* field).
      The result is **also written back** to ``state.color_min`` and
      ``state.color_max`` so that the ColorRange widget reads the live
      frame range when the user later toggles Auto off -- without it,
      the spin boxes would always start from the stale 0/1 defaults
      no matter what the user is currently looking at.

    * **Manual** (``state.color_auto`` False): return the user's
      dialled-in ``(state.color_min, state.color_max)`` unchanged and
      do **not** mutate state.  All-NaN frames degrade to ``(0, 1)``
      so matplotlib's normaliser does not blow up.
    """
    if not state.color_auto:
        return state.color_min, state.color_max

    valid_vals = values[~np.isnan(values)]
    if len(valid_vals) > 0:
        pct = np.percentile(valid_vals, [2, 98])
        vmin, vmax = float(pct[0]), float(pct[1])
    else:
        vmin, vmax = 0.0, 1.0

    state.color_min = vmin
    state.color_max = vmax
    return vmin, vmax

if TYPE_CHECKING:
    from staq_dic.gui.controllers.image_controller import ImageController
    from staq_dic.gui.controllers.roi_controller import ROIController


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ZOOM_FACTOR = 1.15
_ZOOM_MIN = 0.10
_ZOOM_MAX = 20.0

def _centered_arange(lo: int, hi: int, step: int) -> NDArray[np.float64]:
    """Generate evenly-spaced grid points centered within [lo, hi]."""
    span = hi - lo
    n_steps = span // step
    offset = (span - n_steps * step) // 2
    return np.arange(lo + offset, hi + 1, step, dtype=np.float64)


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

    # Signal emitted when a drawing operation finishes or is canceled
    drawing_finished = Signal()

    # Signal emitted when the viewport transform changes (zoom, pan, scroll)
    view_changed = Signal()

    # Signal emitted with scene (x, y) when mouse moves over the canvas
    scene_hover = Signal(float, float)

    # Signal emitted when mouse leaves the canvas
    hover_left = Signal()

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

        # --- Layer 6: brush refinement mask overlay (cyan) ---
        self._refine_item = QGraphicsPixmapItem()
        self._scene.addItem(self._refine_item)
        self._refine_item.setZValue(6)

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

        # Enable mouse tracking for hover detection (no button press needed)
        self.setMouseTracking(True)

        # --- Interaction state ---
        self._current_tool: str = "select"   # select, pan, rect, polygon, circle, brush
        self._drawing_mode: str = "add"      # add or cut
        self._draw_state: dict | None = None  # in-progress drawing data
        self._preview_items: list = []        # temp graphics items while drawing
        self._roi_ctrl: ROIController | None = None

        # Brush sub-tool state (paints state.refine_brush_mask via _brush_ctrl)
        self._brush_ctrl: ROIController | None = None
        self._brush_radius: int = 16
        self._brush_mode: str = "paint"      # "paint" or "erase"
        self._brush_last_pos: QPointF | None = None

        # Pan state (middle-button or left-button when tool == "pan")
        self._panning = False
        self._pan_start = QPointF()

        # Current zoom level (1.0 = 100%)
        self._zoom_level: float = 1.0

    # ----- public API -----

    def set_tool(self, tool: str) -> None:
        """Set the active tool: select, pan, rect, polygon, circle, brush."""
        self._current_tool = tool
        self._cancel_drawing()
        # Reset brush stroke state whenever tool changes
        self._brush_last_pos = None
        if tool == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif tool in ("rect", "polygon", "circle", "brush"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_drawing_mode(self, mode: str) -> None:
        """Set drawing mode: 'add' or 'cut'."""
        self._drawing_mode = mode

    def set_roi_controller(self, ctrl: ROIController) -> None:
        """Attach the ROI controller for mask operations."""
        self._roi_ctrl = ctrl

    def set_brush_controller(self, ctrl: ROIController | None) -> None:
        """Attach the brush refinement controller (frame-0 mask buffer)."""
        self._brush_ctrl = ctrl

    def set_brush_radius(self, radius: int) -> None:
        """Set the freehand-brush radius in pixels."""
        self._brush_radius = max(1, int(radius))

    def set_brush_mode(self, mode: str) -> None:
        """Set the brush mode: 'paint' (add) or 'erase' (cut)."""
        if mode not in ("paint", "erase"):
            raise ValueError(f"brush mode must be 'paint' or 'erase', got {mode!r}")
        self._brush_mode = mode

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
        """Refresh the ROI overlay from per_frame_rois[current_frame].

        Single source of truth: the blue overlay always reflects what is
        persisted in app state for the current frame.  External mutations
        (batch import, context-menu import) therefore become visible
        immediately via the roi_changed signal, without having to re-enter
        ROI editing mode first.

        The in-memory ``self._roi_ctrl`` buffer is a transient stamping
        surface for in-progress drawing operations; it is no longer the
        display source.
        """
        state = AppState.instance()
        mask = state.per_frame_rois.get(state.current_frame)
        if mask is None or not mask.any():
            self._roi_item.setPixmap(QPixmap())
            self._roi_item.setPos(0, 0)
            return
        h, w = mask.shape
        # Build RGBA image — ensure contiguous buffer for QImage
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask, :] = [59, 130, 246, 80]  # blue semi-transparent
        rgba = np.ascontiguousarray(rgba)
        bytes_per_line = w * 4
        qimg = QImage(
            rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
        ).copy()  # deep-copy so pixmap owns the pixel data
        self._roi_item.setPixmap(QPixmap.fromImage(qimg))
        self._roi_item.setPos(0, 0)

    def update_refine_overlay(self) -> None:
        """Refresh the cyan brush refinement overlay from state.refine_brush_mask.

        The brush mask lives in frame-0 image coordinates.  When the user
        is on a non-zero frame the overlay still paints the *frame-0*
        coordinates, because painting is gated to frame 0 only.  Future
        ref-frame display warping is performed by the pipeline at run
        time, not in the GUI.
        """
        state = AppState.instance()
        mask = state.refine_brush_mask
        if mask is None or not mask.any():
            self._refine_item.setPixmap(QPixmap())
            self._refine_item.setPos(0, 0)
            return
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask, :] = [20, 220, 200, 110]  # cyan semi-transparent
        rgba = np.ascontiguousarray(rgba)
        bytes_per_line = w * 4
        qimg = QImage(
            rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
        ).copy()
        self._refine_item.setPixmap(QPixmap.fromImage(qimg))
        self._refine_item.setPos(0, 0)

    def fit_to_view(self) -> None:
        """Fit the entire scene into the viewport."""
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = self.transform().m11()
        self.view_changed.emit()

    def zoom_to_100(self) -> None:
        """Reset zoom to 100% (1:1 pixels)."""
        self.resetTransform()
        self._zoom_level = 1.0
        self.view_changed.emit()

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
        self.view_changed.emit()

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

        # Brush sub-tool: paint a single dot at click site, prime drag state
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._current_tool == "brush"
        ):
            if self._brush_ctrl is None:
                AppState.instance().log_message.emit(
                    "Load images first before using the brush.", "warn"
                )
                event.accept()
                return
            scene_pos = self.mapToScene(event.position().toPoint())
            self._brush_last_pos = scene_pos
            self._stroke_at(scene_pos, scene_pos)
            event.accept()
            return

        super().mousePressEvent(event)

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802
        super().scrollContentsBy(dx, dy)
        self.view_changed.emit()

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

        # Brush drag-paint: while LeftButton held, accumulate strokes
        if (
            self._current_tool == "brush"
            and (event.buttons() & Qt.MouseButton.LeftButton)
            and self._brush_ctrl is not None
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            last = self._brush_last_pos
            if last is not None:
                self._stroke_at(last, scene_pos)
            self._brush_last_pos = scene_pos
            event.accept()
            return

        # Emit scene hover for mesh overlay subset window
        sp = self.mapToScene(event.position().toPoint())
        self.scene_hover.emit(sp.x(), sp.y())

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

        # Brush stroke finalize: persist current buffer to app state.
        # Tool stays in "brush" mode for continuous painting.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._current_tool == "brush"
            and self._brush_ctrl is not None
        ):
            self._brush_last_pos = None
            AppState.instance().set_refine_brush_mask(
                self._brush_ctrl.mask.copy()
            )
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

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.hover_left.emit()
        super().leaveEvent(event)

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
            AppState.instance().log_message.emit(
                "Load images first before drawing ROI.", "warn"
            )
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
            if self._roi_ctrl is None:
                AppState.instance().log_message.emit(
                    "Load images first before drawing ROI.", "warn"
                )
            return
        pts = self._draw_state.get("points", [])
        if len(pts) < 3:
            self._cancel_drawing()
            return
        int_pts = [(int(p.x()), int(p.y())) for p in pts]
        self._roi_ctrl.add_polygon(int_pts, self._drawing_mode)
        self._finish_drawing()

    def _finish_drawing(self) -> None:
        """Clean up preview items, save the stamp to per-frame state,
        refresh the overlay, and reset to select mode."""
        self._remove_preview_items()
        self._draw_state = None
        # Persist the stamp to per-frame state BEFORE refreshing the
        # overlay. After Q3 the overlay reads its data from
        # per_frame_rois[current_frame], so state must be written first
        # or the first refresh will paint with the previous (stale) mask.
        if self._roi_ctrl is not None:
            state = AppState.instance()
            state.set_frame_roi(
                state.current_frame, self._roi_ctrl.mask.copy()
            )
        self.update_roi_overlay()
        # One-shot: reset to select mode after completing a shape
        self._current_tool = "select"
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.drawing_finished.emit()

    def _cancel_drawing(self) -> None:
        """Cancel in-progress drawing, remove preview items, and reset to select."""
        was_drawing = self._draw_state is not None
        self._remove_preview_items()
        self._draw_state = None
        if was_drawing:
            self._current_tool = "select"
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.drawing_finished.emit()

    def _remove_preview_items(self) -> None:
        for item in self._preview_items:
            self._scene.removeItem(item)
        self._preview_items = []

    def _stroke_at(self, p1: QPointF, p2: QPointF) -> None:
        """Apply a single brush stroke segment and refresh the overlay.

        Pushes the segment into ``self._brush_ctrl`` (which holds a
        bool buffer in frame-0 image coordinates), then immediately
        rebuilds the cyan overlay so the user sees their stroke.

        AppState is updated only on mouse release to keep paint latency
        low — intermediate strokes mutate the buffer in place.
        """
        if self._brush_ctrl is None:
            return
        sub_mode = "add" if self._brush_mode == "paint" else "cut"
        self._brush_ctrl.stroke_segment(
            int(p1.x()), int(p1.y()),
            int(p2.x()), int(p2.y()),
            radius=self._brush_radius,
            mode=sub_mode,
        )
        # Render directly from the controller buffer for instant feedback —
        # state.refine_brush_mask gets the same buffer on mouse release.
        self._render_brush_buffer()

    def _render_brush_buffer(self) -> None:
        """Render the in-progress brush controller buffer to the cyan overlay.

        Called between mouse-down and mouse-up so the user sees their
        stroke before the buffer is committed to AppState.  Identical
        rendering logic to ``update_refine_overlay`` but reads from
        ``self._brush_ctrl.mask`` instead of ``state.refine_brush_mask``.
        """
        if self._brush_ctrl is None:
            return
        mask = self._brush_ctrl.mask
        if mask is None or not mask.any():
            self._refine_item.setPixmap(QPixmap())
            self._refine_item.setPos(0, 0)
            return
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask, :] = [20, 220, 200, 110]  # cyan semi-transparent
        rgba = np.ascontiguousarray(rgba)
        bytes_per_line = w * 4
        qimg = QImage(
            rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
        ).copy()
        self._refine_item.setPixmap(QPixmap.fromImage(qimg))
        self._refine_item.setPos(0, 0)


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
        btn_fit.setFixedWidth(60)
        if _HAS_ICONS:
            btn_fit.setIcon(icon_maximize())

        btn_100 = QPushButton("100%")
        btn_100.setToolTip("Zoom to 100% (1:1)")
        btn_100.setFixedWidth(60)

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setToolTip("Zoom in")
        btn_zoom_in.setFixedWidth(28)
        if _HAS_ICONS:
            btn_zoom_in.setIcon(icon_zoom_in())
            btn_zoom_in.setText("")

        btn_zoom_out = QPushButton("\u2013")  # en-dash as minus
        btn_zoom_out.setToolTip("Zoom out")
        btn_zoom_out.setFixedWidth(28)
        if _HAS_ICONS:
            btn_zoom_out.setIcon(icon_zoom_out())
            btn_zoom_out.setText("")

        tb_layout.addWidget(btn_fit)
        tb_layout.addWidget(btn_100)
        tb_layout.addWidget(btn_zoom_in)
        tb_layout.addWidget(btn_zoom_out)
        tb_layout.addStretch()

        # --- Mesh overlay toggles ---
        self._btn_grid = QCheckBox("Show Grid")
        self._btn_grid.setToolTip("Show/hide computational mesh grid")
        self._btn_grid.setChecked(True)

        self._btn_window = QCheckBox("Show Subset")
        self._btn_window.setToolTip("Show subset window on hover (requires Grid)")
        self._btn_window.setChecked(False)

        tb_layout.addWidget(self._btn_grid)
        tb_layout.addWidget(self._btn_window)

        layout.addWidget(toolbar)

        # --- ROI editing banner (hidden by default) ---
        self._roi_banner = QLabel()
        self._roi_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._roi_banner.setFixedHeight(36)
        self._roi_banner.setVisible(False)
        self._roi_banner.setStyleSheet(
            f"background: {COLORS.WARNING}; color: #000; "
            f"font-size: 14px; font-weight: bold; padding: 4px;"
        )
        layout.addWidget(self._roi_banner)

        # --- Canvas ---
        self._canvas = ImageCanvas()
        layout.addWidget(self._canvas, stretch=1)

        # --- Colorbar overlay (child of canvas viewport, positioned in resizeEvent) ---
        self._colorbar = ColorbarOverlay(self._canvas.viewport())

        # --- Mesh overlay (child of canvas viewport, same pattern as colorbar) ---
        self._mesh_overlay = MeshOverlay(self._canvas.viewport())

        # The viewport can resize independently of CanvasArea (e.g. when
        # scrollbars appear/disappear on zoom).  Watch its resize events so
        # we can keep both overlays sized to the actual viewport rect.
        self._canvas.viewport().installEventFilter(self)

        # Debounce timer for preview mesh generation
        self._mesh_preview_timer = QTimer()
        self._mesh_preview_timer.setSingleShot(True)
        self._mesh_preview_timer.setInterval(300)
        self._mesh_preview_timer.timeout.connect(self._generate_preview_mesh)

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
        self._state.results_changed.connect(self._on_results_changed)
        self._state.display_changed.connect(self._on_display_changed)
        self._state.roi_changed.connect(self._on_roi_changed)
        self._state.params_changed.connect(self._on_params_changed_mesh)

        # Mesh overlay toggle wiring
        self._btn_grid.toggled.connect(self._on_grid_toggled)
        self._btn_window.toggled.connect(self._on_window_toggled)

        # Lightweight repaint on pan/zoom (paths already built, just update transform)
        self._canvas.view_changed.connect(self._sync_mesh_view_transform)

        # Subset window hover
        self._canvas.scene_hover.connect(self._on_scene_hover)
        self._canvas.hover_left.connect(self._on_hover_left)

        # Current mesh coords + validity for nearest-node search (scene coords)
        self._hover_mesh_coords: NDArray[np.float64] | None = None
        self._hover_valid: NDArray[np.bool_] | None = None

    @property
    def canvas(self) -> ImageCanvas:
        """Access the underlying ImageCanvas."""
        return self._canvas

    def _sync_overlay_geometry(self) -> None:
        """Resize child overlays (colorbar, mesh) to fill the viewport rect."""
        vp = self._canvas.viewport()
        rect = (0, 0, vp.width(), vp.height())
        self._colorbar.setGeometry(*rect)
        self._mesh_overlay.setGeometry(*rect)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Reposition overlays to fill the canvas viewport."""
        super().resizeEvent(event)
        self._sync_overlay_geometry()
        self._refresh_mesh_overlay()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        """Catch viewport resize events that don't propagate to CanvasArea.

        The QGraphicsView's viewport widget can be resized independently
        of CanvasArea — most notably when scrollbars appear/disappear as
        the user zooms past the Fit threshold.  Without this, the overlay
        widget keeps its old (smaller) geometry and clips mesh elements
        in the newly-uncovered viewport region.
        """
        if obj is self._canvas.viewport() and event.type() == QEvent.Type.Resize:
            self._sync_overlay_geometry()
        return super().eventFilter(obj, event)

    def set_roi_editing_banner(self, frame: int | None) -> None:
        """Show or hide the ROI editing banner for the given frame."""
        if frame is not None:
            self._roi_banner.setText(f"EDITING ROI FOR FRAME {frame:02d}")
            self._roi_banner.setVisible(True)
        else:
            self._roi_banner.setVisible(False)

    def _on_images_changed(self) -> None:
        """Load first image when images are set."""
        if self._state.image_files:
            self._load_frame(0)

    def _on_frame_changed(self, idx: int) -> None:
        """Update display when frame changes.

        Navigating frames while in ROI editing mode (with results available)
        exits ROI editing and returns to results view.
        """
        state = self._state
        # Exit ROI editing when user navigates frames (wants to see results)
        if state.roi_editing and state.results is not None:
            state.roi_editing = False

        self._update_background()
        self._refresh_overlay()
        self._refresh_mesh_overlay()

    def _on_roi_changed(self) -> None:
        """ROI masks changed — invalidate mask-dependent viz caches and refresh."""
        self._viz_ctrl.invalidate_masks()
        self._refresh_overlay()
        # Always refresh mesh: clear grid when ROI is removed,
        # regenerate preview when ROI is (re-)drawn.
        self._refresh_mesh_overlay()

    def _on_results_changed(self) -> None:
        """Refresh overlays when results change."""
        self._refresh_overlay()
        self._refresh_mesh_overlay()

    def _on_display_changed(self) -> None:
        """Handle display settings change (field, color, deformed, roi_editing)."""
        self._update_background()
        self._refresh_overlay()
        self._refresh_mesh_overlay()

    def _update_background(self) -> None:
        """Set the background image based on current mode."""
        state = self._state
        if state.roi_editing:
            # ROI editing: show the frame being edited
            self._load_frame(state.current_frame)
        elif state.results is not None:
            if state.show_deformed:
                self._load_frame(state.current_frame)
            else:
                self._load_frame(0)
        else:
            self._load_frame(state.current_frame)

    def _load_frame(self, idx: int) -> None:
        """Load and display an image frame by index."""
        try:
            rgb = self._image_ctrl.read_image_rgb(idx)
            self._canvas.set_image(rgb)
        except (IndexError, FileNotFoundError, ValueError):
            pass

    def _refresh_overlay(self, *_args: object) -> None:
        """Update field and ROI overlays based on current mode.

        Two mutually exclusive display modes:
        - ROI editing: show ROI mask overlay (blue), hide field overlay
        - Results view: show field overlay (colormap), hide ROI overlay
        """
        import traceback as _tb

        state = self._state
        overlay = self._canvas._overlay_item
        roi_item = self._canvas._roi_item

        # --- ROI editing mode: show ROI overlay, hide field ---
        if state.roi_editing:
            overlay.setPixmap(QPixmap())
            overlay.setScale(1.0)
            overlay.setPos(0, 0)
            self._canvas.update_roi_overlay()
            self.set_roi_editing_banner(state.current_frame)
            self._colorbar.setVisible(False)
            return

        # Hide banner when not editing
        self.set_roi_editing_banner(None)

        # --- No results yet: show ROI overlay (if any), hide field ---
        if state.results is None:
            overlay.setPixmap(QPixmap())
            overlay.setScale(1.0)
            overlay.setPos(0, 0)
            # Keep ROI visible before DIC has run
            self._canvas.update_roi_overlay()
            self._colorbar.setVisible(False)
            return

        # --- Results mode: show field overlay, hide ROI ---
        roi_item.setPixmap(QPixmap())

        result = state.results
        # Frame 0 is reference; results start at index 0 for frame pair 0->1
        frame = state.current_frame - 1
        if frame >= len(result.result_disp):
            overlay.setPixmap(QPixmap())
            overlay.setScale(1.0)
            self._colorbar.setVisible(False)
            return

        try:
            ref_nodes = result.dic_mesh.coordinates_fem

            if frame < 0:
                # Reference frame (current_frame=0): zero displacement
                n_nodes = ref_nodes.shape[0]
                u = np.zeros(n_nodes)
                v = np.zeros(n_nodes)
            else:
                u_accum = result.result_disp[frame].U_accum
                if u_accum is None:
                    overlay.setPixmap(QPixmap())
                    overlay.setScale(1.0)
                    return
                u, v = split_uv(u_accum)

            values = u if state.display_field == "disp_u" else v

            # Deformed mode: shift nodes by accumulated displacement
            # (reference frame always shows undeformed config)
            is_deformed = state.show_deformed and frame >= 0
            if is_deformed:
                nodes = ref_nodes + np.column_stack([u, v])
            else:
                nodes = ref_nodes

            # Auto mode also writes the resolved range back to AppState
            # so the ColorRange widget can pre-populate its spin boxes
            # with the live frame range when the user toggles Auto off.
            vmin, vmax = resolve_color_range(state, values)

            # In deformed mode, warp the ROI mask from reference to deformed
            # coordinates via inverse displacement lookup.  This preserves
            # internal holes that would otherwise be filled by interpolation.
            ref_uv = (u, v) if is_deformed else None

            # Deformed mask priority:
            # 1. Explicit deformed masks (e.g., from segmentation)
            # 2. Per-frame ROI (auto-used when available in deformed mode)
            # 3. Warped mask (handled downstream in viz_controller)
            def_mask = None
            if is_deformed:
                if state.deformed_masks is not None:
                    def_mask = state.deformed_masks.get(frame)
                if def_mask is None:
                    per_frame_roi = state.per_frame_rois.get(
                        state.current_frame
                    )
                    if per_frame_roi is not None:
                        def_mask = per_frame_roi

            cmap = state.colormap

            pixmap, xg, yg, out_step = self._viz_ctrl.render_field(
                frame,
                state.display_field,
                nodes,
                values,
                img_shape=result.dic_para.img_size,
                mesh_step=result.dic_para.winstepsize,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                roi_mask=state.roi_mask,
                deformed=is_deformed,
                ref_uv=ref_uv,
                deformed_mask=def_mask,
            )

            overlay.setPixmap(pixmap)
            overlay.setScale(float(out_step))
            if xg is not None and yg is not None:
                overlay.setPos(float(xg.min()), float(yg.min()))

            # Update colorbar
            field_label = {
                "disp_u": "U (px)",
                "disp_v": "V (px)",
            }.get(state.display_field, state.display_field)
            self._colorbar.update_params(cmap, vmin, vmax, field_label)
            # Ensure colorbar fills the full viewport (may have been
            # initially sized to 0×0 before the canvas was laid out).
            vp = self._canvas.viewport()
            self._colorbar.setGeometry(0, 0, vp.width(), vp.height())
            self._colorbar.setVisible(True)
        except Exception as e:
            tb_str = _tb.format_exc()
            print(f"[_refresh_overlay] {e}\n{tb_str}", flush=True)
            state.log_message.emit(
                f"Overlay error: {type(e).__name__}: {e}", "error"
            )
            overlay.setPixmap(QPixmap())
            overlay.setScale(1.0)
            self._colorbar.setVisible(False)

    # --- Mesh overlay logic ---

    def _on_grid_toggled(self, checked: bool) -> None:
        self._state.set_show_mesh(checked)
        self._btn_window.setEnabled(checked)
        if not checked:
            self._btn_window.setChecked(False)
            self._mesh_overlay.setVisible(False)
            self._hover_mesh_coords = None
        else:
            self._refresh_mesh_overlay()

    def _on_window_toggled(self, checked: bool) -> None:
        self._state.set_show_subset_window(checked)
        if not checked:
            self._mesh_overlay.set_hover_node(None)

    def _on_params_changed_mesh(self) -> None:
        """Debounced mesh preview update when params change.

        Two cases require a preview rebuild:
        1. No results yet — preview is the only mesh source.
        2. ROI editing on frame 0 — even when stale results exist,
           ``_refresh_mesh_overlay`` routes editing to the preview
           path (see Q4), so params changes must rebuild it.

        Without the second case, changing ``subset_step`` while
        editing post-run leaves the preview mesh stale until the
        user manually toggles 'display grid' off and on.
        """
        state = self._state
        if not state.show_mesh:
            return
        if state.roi_editing or state.results is None:
            self._mesh_preview_timer.start()

    def _sync_mesh_view_transform(self) -> None:
        """Lightweight sync — update only the transform, no path rebuild."""
        if self._mesh_overlay.isVisible():
            self._mesh_overlay.set_view_transform(
                self._canvas.viewportTransform(),
            )

    def _refresh_mesh_overlay(self) -> None:
        """Rebuild mesh overlay data (paths + transform).

        Routing rules:
        - User toggled mesh off           -> hide
        - ROI editing on frame K != 0     -> hide (preview mesh only
                                              models frame-0 geometry,
                                              would be misleading) [Q5]
        - ROI editing on frame 0          -> preview mesh from current
                                              params (ignore results,
                                              the user is editing the
                                              ROI not inspecting) [Q4]
        - Otherwise + results             -> results mesh
        - Otherwise + frame-0 ROI         -> preview mesh
        - Else                            -> hide

        Whenever we take a hide branch (or the results branch, which
        overrides the preview), stop any pending preview timer so a
        stale debounced fire cannot undo the hide or clobber results.
        """
        state = self._state
        if not state.show_mesh:
            self._mesh_preview_timer.stop()
            self._mesh_overlay.setVisible(False)
            return

        if state.roi_editing:
            if state.current_frame != 0:
                self._mesh_preview_timer.stop()
                self._mesh_overlay.set_mesh(None, None)
                self._mesh_overlay.setVisible(False)
                return
            # Frame-0 editing: always go through preview path
            if state.roi_mask is not None:
                self._mesh_preview_timer.start()
            else:
                self._mesh_preview_timer.stop()
                self._mesh_overlay.set_mesh(None, None)
                self._mesh_overlay.setVisible(False)
            return

        if state.results is not None:
            self._mesh_preview_timer.stop()
            self._show_results_mesh()
        elif state.roi_mask is not None:
            self._mesh_preview_timer.start()
        else:
            self._mesh_preview_timer.stop()
            self._mesh_overlay.set_mesh(None, None)
            self._mesh_overlay.setVisible(False)

    def _node_valid_mask(
        self,
        coords: NDArray[np.float64],
        roi_mask: NDArray[np.bool_] | None,
    ) -> NDArray[np.bool_]:
        """Return per-node boolean: True if node is inside the ROI."""
        n = coords.shape[0]
        valid = np.ones(n, dtype=bool)
        if roi_mask is None:
            return valid
        h, w = roi_mask.shape
        ix = np.clip(np.round(coords[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(coords[:, 1]).astype(int), 0, h - 1)
        valid = roi_mask[iy, ix]
        return valid

    def _show_results_mesh(self) -> None:
        """Show mesh from pipeline results for the current frame."""
        state = self._state
        result = state.results
        if result is None:
            return

        frame = state.current_frame - 1
        meshes = result.result_fe_mesh_each_frame
        if not meshes:
            self._mesh_overlay.set_mesh(None, None)
            self._mesh_overlay.setVisible(False)
            return

        # Always draw the canonical (frame-0) mesh.
        #
        # ``_compute_cumulative_displacements_tree`` defines
        # ``result_disp[i].U_accum`` on the frame-0 mesh for *every*
        # frame, so the per-frame mesh in ``meshes[i]`` may have a
        # different node count when incremental + refined + per-frame
        # ROIs cause the mask (and therefore the trimmed mesh) to vary
        # across reference frames.  Drawing the canonical mesh keeps
        # the overlay shape consistent with ``U_accum`` and also keeps
        # the mesh topology stable when the user toggles the deformed
        # switch.
        canonical_mesh = meshes[0]
        coords = canonical_mesh.coordinates_fem.copy()
        elements = canonical_mesh.elements_fem

        # In deformed mode, offset canonical nodes by accumulated
        # displacement (both have the same node count by construction).
        is_deformed = state.show_deformed and frame >= 0
        if is_deformed and frame < len(result.result_disp):
            u_accum = result.result_disp[frame].U_accum
            if u_accum is not None:
                u, v = split_uv(u_accum)
                coords = coords + np.column_stack([u, v])

        # Choose mask for node validity:
        # - Deformed mode: use deformed frame's mask (per-frame ROI or warped)
        # - Reference mode: use frame-0 ROI mask
        if is_deformed:
            mask_for_valid = state.per_frame_rois.get(state.current_frame)
            if mask_for_valid is None:
                mask_for_valid = state.roi_mask
        else:
            mask_for_valid = state.roi_mask
        valid = self._node_valid_mask(coords, mask_for_valid)

        self._hover_mesh_coords = coords
        self._hover_valid = valid
        self._mesh_overlay.set_mesh(coords, elements, valid)
        self._mesh_overlay.set_view_transform(self._canvas.viewportTransform())
        self._mesh_overlay.setVisible(True)

    def _generate_preview_mesh(self) -> None:
        """Generate a preview mesh from current ROI and params (debounced)."""
        state = self._state
        if not state.show_mesh:
            return
        roi_mask = state.roi_mask
        if roi_mask is None:
            self._mesh_overlay.set_mesh(None, None)
            self._mesh_overlay.setVisible(False)
            return

        step = state.subset_step
        h, w = roi_mask.shape
        half_w = state.subset_size // 2

        # Match actual DIC padding (integer_search.py): only `half_w` is
        # required for the IC-GN subset to fit.  Edge nodes too close to
        # the border for NCC search are NaN'd then inpainted; partial-edge
        # subsets are handled by the masked-subset code path.
        pad = half_w
        min_x, max_x = pad, w - 1 - pad
        min_y, max_y = pad, h - 1 - pad
        if max_x <= min_x or max_y <= min_y:
            self._mesh_overlay.set_mesh(None, None)
            self._mesh_overlay.setVisible(False)
            return

        x0 = _centered_arange(min_x, max_x, step)
        y0 = _centered_arange(min_y, max_y, step)
        if len(x0) < 2 or len(y0) < 2:
            self._mesh_overlay.set_mesh(None, None)
            self._mesh_overlay.setVisible(False)
            return

        # Build Q4 mesh (same as mesh_setup but without DICPara)
        nx, ny = len(x0), len(y0)
        xx, yy = np.meshgrid(x0, y0, indexing="ij")
        coords = np.column_stack([xx.ravel(), yy.ravel()])

        ii, jj = np.meshgrid(
            np.arange(nx - 1), np.arange(ny - 1), indexing="ij",
        )
        ii_flat, jj_flat = ii.ravel(), jj.ravel()
        n0 = ii_flat * ny + jj_flat
        n1 = (ii_flat + 1) * ny + jj_flat
        n2 = (ii_flat + 1) * ny + (jj_flat + 1)
        n3 = ii_flat * ny + (jj_flat + 1)
        elements = np.full((len(n0), 8), -1, dtype=np.int64)
        elements[:, 0] = n0
        elements[:, 1] = n1
        elements[:, 2] = n2
        elements[:, 3] = n3

        # Trim elements to ROI mask
        from staq_dic.mesh.mark_inside import mark_inside

        f_mask = roi_mask.astype(np.float64)
        _, outside_idx = mark_inside(coords, elements, f_mask)
        if len(outside_idx) < elements.shape[0]:
            elements = elements[outside_idx]

        # Optional: apply user-configured refinement so the preview reflects
        # what the pipeline will actually compute.  Best-effort: if anything
        # goes wrong we silently fall back to the uniform mesh.
        if state.refine_inner or state.refine_outer:
            refined = self._apply_preview_refinement(coords, elements, f_mask)
            if refined is not None:
                coords, elements = refined

        # Compute per-node ROI validity
        valid = self._node_valid_mask(coords, roi_mask)

        self._hover_mesh_coords = coords
        self._hover_valid = valid
        self._mesh_overlay.set_mesh(coords, elements, valid)
        self._mesh_overlay.set_view_transform(self._canvas.viewportTransform())
        self._mesh_overlay.setVisible(True)

    def _apply_preview_refinement(
        self,
        coords: NDArray[np.float64],
        elements: NDArray[np.int64],
        f_mask: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
        """Apply the current refinement policy to a uniform preview mesh.

        Returns the refined ``(coords, elements)`` tuple, or ``None`` if
        refinement is not applicable or the driver fails for any reason.
        Refinement is best-effort for preview — failures must never block
        the GUI.
        """
        state = self._state
        try:
            from staq_dic.core.data_structures import DICMesh
            from staq_dic.mesh.refinement import (
                RefinementContext,
                build_refinement_policy,
                refine_mesh,
            )

            policy = build_refinement_policy(
                refine_inner_boundary=state.refine_inner,
                refine_outer_boundary=state.refine_outer,
                min_element_size=state.compute_refinement_min_size(),
                half_win=max(1, state.subset_size // 2),
            )
            if policy is None or not policy.has_pre_solve:
                return None

            h, w = f_mask.shape
            starting_mesh = DICMesh(
                coordinates_fem=coords,
                elements_fem=elements,
                element_min_size=int(state.subset_step),
            )
            ctx = RefinementContext(mesh=starting_mesh, mask=f_mask)
            U0 = np.zeros(2 * coords.shape[0], dtype=np.float64)

            refined_mesh, _ = refine_mesh(
                starting_mesh,
                policy.pre_solve,
                ctx,
                U0=U0,
                mask=f_mask,
                img_size=(h, w),
            )
            return refined_mesh.coordinates_fem, refined_mesh.elements_fem
        except Exception:  # noqa: BLE001
            return None

    def _on_scene_hover(self, sx: float, sy: float) -> None:
        """Find nearest valid mesh node and show subset window."""
        state = self._state
        if (
            not state.show_mesh
            or not state.show_subset_window
            or self._hover_mesh_coords is None
        ):
            return

        coords = self._hover_mesh_coords
        valid = getattr(self, "_hover_valid", None)

        dx = coords[:, 0] - sx
        dy = coords[:, 1] - sy
        dist_sq = dx * dx + dy * dy

        # Only consider valid (inside-ROI, non-NaN) nodes
        mask = ~np.isnan(dist_sq)
        if valid is not None:
            mask &= valid
        if not np.any(mask):
            self._mesh_overlay.set_hover_node(None)
            return

        dist_sq_masked = np.where(mask, dist_sq, np.inf)
        min_idx = int(np.argmin(dist_sq_masked))

        threshold = state.subset_step * 1.5
        if dist_sq_masked[min_idx] > threshold * threshold:
            self._mesh_overlay.set_hover_node(None)
            return

        self._mesh_overlay.set_hover_node(min_idx, float(state.subset_size))

    def _on_hover_left(self) -> None:
        """Clear hover when mouse leaves the canvas."""
        self._mesh_overlay.set_hover_node(None)

