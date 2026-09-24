"""The Analysis tab's canvas: place, select and edit probes on the reference image.

A ``StrainCanvas`` subclass, so the Strain Field tab is untouched: zoom and the
image and field layers are inherited; probes, placement and editing are added.

The background is always the **reference** frame. Probe coordinates are frame-0
image pixels, so a marker drawn over a deformed image would sit beside the
material it measures rather than on it.

Placement mirrors drawing tools: one click for a point, two for a line,
rectangle or circle, N clicks and a double-click for a polygon, Escape to
cancel. With no tool armed, a click selects the probe under it, a drag moves
it, the selected probe's handles reshape it, a double-click renames it, and a
drag on empty space pans. Edits are previewed live and reported once, on
release, so the chart is recomputed per edit rather than per mouse event.

Markers, handles and labels keep their size on screen at any zoom. Drawn in
image pixels, a 5 px marker became a 100 px disc at 20x and a speck at 0.1x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Literal

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QWidget,
)

from al_dic.analysis.geometry_edit import (
    control_points,
    encloses_area,
    hit_test,
    move_control_point,
    translate,
)
from al_dic.analysis.probes import (
    AreaGeom,
    Geometry,
    LineGeom,
    PointGeom,
    Probe,
    ProbeSet,
)
from al_dic.gui.panels.strain_canvas import StrainCanvas
from al_dic.gui.theme import COLORS
from al_dic.utils.locale_format import format_number

Tool = Literal[
    "none", "point", "line", "area_rect", "area_circle", "area_polygon",
    "extensometer", "crack_gauge",
]

#: Tools that draw a two-point line. An extensometer and a crack gauge are
#: lines; what differs is what the tab plots once one is placed.
LINE_TOOLS = frozenset({"line", "extensometer", "crack_gauge"})

# Sizes in screen pixels, whatever the zoom.
_POINT_RADIUS = 5.0
_TICK_HALF = 6.0
_HANDLE_SIZE = 7.0
_HIT_TOLERANCE = 6.0
_DRAG_THRESHOLD = 3.0
_NORMAL_WIDTH = 1.6
_SELECTED_WIDTH = 2.6

_HALO_Z = 19
_MARK_Z = 20
_PREVIEW_Z = 21
_LABEL_Z = 22
_HANDLE_Z = 23


def _default_length(px: float) -> str:
    return f"{format_number(px, 1)} px"


@dataclass(frozen=True)
class _Drag:
    """A move or reshape in progress. Computed from ``original`` every time."""

    probe_id: int
    original: Geometry
    handle: int | None          # None: the whole probe moves
    press_view: QPointF
    press_scene: QPointF
    current: Geometry
    moved: bool = False         # past the jitter threshold yet


class ProbeLabel(QGraphicsItem):
    """A probe's name -- and a line's length -- at a fixed size on screen."""

    def __init__(self, text: str, colour: str, anchor: QPointF) -> None:
        super().__init__()
        self._text = text
        self._colour = QColor(colour)
        self._font = QFont()
        size = self._font.pointSizeF()
        if size > 0:
            self._font.setPointSizeF(max(size - 0.5, 7.0))
        metrics = QFontMetricsF(self._font)
        width = metrics.horizontalAdvance(text) + 8.0
        height = metrics.height() + 2.0
        # Above and right of the anchor, clear of a point marker.
        self._rect = QRectF(_POINT_RADIUS + 3.0, -_POINT_RADIUS - 3.0 - height,
                            width, height)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setPos(anchor)
        self.setZValue(_LABEL_Z)

    def text(self) -> str:
        return self._text

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        background = QColor(COLORS.BG_PANEL)
        background.setAlpha(210)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(self._rect, 3.0, 3.0)
        painter.setPen(self._colour)
        painter.setFont(self._font)
        painter.drawText(self._rect, Qt.AlignmentFlag.AlignCenter, self._text)


class ProbeCanvas(StrainCanvas):
    """Reference-frame canvas with probes: placement, selection and editing."""

    probe_requested = Signal(str, object)     # (kind, geometry): a new probe
    probe_selected = Signal(object)           # id, or None: picked on the canvas
    probe_edited = Signal(int, object)        # (id, geometry): on release
    probe_activated = Signal(int)             # double-click: rename it
    placement_cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tool: Tool = "none"
        self._pending: list[QPointF] = []
        self._probes: ProbeSet | None = None
        self._selected_id: int | None = None
        self._drag: _Drag | None = None
        self._format_length: Callable[[float], str] = _default_length
        self._items: list[QGraphicsItem] = []

        self._preview = QGraphicsPathItem()
        self._preview.setZValue(_PREVIEW_Z)
        pen = QPen(QColor(COLORS.TEXT_PRIMARY), 1.2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._preview.setPen(pen)
        self._preview.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._scene.addItem(self._preview)
        self._preview_label: ProbeLabel | None = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # -- tool state -------------------------------------------------------

    @property
    def tool(self) -> Tool:
        return self._tool

    @property
    def selected_id(self) -> int | None:
        return self._selected_id

    def set_tool(self, tool: Tool) -> None:
        self._tool = tool
        self._pending.clear()
        self._drag = None
        self._clear_preview()
        self.setCursor(
            Qt.CursorShape.ArrowCursor if tool == "none"
            else Qt.CursorShape.CrossCursor
        )
        self._redraw_overlay()          # handles only show with no tool armed

    def cancel_placement(self) -> None:
        if self._tool != "none":
            self.set_tool("none")
            self.placement_cancelled.emit()

    # -- overlay ----------------------------------------------------------

    def set_probes(self, probes: ProbeSet, selected_id: int | None) -> None:
        self._probes = probes
        self._selected_id = selected_id
        if self._drag is not None and self._probe(self._drag.probe_id) is None:
            self._drag = None
        self._redraw_overlay()

    def set_length_format(self, fmt: Callable[[float], str]) -> None:
        """How a line's length (in image pixels) is written on its label."""
        self._format_length = fmt
        self._redraw_overlay()

    def _probe(self, probe_id: int | None) -> Probe | None:
        if self._probes is None or probe_id is None:
            return None
        try:
            return self._probes.get(probe_id)
        except KeyError:
            return None

    def _shown_geometry(self, probe: Probe) -> Geometry:
        """The drag preview for the probe being edited, else its geometry."""
        if self._drag is not None and self._drag.probe_id == probe.id:
            return self._drag.current
        return probe.geometry

    def _add(self, item: QGraphicsItem) -> None:
        self._scene.addItem(item)
        self._items.append(item)

    def _redraw_overlay(self) -> None:
        for item in self._items:
            self._scene.removeItem(item)
        self._items = []
        if self._probes is None:
            return
        for probe in self._probes:
            if probe.visible:
                self._draw_probe(probe, self._shown_geometry(probe))
        selected = self._probe(self._selected_id)
        if selected is not None and selected.visible and self._tool == "none":
            for x, y in _handle_points(self._shown_geometry(selected)):
                self._add(_handle_item(x, y))

    def _draw_probe(self, probe: Probe, geom: Geometry) -> None:
        width = _SELECTED_WIDTH if probe.id == self._selected_id else _NORMAL_WIDTH
        colour = QColor(probe.color)
        if isinstance(geom, PointGeom):
            at = QPointF(geom.x, geom.y)
            self._add(_path_item(_marker_path(), _halo_pen(width), _HALO_Z, at=at))
            self._add(_path_item(_marker_path(), _pen(colour, width), _MARK_Z, at=at))
        else:
            fill = None
            if isinstance(geom, AreaGeom):
                fill = QColor(colour)
                fill.setAlphaF(0.15)
            outline = _outline_path(geom)
            self._add(_path_item(outline, _halo_pen(width), _HALO_Z))
            self._add(_path_item(outline, _pen(colour, width), _MARK_Z, brush=fill))
            if isinstance(geom, LineGeom):
                angle = math.degrees(math.atan2(geom.y1 - geom.y0, geom.x1 - geom.x0))
                for x, y in ((geom.x0, geom.y0), (geom.x1, geom.y1)):
                    at = QPointF(x, y)
                    self._add(_path_item(_tick_path(), _halo_pen(width), _HALO_Z,
                                         at=at, rotation=angle))
                    self._add(_path_item(_tick_path(), _pen(colour, width), _MARK_Z,
                                         at=at, rotation=angle))
        self._add(ProbeLabel(self._label_text(probe, geom), probe.color,
                             _label_anchor(geom)))

    def _label_text(self, probe: Probe, geom: Geometry) -> str:
        if isinstance(geom, LineGeom):
            return f"{probe.label} · {self._format_length(geom.length())}"
        return probe.label

    # -- hit testing --------------------------------------------------------

    def _scale(self) -> float:
        return max(abs(self.transform().m11()), 1e-6)

    def _hit(self, scene: QPointF) -> int | None:
        if self._probes is None:
            return None
        return hit_test(self._probes, scene.x(), scene.y(),
                        tolerance=_HIT_TOLERANCE / self._scale())

    def _handle_at(self, scene: QPointF) -> int | None:
        """Index of the selected probe's handle under *scene*, if any."""
        probe = self._probe(self._selected_id)
        if probe is None or not probe.visible or self._tool != "none":
            return None
        if isinstance(probe.geometry, PointGeom):
            return None                 # a point has nothing to reshape
        tolerance = _HIT_TOLERANCE / self._scale()
        for index, (x, y) in enumerate(control_points(probe.geometry)):
            if math.hypot(scene.x() - x, scene.y() - y) <= tolerance:
                return index
        return None

    def _select(self, probe_id: int | None) -> None:
        if probe_id == self._selected_id:
            return
        self._selected_id = probe_id
        self._redraw_overlay()
        self.probe_selected.emit(probe_id)

    # -- interaction ------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        scene = self.mapToScene(event.position().toPoint())
        if self._tool != "none":
            self._pending.append(scene)
            self._commit_if_complete()
            event.accept()
            return

        handle = self._handle_at(scene)
        target = self._selected_id if handle is not None else self._hit(scene)
        if target is None:
            self._select(None)
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self._select(target)
            original = self._probes.get(target).geometry   # type: ignore[union-attr]
            self._drag = _Drag(target, original, handle, event.position(), scene,
                               current=original)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag is not None:
            self._continue_drag(event)
            event.accept()
            return
        if self._tool != "none":
            if self._pending:
                self._update_preview(self.mapToScene(event.position().toPoint()))
        elif not self._panning:
            self._update_hover_cursor(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag is not None:
                self._finish_drag()
                event.accept()
                return
            if self._panning:
                self._panning = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._tool == "area_polygon":
            # With fewer than three vertices there is nothing to close yet:
            # the polygon stays open for more clicks.
            if len(self._pending) >= 3:
                self._emit_polygon()
            event.accept()
            return
        if self._tool == "none" and event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit(self.mapToScene(event.position().toPoint()))
            if hit is not None:
                self._drag = None
                self.probe_activated.emit(hit)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            if self._drag is not None:
                self._drag = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self._redraw_overlay()
            elif self._tool != "none":
                self.cancel_placement()
            else:
                self._select(None)
            event.accept()
            return
        super().keyPressEvent(event)

    def _continue_drag(self, event: QMouseEvent) -> None:
        drag = self._drag
        assert drag is not None
        if not drag.moved:
            delta = event.position() - drag.press_view
            if abs(delta.x()) + abs(delta.y()) < _DRAG_THRESHOLD:
                return
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        scene = self.mapToScene(event.position().toPoint())
        dx = scene.x() - drag.press_scene.x()
        dy = scene.y() - drag.press_scene.y()
        if drag.handle is None:
            current = translate(drag.original, dx, dy)
        else:
            hx, hy = control_points(drag.original)[drag.handle]
            current = move_control_point(drag.original, drag.handle,
                                         hx + dx, hy + dy) or drag.current
        self._drag = replace(drag, current=current, moved=True)
        self._redraw_overlay()

    def _finish_drag(self) -> None:
        drag, self._drag = self._drag, None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if drag is not None and drag.moved and drag.current != drag.original:
            self.probe_edited.emit(drag.probe_id, drag.current)
        self._redraw_overlay()

    def _update_hover_cursor(self, scene: QPointF) -> None:
        if self._handle_at(scene) is not None:
            shape = Qt.CursorShape.SizeAllCursor
        elif self._hit(scene) is not None:
            shape = Qt.CursorShape.OpenHandCursor
        else:
            shape = Qt.CursorShape.ArrowCursor
        if self.cursor().shape() != shape:
            self.setCursor(shape)

    # -- shape assembly ---------------------------------------------------

    def _commit_if_complete(self) -> None:
        tool, pts = self._tool, self._pending
        if tool == "point":
            self._emit("point", PointGeom(pts[0].x(), pts[0].y()))
        elif tool in LINE_TOOLS and len(pts) == 2:
            self._emit_guarded(
                "line",
                lambda: LineGeom(pts[0].x(), pts[0].y(), pts[1].x(), pts[1].y()),
            )
        elif tool == "area_rect" and len(pts) == 2:
            self._emit_guarded(
                "area",
                lambda: AreaGeom.rect(
                    pts[0].x(), pts[0].y(), pts[1].x(), pts[1].y()
                ),
            )
        elif tool == "area_circle" and len(pts) == 2:
            radius = (pts[1] - pts[0])
            self._emit_guarded(
                "area",
                lambda: AreaGeom.circle(
                    pts[0].x(), pts[0].y(),
                    (radius.x() ** 2 + radius.y() ** 2) ** 0.5,
                ),
            )

    def _emit_polygon(self) -> None:
        pts = [(p.x(), p.y()) for p in self._pending]

        def build() -> AreaGeom:
            # The floor a reshape applies, so placing cannot store what
            # editing would refuse.
            if not encloses_area(pts):
                raise ValueError("Polygon probe encloses less than a pixel.")
            return AreaGeom.polygon(pts)

        self._emit_guarded("area", build)

    def _emit_guarded(self, kind: str, build) -> None:
        """Build the geometry, discarding a degenerate one silently.

        A zero-length line or zero-radius circle is a double-click, not an
        instruction; refusing it beats storing a probe that can never measure.
        """
        try:
            geometry = build()
        except ValueError:
            self._pending.clear()
            self._clear_preview()
            return
        self._emit(kind, geometry)

    def _emit(self, kind: str, geometry) -> None:
        self._pending.clear()
        self._clear_preview()
        self.set_tool("none")
        self.probe_requested.emit(kind, geometry)

    def _clear_preview(self) -> None:
        self._preview.setPath(QPainterPath())
        if self._preview_label is not None:
            self._scene.removeItem(self._preview_label)
            self._preview_label = None

    def _update_preview(self, cursor: QPointF) -> None:
        path = QPainterPath()
        first = self._pending[0]
        label = None
        if self._tool in LINE_TOOLS:
            path.moveTo(first)
            path.lineTo(cursor)
            # The gauge length as it is drawn: standards fix L0, so people
            # place an extensometer to a length, not just between two spots.
            label = self._format_length(
                math.hypot(cursor.x() - first.x(), cursor.y() - first.y()))
        elif self._tool == "area_rect":
            path.addRect(
                min(first.x(), cursor.x()), min(first.y(), cursor.y()),
                abs(cursor.x() - first.x()), abs(cursor.y() - first.y()),
            )
        elif self._tool == "area_circle":
            r = ((cursor.x() - first.x()) ** 2
                 + (cursor.y() - first.y()) ** 2) ** 0.5
            path.addEllipse(first, r, r)
        elif self._tool == "area_polygon":
            path.moveTo(first)
            for p in self._pending[1:]:
                path.lineTo(p)
            path.lineTo(cursor)
        self._preview.setPath(path)
        if self._preview_label is not None:
            self._scene.removeItem(self._preview_label)
            self._preview_label = None
        if label is not None:
            self._preview_label = ProbeLabel(label, COLORS.TEXT_PRIMARY, cursor)
            self._scene.addItem(self._preview_label)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _pen(colour: QColor, width: float) -> QPen:
    pen = QPen(colour, width)
    pen.setCosmetic(True)               # constant width on screen
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _halo_pen(width: float) -> QPen:
    """A dark rim under every mark, so any probe colour reads on any field."""
    return _pen(QColor(0, 0, 0, 150), width + 2.0)


def _path_item(path: QPainterPath, pen: QPen, z: float, *,
               brush: QColor | None = None, at: QPointF | None = None,
               rotation: float = 0.0) -> QGraphicsPathItem:
    """A path in scene pixels, or -- with *at* -- in screen pixels at *at*."""
    item = QGraphicsPathItem(path)
    item.setPen(pen)
    item.setBrush(QBrush(brush) if brush is not None
                  else QBrush(Qt.BrushStyle.NoBrush))
    item.setZValue(z)
    if at is not None:
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        item.setPos(at)
        item.setRotation(rotation)
    return item


def _marker_path() -> QPainterPath:
    """A ringed cross in screen pixels, open at the centre so the exact spot
    stays visible."""
    r, arm = _POINT_RADIUS, _POINT_RADIUS * 1.8
    path = QPainterPath()
    path.addEllipse(QPointF(0.0, 0.0), r, r)
    for x0, y0, x1, y1 in ((-arm, 0, -r, 0), (r, 0, arm, 0),
                           (0, -arm, 0, -r), (0, r, 0, arm)):
        path.moveTo(x0, y0)
        path.lineTo(x1, y1)
    return path


def _tick_path() -> QPainterPath:
    """A knife edge across a gauge end, drawn perpendicular once rotated."""
    path = QPainterPath()
    path.moveTo(0.0, -_TICK_HALF)
    path.lineTo(0.0, _TICK_HALF)
    return path


def _outline_path(geom: Geometry) -> QPainterPath:
    path = QPainterPath()
    if isinstance(geom, LineGeom):
        path.moveTo(geom.x0, geom.y0)
        path.lineTo(geom.x1, geom.y1)
    elif geom.shape == "rect":
        x0, y0, x1, y1 = geom.data  # type: ignore[misc]
        path.addRect(x0, y0, x1 - x0, y1 - y0)
    elif geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        path.addEllipse(QPointF(cx, cy), r, r)
    else:
        vertices = control_points(geom)
        path.moveTo(*vertices[0])
        for x, y in vertices[1:]:
            path.lineTo(x, y)
        path.closeSubpath()
    return path


def _handle_points(geom: Geometry) -> list[tuple[float, float]]:
    return [] if isinstance(geom, PointGeom) else control_points(geom)


def _handle_item(x: float, y: float) -> QGraphicsRectItem:
    half = _HANDLE_SIZE / 2.0
    item = QGraphicsRectItem(-half, -half, _HANDLE_SIZE, _HANDLE_SIZE)
    item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
    item.setPos(x, y)
    item.setPen(QPen(QColor(COLORS.BG_DARKEST), 1.0))
    item.setBrush(QBrush(QColor(COLORS.TEXT_PRIMARY)))
    item.setZValue(_HANDLE_Z)
    return item


def _label_anchor(geom: Geometry) -> QPointF:
    if isinstance(geom, PointGeom):
        return QPointF(geom.x, geom.y)
    if isinstance(geom, LineGeom):
        return QPointF((geom.x0 + geom.x1) / 2, (geom.y0 + geom.y1) / 2)
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return QPointF(cx, cy - r)
    x, y = min(control_points(geom), key=lambda p: (p[1], p[0]))
    return QPointF(x, y)


__all__ = ["LINE_TOOLS", "ProbeCanvas", "ProbeLabel", "Tool"]
