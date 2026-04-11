"""Mesh overlay — element edges, node dots, and hover subset window.

Draws in **scene coordinates** using ``QPainter.setTransform`` with cosmetic
pens (constant-pixel-width regardless of zoom).  QPainterPath objects are
pre-built once in ``set_mesh()`` and reused across pan/zoom repaints —
only ``set_view_transform()`` triggers lightweight repaints.

Parented to the CanvasArea viewport.  Transparent to mouse events.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QTransform
from PySide6.QtWidgets import QWidget

# Visual constants
_EDGE_COLOR = "#ffffff"
_EDGE_WIDTH = 1
_EDGE_OPACITY = 1.0

_NODE_COLOR = "#22c55e"
_NODE_DIAMETER = 3
_NODE_OPACITY = 0.7

_HOVER_COLOR = "#facc15"
_HOVER_WIDTH = 2
_HOVER_OPACITY = 0.8

# Performance thresholds
_MAX_NODES_FOR_DOTS = 4000


class MeshOverlay(QWidget):
    """QPainter-based overlay that draws mesh elements, nodes, and hover window.

    **Architecture**: paths are pre-built in scene coordinates when mesh data
    changes (``set_mesh``).  On pan/zoom the caller updates only the
    QTransform (``set_view_transform``), and ``paintEvent`` applies it via
    ``QPainter.setTransform`` with cosmetic pens — no per-node Python loop.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setVisible(False)

        # Pre-built draw data (scene coordinates)
        self._edge_path: QPainterPath | None = None
        self._node_points: list[QPointF] = []
        self._show_nodes: bool = True

        # Raw data kept for hover lookup
        self._coords: NDArray[np.float64] | None = None
        self._valid: NDArray[np.bool_] | None = None

        # View transform: scene → viewport (updated on pan/zoom)
        self._vt: QTransform = QTransform()

        # Hover state
        self._hover_idx: int | None = None
        self._hover_winsize: float = 0.0

        # User-configurable edge appearance (matches AppState defaults)
        self._edge_color_str: str = _EDGE_COLOR
        self._edge_width: int = _EDGE_WIDTH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mesh(
        self,
        coords: NDArray[np.float64] | None,
        elements: NDArray[np.int64] | None,
        valid: NDArray[np.bool_] | None = None,
    ) -> None:
        """Update mesh data and pre-build draw paths.

        Args:
            coords: (n_nodes, 2) node positions [x, y] in scene coordinates.
            elements: (n_elem, 8) Q4/Q8 element connectivity.
            valid: (n_nodes,) boolean mask — ``True`` for nodes inside ROI.
                   If *None*, all non-NaN nodes are considered valid.
        """
        self._hover_idx = None
        self._hover_winsize = 0.0

        if coords is None or elements is None:
            self._edge_path = None
            self._node_points = []
            self._coords = None
            self._valid = None
            self.update()
            return

        n_nodes = coords.shape[0]

        # Determine valid nodes (inside ROI + non-NaN)
        if valid is None:
            valid = np.ones(n_nodes, dtype=bool)
        valid = valid & ~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1])
        self._coords = coords
        self._valid = valid

        # --- Pre-build element edge path (scene coords) ---
        corners = elements[:, :4]
        elem_ok = np.all(corners >= 0, axis=1)
        # All 4 corner nodes must be valid
        for j in range(4):
            elem_ok &= valid[np.clip(corners[:, j], 0, n_nodes - 1)]
        valid_corners = corners[elem_ok]

        edge_path = QPainterPath()
        cx = coords[valid_corners[:, 0], 0] if len(valid_corners) > 0 else np.array([])
        # Build path — one moveTo + 3 lineTo + close per element
        for i in range(len(valid_corners)):
            c = valid_corners[i]
            edge_path.moveTo(coords[c[0], 0], coords[c[0], 1])
            edge_path.lineTo(coords[c[1], 0], coords[c[1], 1])
            edge_path.lineTo(coords[c[2], 0], coords[c[2], 1])
            edge_path.lineTo(coords[c[3], 0], coords[c[3], 1])
            edge_path.closeSubpath()
        self._edge_path = edge_path

        # --- Pre-build node points (scene coords) ---
        valid_idx = np.where(valid)[0]
        self._show_nodes = len(valid_idx) <= _MAX_NODES_FOR_DOTS
        if self._show_nodes:
            self._node_points = [
                QPointF(float(coords[i, 0]), float(coords[i, 1]))
                for i in valid_idx
            ]
        else:
            self._node_points = []

        self.update()

    def set_appearance(self, color: str, width: int) -> None:
        """Update mesh edge color and width, then repaint.

        Args:
            color: CSS hex string, e.g. ``"#ffffff"`` or ``"#3b82f6"``.
            width: Line width in screen pixels (cosmetic pen, zoom-invariant).
        """
        changed = color != self._edge_color_str or width != self._edge_width
        self._edge_color_str = color
        self._edge_width = max(1, int(width))
        if changed:
            self.update()

    def set_view_transform(self, vt: QTransform) -> None:
        """Update the scene-to-viewport transform (call on every pan/zoom)."""
        self._vt = vt
        self.update()

    def set_hover_node(
        self,
        node_idx: int | None,
        winsize: float = 0.0,
    ) -> None:
        """Set/clear the hover node for subset window display."""
        changed = node_idx != self._hover_idx or winsize != self._hover_winsize
        self._hover_idx = node_idx
        self._hover_winsize = winsize
        if changed:
            self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self.isVisible() or self._edge_path is None:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw in scene coordinates; QPainter handles the transform.
        # Cosmetic pens keep line width constant in screen pixels.
        p.setTransform(self._vt)

        # --- Element edges ---
        edge_color = QColor(self._edge_color_str)
        if not edge_color.isValid():
            edge_color = QColor(_EDGE_COLOR)
        edge_color.setAlphaF(_EDGE_OPACITY)
        pen = QPen(edge_color, self._edge_width)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(self._edge_path)

        # --- Node dots (skip for dense meshes) ---
        if self._show_nodes and self._node_points:
            node_color = QColor(_NODE_COLOR)
            node_color.setAlphaF(_NODE_OPACITY)
            node_pen = QPen(node_color, _NODE_DIAMETER)
            node_pen.setCosmetic(True)
            node_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(node_pen)
            p.drawPoints(self._node_points)

        # --- Hover subset window ---
        if (
            self._hover_idx is not None
            and self._coords is not None
            and self._valid is not None
            and 0 <= self._hover_idx < len(self._valid)
            and self._valid[self._hover_idx]
            and self._hover_winsize > 0
        ):
            hover_color = QColor(_HOVER_COLOR)
            hover_color.setAlphaF(_HOVER_OPACITY)
            hover_pen = QPen(hover_color, _HOVER_WIDTH, Qt.PenStyle.DashLine)
            hover_pen.setCosmetic(True)
            p.setPen(hover_pen)
            p.setBrush(Qt.BrushStyle.NoBrush)

            sx = float(self._coords[self._hover_idx, 0])
            sy = float(self._coords[self._hover_idx, 1])
            half = self._hover_winsize / 2.0
            p.drawRect(QRectF(sx - half, sy - half, self._hover_winsize, self._hover_winsize))

        p.end()
