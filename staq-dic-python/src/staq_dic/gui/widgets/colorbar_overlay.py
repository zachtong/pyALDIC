"""Colorbar overlay — vertical gradient bar with ticks, drawn on the canvas.

Positioned absolutely in the parent CanvasArea.  Transparent to mouse
events so it does not interfere with canvas pan/zoom/drawing.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QWidget

from staq_dic.gui.theme import COLORS

# Number of color stops in the gradient
_N_STOPS = 256

# Layout constants
_BAR_WIDTH = 18
_TICK_LEN = 5
_LABEL_MARGIN = 4
_TOP_MARGIN = 32
_BOTTOM_MARGIN = 16
_RIGHT_MARGIN = 14


def _nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    """Generate ~n nicely-rounded tick values between vmin and vmax."""
    if vmax <= vmin:
        return [vmin]
    raw_step = (vmax - vmin) / max(n - 1, 1)
    # Round step to 1, 2, or 5 × 10^k
    magnitude = 10 ** np.floor(np.log10(max(abs(raw_step), 1e-15)))
    residual = raw_step / magnitude
    if residual <= 1.5:
        nice_step = 1.0 * magnitude
    elif residual <= 3.5:
        nice_step = 2.0 * magnitude
    elif residual <= 7.5:
        nice_step = 5.0 * magnitude
    else:
        nice_step = 10.0 * magnitude

    start = np.ceil(vmin / nice_step) * nice_step
    ticks = []
    val = start
    while val <= vmax + nice_step * 0.01:
        ticks.append(float(val))
        val += nice_step
    # Always include endpoints if they're missing
    if not ticks or ticks[0] > vmin + nice_step * 0.1:
        ticks.insert(0, float(vmin))
    if ticks[-1] < vmax - nice_step * 0.1:
        ticks.append(float(vmax))
    return ticks


def _format_tick(val: float) -> str:
    """Format a tick value compactly."""
    if abs(val) < 1e-10:
        return "0"
    if abs(val) >= 100 or (abs(val) < 0.01 and val != 0):
        return f"{val:.2e}"
    if abs(val) < 1:
        return f"{val:.3f}"
    return f"{val:.2f}"


class ColorbarOverlay(QWidget):
    """Vertical colorbar overlay widget.

    Call ``update_params()`` whenever the colormap, range, or label changes.
    The widget repaints itself with QPainter.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Transparent to mouse events
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setVisible(False)

        self._cmap_name: str = "jet"
        self._vmin: float = 0.0
        self._vmax: float = 1.0
        self._label: str = ""

    def update_params(
        self,
        cmap: str,
        vmin: float,
        vmax: float,
        label: str = "",
    ) -> None:
        """Update colorbar parameters and repaint."""
        changed = (
            cmap != self._cmap_name
            or vmin != self._vmin
            or vmax != self._vmax
            or label != self._label
        )
        self._cmap_name = cmap
        self._vmin = vmin
        self._vmax = vmax
        self._label = label
        if changed:
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self.isVisible():
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Compute bar geometry
        bar_x = w - _RIGHT_MARGIN - _BAR_WIDTH
        bar_top = _TOP_MARGIN
        bar_bottom = h - _BOTTOM_MARGIN
        bar_height = bar_bottom - bar_top
        if bar_height < 40:
            p.end()
            return

        # --- Draw semi-transparent background panel ---
        panel_x = bar_x - 50
        panel_w = w - panel_x
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(11, 15, 26, 180))  # BG_DARKEST with alpha
        p.drawRoundedRect(
            QRectF(panel_x, bar_top - 20, panel_w, bar_height + 36),
            6, 6,
        )

        # --- Build gradient from matplotlib colormap ---
        try:
            cm = colormaps[self._cmap_name]
        except KeyError:
            cm = colormaps["jet"]

        gradient = QLinearGradient(bar_x, bar_bottom, bar_x, bar_top)
        for i in range(_N_STOPS):
            t = i / (_N_STOPS - 1)
            r, g, b, a = cm(t)
            gradient.setColorAt(t, QColor(int(r * 255), int(g * 255), int(b * 255)))

        # Draw gradient bar
        p.setPen(QPen(QColor(COLORS.TEXT_MUTED), 1))
        p.setBrush(gradient)
        bar_rect = QRectF(bar_x, bar_top, _BAR_WIDTH, bar_height)
        p.drawRect(bar_rect)

        # --- Ticks and labels ---
        tick_font = QFont("Consolas", 9)
        tick_font.setStyleHint(QFont.StyleHint.Monospace)
        p.setFont(tick_font)
        p.setPen(QPen(QColor(COLORS.TEXT_PRIMARY), 1))

        ticks = _nice_ticks(self._vmin, self._vmax, n=5)
        for val in ticks:
            if self._vmax > self._vmin:
                frac = (val - self._vmin) / (self._vmax - self._vmin)
            else:
                frac = 0.5
            frac = max(0.0, min(1.0, frac))
            # Bottom = vmin, top = vmax
            y = bar_bottom - frac * bar_height

            # Tick mark
            p.drawLine(
                int(bar_x + _BAR_WIDTH), int(y),
                int(bar_x + _BAR_WIDTH + _TICK_LEN), int(y),
            )

            # Tick label (to the left of the bar)
            text = _format_tick(val)
            text_rect = p.fontMetrics().boundingRect(text)
            tx = bar_x - _LABEL_MARGIN - text_rect.width()
            ty = y + text_rect.height() / 2 - 2
            p.drawText(int(tx), int(ty), text)

        # --- Label at top ---
        if self._label:
            label_font = QFont("Segoe UI", 10)
            label_font.setBold(True)
            p.setFont(label_font)
            p.setPen(QPen(QColor(COLORS.TEXT_SECONDARY), 1))
            label_rect = p.fontMetrics().boundingRect(self._label)
            lx = bar_x + _BAR_WIDTH / 2 - label_rect.width() / 2
            ly = bar_top - 6
            p.drawText(int(lx), int(ly), self._label)

        p.end()
