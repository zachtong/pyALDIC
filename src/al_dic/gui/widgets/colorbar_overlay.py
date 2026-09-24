"""Colorbar overlay — vertical gradient bar with ticks, drawn on the canvas.

Positioned absolutely in the parent CanvasArea.  Transparent to mouse
events so it does not interfere with canvas pan/zoom/drawing.
"""

from __future__ import annotations

import numpy as np
from al_dic.core.colormaps import resolve
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QWidget

from al_dic.gui.theme import COLORS

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
    """Generate exactly *n* evenly-spaced tick values from vmin to vmax.

    The bottom tick is always vmin and the top tick is always vmax, with
    n-2 interior ticks dividing the range into n-1 equal intervals.
    Example: vmin=-8, vmax=12, n=5  ->  [-8, -3, 2, 7, 12]
    """
    if vmax <= vmin:
        return [vmin]
    step = (vmax - vmin) / max(n - 1, 1)
    return [float(vmin + i * step) for i in range(n)]


def _format_tick(val: float) -> str:
    """Format a tick value compactly."""
    if abs(val) < 1e-10:
        return "0"
    if abs(val) >= 1000 or (abs(val) < 0.01 and val != 0):
        return f"{val:.2e}"
    # Use integer format when the value is close to an integer
    if abs(val - round(val)) < 1e-6:
        return f"{int(round(val))}"
    if abs(val) < 1:
        return f"{val:.3f}"
    return f"{val:.2f}"


def panel_layout(
    width: int, bar_x: float, tick_w: float, label_w: float,
) -> tuple[float, float]:
    """Left edge of the background panel, and where the label starts.

    The panel used to begin a hard-coded 50 px left of the bar, which clipped
    two different things: any label longer than a short field name, and any
    tick in scientific notation. It is now sized from what it has to hold.

    The bar sits against the right edge, so a label centred on it runs off the
    *widget* long before it runs off the panel -- hence the clamp on both
    sides, computed before the panel rather than after it.

    Args:
        width:    widget width in px.
        bar_x:    left edge of the gradient bar.
        tick_w:   width of the widest tick label.
        label_w:  width of the (already elided) top label; 0 if there is none.
    Returns:
        (panel_x, label_x), both in widget coordinates.
    """
    label_x = bar_x + _BAR_WIDTH / 2 - label_w / 2
    label_x = min(label_x, width - _RIGHT_MARGIN - label_w)
    label_x = max(4.0, label_x)

    panel_x = min(
        bar_x - max(50.0, tick_w + _LABEL_MARGIN + 8.0),
        label_x - 8.0 if label_w else float("inf"),
    )
    return max(0.0, panel_x), label_x


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

        # --- Size the panel to its contents, then draw it ---
        # The left edge used to be a hard-coded 50 px, which elided any label
        # longer than a short field name ("von Mises strain" -> "von M...")
        # and was also narrower than a tick in scientific notation.
        label_font = QFont("Segoe UI", 10)
        label_font.setBold(True)
        tick_font = QFont("Consolas", 9)
        tick_font.setStyleHint(QFont.StyleHint.Monospace)

        ticks = _nice_ticks(self._vmin, self._vmax, n=5)
        tick_texts = [_format_tick(v) for v in ticks]
        p.setFont(tick_font)
        tick_w = max(
            (p.fontMetrics().horizontalAdvance(t) for t in tick_texts),
            default=0,
        )

        p.setFont(label_font)
        fm_label = p.fontMetrics()
        # A long label may still be elided, but only once it would take more
        # than half the canvas -- at that point it is the image that suffers.
        label_cap = max(60, int(w * 0.5))
        display_label = (
            fm_label.elidedText(
                self._label, Qt.TextElideMode.ElideRight, label_cap)
            if self._label else ""
        )
        label_w = fm_label.horizontalAdvance(display_label)

        panel_x, label_x = panel_layout(w, bar_x, tick_w, label_w)
        panel_w = w - panel_x
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(11, 15, 26, 180))  # BG_DARKEST with alpha
        p.drawRoundedRect(
            QRectF(panel_x, bar_top - 20, panel_w, bar_height + 36),
            6, 6,
        )

        # --- Build gradient from matplotlib colormap ---
        cm = resolve(self._cmap_name)

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

        # --- Ticks and labels (measured above) ---
        p.setFont(tick_font)
        p.setPen(QPen(QColor(COLORS.TEXT_PRIMARY), 1))

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

        # --- Label at top (the panel was widened to fit it) ---
        if display_label:
            p.setFont(label_font)
            p.setPen(QPen(QColor(COLORS.TEXT_SECONDARY), 1))
            ly = bar_top - 6
            p.drawText(int(label_x), int(ly), display_label)

        p.end()
