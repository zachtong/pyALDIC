"""Small legend widget shown in the canvas corner during seed mode.

Reminds the user what the yellow markers / red region fills mean
while they're placing or reviewing Starting Points. Hidden in all
other init-guess modes.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from al_dic.gui.theme import COLORS


# Colors mirror canvas_area: seed marker (yellow) + unseeded region (red).
_COLOR_SEED = QColor(234, 179, 8)       # YELLOW
_COLOR_UNSEEDED = QColor(239, 68, 68)   # RED
_BG = QColor(0, 0, 0, 180)
_TEXT = QColor(255, 255, 255)


class SeedLegendOverlay(QWidget):
    """Tiny top-right overlay explaining seed mode glyphs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Let mouse events fall through to the canvas (zoom / pan etc)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedSize(190, 70)
        self.setVisible(False)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        p.setBrush(_BG)
        p.setPen(QPen(QColor(COLORS.BORDER), 1))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        p.setPen(QPen(_TEXT, 1))
        font = p.font()
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)

        # Title
        p.drawText(10, 18, "Starting Points")

        font.setBold(False)
        font.setPointSize(8)
        p.setFont(font)

        # Row 1 — yellow dot + 'Starting Point'
        p.setBrush(_COLOR_SEED)
        p.setPen(QPen(_TEXT.darker(150), 1))
        cx, cy = 17, 36
        r = 5
        p.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
        p.setPen(QPen(_TEXT, 1))
        p.drawText(32, cy + 4, "Starting Point")

        # Row 2 — red square + 'Needs a Starting Point'
        p.setBrush(QColor(_COLOR_UNSEEDED.red(), _COLOR_UNSEEDED.green(),
                          _COLOR_UNSEEDED.blue(), 180))
        p.setPen(QPen(_TEXT.darker(150), 1))
        sx, sy = 12, 50
        p.drawRect(sx, sy, 10, 10)
        p.setPen(QPen(_TEXT, 1))
        p.drawText(32, sy + 9, "Region needs one")
