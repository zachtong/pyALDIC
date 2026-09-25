"""Shared ⓘ info icon widget.

A small clickable ⓘ glyph that surfaces its tooltip on hover OR click.

Tooltip alone is fragile: touchscreens never trigger hover, and some
users miss that the icon is interactive. A click also shows the same
tooltip, pinned at the cursor, so discoverability doesn't depend on
knowing the hover convention.

Originally lived inside ``init_guess_widget.py``; extracted here so
``strain_param_panel.py`` (and future widgets) can reuse the same
affordance.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolTip, QWidget

from al_dic.gui.theme_manager import themed


class InfoIcon(QLabel):
    """Clickable ⓘ glyph that shows its tooltip on both hover and click."""

    def __init__(self, tip: str, parent: QWidget | None = None) -> None:
        super().__init__("\u24d8", parent)  # U+24D8 CIRCLED LATIN SMALL LETTER I
        self.setToolTip(tip)
        self._tip_text = tip
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        themed(self,
            "color: {TEXT_SECONDARY}; font-size: 13px; "
            "padding: 0 4px;"
        )

    def set_tip(self, tip: str) -> None:
        """Update the tooltip text (used by retranslate_ui hooks)."""
        self._tip_text = tip
        self.setToolTip(tip)

    def mousePressEvent(self, event):  # noqa: N802 (Qt override)
        QToolTip.showText(event.globalPos(), self._tip_text, self)
        super().mousePressEvent(event)


def make_info_icon(tip: str) -> InfoIcon:
    """Factory that returns a ready-to-add InfoIcon."""
    return InfoIcon(tip)


def label_with_info(label: QLabel, tip: str) -> QHBoxLayout:
    """Compose a horizontal row: label + stretch + InfoIcon.

    Convenience for form-row layouts where the icon should sit at the
    right edge next to the label.
    """
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    row.addWidget(label)
    row.addStretch()
    row.addWidget(InfoIcon(tip))
    return row
