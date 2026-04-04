"""Collapsible section widget — clickable header that shows/hides content."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from staq_dic.gui.theme import COLORS


class CollapsibleSection(QWidget):
    """Section with a clickable header that toggles content visibility.

    The header shows a triangle arrow (▸ / ▾) plus a title and optional
    badge.  Content widgets are added via ``add_widget()`` or
    ``set_content_layout()``.
    """

    def __init__(
        self,
        title: str,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = expanded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Header (clickable) ---
        self._header = QWidget()
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setFixedHeight(28)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(12, 4, 12, 4)
        header_layout.setSpacing(6)

        self._arrow = QLabel("▾" if expanded else "▸")
        self._arrow.setFixedWidth(12)
        self._arrow.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 10px; background: transparent;"
        )
        header_layout.addWidget(self._arrow)

        self._label = QLabel(title)
        self._label.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; "
            f"font-weight: bold; letter-spacing: 1px; background: transparent;"
        )
        header_layout.addWidget(self._label)

        self._badge = QLabel("")
        self._badge.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 10px; "
            f"background: {COLORS.BG_INPUT}; border-radius: 7px; "
            f"padding: 1px 6px;"
        )
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.hide()
        header_layout.addWidget(self._badge)

        header_layout.addStretch()
        outer.addWidget(self._header)

        # --- Content area ---
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._content.setVisible(expanded)
        outer.addWidget(self._content)

        # Click handler
        self._header.mousePressEvent = self._on_header_click

    def _on_header_click(self, _event) -> None:
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._arrow.setText("▾" if self._expanded else "▸")

    def set_badge(self, text: str) -> None:
        """Show or hide the badge (e.g., image count)."""
        if text:
            self._badge.setText(text)
            self._badge.show()
        else:
            self._badge.hide()

    def add_widget(self, widget: QWidget, **kwargs) -> None:
        """Add a widget to the content area."""
        self._content_layout.addWidget(widget, **kwargs)

    def content_layout(self) -> QVBoxLayout:
        """Access the content layout for custom additions."""
        return self._content_layout

    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Programmatically expand or collapse the section."""
        self._expanded = expanded
        self._content.setVisible(expanded)
        self._arrow.setText("▾" if expanded else "▸")
