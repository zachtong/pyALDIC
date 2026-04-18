"""Sticky-header overlay for a QScrollArea of CollapsibleSection widgets.

As the user scrolls the settings area, each CollapsibleSection header
that has scrolled past the top of the viewport is pinned inside a
stacked bar at the top. Clicking a pinned header toggles that section
just like clicking the original (Q2-A).

All headers pin even when their section is collapsed (Q3-A), so the
full outline is always visible.

Stacked-style pinning (Q1-B): later sections appear BELOW earlier
sticky sections as the user scrolls further, freezing each at its
natural order. No 'iOS push-out' behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from al_dic.gui.theme import COLORS

if TYPE_CHECKING:
    from al_dic.gui.widgets.collapsible_section import CollapsibleSection


class _StickyHeaderProxy(QWidget):
    """Lightweight clone of CollapsibleSection's header.

    Mirrors arrow + title so the user can't tell a stuck header apart
    from the real one. Click forwards to the owning section's toggle.
    """

    def __init__(self, section: "CollapsibleSection") -> None:
        super().__init__()
        self._section = section
        self.setFixedHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Slightly stronger BG than content so the stack is visually
        # distinct from scrolled-through content below.
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            f"background: {COLORS.BG_CANVAS}; "
            f"border-bottom: 1px solid {COLORS.BORDER};"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.setSpacing(6)

        self._arrow = QLabel("▾" if section.expanded else "▸")
        self._arrow.setFixedWidth(12)
        self._arrow.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 10px; "
            f"background: transparent;"
        )
        lay.addWidget(self._arrow)

        self._label = QLabel(section.title)
        self._label.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; "
            f"font-weight: bold; letter-spacing: 1px; "
            f"background: transparent;"
        )
        lay.addWidget(self._label)
        lay.addStretch()

        section.toggled.connect(self._on_section_toggled)

    def _on_section_toggled(self, expanded: bool) -> None:
        self._arrow.setText("▾" if expanded else "▸")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # Re-use the section's own click handler so toggle behaviour
        # is identical. Q2-A semantics.
        self._section._on_header_click(event)


class StickyHeadersOverlay(QWidget):
    """Pinned stack of section headers at the top of a scroll viewport."""

    def __init__(
        self,
        scroll_area,  # QScrollArea
        sections: "list[CollapsibleSection]",
    ) -> None:
        super().__init__(scroll_area.viewport())
        self._scroll = scroll_area
        self._sections = sections

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Pass-through for areas NOT occupied by proxy rows; the
        # proxies themselves stay clickable because they're child
        # widgets with their own mouse tracking.
        self.setStyleSheet("background: transparent;")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._proxies: list[_StickyHeaderProxy] = []
        for s in sections:
            p = _StickyHeaderProxy(s)
            p.hide()
            self._layout.addWidget(p)
            self._proxies.append(p)

        # Recompute stack whenever scroll position or geometry changes.
        self._scroll.verticalScrollBar().valueChanged.connect(self._refresh)
        self._scroll.viewport().installEventFilter(self)
        # Toggling a section changes its geometry; recompute afterwards.
        for s in sections:
            s.toggled.connect(lambda *_: self._refresh())

        self._refresh()

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._refresh()
        return super().eventFilter(obj, event)

    def _refresh(self) -> None:
        """Compute which sections' headers should be stuck at the top.

        A header is considered 'stuck' iff, in viewport coordinates, its
        top edge would be above the cumulative height of the stuck stack
        so far (stacked style). Walking sections in order, we keep
        appending to the stack until a section's header edge falls
        below the growing stack bottom — that section is still below
        the stack and its original header is visible in the scrolled
        content.
        """
        vp = self._scroll.viewport()
        vp_w = vp.width()
        if vp_w <= 0:
            return
        self.setFixedWidth(vp_w)
        stack_bottom = 0
        any_visible = False
        for proxy, section in zip(self._proxies, self._sections):
            # Header widget's top-left in viewport coordinates.
            pt = section.header_widget.mapTo(
                vp, section.header_widget.rect().topLeft(),
            )
            # A section is 'above the stack' if its original header
            # top is at or above stack_bottom (proxy would cover it).
            if pt.y() <= stack_bottom:
                proxy.show()
                proxy.setFixedWidth(vp_w)
                stack_bottom += proxy.height()
                any_visible = True
            else:
                proxy.hide()

        # Resize self to just the stack (or hide entirely).
        if any_visible:
            self.setFixedHeight(stack_bottom)
            self.move(0, 0)
            self.show()
            self.raise_()
        else:
            self.hide()
