"""Seven-button exclusive selector for strain fields.

Layout (3 rows, 3 columns):

    [exx]    [eyy]    [exy]
    [e1]     [e2]     [gmax]
    [        von Mises (3 cols)        ]

Mirrors the visual style of :class:`FieldSelector` (ACCENT for active,
BG_INPUT for inactive). Field names match :class:`StrainResult` attribute
names so consumers can ``getattr(result, name)`` directly.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QPushButton,
    QWidget,
)

from staq_dic.gui.theme import COLORS

# Canonical order of fields exposed by the selector. The labels in
# ``_FIELD_LABELS`` map each strain attribute to its on-screen text.
STRAIN_FIELD_NAMES: tuple[str, ...] = (
    "strain_exx",
    "strain_eyy",
    "strain_exy",
    "strain_principal_max",
    "strain_principal_min",
    "strain_maxshear",
    "strain_von_mises",
)

_FIELD_LABELS: dict[str, str] = {
    "strain_exx": "\u03b5xx",                # ε
    "strain_eyy": "\u03b5yy",
    "strain_exy": "\u03b5xy",
    "strain_principal_max": "\u03b5\u2081",   # ε₁
    "strain_principal_min": "\u03b5\u2082",   # ε₂
    "strain_maxshear": "\u03b3 max",          # γ max
    "strain_von_mises": "von Mises",
}


class StrainFieldSelector(QWidget):
    """Seven-button exclusive selector for strain visualization fields."""

    field_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        # Row 0: exx, eyy, exy
        # Row 1: e1, e2, gmax
        # Row 2: von Mises (spans 3 cols)
        positions: dict[str, tuple[int, int, int]] = {
            "strain_exx": (0, 0, 1),
            "strain_eyy": (0, 1, 1),
            "strain_exy": (0, 2, 1),
            "strain_principal_max": (1, 0, 1),
            "strain_principal_min": (1, 1, 1),
            "strain_maxshear": (1, 2, 1),
            "strain_von_mises": (2, 0, 3),
        }
        for name in STRAIN_FIELD_NAMES:
            row, col, span = positions[name]
            btn = QPushButton(_FIELD_LABELS[name])
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            layout.addWidget(btn, row, col, 1, span)
            self._group.addButton(btn)
            self._buttons[name] = btn
            btn.clicked.connect(
                lambda _checked=False, n=name: self._on_clicked(n)
            )

        self._current: str = "strain_exx"
        self._buttons[self._current].setChecked(True)
        self._update_styles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_field(self) -> str:
        """Return the currently selected strain field name."""
        return self._current

    def set_current_field(self, name: str) -> None:
        """Programmatically activate *name*. Emits ``field_changed`` only on
        an actual change."""
        if name not in self._buttons:
            raise ValueError(
                f"Unknown strain field '{name}'. "
                f"Allowed: {STRAIN_FIELD_NAMES}"
            )
        if name == self._current:
            return
        self._current = name
        self._buttons[name].setChecked(True)
        self._update_styles()
        self.field_changed.emit(name)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_clicked(self, name: str) -> None:
        """Handle a user-driven button click."""
        if name == self._current:
            # Re-clicking the active button keeps it active.
            self._buttons[name].setChecked(True)
            return
        self._current = name
        self._update_styles()
        self.field_changed.emit(name)

    def _update_styles(self) -> None:
        for name, btn in self._buttons.items():
            active = name == self._current
            if active:
                btn.setStyleSheet(
                    f"background: {COLORS.ACCENT}; color: white; "
                    f"border: none; border-radius: 4px; font-weight: bold;"
                )
            else:
                btn.setStyleSheet(
                    f"background: {COLORS.BG_INPUT}; "
                    f"color: {COLORS.TEXT_SECONDARY}; "
                    f"border: 1px solid {COLORS.BORDER}; border-radius: 4px;"
                )
