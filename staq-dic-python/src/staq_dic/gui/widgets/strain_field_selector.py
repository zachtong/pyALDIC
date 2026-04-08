"""Field selector for the strain post-processing window.

Exposes displacement, derived displacement, and strain fields in three
clearly separated sub-grids:

    DISPLACEMENT
        [Disp U]    [Disp V]
        [Magnitude] [Velocity]

    STRAIN
        [exx]    [eyy]    [exy]
        [e1]     [e2]     [gmax]
        [von Mises]  [Rotation w]  [Mean e]

Displacement fields (disp_u / disp_v / disp_magnitude / velocity) are
populated directly from ``result_disp`` and are visible *before* running
Compute Strain.  Strain fields require a completed strain computation.

All field names in :data:`FIELD_NAMES` match :class:`StrainResult` attribute
names OR are special-cased computed fields handled by
``StrainWindow._get_field_values()``.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from staq_dic.gui.theme import COLORS

# ---- Displacement section -------------------------------------------------

DISP_FIELD_NAMES: tuple[str, ...] = (
    "disp_u",
    "disp_v",
    "disp_magnitude",
    "velocity",
)

# ---- Strain section -------------------------------------------------------

STRAIN_FIELD_NAMES: tuple[str, ...] = (
    "strain_exx",
    "strain_eyy",
    "strain_exy",
    "strain_principal_max",
    "strain_principal_min",
    "strain_maxshear",
    "strain_von_mises",
    "strain_rotation",
    "strain_mean_normal",
)

# ---- Combined ---------------------------------------------------------------

FIELD_NAMES: tuple[str, ...] = DISP_FIELD_NAMES + STRAIN_FIELD_NAMES

_FIELD_LABELS: dict[str, str] = {
    "disp_u":                "\u0055",               # U
    "disp_v":                "\u0056",               # V
    "disp_magnitude":        "Magnitude",
    "velocity":              "Velocity",
    "strain_exx":            "\u03b5xx",             # εxx
    "strain_eyy":            "\u03b5yy",
    "strain_exy":            "\u03b5xy",
    "strain_principal_max":  "\u03b5\u2081",         # ε₁
    "strain_principal_min":  "\u03b5\u2082",         # ε₂
    "strain_maxshear":       "\u03b3 max",           # γ max
    "strain_von_mises":      "von Mises",
    "strain_rotation":       "\u03c9 rot",           # ω rot
    "strain_mean_normal":    "\u03b5\u0305 mean",    # ε̄ mean
}

# (row, col, colspan) inside each sub-grid, all using 2 columns for disp
# and 3 columns for strain.
_DISP_POSITIONS: dict[str, tuple[int, int, int]] = {
    "disp_u":         (0, 0, 1),
    "disp_v":         (0, 1, 1),
    "disp_magnitude": (1, 0, 1),
    "velocity":       (1, 1, 1),
}

_STRAIN_POSITIONS: dict[str, tuple[int, int, int]] = {
    "strain_exx":           (0, 0, 1),
    "strain_eyy":           (0, 1, 1),
    "strain_exy":           (0, 2, 1),
    "strain_principal_max": (1, 0, 1),
    "strain_principal_min": (1, 1, 1),
    "strain_maxshear":      (1, 2, 1),
    "strain_von_mises":     (2, 0, 1),
    "strain_rotation":      (2, 1, 1),
    "strain_mean_normal":   (2, 2, 1),
}


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px; "
        f"font-weight: bold; letter-spacing: 1px; margin-top: 4px;"
    )
    return lbl


class StrainFieldSelector(QWidget):
    """Three-section exclusive selector for displacement and strain fields."""

    field_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Single shared button group — ensures cross-section mutual exclusion.
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        # --- DISPLACEMENT section ---
        outer.addWidget(_section_label("DISPLACEMENT"))
        disp_grid = QGridLayout()
        disp_grid.setContentsMargins(0, 0, 0, 0)
        disp_grid.setHorizontalSpacing(4)
        disp_grid.setVerticalSpacing(4)
        for name in DISP_FIELD_NAMES:
            row, col, span = _DISP_POSITIONS[name]
            self._add_button(name, disp_grid, row, col, span)
        outer.addLayout(disp_grid)

        # --- STRAIN section ---
        outer.addWidget(_section_label("STRAIN"))
        strain_grid = QGridLayout()
        strain_grid.setContentsMargins(0, 0, 0, 0)
        strain_grid.setHorizontalSpacing(4)
        strain_grid.setVerticalSpacing(4)
        for name in STRAIN_FIELD_NAMES:
            row, col, span = _STRAIN_POSITIONS[name]
            self._add_button(name, strain_grid, row, col, span)
        outer.addLayout(strain_grid)

        self._current: str = "disp_u"
        self._buttons[self._current].setChecked(True)
        self._update_styles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_field(self) -> str:
        """Return the currently selected field name."""
        return self._current

    def set_current_field(self, name: str) -> None:
        """Programmatically activate *name*. Emits ``field_changed`` only on
        an actual change.

        Raises:
            ValueError: If *name* is not a recognised field.
        """
        if name not in self._buttons:
            raise ValueError(
                f"Unknown field '{name}'. Allowed: {FIELD_NAMES}"
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

    def _add_button(
        self,
        name: str,
        grid: QGridLayout,
        row: int,
        col: int,
        col_span: int,
    ) -> None:
        btn = QPushButton(_FIELD_LABELS[name])
        btn.setCheckable(True)
        btn.setFixedHeight(28)
        grid.addWidget(btn, row, col, 1, col_span)
        self._group.addButton(btn)
        self._buttons[name] = btn
        btn.clicked.connect(
            lambda _checked=False, n=name: self._on_clicked(n)
        )

    def _on_clicked(self, name: str) -> None:
        if name == self._current:
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
