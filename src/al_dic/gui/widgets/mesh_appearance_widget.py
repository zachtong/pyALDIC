"""Mesh appearance controls — line color picker and width spinner."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.app_state import AppState
from al_dic.gui.theme import COLORS


class MeshAppearanceWidget(QWidget):
    """Controls for mesh edge color and line width.

    Writes changes directly to AppState and emits ``display_changed``
    so CanvasArea can pick them up via ``_on_display_changed``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- Line color row ---
        color_row = QHBoxLayout()
        color_row.setSpacing(6)
        color_row.addWidget(QLabel(self.tr("Mesh color")))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(24, 18)
        self._color_btn.setToolTip(self.tr("Click to choose mesh line color"))
        self._color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self._color_btn)
        color_row.addStretch()
        layout.addLayout(color_row)

        # --- Line width row ---
        width_row = QHBoxLayout()
        width_row.setSpacing(6)
        width_row.addWidget(QLabel(self.tr("Line width")))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 8)
        self._width_spin.setValue(1)
        self._width_spin.setSuffix(" px")
        self._width_spin.setFixedWidth(72)
        self._width_spin.valueChanged.connect(self._on_width_changed)
        width_row.addWidget(self._width_spin)
        width_row.addStretch()
        layout.addLayout(width_row)

        # Sync from current state on construction
        self._sync_from_state()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sync_from_state(self) -> None:
        state = AppState.instance()
        self._set_button_color(state.mesh_line_color)
        self._width_spin.blockSignals(True)
        self._width_spin.setValue(state.mesh_line_width)
        self._width_spin.blockSignals(False)

    def _set_button_color(self, hex_color: str) -> None:
        """Apply hex_color as the button's background."""
        self._color_btn.setStyleSheet(
            f"QPushButton {{ background: {hex_color}; "
            f"border: 1px solid {COLORS.BORDER}; border-radius: 3px; }}"
        )

    def _pick_color(self) -> None:
        state = AppState.instance()
        initial = QColor(state.mesh_line_color)
        color = QColorDialog.getColor(
            initial, self, "Choose mesh line color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not color.isValid():
            return
        hex_color = color.name()   # "#rrggbb"
        state.mesh_line_color = hex_color
        self._set_button_color(hex_color)
        state.display_changed.emit()

    def _on_width_changed(self, value: int) -> None:
        state = AppState.instance()
        state.mesh_line_width = value
        state.display_changed.emit()
