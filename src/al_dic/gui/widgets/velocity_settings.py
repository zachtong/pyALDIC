"""Physics-unit settings for the Velocity field in the strain window.

When the user selects the Velocity display field, this widget appears
in the right panel offering an optional physical-unit conversion:

    velocity [px/frame]  ->  velocity [<unit>/s]

Conversion: vel_physical = vel_px_frame * pixel_size [unit/px] * fps [frame/s]

The widget is HIDDEN when a non-velocity field is selected; the parent
window shows/hides it via :meth:`set_visible_for_field`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
)


_UNIT_OPTIONS: tuple[str, ...] = ("nm", "µm", "mm", "cm", "m", "inch")


class VelocitySettingsWidget(QWidget):
    """Toggle physical unit conversion for the velocity display field."""

    settings_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toggle physical units
        self._use_phy = QCheckBox(self.tr("Use physical units"))
        self._use_phy.setChecked(False)
        layout.addRow(self._use_phy)

        # Pixel size: [spinbox] [unit combo] / pixel
        self._px_size_spin = QDoubleSpinBox()
        self._px_size_spin.setDecimals(4)
        self._px_size_spin.setRange(1e-9, 1e9)
        self._px_size_spin.setValue(1.0)
        self._px_size_spin.setSingleStep(0.1)
        self._px_size_spin.setEnabled(False)

        self._unit_combo = QComboBox()
        for u in _UNIT_OPTIONS:
            self._unit_combo.addItem(u)
        self._unit_combo.setCurrentIndex(2)  # default: mm
        self._unit_combo.setEnabled(False)

        px_row = QHBoxLayout()
        px_row.setSpacing(4)
        px_row.addWidget(self._px_size_spin, 1)
        px_row.addWidget(self._unit_combo)
        px_row.addWidget(QLabel(self.tr("/ px")))
        px_widget = QWidget()
        px_widget.setLayout(px_row)
        layout.addRow("Pixel size", px_widget)

        # Frame rate: [spinbox] frame/s
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setDecimals(2)
        self._fps_spin.setRange(1e-6, 1e9)
        self._fps_spin.setValue(1.0)
        self._fps_spin.setSingleStep(1.0)
        self._fps_spin.setEnabled(False)
        layout.addRow("Frame rate (fps)", self._fps_spin)

        # Computed unit display
        self._unit_label = QLabel(self.tr("Unit: px/frame"))
        self._unit_label.setStyleSheet("font-size: 10px; color: gray;")
        layout.addRow(self._unit_label)

        # Wire
        self._use_phy.toggled.connect(self._on_toggle)
        self._px_size_spin.valueChanged.connect(self._on_value_changed)
        self._unit_combo.currentIndexChanged.connect(self._on_value_changed)
        self._fps_spin.valueChanged.connect(self._on_value_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_visible_for_field(self, field_name: str) -> None:
        """Show this widget only when the velocity field is active."""
        self.setVisible(field_name == "velocity")

    def get_config(self) -> dict[str, object]:
        """Return the current conversion configuration.

        Returns:
            dict with keys:
                ``use_physical`` (bool),
                ``pixel_size`` (float, user-entered value in selected unit/px),
                ``unit`` (str, selected unit label),
                ``fps`` (float, frames per second),
                ``result_unit`` (str, display unit for the velocity field).
        """
        use = self._use_phy.isChecked()
        unit = self._unit_combo.currentText()
        fps = float(self._fps_spin.value())
        px = float(self._px_size_spin.value())
        return {
            "use_physical": use,
            "pixel_size": px,
            "unit": unit,
            "fps": fps,
            "result_unit": f"{unit}/s" if use else "px/frame",
        }

    def apply_conversion(
        self, vel_pxframe: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], str]:
        """Convert velocity from px/frame to physical units (if enabled).

        Args:
            vel_pxframe: Velocity magnitude in pixels per frame.

        Returns:
            (converted_array, unit_label) where ``unit_label`` is e.g.
            ``"mm/s"`` or ``"px/frame"``.
        """
        cfg = self.get_config()
        if not cfg["use_physical"]:
            return vel_pxframe, "px/frame"
        # vel [unit/s] = vel [px/frame] * pixel_size [unit/px] * fps [frame/s]
        converted = vel_pxframe * float(cfg["pixel_size"]) * float(cfg["fps"])
        return converted, str(cfg["result_unit"])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_toggle(self, checked: bool) -> None:
        self._px_size_spin.setEnabled(checked)
        self._unit_combo.setEnabled(checked)
        self._fps_spin.setEnabled(checked)
        self._update_unit_label()
        self.settings_changed.emit()

    def _on_value_changed(self, *_args: object) -> None:
        self._update_unit_label()
        self.settings_changed.emit()

    def _update_unit_label(self) -> None:
        cfg = self.get_config()
        self._unit_label.setText(f"Unit: {cfg['result_unit']}")
