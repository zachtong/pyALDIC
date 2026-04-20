"""Physical-units configuration widget — pixel size and frame rate.

Shared between the main window and the strain post-processing window via
:class:`AppState` signals.  A change in either window is automatically
reflected in the other through the ``physical_units_changed`` signal.

Layout mirrors the VelocitySettingsWidget style:

    [checkbox] Use physical units
    Pixel size  [spinbox] [unit▼ (nm / µm / mm / cm / m / inch)] / px
    Frame rate  [spinbox] fps
    <info label>

When *Use physical units* is checked:

* Displacement fields are multiplied by *pixel_size* and displayed in the
  chosen unit.
* Velocity is multiplied by *pixel_size × frame_rate* and displayed in
  ``<unit>/s``.
* Strain fields remain dimensionless (no conversion applied).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from al_dic.gui.app_state import AppState


_UNIT_OPTIONS: tuple[str, ...] = ("nm", "µm", "mm", "cm", "m", "inch")


class PhysicalUnitsWidget(QWidget):
    """Toggle + pixel-size (with unit dropdown) + frame-rate wired to AppState."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        self._updating = False

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Toggle
        self._enabled_cb = QCheckBox(self.tr("Use physical units"))
        self._enabled_cb.setChecked(self._state.use_physical_units)
        layout.addRow(self._enabled_cb)

        # Pixel size: [spinbox] [unit combo] / px
        self._pixel_spin = QDoubleSpinBox()
        self._pixel_spin.setDecimals(4)
        self._pixel_spin.setRange(1e-9, 1e9)
        self._pixel_spin.setSingleStep(0.1)
        self._pixel_spin.setValue(self._state.pixel_size)
        self._pixel_spin.setEnabled(self._state.use_physical_units)
        self._pixel_spin.setToolTip(self.tr("Physical size of one image pixel"))

        self._unit_combo = QComboBox()
        for u in _UNIT_OPTIONS:
            self._unit_combo.addItem(u)
        stored = self._state.pixel_unit
        idx = _UNIT_OPTIONS.index(stored) if stored in _UNIT_OPTIONS else 2
        self._unit_combo.setCurrentIndex(idx)
        self._unit_combo.setEnabled(self._state.use_physical_units)

        px_row = QHBoxLayout()
        px_row.setSpacing(4)
        px_row.setContentsMargins(0, 0, 0, 0)
        px_row.addWidget(self._pixel_spin, 1)
        px_row.addWidget(self._unit_combo)
        px_row.addWidget(QLabel(self.tr("/ px")))
        px_widget = QWidget()
        px_widget.setLayout(px_row)
        layout.addRow("Pixel size", px_widget)

        # Frame rate
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setDecimals(3)
        self._fps_spin.setRange(1e-6, 1e9)
        self._fps_spin.setSingleStep(1.0)
        self._fps_spin.setValue(self._state.frame_rate)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setEnabled(self._state.use_physical_units)
        self._fps_spin.setToolTip(self.tr("Acquisition frame rate (used for velocity field)"))
        layout.addRow("Frame rate", self._fps_spin)

        # Computed unit info label
        self._unit_label = QLabel()
        self._unit_label.setStyleSheet("font-size: 10px; color: gray;")
        layout.addRow(self._unit_label)
        self._update_unit_label()

        # Wire user controls → AppState
        self._enabled_cb.toggled.connect(self._on_user_changed)
        self._pixel_spin.valueChanged.connect(self._on_user_changed)
        self._unit_combo.currentIndexChanged.connect(self._on_user_changed)
        self._fps_spin.valueChanged.connect(self._on_user_changed)

        # Sync from AppState when the other window makes a change
        self._state.physical_units_changed.connect(self._on_state_changed)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_user_changed(self, *_args: object) -> None:
        if self._updating:
            return
        enabled = self._enabled_cb.isChecked()
        self._pixel_spin.setEnabled(enabled)
        self._unit_combo.setEnabled(enabled)
        self._fps_spin.setEnabled(enabled)
        self._update_unit_label()
        self._state.set_physical_units(
            enabled,
            self._pixel_spin.value(),
            self._fps_spin.value(),
            self._unit_combo.currentText(),
        )

    def _on_state_changed(self) -> None:
        """Sync widget to AppState when changed from elsewhere."""
        self._updating = True
        self._enabled_cb.blockSignals(True)
        self._pixel_spin.blockSignals(True)
        self._unit_combo.blockSignals(True)
        self._fps_spin.blockSignals(True)

        enabled = self._state.use_physical_units
        self._enabled_cb.setChecked(enabled)
        self._pixel_spin.setValue(self._state.pixel_size)
        stored = self._state.pixel_unit
        idx = _UNIT_OPTIONS.index(stored) if stored in _UNIT_OPTIONS else 2
        self._unit_combo.setCurrentIndex(idx)
        self._fps_spin.setValue(self._state.frame_rate)
        self._pixel_spin.setEnabled(enabled)
        self._unit_combo.setEnabled(enabled)
        self._fps_spin.setEnabled(enabled)
        self._update_unit_label()

        self._enabled_cb.blockSignals(False)
        self._pixel_spin.blockSignals(False)
        self._unit_combo.blockSignals(False)
        self._fps_spin.blockSignals(False)
        self._updating = False

    def _update_unit_label(self) -> None:
        unit = self._unit_combo.currentText()
        if self._enabled_cb.isChecked():
            self._unit_label.setText(f"Disp: {unit}  Velocity: {unit}/s")
        else:
            self._unit_label.setText(self.tr("Disp: px  Velocity: px/fr"))
