"""Physical-units configuration widget — pixel size and frame rate.

Shared between the main window and the strain post-processing window via
:class:`AppState` signals.  A change in either window is automatically
reflected in the other through the ``physical_units_changed`` signal.

When *Use physical units* is checked:

* Displacement fields (U, V, magnitude) are multiplied by *pixel_size* and
  displayed in microns (μm).
* Velocity is multiplied by *pixel_size × frame_rate* and displayed in μm/s.
* Strain fields remain dimensionless (no conversion applied).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QWidget,
)

from staq_dic.gui.app_state import AppState


class PhysicalUnitsWidget(QWidget):
    """Toggle + pixel-size + frame-rate controls wired to :class:`AppState`."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        self._updating = False  # guard against re-entrant signal loops

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._enabled_cb = QCheckBox("Use physical units")
        self._enabled_cb.setChecked(self._state.use_physical_units)
        layout.addRow(self._enabled_cb)

        self._pixel_spin = QDoubleSpinBox()
        self._pixel_spin.setRange(0.001, 1_000_000.0)
        self._pixel_spin.setDecimals(4)
        self._pixel_spin.setSingleStep(0.1)
        self._pixel_spin.setValue(self._state.pixel_size)
        self._pixel_spin.setSuffix(" \u03bcm/px")
        self._pixel_spin.setEnabled(self._state.use_physical_units)
        self._pixel_spin.setToolTip("Physical size of one image pixel in micrometres")
        layout.addRow("Pixel size", self._pixel_spin)

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.001, 1_000_000.0)
        self._fps_spin.setDecimals(3)
        self._fps_spin.setSingleStep(1.0)
        self._fps_spin.setValue(self._state.frame_rate)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setEnabled(self._state.use_physical_units)
        self._fps_spin.setToolTip("Acquisition frame rate — used to convert velocity to μm/s")
        layout.addRow("Frame rate", self._fps_spin)

        # Wire user controls → AppState
        self._enabled_cb.toggled.connect(self._on_user_changed)
        self._pixel_spin.valueChanged.connect(self._on_user_changed)
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
        self._fps_spin.setEnabled(enabled)
        self._state.set_physical_units(
            enabled,
            self._pixel_spin.value(),
            self._fps_spin.value(),
        )

    def _on_state_changed(self) -> None:
        """Sync widget to AppState when changed from elsewhere."""
        self._updating = True
        self._enabled_cb.blockSignals(True)
        self._pixel_spin.blockSignals(True)
        self._fps_spin.blockSignals(True)

        self._enabled_cb.setChecked(self._state.use_physical_units)
        self._pixel_spin.setValue(self._state.pixel_size)
        self._fps_spin.setValue(self._state.frame_rate)
        self._pixel_spin.setEnabled(self._state.use_physical_units)
        self._fps_spin.setEnabled(self._state.use_physical_units)

        self._enabled_cb.blockSignals(False)
        self._pixel_spin.blockSignals(False)
        self._fps_spin.blockSignals(False)
        self._updating = False
