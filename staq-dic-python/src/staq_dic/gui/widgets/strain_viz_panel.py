"""Visualization controls for the strain post-processing window.

Exposes:

* **Colormap** -- defaults to ``RdBu_r`` (a diverging map). Strain is
  signed so a zero-centered diverging colormap is the natural default.
* **Use percentile** -- when on, ``vmin`` / ``vmax`` are computed from
  the field's 5-95 percentile (matching the existing displacement
  viz behaviour). When off, the user dials in absolute bounds.
* **vmin / vmax** -- manual bounds, only enabled when percentile is off.
* **Alpha** -- overlay opacity slider 0-100, returned as a float in
  ``[0, 1]`` from :meth:`get_state`.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QSlider,
    QWidget,
)

# Diverging maps come first; sequential maps follow as a convenience for
# users who want to inspect a positive scalar like the von-Mises field.
_COLORMAP_OPTIONS: tuple[str, ...] = (
    "RdBu_r",
    "seismic",
    "coolwarm",
    "jet",
    "viridis",
)


class StrainVizPanel(QWidget):
    """Compose colormap / range / alpha controls with a single dirty signal."""

    viz_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- Colormap ---
        self._cmap_combo = QComboBox()
        for name in _COLORMAP_OPTIONS:
            self._cmap_combo.addItem(name)
        self._cmap_combo.setCurrentIndex(0)
        layout.addRow("Colormap", self._cmap_combo)

        # --- Auto percentile vs. manual ---
        self._pct_check = QCheckBox("Auto (5-95 percentile)")
        self._pct_check.setChecked(True)
        layout.addRow("Range", self._pct_check)

        # --- Manual vmin / vmax (disabled while percentile is on) ---
        self._vmin_spin = QDoubleSpinBox()
        self._vmin_spin.setDecimals(6)
        self._vmin_spin.setRange(-1e6, 1e6)
        self._vmin_spin.setSingleStep(1e-3)
        self._vmin_spin.setValue(-0.01)
        self._vmin_spin.setEnabled(False)
        layout.addRow("vmin", self._vmin_spin)

        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setDecimals(6)
        self._vmax_spin.setRange(-1e6, 1e6)
        self._vmax_spin.setSingleStep(1e-3)
        self._vmax_spin.setValue(0.01)
        self._vmax_spin.setEnabled(False)
        layout.addRow("vmax", self._vmax_spin)

        # --- Alpha ---
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(70)
        layout.addRow("Alpha", self._alpha_slider)

        # Wire signals
        self._cmap_combo.currentIndexChanged.connect(self._emit_changed)
        self._pct_check.toggled.connect(self._on_pct_toggled)
        self._vmin_spin.valueChanged.connect(self._emit_changed)
        self._vmax_spin.valueChanged.connect(self._emit_changed)
        self._alpha_slider.valueChanged.connect(self._emit_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, object]:
        """Return the current viz configuration as a plain dict."""
        return {
            "colormap": self._cmap_combo.currentText(),
            "use_percentile": self._pct_check.isChecked(),
            "vmin": float(self._vmin_spin.value()),
            "vmax": float(self._vmax_spin.value()),
            "alpha": float(self._alpha_slider.value()) / 100.0,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_pct_toggled(self, checked: bool) -> None:
        # Enable/disable manual vmin/vmax based on percentile mode
        self._vmin_spin.setEnabled(not checked)
        self._vmax_spin.setEnabled(not checked)
        self._emit_changed()

    def _emit_changed(self, *_args: object) -> None:
        self.viz_changed.emit()
