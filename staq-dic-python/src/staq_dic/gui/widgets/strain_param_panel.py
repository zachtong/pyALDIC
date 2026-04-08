"""Strain-only parameter panel for the post-processing window.

Exposes:

* ``method_to_compute_strain`` (2 = plane fitting, 3 = FEM nodal)
* ``strain_plane_fit_rad`` (px) -- derived from VSG size; only enabled for
  plane fitting method
* Post-gradient strain smoothing (checkbox) -> ``strain_smoothness`` preset
  (smooths the gradient field F_raw, NOT the displacement field U)
* ``strain_type`` (0 = infinitesimal, 1 = Eulerian, 2 = Green-Lagrangian)

VSG size (Virtual Strain Gauge diameter in pixels) replaces the raw
plane-fit radius. Conversion: ``rad = (VSG - 1) / 2``.  FEM nodal method
hides the VSG control since gauge size is determined by mesh spacing.

Tracks a dirty flag so the window can show a "Stale" hint until the
user explicitly recomputes.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)


# Smoothness presets for pre-smooth dropdown.
# Each value is passed directly to StrainController as strain_smoothness.
# At step=16 px: factor = 500 * smoothness, sigma = spacing * factor
#   Light  : factor~0.25, sigma~4 px
#   Medium : factor~1.0,  sigma~16 px (one mesh step)
#   Strong : factor~4.0,  sigma~64 px
_SMOOTH_PRESETS: tuple[tuple[str, float], ...] = (
    ("Light  (σ ≈ 4 px)",  5e-4),
    ("Medium (σ ≈ 1 step)", 2e-3),
    ("Strong (σ ≈ 4 steps)", 8e-3),
)

# Default VSG size in pixels (must be odd).
# rad = (VSG - 1) / 2 = (41 - 1) / 2 = 20 px  (matches prior default).
_DEFAULT_VSG_PX = 41


class StrainParamPanel(QWidget):
    """Compose method / VSG / pre-smooth / type editors with a dirty flag."""

    params_dirty = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- Method ---
        self._method_combo = QComboBox()
        self._method_codes = (2, 3)
        self._method_combo.addItem("Plane fitting")
        self._method_combo.addItem("FEM nodal")
        self._method_combo.setCurrentIndex(0)   # default: plane fitting (method 2)
        layout.addRow("Method", self._method_combo)

        # --- VSG size (virtual strain gauge diameter, pixels) ---
        # Odd integer: VSG = 2*rad + 1.  Shown only for plane fitting.
        self._vsg_spin = QSpinBox()
        self._vsg_spin.setRange(3, 401)
        self._vsg_spin.setSingleStep(2)
        self._vsg_spin.setSuffix(" px")
        self._vsg_spin.setValue(_DEFAULT_VSG_PX)
        layout.addRow("VSG size", self._vsg_spin)

        # --- Post-gradient strain smoothing ---
        # Applies Gaussian smoothing to the computed gradient field F_raw,
        # NOT to the displacement field U.  Equivalent to smoothing the
        # strain field after differentiation.
        self._presmooth_check = QCheckBox("Smooth strain field (post-grad)")
        self._presmooth_check.setChecked(False)
        layout.addRow("Smoothing", self._presmooth_check)

        self._smooth_combo = QComboBox()
        for label, _ in _SMOOTH_PRESETS:
            self._smooth_combo.addItem(label)
        self._smooth_combo.setCurrentIndex(0)
        self._smooth_combo.setEnabled(False)
        layout.addRow("", self._smooth_combo)

        # --- Strain type ---
        self._type_combo = QComboBox()
        self._type_codes = (0, 1, 2)
        self._type_combo.addItem("Infinitesimal")
        self._type_combo.addItem("Eulerian")
        self._type_combo.addItem("Green-Lagrangian")
        self._type_combo.setCurrentIndex(0)
        layout.addRow("Strain type", self._type_combo)

        self._dirty = False

        # Wire enable/disable for VSG spin based on method
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._on_method_changed(self._method_combo.currentIndex())  # init state

        # Wire pre-smooth toggle
        self._presmooth_check.toggled.connect(self._on_presmooth_toggled)

        # Wire dirty propagation
        self._method_combo.currentIndexChanged.connect(self._mark_dirty)
        self._vsg_spin.valueChanged.connect(self._mark_dirty)
        self._presmooth_check.toggled.connect(self._mark_dirty)
        self._smooth_combo.currentIndexChanged.connect(self._mark_dirty)
        self._type_combo.currentIndexChanged.connect(self._mark_dirty)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_override(self) -> dict[str, object]:
        """Return the current parameter values as a dict suitable for
        :meth:`StrainController.compute_all_frames`.

        VSG size is converted to plane-fit radius: ``rad = (VSG - 1) / 2``.
        """
        method = self._method_codes[self._method_combo.currentIndex()]
        rad = (self._vsg_spin.value() - 1) / 2.0
        return {
            "method_to_compute_strain": method,
            "strain_plane_fit_rad": rad,
            "strain_smoothness": self._resolve_smoothness(),
            "strain_type": self._type_codes[self._type_combo.currentIndex()],
        }

    def is_dirty(self) -> bool:
        """Return True if any parameter changed since the last
        :meth:`mark_clean` call."""
        return self._dirty

    def mark_clean(self) -> None:
        """Reset the dirty flag (typically after a successful compute)."""
        self._dirty = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_smoothness(self) -> float:
        if not self._presmooth_check.isChecked():
            return 0.0
        return _SMOOTH_PRESETS[self._smooth_combo.currentIndex()][1]

    def _on_method_changed(self, index: int) -> None:
        """Show / enable VSG size only for plane fitting (method 2)."""
        code = self._method_codes[index]
        self._vsg_spin.setEnabled(code == 2)

    def _on_presmooth_toggled(self, checked: bool) -> None:
        self._smooth_combo.setEnabled(checked)

    def _mark_dirty(self, *_args: object) -> None:
        self._dirty = True
        self.params_dirty.emit()
