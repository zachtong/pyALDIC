"""Strain-only parameter panel for the post-processing window.

Exposes the four knobs whitelisted by ``StrainController.ALLOWED_OVERRIDES``:

* ``method_to_compute_strain`` (2 = plane fit, 3 = FEM nodal)
* ``strain_plane_fit_rad`` (px)
* ``strain_smoothness``
* ``strain_type`` (0 = infinitesimal, 1 = Eulerian, 2 = Green-Lagrangian)

Tracks a dirty flag so the window can show a "Stale" hint until the
user explicitly recomputes.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QWidget,
)


class StrainParamPanel(QWidget):
    """Compose method/rad/smoothness/type editors with a dirty flag."""

    params_dirty = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- Method ---
        self._method_combo = QComboBox()
        # Display labels paired with the integer code stored in DICPara
        self._method_codes = (2, 3)
        self._method_combo.addItem("Plane fitting (method 2)")
        self._method_combo.addItem("FEM nodal (method 3)")
        self._method_combo.setCurrentIndex(0)
        layout.addRow("Method", self._method_combo)

        # --- Plane-fit radius ---
        self._rad_spin = QDoubleSpinBox()
        self._rad_spin.setDecimals(2)
        self._rad_spin.setRange(1.0, 200.0)
        self._rad_spin.setSingleStep(1.0)
        self._rad_spin.setSuffix(" px")
        self._rad_spin.setValue(20.0)
        layout.addRow("Plane-fit rad", self._rad_spin)

        # --- Smoothness ---
        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setDecimals(7)
        self._smooth_spin.setRange(0.0, 1.0)
        self._smooth_spin.setSingleStep(1e-5)
        self._smooth_spin.setValue(1e-5)
        layout.addRow("Smoothness", self._smooth_spin)

        # --- Strain type ---
        self._type_combo = QComboBox()
        self._type_codes = (0, 1, 2)
        self._type_combo.addItem("Infinitesimal")
        self._type_combo.addItem("Eulerian")
        self._type_combo.addItem("Green-Lagrangian")
        self._type_combo.setCurrentIndex(0)
        layout.addRow("Strain type", self._type_combo)

        self._dirty = False

        # Wire dirty propagation
        self._method_combo.currentIndexChanged.connect(self._mark_dirty)
        self._rad_spin.valueChanged.connect(self._mark_dirty)
        self._smooth_spin.valueChanged.connect(self._mark_dirty)
        self._type_combo.currentIndexChanged.connect(self._mark_dirty)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_override(self) -> dict[str, object]:
        """Return the current parameter values as a dict suitable for
        :meth:`StrainController.compute_all_frames`."""
        return {
            "method_to_compute_strain": self._method_codes[
                self._method_combo.currentIndex()
            ],
            "strain_plane_fit_rad": float(self._rad_spin.value()),
            "strain_smoothness": float(self._smooth_spin.value()),
            "strain_type": self._type_codes[
                self._type_combo.currentIndex()
            ],
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

    def _mark_dirty(self, *_args: object) -> None:
        self._dirty = True
        self.params_dirty.emit()
