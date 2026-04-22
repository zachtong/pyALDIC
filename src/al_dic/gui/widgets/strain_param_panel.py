"""Strain-only parameter panel for the post-processing window.

Exposes:

* ``method_to_compute_strain`` (2 = plane fitting, 3 = FEM nodal)
* ``strain_plane_fit_rad`` (px) -- derived from VSG size; only enabled for
  plane fitting method
* Strain field smoothing dropdown -> ``strain_smoothness`` preset
  (smooths the strain tensor field after computation; "Off" disables it)
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
    QComboBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from al_dic.gui.app_state import AppState
from al_dic.i18n import tr_args


# Smoothness presets for the strain-field smoothing dropdown.
# Each value is passed directly to StrainController as strain_smoothness.
# Internal scaling: factor = 500 * smoothness, sigma = node_local_spacing * factor
#
# Design rationale (sigma/step ratio determines effective smoothing):
#   Off            : no smoothing
#   Light (0.5x)   : nearest neighbors contribute ~38% of value
#   Medium (1x)    : nearest neighbors contribute ~84% (recommended)
#   Strong (2x)    : nearest neighbors contribute ~96% (may blur real gradients)
# Below sigma/step = 0.25 the Gaussian is too narrow to reach any neighbor
# (neighbor weight ~0.03%), so smoothing effectively does nothing.
_SMOOTH_PRESETS: tuple[tuple[str, float], ...] = (
    ("Off",                       0.0),
    ("Light (\u03c3 = 0.5 \u00d7 step)",    1e-3),
    ("Medium (\u03c3 = 1 \u00d7 step)",     2e-3),
    ("Strong (\u03c3 = 2 \u00d7 step) ⚠", 4e-3),
)

# Default VSG size in pixels (must be odd).
# rad = (VSG - 1) / 2 = (41 - 1) / 2 = 20 px  (matches prior default).
_DEFAULT_VSG_PX = 41


class StrainParamPanel(QWidget):
    """Compose method / VSG / smoothing / type editors with a dirty flag."""

    params_dirty = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- Method ---
        self._method_combo = QComboBox()
        self._method_codes = (2, 3)
        self._method_combo.addItem(self.tr("Plane fitting"))
        self._method_combo.addItem(self.tr("FEM nodal"))
        self._method_combo.setCurrentIndex(0)   # default: plane fitting (method 2)
        layout.addRow(self.tr("Method"), self._method_combo)

        # --- VSG size (virtual strain gauge diameter, pixels) ---
        # Odd integer: VSG = 2*rad + 1.  Shown only for plane fitting.
        self._vsg_spin = QSpinBox()
        self._vsg_spin.setRange(3, 401)
        self._vsg_spin.setSingleStep(2)
        self._vsg_spin.setSuffix(" px")
        self._vsg_spin.setValue(_DEFAULT_VSG_PX)
        layout.addRow(self.tr("VSG size"), self._vsg_spin)

        # Inline warning: plane fit needs VSG radius >= subset_step for
        # every node to find >= 3 neighbours; otherwise the strain
        # field collapses to zero. Updated live as VSG or subset step
        # change.
        self._vsg_warning = QLabel("")
        self._vsg_warning.setWordWrap(True)
        self._vsg_warning.setStyleSheet(
            "color: #d97706; font-size: 10px; padding-left: 4px;"
        )
        self._vsg_warning.setVisible(False)
        layout.addRow("", self._vsg_warning)

        # --- Strain field smoothing ---
        # Applies Gaussian smoothing to the computed strain tensor field
        # after differentiation (not to the displacement field).
        # Kernel width scales with local mesh spacing.
        self._smooth_combo = QComboBox()
        for label, _ in _SMOOTH_PRESETS:
            self._smooth_combo.addItem(self.tr(label))
        self._smooth_combo.setCurrentIndex(0)   # default: Off
        self._smooth_combo.setToolTip(self.tr(
            "Gaussian smoothing of the strain field after computation.\n"
            "σ is the Gaussian kernel width; 'step' = DIC node spacing.\n"
            "  Light  (0.5 × step):  subtle, preserves fine features.\n"
            "  Medium (1 × step):    balanced, recommended for noisy data.\n"
            "  Strong (2 × step) ⚠:  aggressive, may blur real gradients."
        ))
        layout.addRow(self.tr("Strain field smoothing"), self._smooth_combo)

        # --- Strain type ---
        self._type_combo = QComboBox()
        self._type_codes = (0, 1, 2)
        self._type_combo.addItem(self.tr("Infinitesimal"))
        self._type_combo.addItem(self.tr("Eulerian"))
        self._type_combo.addItem(self.tr("Green-Lagrangian"))
        self._type_combo.setCurrentIndex(0)
        layout.addRow(self.tr("Strain type"), self._type_combo)

        self._dirty = False

        # Wire enable/disable for VSG spin based on method
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._on_method_changed(self._method_combo.currentIndex())  # init state

        # Wire dirty propagation
        self._method_combo.currentIndexChanged.connect(self._mark_dirty)
        self._vsg_spin.valueChanged.connect(self._on_vsg_value_changed)
        self._smooth_combo.currentIndexChanged.connect(self._mark_dirty)
        self._type_combo.currentIndexChanged.connect(self._mark_dirty)

        # Refresh VSG warning whenever the DIC subset_step changes
        # (mesh refinement, param panel edit, session load, …).
        AppState.instance().params_changed.connect(self._refresh_vsg_warning)
        self._refresh_vsg_warning()

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
        return _SMOOTH_PRESETS[self._smooth_combo.currentIndex()][1]

    def _on_vsg_value_changed(self, value: int) -> None:
        """Snap even inputs to the next odd integer, then mark dirty."""
        if value % 2 == 0:
            self._vsg_spin.blockSignals(True)
            self._vsg_spin.setValue(value + 1)
            self._vsg_spin.blockSignals(False)
        self._refresh_vsg_warning()
        self._mark_dirty()

    def _on_method_changed(self, index: int) -> None:
        """Show / enable VSG size only for plane fitting (method 2)."""
        code = self._method_codes[index]
        self._vsg_spin.setEnabled(code == 2)
        self._refresh_vsg_warning()

    def _refresh_vsg_warning(self) -> None:
        """Warn when VSG radius is smaller than the DIC node spacing.

        Plane fit needs >= 3 valid neighbours within the VSG radius at
        every node. When `rad < subset_step`, many nodes (or all of
        them, if rad < step) have zero neighbours within radius, the
        whole F field comes back NaN, and fill_nan_idw's all-NaN path
        kicks in. The strain compute now raises loudly in that case,
        but warning the user *before* they hit Compute is nicer.
        """
        if self._method_combo.currentIndex() != 0:  # Plane fitting only
            self._vsg_warning.setVisible(False)
            return
        subset_step = int(getattr(
            AppState.instance(), "subset_step", 8,
        ) or 8)
        rad = (self._vsg_spin.value() - 1) / 2.0
        recommended_vsg = 2 * subset_step + 1
        if rad < subset_step:
            msg = tr_args(
                self.tr(
                    "⚠ VSG radius (%1 px) < DIC node spacing (%2 px); "
                    "plane fit will fail. Use VSG ≥ %3 px or switch "
                    "Method to FEM nodal."
                ),
                int(rad), subset_step, recommended_vsg,
            )
            self._vsg_warning.setText(msg)
            self._vsg_warning.setVisible(True)
        else:
            self._vsg_warning.setVisible(False)

    def _mark_dirty(self, *_args: object) -> None:
        self._dirty = True
        self.params_dirty.emit()
