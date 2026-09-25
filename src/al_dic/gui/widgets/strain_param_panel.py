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
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from al_dic.gui.app_state import AppState
from al_dic.gui.theme_manager import themed
from al_dic.gui.widgets.double_spin import LocaleSafeDoubleSpinBox
from al_dic.gui.widgets.info_icon import InfoIcon
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
# Smoothness presets. The labels are stored as raw English source keys
# and translated at combo-population time via ``self.tr(...)``; that keeps
# the preset->smoothness float lookup keyed on the source string, so
# ``_resolve_smoothness`` works identically across locales.
_SMOOTH_PRESETS: tuple[tuple[str, float], ...] = (
    ("Off",                              0.0),
    ("Light (\u03c3 = 0.5 \u00d7 step)", 1e-3),
    ("Medium (\u03c3 = 1 \u00d7 step)",  2e-3),
    ("Strong (\u03c3 = 2 \u00d7 step) \u26a0", 4e-3),
)

# Default VSG size in pixels (must be odd).
# rad = (VSG - 1) / 2 = (41 - 1) / 2 = 20 px  (matches prior default).
_DEFAULT_VSG_PX = 41

# Default edge-trim coefficient (alpha). 0.70 is calibrated to where the
# plane-fit error returns to the interior baseline; see DICPara docstring
# and reports/strain_edge_trim_validation.pdf.
_DEFAULT_EDGE_TRIM_ALPHA = 0.70


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

        # VSG label + ⓘ info icon explaining what VSG means, since
        # "Virtual Strain Gauge" is DIC-specific jargon users may not
        # know coming from a general mechanics background.
        vsg_tip = self.tr(
            "VSG (Virtual Strain Gauge) size is the diameter, in pixels, "
            "of the circular region around each mesh node used to fit a "
            "local displacement plane. Strain is then taken as the "
            "plane's slope.\n\n"
            "• Larger VSG → smoother strain, lower spatial resolution.\n"
            "• Smaller VSG → sharper strain, more noise.\n"
            "• Rule of thumb: VSG ≥ 2 × subset step + 1 (default: 41 px).\n\n"
            "Not used when Method = FEM nodal (there, mesh spacing itself "
            "sets the gauge size)."
        )
        vsg_label_row = QHBoxLayout()
        vsg_label_row.setContentsMargins(0, 0, 0, 0)
        vsg_label_row.setSpacing(2)
        vsg_lbl = QLabel(self.tr("VSG size"))
        vsg_label_row.addWidget(vsg_lbl)
        vsg_label_row.addWidget(InfoIcon(vsg_tip))
        vsg_label_row.addStretch()
        vsg_label_widget = QWidget()
        vsg_label_widget.setLayout(vsg_label_row)
        layout.addRow(vsg_label_widget, self._vsg_spin)

        # Live readout: the VSG radius translated to an N x N node window on a
        # uniform mesh (2*floor(rad/step)+1 nodes per axis). Tells the user how
        # many data points the plane fit actually spans. Informational.
        self._vsg_window_readout = QLabel("")
        themed(self._vsg_window_readout,
            "color: {TEXT_MUTED}; font-size: 10px; padding-left: 4px;"
        )
        self._vsg_window_readout.setToolTip(self.tr(
            "Number of mesh nodes per axis inside the circular VSG window on a "
            "uniform mesh: 2 × floor(VSG radius / node spacing) + 1. The plane "
            "fit uses every node within the radius; on a refined mesh the count "
            "varies locally."
        ))
        self._vsg_window_readout.setVisible(False)
        layout.addRow("", self._vsg_window_readout)

        # Inline warning: plane fit needs VSG radius >= subset_step for
        # every node to find >= 3 neighbours; otherwise the strain
        # field collapses to zero. Updated live as VSG or subset step
        # change.
        self._vsg_warning = QLabel("")
        self._vsg_warning.setWordWrap(True)
        themed(self._vsg_warning,
            "color: {WARNING}; font-size: 10px; padding-left: 4px;"
        )
        self._vsg_warning.setVisible(False)
        layout.addRow("", self._vsg_warning)

        # --- Trim low-confidence edges (plane fitting only) ---
        # ROI/hole-edge nodes whose VSG window crosses the boundary get a
        # one-sided, unreliable plane fit. This coefficient (alpha) sets how
        # wide a boundary band to drop: trim band = alpha * VSG radius.
        # 0 = keep all, 0.7 = recommended, 1 = strictest.
        self._edge_trim_spin = LocaleSafeDoubleSpinBox()
        self._edge_trim_spin.setRange(0.0, 1.0)
        self._edge_trim_spin.setSingleStep(0.05)
        self._edge_trim_spin.setDecimals(2)
        self._edge_trim_spin.setValue(_DEFAULT_EDGE_TRIM_ALPHA)
        edge_tip = self.tr(
            "Hides low-confidence strain at ROI / hole edges, where the VSG "
            "window crosses the boundary and the local plane fit becomes "
            "one-sided and unreliable.\n\n"
            "• Coefficient × VSG radius = width of the trimmed boundary band.\n"
            "• 0.00 = keep every node (no trimming).\n"
            "• 0.70 = recommended (trims where edge error rises sharply).\n"
            "• 1.00 = strictest (trim any node whose window touches the edge).\n\n"
            "Only applies when Method = Plane fitting."
        )
        edge_label_row = QHBoxLayout()
        edge_label_row.setContentsMargins(0, 0, 0, 0)
        edge_label_row.setSpacing(2)
        edge_lbl = QLabel(self.tr("Trim low-confidence edges"))
        edge_label_row.addWidget(edge_lbl)
        edge_label_row.addWidget(InfoIcon(edge_tip))
        edge_label_row.addStretch()
        edge_label_widget = QWidget()
        edge_label_widget.setLayout(edge_label_row)
        layout.addRow(edge_label_widget, self._edge_trim_spin)

        # Live readout: how many nodes the current trim removes. Updated by
        # the window after each strain render. Informational, not a warning.
        self._edge_trim_readout = QLabel("")
        themed(self._edge_trim_readout,
            "color: {TEXT_MUTED}; font-size: 10px; padding-left: 4px;"
        )
        self._edge_trim_readout.setVisible(False)
        layout.addRow("", self._edge_trim_readout)

        # --- Strain field smoothing ---
        # Applies Gaussian smoothing to the computed strain tensor field
        # after differentiation (not to the displacement field).
        # Kernel width scales with local mesh spacing.
        self._smooth_combo = QComboBox()
        # Literal tr() calls per preset — pyside6-lupdate only picks
        # strings up when they appear as literal arguments to tr(),
        # so iterating `self.tr(label)` over a variable would leave
        # these four strings out of the .ts catalog.
        _smooth_labels = (
            self.tr("Off", "Strain smoothing preset"),
            self.tr("Light (σ = 0.5 × step)"),
            self.tr("Medium (σ = 1 × step)"),
            self.tr("Strong (σ = 2 × step) ⚠"),
        )
        for lbl in _smooth_labels:
            self._smooth_combo.addItem(lbl)
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
        self._edge_trim_spin.valueChanged.connect(self._mark_dirty)
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
            "strain_edge_trim_alpha": float(self._edge_trim_spin.value()),
        }

    def is_dirty(self) -> bool:
        """Return True if any parameter changed since the last
        :meth:`mark_clean` call."""
        return self._dirty

    def mark_clean(self) -> None:
        """Reset the dirty flag (typically after a successful compute)."""
        self._dirty = False

    def set_trim_readout(self, n_trimmed: int, n_total: int) -> None:
        """Update the 'Trimmed: N nodes (M%)' readout.

        Called by the strain window after rendering a strain frame, using that
        frame's ``StrainResult.strain_valid``. Pass ``n_total <= 0`` (or call
        while Method != Plane fitting) to hide the readout.
        """
        plane_fit = self._method_combo.currentIndex() == 0
        if n_total <= 0 or not plane_fit:
            self._edge_trim_readout.setVisible(False)
            return
        pct = 100.0 * n_trimmed / n_total
        self._edge_trim_readout.setText(
            tr_args(self.tr("Trimmed: %1 nodes (%2%)"), n_trimmed, f"{pct:.0f}")
        )
        self._edge_trim_readout.setVisible(True)

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
        """Show / enable VSG size + edge trim only for plane fitting (method 2)."""
        code = self._method_codes[index]
        self._vsg_spin.setEnabled(code == 2)
        self._edge_trim_spin.setEnabled(code == 2)
        if code != 2:
            self._edge_trim_readout.setVisible(False)
        self._refresh_vsg_warning()

    def _set_vsg_minimum(self, min_vsg: int) -> None:
        """Raise the VSG spinbox floor to ``min_vsg`` (forced odd, >= 3).

        Keeps the current value when it already meets the floor; otherwise bumps
        it up. Signals are blocked so re-deriving the floor from the DIC step
        does not spuriously mark the panel dirty or recurse through the warning.
        """
        min_vsg = max(3, min_vsg | 1)  # odd and at least 3
        if self._vsg_spin.minimum() == min_vsg:
            return
        self._vsg_spin.blockSignals(True)
        self._vsg_spin.setMinimum(min_vsg)
        if self._vsg_spin.value() < min_vsg:
            self._vsg_spin.setValue(min_vsg)
        self._vsg_spin.blockSignals(False)

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
            self._vsg_window_readout.setVisible(False)
            return
        subset_step = int(getattr(
            AppState.instance(), "subset_step", 8,
        ) or 8)
        recommended_vsg = 2 * subset_step + 1
        # Tie the VSG floor to the DIC node spacing (a plane fit needs its
        # radius >= the node spacing to find neighbours), replacing the abstract
        # fixed 3-px minimum. rad >= step then holds, so the warning below is a
        # belt-and-suspenders that only fires if a caller sets a smaller value.
        self._set_vsg_minimum(recommended_vsg)
        rad = (self._vsg_spin.value() - 1) / 2.0

        # Translate the VSG radius to the N x N node window on a uniform mesh:
        # a node's k-th axial neighbour sits k*step away, so k <= floor(rad/step)
        # nodes fit on each side -> 2*floor(rad/step)+1 per axis. This is exactly
        # what query_ball_point(rad) selects on a regular grid.
        n_axis = 2 * (int(rad) // subset_step) + 1 if subset_step > 0 else 1
        self._vsg_window_readout.setText(
            tr_args(self.tr("Strain window ≈ %1×%2 nodes"), n_axis, n_axis)
        )
        self._vsg_window_readout.setVisible(True)

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
