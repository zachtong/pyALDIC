"""Visualization controls for the strain post-processing window.

Exposes:

* **Colormap** -- defaults to ``jet`` (consistent with main window default).
* **Auto range** -- when on, vmin/vmax are derived from the field's
  data range. When off, the user dials in absolute bounds. Emits
  :attr:`auto_disabled` when the user switches from Auto → Manual so
  the parent window can populate the spinboxes with the current field's
  min/max.
* **vmin / vmax** -- manual bounds, only enabled when auto is off.
* **Opacity** -- overlay opacity slider 0-100, returned as a float in
  ``[0, 1]`` from :meth:`get_state`.
* **Show on deformed** -- when True, the parent window renders the
  overlay at displaced node positions and loads the deformed frame image
  as the background.
* **Fill trimmed edges** -- off by default. When on, the edge-trimmed
  strain band is re-interpolated from reliable interior nodes rather than
  blanked, for the on-screen view and exported images/animations only.
  Exported data files (NPZ/MAT/CSV) always keep the trim as NaN.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QWidget,
)

from al_dic.core.colormaps import COLORMAP_NAMES
from al_dic.gui.widgets.double_spin import LocaleSafeDoubleSpinBox
from al_dic.gui.widgets.range_mode import AutoFixedSelector

#: Alias so existing references read unchanged; the list itself lives in
#: al_dic.core.colormaps, so every chooser offers the same options.
_COLORMAP_OPTIONS: tuple[str, ...] = COLORMAP_NAMES


class StrainVizPanel(QWidget):
    """Compose colormap / range / opacity / deformed controls."""

    viz_changed = Signal()

    # Fires when the user switches from Auto → Manual so the parent window
    # can push the current field's min/max into the spinboxes.
    auto_disabled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- Where the field goes (first: sets rendering mode before the
        # styling controls).  Geometry and background are separate questions;
        # this mirrors the main window's sidebar exactly.
        self._geometry_combo = QComboBox()
        self._geometry_combo.addItem(self.tr("Deformed frame"), True)
        self._geometry_combo.addItem(self.tr("Reference frame"), False)
        self._geometry_combo.setToolTip(self.tr(
            "Plot the field at the deformed node positions, or at their "
            "positions in the reference frame."
        ))
        layout.addRow(self.tr("Show on"), self._geometry_combo)

        self._background_check = QCheckBox(self.tr("Show background image"))
        self._background_check.setChecked(True)
        self._background_check.setToolTip(self.tr(
            "Uncheck to show the field on its own, with no speckle image "
            "behind it."
        ))
        layout.addRow(self.tr("Background"), self._background_check)

        self._hidden_bg_combo = QComboBox()
        for _lbl, _val in ((self.tr("White"), "white"),
                           (self.tr("Black"), "black"),
                           (self.tr("Transparent"), "transparent")):
            self._hidden_bg_combo.addItem(_lbl, _val)
        self._hidden_bg_combo.setToolTip(self.tr(
            "What replaces the image when it is hidden. Transparency is "
            "kept for PNG and TIFF on export; other formats get white."
        ))
        self._hidden_bg_combo.setEnabled(False)
        layout.addRow(self.tr("Hidden background"), self._hidden_bg_combo)

        # --- Colormap ---
        self._cmap_combo = QComboBox()
        for name in _COLORMAP_OPTIONS:
            self._cmap_combo.addItem(name)
        self._cmap_combo.setCurrentIndex(0)   # jet
        layout.addRow(self.tr("Colormap"), self._cmap_combo)

        # --- Range mode: explicit Auto / Fixed choice ---
        self._auto_check = AutoFixedSelector()
        layout.addRow(self.tr("Range"), self._auto_check)

        # --- Manual Min / Max (disabled while auto is on) ---
        self._vmin_spin = LocaleSafeDoubleSpinBox()
        self._vmin_spin.setDecimals(6)
        self._vmin_spin.setRange(-1e9, 1e9)
        self._vmin_spin.setSingleStep(1e-3)
        self._vmin_spin.setValue(-0.01)
        self._vmin_spin.setEnabled(False)

        self._vmax_spin = LocaleSafeDoubleSpinBox()
        self._vmax_spin.setDecimals(6)
        self._vmax_spin.setRange(-1e9, 1e9)
        self._vmax_spin.setSingleStep(1e-3)
        self._vmax_spin.setValue(0.01)
        self._vmax_spin.setEnabled(False)

        minmax_row = QHBoxLayout()
        minmax_row.setSpacing(4)
        minmax_row.setContentsMargins(0, 0, 0, 0)
        minmax_row.addWidget(QLabel(self.tr("Min")))
        minmax_row.addWidget(self._vmin_spin, 1)
        minmax_row.addWidget(QLabel(self.tr("Max")))
        minmax_row.addWidget(self._vmax_spin, 1)
        minmax_widget = QWidget()
        minmax_widget.setLayout(minmax_row)
        layout.addRow(minmax_widget)

        # --- Opacity ---
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(70)
        layout.addRow(self.tr("Opacity"), self._opacity_slider)

        # --- Fill trimmed edges (display only) ---
        # Off by default: the edge-trimmed strain band is blanked so the trim
        # is visible.  When on, that band is re-interpolated from the reliable
        # interior nodes -- for the on-screen view AND exported images /
        # animations only.  Exported data files (NPZ/MAT/CSV) always keep the
        # trimmed edge as NaN regardless of this toggle.
        self._fill_edges_check = QCheckBox(
            self.tr("Fill trimmed edges (display only)")
        )
        self._fill_edges_check.setChecked(False)
        self._fill_edges_check.setToolTip(self.tr(
            "Re-interpolate the edge-trimmed strain band from reliable "
            "interior nodes. Affects the on-screen view and exported "
            "images/animations; exported data files always keep the "
            "trimmed edge as NaN."
        ))
        layout.addRow(self.tr("Edges"), self._fill_edges_check)

        # Wire signals
        self._cmap_combo.currentIndexChanged.connect(self._emit_changed)
        self._auto_check.toggled.connect(self._on_auto_toggled)
        self._vmin_spin.valueChanged.connect(self._emit_changed)
        self._vmax_spin.valueChanged.connect(self._emit_changed)
        self._opacity_slider.valueChanged.connect(self._emit_changed)
        self._geometry_combo.currentIndexChanged.connect(self._emit_changed)
        self._background_check.toggled.connect(self._on_background_toggled)
        self._hidden_bg_combo.currentIndexChanged.connect(self._emit_changed)
        self._fill_edges_check.toggled.connect(self._emit_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, object]:
        """Return the current viz configuration as a plain dict."""
        return {
            "colormap": self._cmap_combo.currentText(),
            "use_percentile": self._auto_check.isChecked(),
            "vmin": float(self._vmin_spin.value()),
            "vmax": float(self._vmax_spin.value()),
            "alpha": float(self._opacity_slider.value()) / 100.0,
            "show_deformed": bool(self._geometry_combo.currentData()),
            "show_background": self._background_check.isChecked(),
            "hidden_bg_color": str(self._hidden_bg_combo.currentData()),
            "fill_trimmed_edges": self._fill_edges_check.isChecked(),
        }

    def _on_background_toggled(self, shown: bool) -> None:
        self._hidden_bg_combo.setEnabled(not shown)
        self._emit_changed()

    def set_hidden_bg_color(self, color: str) -> None:
        """Adopt the shared fill without re-emitting a change."""
        idx = self._hidden_bg_combo.findData(color)
        if idx < 0 or idx == self._hidden_bg_combo.currentIndex():
            return
        self._hidden_bg_combo.blockSignals(True)
        self._hidden_bg_combo.setCurrentIndex(idx)
        self._hidden_bg_combo.blockSignals(False)

    def set_range(self, vmin: float, vmax: float) -> None:
        """Populate vmin/vmax spinboxes programmatically without extra signal.

        Called by StrainWindow when the user switches to manual mode so the
        spinboxes start at the current field's data range rather than the
        arbitrary defaults.
        """
        self._vmin_spin.blockSignals(True)
        self._vmax_spin.blockSignals(True)
        self._vmin_spin.setValue(vmin)
        self._vmax_spin.setValue(vmax)
        self._vmin_spin.blockSignals(False)
        self._vmax_spin.blockSignals(False)
        self._emit_changed()

    def load_field_state(
        self,
        auto: bool,
        vmin: float,
        vmax: float,
        colormap: str,
    ) -> None:
        """Load per-field color state into controls without emitting signals.

        Called by StrainWindow when switching fields so each field "remembers"
        its own colormap and auto/manual range setting.
        """
        self._auto_check.blockSignals(True)
        self._cmap_combo.blockSignals(True)
        self._vmin_spin.blockSignals(True)
        self._vmax_spin.blockSignals(True)
        try:
            self._auto_check.setChecked(auto)
            self._vmin_spin.setEnabled(not auto)
            self._vmax_spin.setEnabled(not auto)
            self._vmin_spin.setValue(vmin)
            self._vmax_spin.setValue(vmax)
            idx = self._cmap_combo.findText(colormap)
            if idx >= 0:
                self._cmap_combo.setCurrentIndex(idx)
        finally:
            self._auto_check.blockSignals(False)
            self._cmap_combo.blockSignals(False)
            self._vmin_spin.blockSignals(False)
            self._vmax_spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_auto_toggled(self, checked: bool) -> None:
        self._vmin_spin.setEnabled(not checked)
        self._vmax_spin.setEnabled(not checked)
        if not checked:
            # Signal parent window to push current field range into spinboxes
            self.auto_disabled.emit()
        self._emit_changed()

    def _emit_changed(self, *_args: object) -> None:
        self.viz_changed.emit()
