"""DIC parameter input panel.

Subset / mesh / refinement parameters. Workflow-level choices
(tracking mode, solver, reference-update policy) live in
:class:`WorkflowTypePanel` so they can be surfaced above the Region
of Interest section in the sidebar.
"""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.app_state import AppState


class ParamPanel(QWidget):
    """Parameter inputs: Subset Size, Subset Step, Search Range, Refinement."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        state = AppState.instance()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 4, 0)

        # --- Subset Size (display odd, store even internally) ---
        # User sees 21, 41 etc. (odd = 2*half+1 centered window).
        # Internal winsize is even (20, 40 etc. = half-width parameter).
        self._subset_size = self._add_spinbox(
            layout,
            self.tr("Subset Size"),
            state.subset_size + 1,  # internal 40 -> display 41
            minimum=11,
            maximum=201,
            step=2,
            tooltip=self.tr(
                "IC-GN subset window size in pixels (odd number)"),
        )
        self._subset_size.valueChanged.connect(self._on_subset_size_changed)

        # --- Subset Step ---
        self._subset_step = self._add_power_of_2_combo(
            layout,
            self.tr("Subset Step"),
            state.subset_step,
            options=[4, 8, 16, 32, 64],
            tooltip=self.tr("Node spacing in pixels (must be power of 2)"),
        )
        self._subset_step.currentTextChanged.connect(self._on_subset_step_changed)

        # --- Search Range (maximum detectable per-frame displacement) ---
        # Maps to DICPara.size_of_fft_search_region. Label + tooltip swap
        # when init_guess_mode == 'seed_propagation' (see _sync_search_label).
        search_row = QHBoxLayout()
        self._search_lbl = QLabel(self.tr("Search Range"))
        self._search_lbl.setFixedWidth(120)
        self._search_range = QSpinBox()
        self._search_range.setRange(4, 512)
        self._search_range.setSingleStep(2)
        self._search_range.setValue(state.search_range)
        self._search_range.setSuffix(" px")
        search_row.addWidget(self._search_lbl)
        search_row.addWidget(self._search_range)
        layout.addLayout(search_row)
        self._search_range.valueChanged.connect(self._on_search_range_changed)
        state.params_changed.connect(self._sync_search_label)
        self._sync_search_label()

        # --- Mesh Refinement (inner / outer boundary) ---
        self._refine_inner_cb = QCheckBox(self.tr("Refine Inner Boundary"))
        self._refine_inner_cb.setChecked(state.refine_inner)
        self._refine_inner_cb.setToolTip(self.tr(
            "Locally refine the mesh along internal mask boundaries\n"
            "(holes inside the Region of Interest). Useful for bubble / "
            "void edges."
        ))
        self._refine_outer_cb = QCheckBox(self.tr("Refine Outer Boundary"))
        self._refine_outer_cb.setChecked(state.refine_outer)
        self._refine_outer_cb.setToolTip(self.tr(
            "Locally refine the mesh along the outer Region of Interest\n"
            "boundary."
        ))
        # Indent the checkboxes so they visually belong to the subset_step row
        inner_row = QHBoxLayout()
        inner_row.setContentsMargins(120, 0, 0, 0)
        inner_row.addWidget(self._refine_inner_cb)
        layout.addLayout(inner_row)
        outer_row = QHBoxLayout()
        outer_row.setContentsMargins(120, 0, 0, 0)
        outer_row.addWidget(self._refine_outer_cb)
        layout.addLayout(outer_row)

        # Level selector
        self._refine_level = QComboBox()
        self._refine_level.setToolTip(self.tr(
            "Refinement aggressiveness. min element size = "
            "max(2, subset_step / 2^level). Applies uniformly to inner-, "
            "outer-boundary AND brush-painted refinement zones. Available "
            "levels depend on subset size and subset step."
        ))
        level_row = QHBoxLayout()
        self._refine_level_lbl = QLabel(self.tr("Refinement Level"))
        self._refine_level_lbl.setFixedWidth(120)
        level_row.addWidget(self._refine_level_lbl)
        level_row.addWidget(self._refine_level)
        layout.addLayout(level_row)

        self._refine_info_lbl = QLabel()
        self._refine_info_lbl.setContentsMargins(120, 0, 0, 0)
        self._refine_info_lbl.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self._refine_info_lbl)

        # Wire refinement signals
        self._refine_inner_cb.toggled.connect(self._on_refine_inner_toggled)
        self._refine_outer_cb.toggled.connect(self._on_refine_outer_toggled)
        self._refine_level.currentIndexChanged.connect(self._on_refine_level_changed)
        state.roi_changed.connect(self._update_refinement_ui)
        self._update_refinement_ui()

        # Prevent mouse wheel from changing values when widget is unfocused
        for widget in [
            self._subset_size, self._subset_step, self._search_range,
            self._refine_level,
        ]:
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            widget.installEventFilter(self)

    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        """Block wheel events on unfocused spinboxes and combos."""
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)

    def _on_subset_size_changed(self, display_value: int) -> None:
        """Convert odd display value to even internal winsize."""
        if display_value % 2 == 0:
            display_value = display_value + 1
            self._subset_size.blockSignals(True)
            self._subset_size.setValue(display_value)
            self._subset_size.blockSignals(False)
        AppState.instance().set_param("subset_size", display_value - 1)
        self._update_refinement_ui()

    def _on_subset_step_changed(self, value_text: str) -> None:
        """Update subset_step in state and refresh refinement readout."""
        AppState.instance().set_param("subset_step", int(value_text))
        self._update_refinement_ui()

    def _sync_search_label(self) -> None:
        """Switch the Search Range label/tooltip based on init_guess_mode.

        In seed_propagation mode, the parameter only governs the seed
        bootstrap search radius (auto-expands on clipped peaks) rather
        than every-node FFT search; the label makes that explicit.
        """
        state = AppState.instance()
        fft_tip = self.tr(
            "Maximum per-frame displacement the FFT search can detect "
            "(pixels).\n"
            "Set comfortably larger than the expected inter-frame motion.\n"
            "For large rotations in incremental mode, this must cover\n"
            "  radius × sin(per-step angle)."
        )
        seed_tip = self.tr(
            "Initial half-width (pixels) of the single-point NCC search "
            "at each Starting Point.\n"
            "Auto-expands 2x per retry if the peak is clipped, up to "
            "image half-size.\n"
            "Only affects Starting Point bootstrap; other nodes use "
            "F-aware propagation (no per-node search)."
        )
        if state.init_guess_mode == "seed_propagation":
            self._search_lbl.setText(self.tr("Initial Seed Search"))
            self._search_lbl.setToolTip(seed_tip)
            self._search_range.setToolTip(seed_tip)
        else:
            self._search_lbl.setText(self.tr("Search Range"))
            self._search_lbl.setToolTip(fft_tip)
            self._search_range.setToolTip(fft_tip)

    def _on_search_range_changed(self, value: int) -> None:
        """Update search_range in state (mirrors InitGuessWidget pattern)."""
        state = AppState.instance()
        state.search_range = value
        state.params_changed.emit()

    def _on_refine_inner_toggled(self, on: bool) -> None:
        AppState.instance().set_refine_inner(on)
        self._update_refinement_ui()

    def _on_refine_outer_toggled(self, on: bool) -> None:
        AppState.instance().set_refine_outer(on)
        self._update_refinement_ui()

    def _on_refine_level_changed(self, index: int) -> None:
        if index < 0:
            return
        AppState.instance().set_refinement_level(index + 1)
        self._update_min_size_label()

    def _rebuild_level_dropdown(self) -> None:
        """Repopulate the level combo to reflect the current max level."""
        state = AppState.instance()
        max_level = state.compute_max_refinement_level()
        labels = ["Light", "Medium", "Heavy", "Extra Heavy", "Ultra"]
        items = [
            f"{labels[i] if i < len(labels) else f'L{i + 1}'} (L{i + 1})"
            for i in range(max_level)
        ]
        target_level = max(1, min(state.refinement_level, max_level))
        if target_level != state.refinement_level:
            state.set_refinement_level(target_level)

        self._refine_level.blockSignals(True)
        self._refine_level.clear()
        self._refine_level.addItems(items)
        self._refine_level.setCurrentIndex(target_level - 1)
        self._refine_level.blockSignals(False)

    def _update_min_size_label(self) -> None:
        """Refresh just the min-size readout label (no dropdown rebuild)."""
        state = AppState.instance()
        any_on = self._any_refinement_on()
        if any_on:
            min_size = state.compute_refinement_min_size()
            from al_dic.i18n import tr_args
            self._refine_info_lbl.setText(
                tr_args(
                    self.tr(
                        "min element size = %1 px  "
                        "(subset_step=%2, level=%3)"
                    ),
                    min_size, state.subset_step, state.refinement_level,
                )
            )
            self._refine_info_lbl.setVisible(True)
        else:
            self._refine_info_lbl.setVisible(False)

    def _any_refinement_on(self) -> bool:
        """True if any refinement source is currently active."""
        state = AppState.instance()
        if state.refine_inner or state.refine_outer:
            return True
        brush = state.refine_brush_mask
        return brush is not None and bool(brush.any())

    def _update_refinement_ui(self) -> None:
        """Rebuild dropdown + refresh enabled state + refresh readout."""
        self._rebuild_level_dropdown()
        any_on = self._any_refinement_on()
        self._refine_level.setEnabled(any_on)
        self._refine_level_lbl.setEnabled(any_on)
        self._update_min_size_label()

    def _add_spinbox(
        self,
        parent_layout: QVBoxLayout,
        label: str,
        default: int,
        minimum: int = 0,
        maximum: int = 999,
        step: int = 1,
        tooltip: str = "",
    ) -> QSpinBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(120)
        lbl.setToolTip(tooltip)
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(default)
        spin.setToolTip(tooltip)
        row.addWidget(lbl)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return spin

    def _add_power_of_2_combo(
        self,
        parent_layout: QVBoxLayout,
        label: str,
        default: int,
        options: list[int] | None = None,
        tooltip: str = "",
    ) -> QComboBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(120)
        lbl.setToolTip(tooltip)
        combo = QComboBox()
        combo.addItems([str(v) for v in (options or [])])
        combo.setCurrentText(str(default))
        combo.setToolTip(tooltip)
        row.addWidget(lbl)
        row.addWidget(combo)
        parent_layout.addLayout(row)
        return combo
