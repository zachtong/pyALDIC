"""DIC parameter input panel."""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.app_state import AppState


class ParamPanel(QWidget):
    """Parameter inputs: Subset Size, Subset Step, Search Range, Tracking Mode."""

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
            "Subset Size",
            state.subset_size + 1,  # internal 40 -> display 41
            minimum=11,
            maximum=201,
            step=2,
            tooltip="IC-GN subset window size in pixels (odd number)",
        )
        self._subset_size.valueChanged.connect(self._on_subset_size_changed)

        # --- Subset Step ---
        self._subset_step = self._add_power_of_2_combo(
            layout,
            "Subset Step",
            state.subset_step,
            options=[4, 8, 16, 32, 64],
            tooltip="Node spacing in pixels (must be power of 2)",
        )
        self._subset_step.currentTextChanged.connect(self._on_subset_step_changed)

        # --- Search Range (maximum detectable per-frame displacement) ---
        # Maps to DICPara.size_of_fft_search_region. Fundamental DIC parameter;
        # promoted from the ADVANCED collapsible so it is visible by default.
        self._search_range = self._add_spinbox(
            layout,
            "Search Range",
            state.search_range,
            minimum=4,
            maximum=512,
            step=2,
            tooltip=(
                "Maximum per-frame displacement the FFT search can detect "
                "(pixels).\n"
                "Set comfortably larger than the expected inter-frame motion.\n"
                "For large rotations in incremental mode, this must cover\n"
                "  radius \u00d7 sin(per-step angle)."
            ),
        )
        self._search_range.setSuffix(" px")
        self._search_range.valueChanged.connect(self._on_search_range_changed)

        # --- Mesh Refinement (inner / outer boundary) ---
        # Two independent toggles + a single level selector that controls how
        # aggressively the mesh is refined near the active criteria.
        self._refine_inner_cb = QCheckBox("Refine Inner Boundary")
        self._refine_inner_cb.setChecked(state.refine_inner)
        self._refine_inner_cb.setToolTip(
            "Locally refine the mesh along internal mask boundaries\n"
            "(holes inside the Region of Interest). Useful for bubble / "
            "void edges."
        )
        self._refine_outer_cb = QCheckBox("Refine Outer Boundary")
        self._refine_outer_cb.setChecked(state.refine_outer)
        self._refine_outer_cb.setToolTip(
            "Locally refine the mesh along the outer Region of Interest\n"
            "boundary."
        )
        # Indent the checkboxes so they visually belong to the subset_step row
        inner_row = QHBoxLayout()
        inner_row.setContentsMargins(120, 0, 0, 0)
        inner_row.addWidget(self._refine_inner_cb)
        layout.addLayout(inner_row)
        outer_row = QHBoxLayout()
        outer_row.setContentsMargins(120, 0, 0, 0)
        outer_row.addWidget(self._refine_outer_cb)
        layout.addLayout(outer_row)

        # Level selector — items are populated dynamically from
        # AppState.compute_max_refinement_level() so the choices honor both
        # the integer constraint (subset_step / 2^level >= 2) and the
        # physical constraint (node spacing >= subset_size / 4).
        self._refine_level = QComboBox()
        self._refine_level.setToolTip(
            "Refinement aggressiveness. min element size = "
            "max(2, subset_step / 2^level). Applies uniformly to inner-, "
            "outer-boundary AND brush-painted refinement zones. Available "
            "levels depend on subset size and subset step."
        )
        level_row = QHBoxLayout()
        self._refine_level_lbl = QLabel("Refinement Level")
        self._refine_level_lbl.setFixedWidth(120)
        level_row.addWidget(self._refine_level_lbl)
        level_row.addWidget(self._refine_level)
        layout.addLayout(level_row)

        # Live readout of the computed min element size
        self._refine_info_lbl = QLabel()
        self._refine_info_lbl.setContentsMargins(120, 0, 0, 0)
        self._refine_info_lbl.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self._refine_info_lbl)

        # Wire refinement signals
        self._refine_inner_cb.toggled.connect(self._on_refine_inner_toggled)
        self._refine_outer_cb.toggled.connect(self._on_refine_outer_toggled)
        self._refine_level.currentIndexChanged.connect(self._on_refine_level_changed)
        # External paths (brush stroke, brush clear, ROI clear cascade)
        # mutate refine_brush_mask via roi_changed; refresh the level
        # selector enabled state + min-size readout when that happens.
        state.roi_changed.connect(self._update_refinement_ui)
        self._update_refinement_ui()

        # --- Tracking Mode ---
        self._tracking_mode = QComboBox()
        self._tracking_mode.addItems(["Incremental", "Accumulative"])
        self._tracking_mode.setCurrentText(state.tracking_mode.capitalize())
        row = QHBoxLayout()
        lbl = QLabel("Tracking Mode")
        lbl.setFixedWidth(120)
        row.addWidget(lbl)
        row.addWidget(self._tracking_mode)
        layout.addLayout(row)

        # --- Solver ---
        self._solver = QComboBox()
        self._solver.addItems(["AL-DIC", "Local DIC"])
        self._solver.setCurrentIndex(0 if state.use_admm else 1)
        self._solver.setToolTip(
            "Local DIC: Independent subset matching (IC-GN). Fast,\n"
            "preserves sharp local features. Best for small\n"
            "deformations or high-quality images.\n\n"
            "AL-DIC: Augmented Lagrangian with global FEM\n"
            "regularization. Enforces displacement compatibility\n"
            "between subsets. Best for large deformations, noisy\n"
            "images, or when strain accuracy matters."
        )
        solver_row = QHBoxLayout()
        solver_lbl = QLabel("Solver")
        solver_lbl.setFixedWidth(120)
        solver_row.addWidget(solver_lbl)
        solver_row.addWidget(self._solver)
        layout.addLayout(solver_row)

        # ADMM iterations sub-panel (hidden when Local DIC)
        self._admm_panel = QWidget()
        admm_layout = QVBoxLayout(self._admm_panel)
        admm_layout.setContentsMargins(12, 0, 0, 0)
        admm_layout.setSpacing(4)
        self._admm_iter_spin = QSpinBox()
        self._admm_iter_spin.setRange(1, 10)
        self._admm_iter_spin.setValue(state.admm_max_iter)
        self._admm_iter_spin.setToolTip(
            "Number of ADMM alternating minimization cycles.\n"
            "1 = single global pass (fastest), 3 = default,\n"
            "5+ = diminishing returns for most cases."
        )
        admm_row = QHBoxLayout()
        admm_lbl = QLabel("ADMM Iterations")
        admm_lbl.setFixedWidth(108)
        admm_row.addWidget(admm_lbl)
        admm_row.addWidget(self._admm_iter_spin)
        admm_layout.addLayout(admm_row)
        layout.addWidget(self._admm_panel)
        self._admm_panel.setVisible(state.use_admm)

        self._solver.currentTextChanged.connect(self._on_solver_changed)
        self._admm_iter_spin.valueChanged.connect(self._on_admm_iter_changed)

        # --- Incremental sub-mode panel (hidden when Accumulative) ---
        self._inc_panel = QWidget()
        inc_layout = QVBoxLayout(self._inc_panel)
        inc_layout.setContentsMargins(12, 0, 0, 0)
        inc_layout.setSpacing(4)

        # Ref update mode
        self._ref_mode = QComboBox()
        self._ref_mode.addItems(["Every Frame", "Every N Frames", "Custom Frames"])
        self._ref_mode.setCurrentIndex(0)
        self._ref_mode.setToolTip(
            "How to choose reference frames for incremental tracking"
        )
        ref_row = QHBoxLayout()
        ref_lbl = QLabel("Reference Update")
        ref_lbl.setFixedWidth(108)
        ref_row.addWidget(ref_lbl)
        ref_row.addWidget(self._ref_mode)
        inc_layout.addLayout(ref_row)

        # Interval spinner
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(2, 100)
        self._interval_spin.setValue(state.inc_ref_interval)
        self._interval_spin.setToolTip("Update reference every N frames")
        self._interval_row = QHBoxLayout()
        self._interval_lbl = QLabel("Interval")
        self._interval_lbl.setFixedWidth(108)
        self._interval_row.addWidget(self._interval_lbl)
        self._interval_row.addWidget(self._interval_spin)
        inc_layout.addLayout(self._interval_row)

        # Custom frames line edit
        self._custom_edit = QLineEdit()
        self._custom_edit.setPlaceholderText("e.g. 0, 5, 10, 20")
        self._custom_edit.setToolTip(
            "Comma-separated frame indices to use as reference frames"
        )
        self._custom_row = QHBoxLayout()
        self._custom_lbl = QLabel("Reference Frames")
        self._custom_lbl.setFixedWidth(108)
        self._custom_row.addWidget(self._custom_lbl)
        self._custom_row.addWidget(self._custom_edit)
        inc_layout.addLayout(self._custom_row)

        layout.addWidget(self._inc_panel)

        # Wire signals
        self._ref_mode.currentIndexChanged.connect(self._on_ref_mode_changed)
        self._interval_spin.valueChanged.connect(
            lambda v: state.set_param("inc_ref_interval", v)
        )
        self._custom_edit.editingFinished.connect(self._on_custom_refs_changed)
        self._tracking_mode.currentTextChanged.connect(
            self._on_tracking_mode_changed
        )

        # Initial visibility
        self._on_tracking_mode_changed(self._tracking_mode.currentText())
        self._on_ref_mode_changed(0)

        # Prevent mouse wheel from changing values when widget is unfocused
        for widget in [
            self._subset_size, self._subset_step, self._search_range,
            self._tracking_mode, self._solver, self._admm_iter_spin,
            self._ref_mode, self._interval_spin,
            self._refine_level,
        ]:
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            widget.installEventFilter(self)

    def _on_solver_changed(self, text: str) -> None:
        """Toggle ADMM sub-panel and update state."""
        state = AppState.instance()
        use_admm = text == "AL-DIC"
        state.set_param("use_admm", use_admm)
        self._admm_panel.setVisible(use_admm)

    def _on_admm_iter_changed(self, value: int) -> None:
        AppState.instance().set_param("admm_max_iter", value)

    def _on_tracking_mode_changed(self, text: str) -> None:
        """Set tracking mode param and toggle incremental sub-panel visibility.

        Also resets the initial-guess mode to a sensible default for the
        selected tracking mode:
          accumulative  -> "previous"       (warm-start within same reference)
          incremental   -> "fft_ref_update" (FFT whenever reference changes)
        """
        state = AppState.instance()
        mode = text.lower()
        state.set_param("tracking_mode", mode)
        self._inc_panel.setVisible(mode == "incremental")
        # Auto-select a meaningful init-guess default for the new tracking mode
        if mode == "accumulative":
            state.init_guess_mode = "previous"
        else:
            state.init_guess_mode = "fft_ref_update"
        state.params_changed.emit()

    def _on_ref_mode_changed(self, index: int) -> None:
        """Toggle interval/custom widgets based on ref update mode."""
        state = AppState.instance()
        modes = ["every_frame", "every_n", "custom"]
        mode = modes[index]
        state.set_param("inc_ref_mode", mode)
        self._interval_spin.setVisible(mode == "every_n")
        self._interval_lbl.setVisible(mode == "every_n")
        self._custom_edit.setVisible(mode == "custom")
        self._custom_lbl.setVisible(mode == "custom")

    def _on_custom_refs_changed(self) -> None:
        """Parse comma-separated frame indices and store in AppState."""
        state = AppState.instance()
        text = self._custom_edit.text().strip()
        if not text:
            state.inc_custom_refs = []
            return
        try:
            refs = [int(x.strip()) for x in text.split(",") if x.strip()]
            state.inc_custom_refs = sorted(set(refs))
        except ValueError:
            state.log_message.emit(
                "Invalid frame indices — use comma-separated numbers", "warn"
            )
        state.params_changed.emit()

    def eventFilter(self, obj, event):
        """Block wheel events on unfocused spinboxes and combos."""
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)

    def _on_subset_size_changed(self, display_value: int) -> None:
        """Convert odd display value to even internal winsize."""
        # Snap to nearest odd if user somehow enters even
        if display_value % 2 == 0:
            display_value = display_value + 1
            self._subset_size.blockSignals(True)
            self._subset_size.setValue(display_value)
            self._subset_size.blockSignals(False)
        # Store even value internally (display 41 -> internal 40)
        AppState.instance().set_param("subset_size", display_value - 1)
        # subset_size affects the physical constraint — rebuild dropdown
        self._update_refinement_ui()

    def _on_subset_step_changed(self, value_text: str) -> None:
        """Update subset_step in state and refresh refinement readout."""
        AppState.instance().set_param("subset_step", int(value_text))
        # subset_step affects both the integer constraint and the min-size
        # readout — rebuild dropdown.
        self._update_refinement_ui()

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
        # Ignore spurious -1 emitted while we are repopulating the combo.
        if index < 0:
            return
        AppState.instance().set_refinement_level(index + 1)
        self._update_min_size_label()

    def _rebuild_level_dropdown(self) -> None:
        """Repopulate the level combo to reflect the current max level.

        Called whenever subset_size or subset_step changes. Preserves the
        currently-selected level when possible, otherwise clamps to the new
        maximum.
        """
        state = AppState.instance()
        max_level = state.compute_max_refinement_level()
        labels = ["Light", "Medium", "Heavy", "Extra Heavy", "Ultra"]
        items = [
            f"{labels[i] if i < len(labels) else f'L{i + 1}'} (L{i + 1})"
            for i in range(max_level)
        ]
        # Clamp the stored level to the new range BEFORE rebuilding the
        # combo so set_refinement_level emits at most one signal.
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
            self._refine_info_lbl.setText(
                f"min element size = {min_size} px  "
                f"(subset_step={state.subset_step}, "
                f"level={state.refinement_level})"
            )
            self._refine_info_lbl.setVisible(True)
        else:
            self._refine_info_lbl.setVisible(False)

    def _any_refinement_on(self) -> bool:
        """True if any refinement source is currently active.

        Brush mask is included so the level selector lights up as soon
        as the user starts painting, since BrushRegionCriterion shares
        the same min_element_size as the inner / outer criteria.
        """
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
