"""Workflow-type parameter panel.

Holds the parameters that define WHAT kind of analysis the user is
running — which images are compared against which, which solver runs,
and how often the reference frame refreshes. These choices constrain
the rest of the workflow: accumulative mode compares every frame to
frame 1, so only frame 1 needs a Region of Interest; incremental mode
with a Reference Update schedule needs a Region of Interest at every
ref frame.

Surfaced above the Region of Interest section in the left sidebar so
the user sees which regions are required BEFORE they start drawing.

Exposes:
* ``tracking_mode`` -> AppState.tracking_mode  ("incremental" or
  "accumulative")
* ``use_admm`` / ``admm_max_iter`` -> ``AppState`` (Solver choice)
* ``shape_function_order`` -> AppState (subset shape function, Local DIC)
* ``inc_ref_mode``   -> AppState.inc_ref_mode
* ``inc_ref_interval``, ``inc_custom_refs`` -> AppState
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.app_state import AppState


class WorkflowTypePanel(QWidget):
    """Panel for high-level workflow parameters (tracking, solver, refs)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        state = AppState.instance()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 4, 0)

        # --- Tracking Mode ---
        # Display text is translated; stable English code stored as userData
        # so state-sync doesn't break when language changes.
        self._tracking_mode = QComboBox()
        self._tracking_mode.addItem(self.tr("Incremental"), "incremental")
        self._tracking_mode.addItem(self.tr("Accumulative"), "accumulative")
        _idx = self._tracking_mode.findData(state.tracking_mode)
        if _idx >= 0:
            self._tracking_mode.setCurrentIndex(_idx)
        self._tracking_mode.setToolTip(self.tr(
            "Incremental: each frame is compared to the previous reference "
            "frame.\nSuitable for large accumulated deformation, required "
            "for large rotations.\n\n"
            "Accumulative: every frame is compared to frame 1.\n"
            "Accurate for small, monotonic deformation only."
        ))
        row = QHBoxLayout()
        lbl = QLabel(self.tr("Tracking Mode"))
        lbl.setFixedWidth(120)
        row.addWidget(lbl)
        row.addWidget(self._tracking_mode)
        layout.addLayout(row)

        # --- Solver ---
        # "AL-DIC" is a brand-specific acronym kept literal across locales;
        # "Local DIC" is translatable. Index-based sync stays unchanged.
        self._solver = QComboBox()
        self._solver.addItem("AL-DIC", "aldic")
        self._solver.addItem(self.tr("Local DIC"), "local")
        self._solver.setCurrentIndex(0 if state.use_admm else 1)
        self._solver.setToolTip(self.tr(
            "Local DIC: each subset is matched independently (IC-GN).\n"
            "Fast; keeps sharp local features. The default.\n\n"
            "AL-DIC: adds a global FEM step that makes the\n"
            "displacement field continuous from subset to subset.\n"
            "About 3 times slower than Local DIC."
        ))
        solver_row = QHBoxLayout()
        solver_lbl = QLabel(self.tr("Solver"))
        solver_lbl.setFixedWidth(120)
        solver_row.addWidget(solver_lbl)
        solver_row.addWidget(self._solver)
        layout.addLayout(solver_row)

        # --- Shape function (Local DIC only) ---
        # Stable codes as userData; AL-DIC always runs first order, so the
        # control is disabled there and keeps the user's Local DIC choice.
        self._shape_order = QComboBox()
        self._shape_order.addItem(self.tr("1st order"), "first")
        self._shape_order.addItem(self.tr("2nd order"), "second")
        _shape_idx = self._shape_order.findData(state.shape_function_order)
        if _shape_idx >= 0:
            self._shape_order.setCurrentIndex(_shape_idx)
        self._shape_order.setToolTip(self.tr(
            "1st order: each subset can translate, stretch, shear and\n"
            "rotate. The standard choice.\n\n"
            "2nd order: each subset can also bend. Removes the error a\n"
            "1st-order subset makes where strain changes within the\n"
            "subset, at about twice the noise. Most useful with a larger\n"
            "subset and low image noise. A subset that cannot support it\n"
            "keeps its 1st-order result.\n\n"
            "Available for Local DIC only."
        ))
        shape_row = QHBoxLayout()
        shape_lbl = QLabel(self.tr("Shape Function"))
        shape_lbl.setFixedWidth(120)
        shape_row.addWidget(shape_lbl)
        shape_row.addWidget(self._shape_order)
        layout.addLayout(shape_row)
        self._shape_order.setEnabled(not state.use_admm)

        # ADMM iterations moved to ADVANCED > AdvancedTuningWidget —
        # it's a numerical tuning knob, not a workflow choice.

        # --- Incremental sub-panel (Reference Update) ---
        self._inc_panel = QWidget()
        inc_layout = QVBoxLayout(self._inc_panel)
        inc_layout.setContentsMargins(12, 0, 0, 0)
        inc_layout.setSpacing(4)

        self._ref_mode = QComboBox()
        # Display translated; stable code stored as userData.
        self._ref_mode.addItem(self.tr("Every Frame"), "every_frame")
        self._ref_mode.addItem(self.tr("Every N Frames"), "every_n")
        self._ref_mode.addItem(self.tr("Custom Frames"), "custom")
        self._ref_mode.setCurrentIndex(0)
        self._ref_mode.setToolTip(self.tr(
            "When the reference frame refreshes during incremental "
            "tracking.\n"
            "Every Frame: reset reference every frame (smallest per-step "
            "displacement,\nmost robust for large deformation).\n"
            "Every N Frames: reset every N frames (balance speed vs "
            "robustness).\n"
            "Custom Frames: user-defined list of reference frame indices."
        ))
        ref_row = QHBoxLayout()
        ref_lbl = QLabel(self.tr("Reference Update"))
        ref_lbl.setFixedWidth(108)
        ref_row.addWidget(ref_lbl)
        ref_row.addWidget(self._ref_mode)
        inc_layout.addLayout(ref_row)

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(2, 100)
        self._interval_spin.setValue(state.inc_ref_interval)
        self._interval_spin.setToolTip(self.tr(
            "Update reference every N frames"))
        self._interval_row = QHBoxLayout()
        self._interval_lbl = QLabel(self.tr("Interval"))
        self._interval_lbl.setFixedWidth(108)
        self._interval_row.addWidget(self._interval_lbl)
        self._interval_row.addWidget(self._interval_spin)
        inc_layout.addLayout(self._interval_row)

        self._custom_edit = QLineEdit()
        self._custom_edit.setPlaceholderText("e.g. 0, 5, 10, 20")
        self._custom_edit.setToolTip(self.tr(
            "Comma-separated frame indices to use as reference frames "
            "(0-based)"
        ))
        self._custom_row = QHBoxLayout()
        self._custom_lbl = QLabel(self.tr("Reference Frames"))
        self._custom_lbl.setFixedWidth(108)
        self._custom_row.addWidget(self._custom_lbl)
        self._custom_row.addWidget(self._custom_edit)
        inc_layout.addLayout(self._custom_row)

        layout.addWidget(self._inc_panel)

        # --- Wiring ---
        self._tracking_mode.currentIndexChanged.connect(
            self._on_tracking_mode_changed
        )
        self._solver.currentIndexChanged.connect(self._on_solver_changed)
        state.params_changed.connect(self._sync_shape_order)
        self._shape_order.currentIndexChanged.connect(
            self._on_shape_order_changed
        )
        self._ref_mode.currentIndexChanged.connect(self._on_ref_mode_changed)
        self._interval_spin.valueChanged.connect(
            lambda v: state.set_param("inc_ref_interval", v)
        )
        self._custom_edit.editingFinished.connect(self._on_custom_refs_changed)

        # Initial visibility
        self._on_tracking_mode_changed(self._tracking_mode.currentIndex())
        self._on_ref_mode_changed(0)

        # Block scroll-wheel changes on unfocused widgets
        for widget in [
            self._tracking_mode, self._solver, self._shape_order,
            self._ref_mode, self._interval_spin,
        ]:
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            widget.installEventFilter(self)

    # ------------------------------------------------------------------

    def _on_solver_changed(self, _index: int) -> None:
        """Toggle solver kind in state (read from combobox userData code)."""
        state = AppState.instance()
        use_admm = self._solver.currentData() == "aldic"
        self._shape_order.setEnabled(not use_admm)
        state.set_param("use_admm", use_admm)

    def _sync_shape_order(self) -> None:
        """Follow a shape function set elsewhere (the Apply button of the
        post-run suggestion)."""
        index = self._shape_order.findData(AppState.instance().shape_function_order)
        if index < 0 or index == self._shape_order.currentIndex():
            return
        self._shape_order.blockSignals(True)
        self._shape_order.setCurrentIndex(index)
        self._shape_order.blockSignals(False)

    def _on_shape_order_changed(self, _index: int) -> None:
        """Store the chosen shape-function order (combobox userData code)."""
        state = AppState.instance()
        state.set_param(
            "shape_function_order", self._shape_order.currentData() or "first",
        )

    def _on_tracking_mode_changed(self, _index: int) -> None:
        """Set tracking mode and swap init-guess default + inc sub-panel."""
        state = AppState.instance()
        mode = self._tracking_mode.currentData() or "accumulative"
        state.set_param("tracking_mode", mode)
        self._inc_panel.setVisible(mode == "incremental")
        # Auto-select a meaningful init-guess default for the new mode.
        # Do NOT clobber a user-selected seed_propagation choice — it works
        # for both tracking modes and represents an explicit user intent.
        if state.init_guess_mode != "seed_propagation":
            if mode == "accumulative":
                state.init_guess_mode = "previous"
            else:
                state.init_guess_mode = "fft_ref_update"
        state.params_changed.emit()

    def _on_ref_mode_changed(self, _index: int) -> None:
        """Toggle Interval / Custom row visibility based on ref mode."""
        state = AppState.instance()
        mode = self._ref_mode.currentData() or "every_frame"
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
