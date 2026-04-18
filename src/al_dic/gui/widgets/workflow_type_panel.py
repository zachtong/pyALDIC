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
        self._tracking_mode = QComboBox()
        self._tracking_mode.addItems(["Incremental", "Accumulative"])
        self._tracking_mode.setCurrentText(state.tracking_mode.capitalize())
        self._tracking_mode.setToolTip(
            "Incremental: each frame is compared to the previous reference "
            "frame.\nSuitable for large accumulated deformation, required "
            "for large rotations.\n\n"
            "Accumulative: every frame is compared to frame 1.\n"
            "Accurate for small, monotonic deformation only."
        )
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

        # --- ADMM iterations (visible only for AL-DIC) ---
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

        # --- Incremental sub-panel (Reference Update) ---
        self._inc_panel = QWidget()
        inc_layout = QVBoxLayout(self._inc_panel)
        inc_layout.setContentsMargins(12, 0, 0, 0)
        inc_layout.setSpacing(4)

        self._ref_mode = QComboBox()
        self._ref_mode.addItems(["Every Frame", "Every N Frames", "Custom Frames"])
        self._ref_mode.setCurrentIndex(0)
        self._ref_mode.setToolTip(
            "When the reference frame refreshes during incremental "
            "tracking.\n"
            "Every Frame: reset reference every frame (smallest per-step "
            "displacement,\nmost robust for large deformation).\n"
            "Every N Frames: reset every N frames (balance speed vs "
            "robustness).\n"
            "Custom Frames: user-defined list of reference frame indices."
        )
        ref_row = QHBoxLayout()
        ref_lbl = QLabel("Reference Update")
        ref_lbl.setFixedWidth(108)
        ref_row.addWidget(ref_lbl)
        ref_row.addWidget(self._ref_mode)
        inc_layout.addLayout(ref_row)

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

        self._custom_edit = QLineEdit()
        self._custom_edit.setPlaceholderText("e.g. 0, 5, 10, 20")
        self._custom_edit.setToolTip(
            "Comma-separated frame indices to use as reference frames "
            "(0-based)"
        )
        self._custom_row = QHBoxLayout()
        self._custom_lbl = QLabel("Reference Frames")
        self._custom_lbl.setFixedWidth(108)
        self._custom_row.addWidget(self._custom_lbl)
        self._custom_row.addWidget(self._custom_edit)
        inc_layout.addLayout(self._custom_row)

        layout.addWidget(self._inc_panel)

        # --- Wiring ---
        self._tracking_mode.currentTextChanged.connect(
            self._on_tracking_mode_changed
        )
        self._solver.currentTextChanged.connect(self._on_solver_changed)
        self._admm_iter_spin.valueChanged.connect(self._on_admm_iter_changed)
        self._ref_mode.currentIndexChanged.connect(self._on_ref_mode_changed)
        self._interval_spin.valueChanged.connect(
            lambda v: state.set_param("inc_ref_interval", v)
        )
        self._custom_edit.editingFinished.connect(self._on_custom_refs_changed)

        # Initial visibility
        self._on_tracking_mode_changed(self._tracking_mode.currentText())
        self._on_ref_mode_changed(0)

        # Block scroll-wheel changes on unfocused widgets
        for widget in [
            self._tracking_mode, self._solver, self._admm_iter_spin,
            self._ref_mode, self._interval_spin,
        ]:
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            widget.installEventFilter(self)

    # ------------------------------------------------------------------

    def _on_solver_changed(self, text: str) -> None:
        """Toggle ADMM sub-panel and update state."""
        state = AppState.instance()
        use_admm = text == "AL-DIC"
        state.set_param("use_admm", use_admm)
        self._admm_panel.setVisible(use_admm)

    def _on_admm_iter_changed(self, value: int) -> None:
        AppState.instance().set_param("admm_max_iter", value)

    def _on_tracking_mode_changed(self, text: str) -> None:
        """Set tracking mode and swap init-guess default + inc sub-panel."""
        state = AppState.instance()
        mode = text.lower()
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

    def _on_ref_mode_changed(self, index: int) -> None:
        """Toggle Interval / Custom row visibility based on ref mode."""
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
