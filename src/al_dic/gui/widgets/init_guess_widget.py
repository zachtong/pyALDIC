"""Initial-guess parameter panel inside the ADVANCED collapsible section.

Controls:
    - Mode radio buttons:
        * previous-frame warm-start
        * FFT when reference frame updates (incremental mode default)
        * FFT every frame
        * FFT every-N-frames (with spinbox for N)
    - Auto-expand checkbox: let the pipeline widen the search region
      automatically when FFT peaks are clipped.

Note: the FFT search radius (``state.search_range``) is now edited in
the main ParamPanel as "Search Range" since it is a fundamental DIC
parameter, not an initial-guess tuning knob.

Auto-selection on tracking mode change:
    accumulative  -> "previous"
    incremental   -> "fft_ref_update"
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.app_state import AppState
from al_dic.gui.theme import COLORS


class InitGuessWidget(QWidget):
    """Initial-guess controls wired to AppState."""

    # Emitted when the user clicks "Place Starting Points" — the top-level
    # controller forwards this to canvas.set_tool("seed") so the canvas
    # widget doesn't need to know about this widget directly.
    request_place_seeds = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        self._building = False
        self._seed_ctrl = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        sec_lbl = QLabel("INITIAL GUESS")
        sec_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px; "
            f"font-weight: bold; letter-spacing: 1px;"
        )
        layout.addWidget(sec_lbl)

        # --- Mode: previous frame ---
        self._rb_previous = QRadioButton("Previous frame (default, fastest)")
        self._rb_previous.setToolTip(
            "Use the converged result from the previous frame as the IC-GN "
            "starting point.  Fastest; may propagate errors over many frames.  "
            "Recommended for accumulative mode."
        )
        layout.addWidget(self._rb_previous)

        # --- Mode: FFT when reference frame updates (incremental default) ---
        self._rb_fft_ref_update = QRadioButton("FFT when reference frame updates")
        self._rb_fft_ref_update.setToolTip(
            "Run an FFT search whenever the reference frame changes.  "
            "Within a same-reference segment, the previous frame's result is "
            "used for warm-start.  Recommended default for incremental mode."
        )
        layout.addWidget(self._rb_fft_ref_update)

        # --- Mode: FFT every frame ---
        self._rb_fft_every = QRadioButton("FFT every frame (safest)")
        self._rb_fft_every.setToolTip(
            "Run a fresh FFT cross-correlation search for every frame.  "
            "Eliminates warm-start error propagation; slowest."
        )
        layout.addWidget(self._rb_fft_every)

        # --- Mode: FFT every N frames ---
        reset_row = QHBoxLayout()
        reset_row.setSpacing(6)
        self._rb_reset_n = QRadioButton("Reset FFT every")
        self._rb_reset_n.setToolTip(
            "Perform a full FFT reset every N frames, then resume warm-start.  "
            "Limits error propagation to at most N frames."
        )
        reset_row.addWidget(self._rb_reset_n)

        self._reset_spin = QSpinBox()
        self._reset_spin.setRange(2, 999)
        self._reset_spin.setValue(self._state.fft_reset_interval)
        self._reset_spin.setFixedWidth(52)
        self._reset_spin.setSuffix(" fr")
        self._reset_spin.setToolTip("FFT reset interval in frames")
        reset_row.addWidget(self._reset_spin)
        reset_row.addStretch()
        layout.addLayout(reset_row)

        # --- Mode: seed propagation ---
        self._rb_seed_prop = QRadioButton(
            "Starting Points (large displacement / cracks)"
        )
        self._rb_seed_prop.setToolTip(
            "Place a few 'starting points' manually on the canvas; "
            "pyALDIC bootstraps each with single-point NCC, then propagates "
            "the displacement field across mesh neighbours via F-aware BFS.\n\n"
            "Best for: large inter-frame displacement (>50 px), discontinuous "
            "fields (cracks, shear bands).  At least one starting point per "
            "connected mask region required before the pipeline can run."
        )
        layout.addWidget(self._rb_seed_prop)

        # "Place Starting Points" sub-panel — visible only when this mode
        # is selected. Contains the Place button + region-progress label.
        self._seed_panel = QWidget()
        seed_panel_layout = QVBoxLayout(self._seed_panel)
        seed_panel_layout.setContentsMargins(18, 4, 0, 4)
        seed_panel_layout.setSpacing(4)

        self._btn_place_seeds = QPushButton("Place Starting Points")
        self._btn_place_seeds.setToolTip(
            "Enter placement mode on the canvas.\n"
            "Left-click inside the ROI to drop a point, right-click on a "
            "point to remove it, Esc to exit."
        )
        seed_panel_layout.addWidget(self._btn_place_seeds)

        self._lbl_seed_progress = QLabel("0 / 0 regions seeded")
        self._lbl_seed_progress.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px;"
        )
        seed_panel_layout.addWidget(self._lbl_seed_progress)

        self._seed_panel.setVisible(False)
        layout.addWidget(self._seed_panel)

        self._btn_place_seeds.clicked.connect(self.request_place_seeds.emit)

        # NOTE: "Search Radius" (``state.search_range``) was previously edited
        # here. It has been moved to the main ParamPanel as "Search Range"
        # so users can find it without expanding the ADVANCED section.

        # --- Auto-expand checkbox ---
        self._auto_expand_cb = QCheckBox("Auto-expand search on clipped peaks")
        self._auto_expand_cb.setToolTip(
            "When the NCC peak reaches the edge of the search region, "
            "automatically retry with a larger region (up to image half-size).  "
            "Retries up to 6 times with 2× growth each attempt."
        )
        self._auto_expand_cb.setChecked(self._state.fft_auto_expand)
        layout.addWidget(self._auto_expand_cb)

        # --- Set initial radio state ---
        self._sync_from_state()

        # --- Prevent scroll-wheel from changing unfocused spinboxes ---
        for w in (self._reset_spin,):
            w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            w.installEventFilter(self)

        # --- Connections ---
        self._rb_previous.toggled.connect(self._on_mode_changed)
        self._rb_fft_ref_update.toggled.connect(self._on_mode_changed)
        self._rb_fft_every.toggled.connect(self._on_mode_changed)
        self._rb_reset_n.toggled.connect(self._on_mode_changed)
        self._rb_seed_prop.toggled.connect(self._on_mode_changed)
        self._reset_spin.valueChanged.connect(self._on_interval_changed)
        self._auto_expand_cb.stateChanged.connect(self._on_auto_expand_changed)

        # Listen for external state changes (e.g. tracking mode auto-switch)
        self._state.params_changed.connect(self._on_params_changed_externally)

    # ------------------------------------------------------------------
    # State -> UI
    # ------------------------------------------------------------------

    def _sync_from_state(self) -> None:
        self._building = True
        mode = self._state.init_guess_mode
        self._rb_previous.setChecked(mode == "previous")
        self._rb_fft_ref_update.setChecked(mode == "fft_ref_update")
        self._rb_fft_every.setChecked(mode == "fft_every")
        self._rb_reset_n.setChecked(mode == "fft_reset_n")
        self._rb_seed_prop.setChecked(mode == "seed_propagation")
        self._reset_spin.setValue(self._state.fft_reset_interval)
        self._reset_spin.setEnabled(mode == "fft_reset_n")
        self._auto_expand_cb.setChecked(self._state.fft_auto_expand)
        self._seed_panel.setVisible(mode == "seed_propagation")
        self._refresh_seed_progress()
        self._building = False

    def set_seed_controller(self, ctrl) -> None:
        """Attach the SeedController for region-progress readout."""
        self._seed_ctrl = ctrl
        self._state.seeds_changed.connect(self._refresh_seed_progress)
        self._state.roi_changed.connect(self._refresh_seed_progress)
        self._refresh_seed_progress()

    def _refresh_seed_progress(self) -> None:
        if self._seed_ctrl is None:
            return
        status = self._seed_ctrl.regions_status()
        total = len(status)
        seeded = sum(1 for _, has, _ in status if has)
        self._lbl_seed_progress.setText(
            f"{seeded} / {total} regions seeded"
        )

    def _on_params_changed_externally(self) -> None:
        """Re-sync UI when tracking mode (or other state) changes externally."""
        if not self._building:
            self._sync_from_state()

    # ------------------------------------------------------------------
    # UI -> State
    # ------------------------------------------------------------------

    def _on_mode_changed(self) -> None:
        if self._building:
            return
        if self._rb_previous.isChecked():
            mode = "previous"
        elif self._rb_fft_ref_update.isChecked():
            mode = "fft_ref_update"
        elif self._rb_fft_every.isChecked():
            mode = "fft_every"
        elif self._rb_seed_prop.isChecked():
            mode = "seed_propagation"
        else:
            mode = "fft_reset_n"
        self._reset_spin.setEnabled(mode == "fft_reset_n")
        self._seed_panel.setVisible(mode == "seed_propagation")
        self._state.init_guess_mode = mode
        self._state.params_changed.emit()

    def _on_interval_changed(self, value: int) -> None:
        if self._building:
            return
        self._state.fft_reset_interval = value
        self._state.params_changed.emit()

    def _on_auto_expand_changed(self, _: int) -> None:
        if self._building:
            return
        self._state.fft_auto_expand = self._auto_expand_cb.isChecked()
        self._state.params_changed.emit()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)
