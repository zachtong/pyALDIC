"""Initial-Guess panel.

Three-tier layout:

    - Starting Points    -> DICPara.init_guess_mode = "seed_propagation"
    - FFT                -> one of:
        * every N frames (N=1 is FFT-every-frame) -> "fft_every" if N==1
                                                      "fft_reset_n" otherwise
        * only on reference-frame updates         -> "fft_ref_update"
    - Previous frame     -> "previous"

Each top-level choice carries an inline, one-line guidance note so
users pick a mode matching their data (displacement magnitude,
smoothness, texture discontinuities).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
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


class _InfoIcon(QLabel):
    """A small ⓘ glyph that surfaces its tooltip on hover OR click.

    Tooltip alone is fragile: touchscreens never trigger hover, and
    some users miss that the icon is interactive. A click also shows
    the same tooltip, pinned at the cursor, so discoverability
    doesn't depend on knowing the hover convention.
    """

    def __init__(self, tip: str, parent: QWidget | None = None) -> None:
        super().__init__("ⓘ", parent)
        self.setToolTip(tip)
        self._tip_text = tip
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 13px; "
            f"padding: 0 4px;"
        )

    def mousePressEvent(self, event):  # noqa: N802 (Qt override)
        # Show the tooltip text at the cursor, same behaviour as hover.
        # The global pos plus a small downward offset keeps the popup
        # from sitting directly under the arrow cursor.
        from PySide6.QtWidgets import QToolTip
        QToolTip.showText(event.globalPos(), self._tip_text, self)
        super().mousePressEvent(event)


def _help_icon(tip: str) -> QLabel:
    return _InfoIcon(tip)


def _radio_row(radio: QRadioButton, info_tip: str) -> QHBoxLayout:
    """Lay out a radio button + right-aligned info icon."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    row.addWidget(radio)
    row.addStretch()
    row.addWidget(_help_icon(info_tip))
    return row


class InitGuessWidget(QWidget):
    """Initial-guess controls with tri-mode (seed / FFT / previous) layout.

    Lives directly below WORKFLOW TYPE in the left sidebar — prominent
    placement so users think about which init matches their scenario
    before reaching ROI drawing.
    """

    # Emitted when the user clicks "Place Starting Points" — the top-level
    # controller forwards to canvas.set_tool("seed").
    request_place_seeds = Signal()
    # Emitted when the user clicks "Auto-place".
    request_auto_place_seeds = Signal()
    # Emitted when the user clicks "Clear" to drop every Starting Point.
    request_clear_seeds = Signal()
    # Emitted whenever the user changes the init-guess method via the UI
    # (radio toggle or FFT submode). App.py uses this to jump to frame 0
    # ROI editing so the user always sees a consistent 'setup' view
    # after any init-method action.
    init_mode_user_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        self._building = False
        self._seed_ctrl = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        # ----------------------------------------------------------------
        # Starting Points (seed propagation) — large displacement / cracks
        # ----------------------------------------------------------------
        self._rb_seed_prop = QRadioButton(self.tr("Starting Points"))
        layout.addLayout(_radio_row(
            self._rb_seed_prop,
            self.tr(
                "Place a few points; pyALDIC bootstraps each with a "
                "single-point NCC and propagates the field along mesh "
                "neighbours.\n\n"
                "Best for:\n"
                "• Large inter-frame displacement (> 50 px)\n"
                "• Discontinuous fields (cracks, shear bands)\n"
                "• Scenarios where FFT picks wrong peaks\n\n"
                "Auto-placed per region when you draw or edit an ROI."),
        ))

        self._seed_panel = QWidget()
        seed_layout = QVBoxLayout(self._seed_panel)
        seed_layout.setContentsMargins(18, 2, 0, 2)
        seed_layout.setSpacing(4)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._btn_place_seeds = QPushButton(self.tr("Place Starting Points"))
        self._btn_place_seeds.setCheckable(True)
        self._btn_place_seeds.setToolTip(self.tr(
            "Enter placement mode on the canvas. Left-click to add, "
            "right-click to remove, Esc or click again to exit."
        ))
        btn_row.addWidget(self._btn_place_seeds, stretch=2)
        self._btn_auto_place = QPushButton(self.tr("Auto-place"))
        self._btn_auto_place.setToolTip(self.tr(
            "Fill empty regions with the highest-NCC node in each. "
            "Existing Starting Points are preserved."
        ))
        btn_row.addWidget(self._btn_auto_place, stretch=1)
        self._btn_clear_seeds = QPushButton(self.tr("Clear"))
        self._btn_clear_seeds.setToolTip(self.tr(
            "Remove every Starting Point. Faster than right-clicking "
            "each one individually."
        ))
        btn_row.addWidget(self._btn_clear_seeds, stretch=1)
        seed_layout.addLayout(btn_row)
        self._lbl_seed_progress = QLabel(
            self.tr("%1 / %2 regions ready").arg(0).arg(0))
        self._lbl_seed_progress.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px;"
        )
        seed_layout.addWidget(self._lbl_seed_progress)
        self._seed_panel.setVisible(False)
        layout.addWidget(self._seed_panel)

        self._btn_place_seeds.clicked.connect(self.request_place_seeds.emit)
        self._btn_auto_place.clicked.connect(
            self.request_auto_place_seeds.emit,
        )
        self._btn_clear_seeds.clicked.connect(
            self.request_clear_seeds.emit,
        )

        # ----------------------------------------------------------------
        # FFT (cross-correlation) — standard small-to-moderate motion
        # ----------------------------------------------------------------
        self._rb_fft = QRadioButton(self.tr("FFT (cross-correlation)"))
        layout.addLayout(_radio_row(
            self._rb_fft,
            self.tr(
                "Full-grid normalized cross-correlation. Robust within the "
                "search radius; the search auto-expands when peaks clip.\n\n"
                "Best for:\n"
                "• Small-to-moderate smooth motion\n"
                "• Well-textured speckle\n"
                "• No special user setup needed\n\n"
                "Cost grows with the search radius, so very large "
                "displacements become slow."),
        ))

        # FFT sub-mode selector
        self._fft_panel = QWidget()
        fft_layout = QVBoxLayout(self._fft_panel)
        fft_layout.setContentsMargins(18, 2, 0, 2)
        fft_layout.setSpacing(4)

        every_row = QHBoxLayout()
        every_row.setSpacing(6)
        self._rb_fft_every_n = QRadioButton(self.tr("Every"))
        self._rb_fft_every_n.setToolTip(self.tr(
            "Run FFT every N frames. N = 1 means FFT every frame "
            "(safest, slowest). N > 1 uses warm-start between "
            "resets to limit error propagation to N frames."
        ))
        self._spin_fft_n = QSpinBox()
        self._spin_fft_n.setRange(1, 999)
        self._spin_fft_n.setFixedWidth(60)
        self._spin_fft_n.setSuffix(" fr")
        every_row.addWidget(self._rb_fft_every_n)
        every_row.addWidget(self._spin_fft_n)
        _every_hint = QLabel(self.tr("(N=1 = every frame)"))
        _every_hint.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px;"
        )
        every_row.addWidget(_every_hint)
        every_row.addStretch()
        fft_layout.addLayout(every_row)

        self._rb_fft_ref_update = QRadioButton(self.tr(
            "Only when reference frame updates (incremental only)"
        ))
        self._rb_fft_ref_update.setToolTip(self.tr(
            "Run FFT whenever the reference frame changes; warm-start "
            "within each segment. Typical default for incremental mode."
        ))
        fft_layout.addWidget(self._rb_fft_ref_update)

        self._fft_panel.setVisible(False)
        layout.addWidget(self._fft_panel)

        # ----------------------------------------------------------------
        # Previous — fastest, smooth small motion only
        # ----------------------------------------------------------------
        self._rb_previous = QRadioButton(self.tr("Previous frame"))
        layout.addLayout(_radio_row(
            self._rb_previous,
            self.tr(
                "Use the previous frame's converged displacement as the "
                "initial guess. No cross-correlation runs.\n\n"
                "Best for:\n"
                "• Very small inter-frame motion (a few pixels)\n"
                "• Fastest option when motion is smooth\n\n"
                "Errors can accumulate over long sequences. Prefer FFT or "
                "Starting Points on noisy data or when motion is larger."),
        ))

        # ----------------------------------------------------------------
        # Wheel-focus guard: unfocused spinboxes ignore scroll
        # ----------------------------------------------------------------
        for w in (self._spin_fft_n,):
            w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            w.installEventFilter(self)

        # ----------------------------------------------------------------
        # Connections
        # ----------------------------------------------------------------
        self._rb_seed_prop.toggled.connect(self._on_top_level_changed)
        self._rb_fft.toggled.connect(self._on_top_level_changed)
        self._rb_previous.toggled.connect(self._on_top_level_changed)
        self._rb_fft_every_n.toggled.connect(self._on_fft_submode_changed)
        self._rb_fft_ref_update.toggled.connect(self._on_fft_submode_changed)
        self._spin_fft_n.valueChanged.connect(self._on_fft_n_changed)
        self._state.params_changed.connect(self._on_params_changed_externally)

        self._sync_from_state()

    # ------------------------------------------------------------------
    # State -> UI
    # ------------------------------------------------------------------

    def _sync_from_state(self) -> None:
        self._building = True
        mode = self._state.init_guess_mode
        is_seed = mode == "seed_propagation"
        is_fft = mode in ("fft_every", "fft_reset_n", "fft_ref_update")
        is_prev = mode == "previous"

        self._rb_seed_prop.setChecked(is_seed)
        self._rb_fft.setChecked(is_fft)
        self._rb_previous.setChecked(is_prev)

        if is_fft:
            if mode == "fft_ref_update":
                self._rb_fft_ref_update.setChecked(True)
                self._rb_fft_every_n.setChecked(False)
                self._spin_fft_n.setEnabled(False)
            else:
                self._rb_fft_every_n.setChecked(True)
                self._rb_fft_ref_update.setChecked(False)
                if mode == "fft_every":
                    self._spin_fft_n.setValue(1)
                else:  # fft_reset_n
                    self._spin_fft_n.setValue(
                        max(1, self._state.fft_reset_interval),
                    )
                self._spin_fft_n.setEnabled(True)

        self._seed_panel.setVisible(is_seed)
        self._fft_panel.setVisible(is_fft)
        self._refresh_seed_progress()
        self._building = False

    def _on_params_changed_externally(self) -> None:
        if not self._building:
            self._sync_from_state()

    def set_seed_controller(self, ctrl) -> None:
        self._seed_ctrl = ctrl
        self._state.seeds_changed.connect(self._refresh_seed_progress)
        self._state.roi_changed.connect(self._refresh_seed_progress)
        self._refresh_seed_progress()

    def set_seed_mode_active(self, active: bool) -> None:
        self._btn_place_seeds.setChecked(active)
        self._btn_place_seeds.setText(
            self.tr("Placing... (click to exit)") if active
            else self.tr("Place Starting Points")
        )

    def _refresh_seed_progress(self) -> None:
        if self._seed_ctrl is None:
            return
        status = self._seed_ctrl.regions_status()
        total = len(status)
        seeded = sum(1 for _, has, _ in status if has)
        self._lbl_seed_progress.setText(
            self.tr("%1 / %2 regions ready").arg(seeded).arg(total)
        )

    # ------------------------------------------------------------------
    # UI -> State
    # ------------------------------------------------------------------

    def _on_top_level_changed(self) -> None:
        if self._building:
            return
        if self._rb_seed_prop.isChecked():
            self._state.init_guess_mode = "seed_propagation"
        elif self._rb_fft.isChecked():
            # Switch to whichever FFT sub-mode was previously active, or
            # default to 'every 1' if this is the first time.
            current = self._state.init_guess_mode
            if current not in ("fft_every", "fft_reset_n", "fft_ref_update"):
                self._state.init_guess_mode = "fft_every"
                self._state.fft_reset_interval = 0
        elif self._rb_previous.isChecked():
            self._state.init_guess_mode = "previous"
        self._sync_from_state()
        self._state.params_changed.emit()
        self.init_mode_user_changed.emit()

    def _on_fft_submode_changed(self) -> None:
        if self._building:
            return
        if self._rb_fft_ref_update.isChecked():
            self._state.init_guess_mode = "fft_ref_update"
            self._state.fft_reset_interval = 0
        elif self._rb_fft_every_n.isChecked():
            n = self._spin_fft_n.value()
            if n == 1:
                self._state.init_guess_mode = "fft_every"
                self._state.fft_reset_interval = 0
            else:
                self._state.init_guess_mode = "fft_reset_n"
                self._state.fft_reset_interval = n
        self._sync_from_state()
        self._state.params_changed.emit()
        self.init_mode_user_changed.emit()

    def _on_fft_n_changed(self, value: int) -> None:
        if self._building:
            return
        if not self._rb_fft_every_n.isChecked():
            return
        if value == 1:
            self._state.init_guess_mode = "fft_every"
            self._state.fft_reset_interval = 0
        else:
            self._state.init_guess_mode = "fft_reset_n"
            self._state.fft_reset_interval = value
        self._state.params_changed.emit()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)
