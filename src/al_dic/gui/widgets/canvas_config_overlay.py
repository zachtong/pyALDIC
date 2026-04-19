"""Top-left canvas overlay that summarises the current DIC configuration.

The overlay is a small always-visible legend showing the three
decisions that shape a run — tracking mode, solver, and initial-guess
method. It updates live as the user edits the sidebar, so a glance at
the canvas tells you what will happen if you press Run.

Invisible until at least two images are loaded.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from al_dic.gui.app_state import AppState
from al_dic.gui.theme import COLORS


class CanvasConfigOverlay(QFrame):
    """Small semi-transparent panel pinned to the canvas top-left.

    Shows three labelled rows: ``Mode``, ``Solver``, ``Init``.
    Values follow whatever is currently selected in the sidebar.
    """

    MARGIN = 12  # px from the canvas edge

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(
            f"background-color: rgba(16, 20, 24, 215); "
            f"border: 1px solid {COLORS.BORDER}; "
            f"border-radius: 6px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)

        self._mode_lbl = self._make_row(layout, "Mode")
        self._solver_lbl = self._make_row(layout, "Solver")
        self._init_lbl = self._make_row(layout, "Init")
        self.adjustSize()

        # Keep in sync with the sidebar.
        self._state.params_changed.connect(self._refresh)
        self._state.images_changed.connect(self._refresh)
        self._refresh()

    def _make_row(self, layout: QVBoxLayout, key: str) -> QLabel:
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        key_lbl = QLabel(key.upper())
        key_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 9px; "
            f"font-weight: bold; letter-spacing: 1px; "
            f"background: transparent;"
        )
        row_layout.addWidget(key_lbl)

        val_lbl = QLabel("—")
        val_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 12px; "
            f"font-weight: bold; background: transparent;"
        )
        row_layout.addWidget(val_lbl)
        layout.addWidget(row)
        return val_lbl

    # ----------------------------------------------------------------------
    # State → text formatting
    # ----------------------------------------------------------------------

    def _refresh(self) -> None:
        if len(self._state.image_files) < 2:
            self.setVisible(False)
            return
        self._mode_lbl.setText(self._format_mode())
        self._solver_lbl.setText(self._format_solver())
        self._init_lbl.setText(self._format_init())
        self.adjustSize()
        self.setVisible(True)
        self._reposition()

    def _format_mode(self) -> str:
        raw = getattr(self._state, "tracking_mode", "accumulative")
        return {
            "accumulative": "Accumulative",
            "incremental": "Incremental",
        }.get(str(raw).lower(), str(raw))

    def _format_solver(self) -> str:
        use_admm = getattr(self._state, "use_admm", True)
        if not use_admm:
            return "Local DIC"
        iters = int(getattr(self._state, "admm_max_iter", 3))
        return f"ADMM ({iters} iter)"

    def _format_init(self) -> str:
        mode = str(getattr(self._state, "init_guess_mode", "fft_every"))
        interval = int(getattr(self._state, "fft_reset_interval", 0) or 0)
        if mode == "seed_propagation":
            return "Starting Points"
        if mode in ("previous", "fft_ref_update", "auto"):
            return "Previous frame"
        if mode == "fft_every":
            return "FFT every frame"
        if mode == "fft_reset_n" and interval > 1:
            return f"FFT every {interval} fr"
        return "FFT"

    # ----------------------------------------------------------------------
    # Geometry — pin to top-left of parent viewport
    # ----------------------------------------------------------------------

    def reposition(self) -> None:
        """Re-anchor to the parent's top-left corner. Called from
        ``CanvasArea`` when the viewport resizes."""
        self._reposition()

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.move(self.MARGIN, self.MARGIN)
