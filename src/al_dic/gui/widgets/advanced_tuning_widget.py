"""Advanced numerical tuning knobs (ADMM iterations, FFT auto-expand).

Lives inside the ADVANCED collapsible section. These parameters rarely
need to be changed — defaults work for most use cases — so they are
tucked out of the primary workflow.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.app_state import AppState
from al_dic.gui.theme import COLORS


class AdvancedTuningWidget(QWidget):
    """ADMM iteration count + FFT auto-expand checkbox."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        self._building = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        # --- ADMM iterations (AL-DIC only) -----------------------------
        admm_row = QHBoxLayout()
        admm_row.setSpacing(6)
        admm_lbl = QLabel(self.tr("ADMM Iterations"))
        admm_lbl.setFixedWidth(120)
        admm_row.addWidget(admm_lbl)
        self._admm_iter_spin = QSpinBox()
        self._admm_iter_spin.setRange(1, 10)
        self._admm_iter_spin.setValue(self._state.admm_max_iter)
        self._admm_iter_spin.setFixedWidth(60)
        self._admm_iter_spin.setToolTip(self.tr(
            "Number of ADMM alternating minimization cycles for AL-DIC.\n"
            "1 = single global pass (fastest), 3 = default,\n"
            "5+ = diminishing returns for most cases."
        ))
        admm_row.addWidget(self._admm_iter_spin)
        admm_row.addStretch()
        layout.addLayout(admm_row)
        self._admm_hint = QLabel(self.tr(
            "Only affects AL-DIC solver. Ignored by Local DIC."
        ))
        self._admm_hint.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px; "
            f"padding-left: 6px;"
        )
        layout.addWidget(self._admm_hint)

        # Keep ADMM row disabled when Local DIC is active (use_admm=False)
        self._admm_iter_spin.setEnabled(self._state.use_admm)

        # --- FFT auto-expand -------------------------------------------
        self._auto_expand_cb = QCheckBox(self.tr(
            "Auto-expand FFT search on clipped peaks"
        ))
        self._auto_expand_cb.setToolTip(self.tr(
            "When the NCC peak reaches the edge of the search region, "
            "automatically retry with a larger region "
            "(up to image half-size, 6 retries with 2x growth).\n\n"
            "Only relevant for the FFT init-guess mode."
        ))
        self._auto_expand_cb.setChecked(self._state.fft_auto_expand)
        layout.addWidget(self._auto_expand_cb)

        # --- Focus/scroll guards ---------------------------------------
        self._admm_iter_spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._admm_iter_spin.installEventFilter(self)

        # --- Connections -----------------------------------------------
        self._admm_iter_spin.valueChanged.connect(self._on_admm_changed)
        self._auto_expand_cb.stateChanged.connect(self._on_auto_expand_changed)
        self._state.params_changed.connect(self._on_params_changed_externally)

    def _on_admm_changed(self, value: int) -> None:
        if self._building:
            return
        self._state.set_param("admm_max_iter", value)

    def _on_auto_expand_changed(self, _: int) -> None:
        if self._building:
            return
        self._state.fft_auto_expand = self._auto_expand_cb.isChecked()
        self._state.params_changed.emit()

    def _on_params_changed_externally(self) -> None:
        if self._building:
            return
        self._building = True
        self._admm_iter_spin.setValue(self._state.admm_max_iter)
        self._admm_iter_spin.setEnabled(self._state.use_admm)
        self._auto_expand_cb.setChecked(self._state.fft_auto_expand)
        self._building = False

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)
