"""Color range controls -- auto/manual min/max."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from staq_dic.gui.app_state import AppState
from staq_dic.gui.theme import COLORS


class ColorRange(QWidget):
    """Auto toggle + Min/Max spin boxes for color range."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Auto checkbox
        auto_row = QHBoxLayout()
        self._auto_cb = QCheckBox("Auto")
        self._auto_cb.setChecked(True)
        self._auto_cb.stateChanged.connect(self._on_auto_changed)
        auto_row.addWidget(self._auto_cb)
        auto_row.addStretch()
        layout.addLayout(auto_row)

        # Min/Max row
        range_row = QHBoxLayout()
        range_row.setSpacing(4)

        range_row.addWidget(QLabel("Min"))
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setRange(-1000, 1000)
        self._min_spin.setDecimals(3)
        self._min_spin.setValue(0.0)
        self._min_spin.setSingleStep(0.01)
        self._min_spin.setEnabled(False)  # disabled when auto=True
        self._min_spin.valueChanged.connect(self._on_range_changed)
        range_row.addWidget(self._min_spin)

        range_row.addWidget(QLabel("Max"))
        self._max_spin = QDoubleSpinBox()
        self._max_spin.setRange(-1000, 1000)
        self._max_spin.setDecimals(3)
        self._max_spin.setValue(1.0)
        self._max_spin.setSingleStep(0.01)
        self._max_spin.setEnabled(False)
        self._max_spin.valueChanged.connect(self._on_range_changed)
        range_row.addWidget(self._max_spin)

        layout.addLayout(range_row)

    def _on_auto_changed(self, state: int) -> None:
        """Enable/disable spin boxes based on auto checkbox."""
        auto = state == Qt.CheckState.Checked.value
        self._min_spin.setEnabled(not auto)
        self._max_spin.setEnabled(not auto)
        if not auto:
            # Populate spin boxes with the current auto-computed range
            # so the user starts from the current display, not stale defaults.
            app_state = AppState.instance()
            self._min_spin.blockSignals(True)
            self._max_spin.blockSignals(True)
            self._min_spin.setValue(app_state.color_min)
            self._max_spin.setValue(app_state.color_max)
            self._min_spin.blockSignals(False)
            self._max_spin.blockSignals(False)
        self._emit_range()

    def _on_range_changed(self) -> None:
        """Forward manual range changes to AppState."""
        self._emit_range()

    def _emit_range(self) -> None:
        """Push current auto/min/max to AppState."""
        AppState.instance().set_color_range(
            auto=self._auto_cb.isChecked(),
            vmin=self._min_spin.value(),
            vmax=self._max_spin.value(),
        )

    def set_range(self, vmin: float, vmax: float) -> None:
        """Update displayed range (called when auto-range is computed)."""
        self._min_spin.blockSignals(True)
        self._max_spin.blockSignals(True)
        self._min_spin.setValue(vmin)
        self._max_spin.setValue(vmax)
        self._min_spin.blockSignals(False)
        self._max_spin.blockSignals(False)
