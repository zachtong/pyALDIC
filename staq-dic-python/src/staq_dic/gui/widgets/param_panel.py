"""DIC parameter input panel."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from staq_dic.gui.app_state import AppState


class ParamPanel(QWidget):
    """Parameter inputs: Subset Size, Subset Step, Search Range, Tracking Mode."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        state = AppState.instance()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Subset Size ---
        self._subset_size = self._add_spinbox(
            layout,
            "Subset Size",
            state.subset_size,
            minimum=10,
            maximum=200,
            step=2,
            tooltip="IC-GN subset window size in pixels (must be even)",
        )
        self._subset_size.valueChanged.connect(
            lambda v: state.set_param("subset_size", v)
        )

        # --- Subset Step ---
        self._subset_step = self._add_power_of_2_combo(
            layout,
            "Subset Step",
            state.subset_step,
            options=[4, 8, 16, 32, 64],
            tooltip="Node spacing in pixels (must be power of 2)",
        )
        self._subset_step.currentTextChanged.connect(
            lambda v: state.set_param("subset_step", int(v))
        )

        # --- Initial Search Range ---
        self._search_range = self._add_spinbox(
            layout,
            "Search Range",
            state.search_range,
            minimum=5,
            maximum=100,
            step=5,
            tooltip="FFT cross-correlation search range in pixels",
        )
        self._search_range.valueChanged.connect(
            lambda v: state.set_param("search_range", v)
        )

        # --- Tracking Mode ---
        self._tracking_mode = QComboBox()
        self._tracking_mode.addItems(["Incremental", "Accumulative"])
        self._tracking_mode.setCurrentText(state.tracking_mode.capitalize())
        self._tracking_mode.currentTextChanged.connect(
            lambda v: state.set_param("tracking_mode", v.lower())
        )
        row = QHBoxLayout()
        lbl = QLabel("Tracking Mode")
        lbl.setFixedWidth(120)
        row.addWidget(lbl)
        row.addWidget(self._tracking_mode)
        layout.addLayout(row)

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
