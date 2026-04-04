"""DIC parameter input panel."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
        row = QHBoxLayout()
        lbl = QLabel("Tracking Mode")
        lbl.setFixedWidth(120)
        row.addWidget(lbl)
        row.addWidget(self._tracking_mode)
        layout.addLayout(row)

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
        ref_lbl = QLabel("Ref Update")
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
        self._custom_lbl = QLabel("Ref Frames")
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

    def _on_tracking_mode_changed(self, text: str) -> None:
        """Set tracking mode param and toggle incremental sub-panel visibility."""
        state = AppState.instance()
        state.set_param("tracking_mode", text.lower())
        self._inc_panel.setVisible(text.lower() == "incremental")

    def _on_ref_mode_changed(self, index: int) -> None:
        """Toggle interval/custom widgets based on ref update mode."""
        state = AppState.instance()
        modes = ["every_frame", "every_n", "custom"]
        mode = modes[index]
        state.inc_ref_mode = mode
        self._interval_spin.setVisible(mode == "every_n")
        self._interval_lbl.setVisible(mode == "every_n")
        self._custom_edit.setVisible(mode == "custom")
        self._custom_lbl.setVisible(mode == "custom")
        state.params_changed.emit()

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
            pass
        state.params_changed.emit()

    def _on_subset_size_changed(self, display_value: int) -> None:
        """Convert odd display value to even internal winsize."""
        # Snap to nearest odd if user somehow enters even
        if display_value % 2 == 0:
            display_value = display_value - 1
            self._subset_size.blockSignals(True)
            self._subset_size.setValue(display_value)
            self._subset_size.blockSignals(False)
        # Store even value internally (display 41 -> internal 40)
        AppState.instance().set_param("subset_size", display_value - 1)

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
