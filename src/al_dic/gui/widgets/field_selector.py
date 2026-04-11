"""Display field toggle -- Disp U / Disp V selector."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton

from al_dic.gui.app_state import AppState
from al_dic.gui.theme import COLORS


class FieldSelector(QWidget):
    """Two exclusive toggle buttons: Disp U and Disp V."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._btn_u = QPushButton("Disp U")
        self._btn_v = QPushButton("Disp V")

        for btn in (self._btn_u, self._btn_v):
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            layout.addWidget(btn)

        # Default: Disp U selected
        self._btn_u.setChecked(True)
        self._update_styles()

        self._btn_u.clicked.connect(lambda: self._select("disp_u"))
        self._btn_v.clicked.connect(lambda: self._select("disp_v"))

    def _select(self, field: str) -> None:
        """Set exactly one button active and update AppState."""
        self._btn_u.setChecked(field == "disp_u")
        self._btn_v.setChecked(field == "disp_v")
        self._update_styles()
        AppState.instance().set_display_field(field)

    def _update_styles(self) -> None:
        """Apply active/inactive styles to both buttons."""
        for btn, active in [
            (self._btn_u, self._btn_u.isChecked()),
            (self._btn_v, self._btn_v.isChecked()),
        ]:
            if active:
                btn.setStyleSheet(
                    f"background: {COLORS.ACCENT}; color: white; "
                    f"border: none; border-radius: 4px; font-weight: bold;"
                )
            else:
                btn.setStyleSheet(
                    f"background: {COLORS.BG_INPUT}; color: {COLORS.TEXT_SECONDARY}; "
                    f"border: 1px solid {COLORS.BORDER}; border-radius: 4px;"
                )
