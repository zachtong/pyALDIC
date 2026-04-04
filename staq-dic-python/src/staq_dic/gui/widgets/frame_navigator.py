"""Frame navigator — bottom bar with slider and prev/next buttons."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from staq_dic.gui.app_state import AppState
from staq_dic.gui.theme import COLORS

try:
    from staq_dic.gui.icons import icon_chevron_left, icon_chevron_right
    _HAS_ICONS = True
except ImportError:  # pragma: no cover
    _HAS_ICONS = False


class FrameNavigator(QWidget):
    """Bottom bar: prev/next buttons, frame label, horizontal slider."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()

        self.setFixedHeight(36)
        self.setStyleSheet(
            f"background: {COLORS.BG_PANEL}; "
            f"border-top: 1px solid {COLORS.BORDER};"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(8)

        # Prev button
        self._prev_btn = QPushButton("<")
        self._prev_btn.setFixedWidth(28)
        self._prev_btn.setToolTip("Previous frame")
        if _HAS_ICONS:
            self._prev_btn.setIcon(icon_chevron_left())
            self._prev_btn.setText("")
        self._prev_btn.clicked.connect(self._on_prev)
        layout.addWidget(self._prev_btn)

        # Frame label
        self._label = QLabel("FRAME 0/0")
        self._label.setFixedWidth(90)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; "
            f"font-weight: bold; background: transparent;"
        )
        layout.addWidget(self._label)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, stretch=1)

        # Next button
        self._next_btn = QPushButton(">")
        self._next_btn.setFixedWidth(28)
        self._next_btn.setToolTip("Next frame")
        if _HAS_ICONS:
            self._next_btn.setIcon(icon_chevron_right())
            self._next_btn.setText("")
        self._next_btn.clicked.connect(self._on_next)
        layout.addWidget(self._next_btn)

        # Connect signals
        self._state.images_changed.connect(self._on_images_changed)
        self._state.current_frame_changed.connect(self._on_frame_changed)

    def _on_images_changed(self) -> None:
        n = len(self._state.image_files)
        self._slider.setRange(0, max(0, n - 1))
        self._update_label(0, n)

    def _on_frame_changed(self, idx: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._update_label(idx, len(self._state.image_files))

    def _on_slider_changed(self, value: int) -> None:
        self._state.set_current_frame(value)

    def _on_prev(self) -> None:
        self._state.set_current_frame(self._state.current_frame - 1)

    def _on_next(self) -> None:
        self._state.set_current_frame(self._state.current_frame + 1)

    def _update_label(self, idx: int, total: int) -> None:
        self._label.setText(f"FRAME {idx + 1}/{total}" if total > 0 else "FRAME 0/0")
