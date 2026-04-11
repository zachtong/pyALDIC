"""Frame navigator — bottom bar with slider, prev/next, and playback controls."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from al_dic.gui.app_state import AppState
from al_dic.gui.theme import COLORS

try:
    from al_dic.gui.icons import (
        icon_chevron_left,
        icon_chevron_right,
        icon_play,
        icon_pause,
    )
    _HAS_ICONS = True
except ImportError:  # pragma: no cover
    _HAS_ICONS = False

# Speed presets: label -> interval in ms
_SPEED_PRESETS = [
    ("1 fps", 1000),
    ("2 fps", 500),
    ("5 fps", 200),
    ("10 fps", 100),
    ("30 fps", 33),
]


class FrameNavigator(QWidget):
    """Bottom bar: prev/next, play/pause, speed selector, frame label, slider."""

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
        layout.setSpacing(6)

        # Prev button
        self._prev_btn = QPushButton("<")
        self._prev_btn.setFixedWidth(28)
        self._prev_btn.setToolTip("Previous frame")
        if _HAS_ICONS:
            self._prev_btn.setIcon(icon_chevron_left())
            self._prev_btn.setText("")
        self._prev_btn.clicked.connect(self._on_prev)
        layout.addWidget(self._prev_btn)

        # Play/Pause button
        self._play_btn = QPushButton()
        self._play_btn.setFixedWidth(28)
        self._play_btn.setToolTip("Play animation")
        if _HAS_ICONS:
            self._play_btn.setIcon(icon_play())
        else:
            self._play_btn.setText("\u25B6")
        self._play_btn.clicked.connect(self._on_play_toggle)
        layout.addWidget(self._play_btn)

        # Next button
        self._next_btn = QPushButton(">")
        self._next_btn.setFixedWidth(28)
        self._next_btn.setToolTip("Next frame")
        if _HAS_ICONS:
            self._next_btn.setIcon(icon_chevron_right())
            self._next_btn.setText("")
        self._next_btn.clicked.connect(self._on_next)
        layout.addWidget(self._next_btn)

        # Speed selector
        self._speed_combo = QComboBox()
        for label, _ms in _SPEED_PRESETS:
            self._speed_combo.addItem(label)
        self._speed_combo.setCurrentIndex(1)  # default 2 fps
        self._speed_combo.setFixedWidth(68)
        self._speed_combo.setToolTip("Playback speed")
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        layout.addWidget(self._speed_combo)

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

        # --- Playback timer ---
        self._playing = False
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer_tick)
        self._timer.setInterval(_SPEED_PRESETS[1][1])  # 2 fps default

        # Connect signals
        self._state.images_changed.connect(self._on_images_changed)
        self._state.current_frame_changed.connect(self._on_frame_changed)

    def _on_images_changed(self) -> None:
        n = len(self._state.image_files)
        self._slider.setRange(0, max(0, n - 1))
        self._update_label(0, n)
        # Stop playback when images change
        self._stop_playback()

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

    def _on_play_toggle(self) -> None:
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        n = len(self._state.image_files)
        if n < 2:
            return
        self._playing = True
        if _HAS_ICONS:
            self._play_btn.setIcon(icon_pause())
        else:
            self._play_btn.setText("\u23F8")
        self._play_btn.setToolTip("Pause animation")
        self._timer.start()

    def _stop_playback(self) -> None:
        self._playing = False
        self._timer.stop()
        if _HAS_ICONS:
            self._play_btn.setIcon(icon_play())
        else:
            self._play_btn.setText("\u25B6")
        self._play_btn.setToolTip("Play animation")

    def _on_timer_tick(self) -> None:
        n = len(self._state.image_files)
        if n < 2:
            self._stop_playback()
            return
        next_frame = self._state.current_frame + 1
        if next_frame >= n:
            next_frame = 0  # loop
        self._state.set_current_frame(next_frame)

    def _on_speed_changed(self, index: int) -> None:
        if 0 <= index < len(_SPEED_PRESETS):
            _label, ms = _SPEED_PRESETS[index]
            self._timer.setInterval(ms)

    def _update_label(self, idx: int, total: int) -> None:
        self._label.setText(f"FRAME {idx + 1}/{total}" if total > 0 else "FRAME 0/0")
