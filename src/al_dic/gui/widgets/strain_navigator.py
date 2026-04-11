"""Bottom navigation bar for the strain post-processing window.

Independent of AppState — driven entirely via :meth:`set_state` and the
:attr:`frame_changed` signal.  Mirrors :class:`FrameNavigator` visually
but does not touch ``AppState.current_frame``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

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

_SPEED_PRESETS = [
    ("1 fps", 1000),
    ("2 fps", 500),
    ("5 fps", 200),
    ("10 fps", 100),
    ("30 fps", 33),
]


class StrainNavigator(QWidget):
    """Bottom nav bar: prev / play-pause / next / speed / label / slider.

    Caller is responsible for:
    * Calling :meth:`set_state` whenever the total frame count or current
      frame changes externally.
    * Connecting :attr:`frame_changed` to advance the displayed field.
    """

    frame_changed = Signal(int)   # emitted with the new frame index

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._n_frames: int = 0
        self._current: int = 0
        self._playing: bool = False

        self.setFixedHeight(36)
        self.setStyleSheet(
            f"background: {COLORS.BG_PANEL}; "
            f"border-top: 1px solid {COLORS.BORDER};"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        # Prev
        self._prev_btn = QPushButton()
        self._prev_btn.setFixedWidth(28)
        self._prev_btn.setToolTip("Previous frame")
        if _HAS_ICONS:
            self._prev_btn.setIcon(icon_chevron_left())
        else:
            self._prev_btn.setText("<")
        self._prev_btn.clicked.connect(self._on_prev)
        layout.addWidget(self._prev_btn)

        # Play / Pause
        self._play_btn = QPushButton()
        self._play_btn.setFixedWidth(28)
        self._play_btn.setToolTip("Play animation")
        if _HAS_ICONS:
            self._play_btn.setIcon(icon_play())
        else:
            self._play_btn.setText("\u25B6")
        self._play_btn.clicked.connect(self._on_play_toggle)
        layout.addWidget(self._play_btn)

        # Next
        self._next_btn = QPushButton()
        self._next_btn.setFixedWidth(28)
        self._next_btn.setToolTip("Next frame")
        if _HAS_ICONS:
            self._next_btn.setIcon(icon_chevron_right())
        else:
            self._next_btn.setText(">")
        self._next_btn.clicked.connect(self._on_next)
        layout.addWidget(self._next_btn)

        # Speed selector
        self._speed_combo = QComboBox()
        for label, _ms in _SPEED_PRESETS:
            self._speed_combo.addItem(label)
        self._speed_combo.setCurrentIndex(1)   # default 2 fps
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
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider, stretch=1)

        # Timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_tick)
        self._timer.setInterval(_SPEED_PRESETS[1][1])   # 2 fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_state(self, n_frames: int, current: int) -> None:
        """Sync the slider range and label from outside.

        Called by :class:`StrainWindow` whenever the frame count or the
        external current-frame changes (e.g. after Compute Strain or a
        slider drag that was processed by the parent).
        """
        self._n_frames = max(0, n_frames)
        clamped = max(0, min(current, max(0, self._n_frames - 1)))
        self._current = clamped
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(0, self._n_frames - 1))
        self._slider.setValue(clamped)
        self._slider.blockSignals(False)
        self._update_label()
        if self._n_frames < 2:
            self._stop_playback()

    def stop_playback(self) -> None:
        """Stop playback (e.g. when new results arrive)."""
        self._stop_playback()

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_slider(self, value: int) -> None:
        self._current = value
        self._update_label()
        self.frame_changed.emit(value)

    def _on_prev(self) -> None:
        if self._n_frames < 1:
            return
        new = max(0, self._current - 1)
        self._slider.setValue(new)   # triggers _on_slider → frame_changed

    def _on_next(self) -> None:
        if self._n_frames < 1:
            return
        new = min(self._n_frames - 1, self._current + 1)
        self._slider.setValue(new)

    def _on_play_toggle(self) -> None:
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _on_tick(self) -> None:
        if self._n_frames < 2:
            self._stop_playback()
            return
        next_frame = (self._current + 1) % self._n_frames
        self._slider.setValue(next_frame)

    def _on_speed_changed(self, index: int) -> None:
        if 0 <= index < len(_SPEED_PRESETS):
            self._timer.setInterval(_SPEED_PRESETS[index][1])

    def _start_playback(self) -> None:
        if self._n_frames < 2:
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

    def _update_label(self) -> None:
        if self._n_frames > 0:
            self._label.setText(f"FRAME {self._current + 1}/{self._n_frames}")
        else:
            self._label.setText("FRAME 0/0")
