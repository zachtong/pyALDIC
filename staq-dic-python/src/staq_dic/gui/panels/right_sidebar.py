"""Right sidebar -- run controls, progress, display settings, console."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from staq_dic.gui.app_state import AppState, RunState
from staq_dic.gui.theme import COLORS
from staq_dic.gui.widgets.color_range import ColorRange
from staq_dic.gui.widgets.console_log import ConsoleLog
from staq_dic.gui.widgets.field_selector import FieldSelector


class RightSidebar(QWidget):
    """Right sidebar: run controls, progress, display, console."""

    def __init__(self, pipeline_ctrl, parent=None) -> None:
        super().__init__(parent)
        self._pipeline_ctrl = pipeline_ctrl
        self._state = AppState.instance()
        self.setObjectName("rightSidebar")
        self.setFixedWidth(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Run controls ---
        self._run_btn = QPushButton("Run DIC Analysis")
        self._run_btn.setProperty("class", "btn-primary")
        self._run_btn.setFixedHeight(36)
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        btn_row = QHBoxLayout()
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedHeight(30)
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._on_pause)
        btn_row.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("class", "btn-danger")
        self._stop_btn.setFixedHeight(30)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        layout.addLayout(btn_row)

        self._export_btn = QPushButton("Export Results")
        self._export_btn.setFixedHeight(30)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        layout.addWidget(self._export_btn)

        # --- Progress section ---
        self._add_section_label(layout, "PROGRESS")

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("Ready")
        self._progress_label.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px;"
        )
        layout.addWidget(self._progress_label)

        # Elapsed / remaining
        stats_row = QHBoxLayout()
        self._elapsed_label = QLabel("ELAPSED  --:--")
        self._elapsed_label.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 10px;"
        )
        stats_row.addWidget(self._elapsed_label)
        self._remaining_label = QLabel("REMAINING  --:--")
        self._remaining_label.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 10px;"
        )
        stats_row.addWidget(self._remaining_label)
        layout.addLayout(stats_row)

        # --- Display field section ---
        self._add_section_label(layout, "DISPLAY FIELD")
        self._field_selector = FieldSelector()
        layout.addWidget(self._field_selector)

        # --- Color range section ---
        self._add_section_label(layout, "COLOR RANGE")
        self._color_range = ColorRange()
        layout.addWidget(self._color_range)

        # --- Console section ---
        console_header = QHBoxLayout()
        lbl = QLabel("CONSOLE")
        lbl.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; font-weight: bold;"
        )
        console_header.addWidget(lbl)
        console_header.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(40, 20)
        clear_btn.setStyleSheet(
            f"font-size: 10px; color: {COLORS.TEXT_MUTED}; border: none;"
        )
        clear_btn.clicked.connect(lambda: self._console.clear_log())
        console_header.addWidget(clear_btn)
        layout.addLayout(console_header)

        self._console = ConsoleLog()
        layout.addWidget(self._console)

        # Spacer at bottom
        layout.addStretch()

        # --- Connect signals ---
        self._state.run_state_changed.connect(self._on_run_state)
        self._state.progress_updated.connect(self._on_progress)
        self._state.log_message.connect(self._console.append_log)

        # Timer for elapsed display
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_elapsed)
        self._timer.setInterval(1000)

    @property
    def console(self) -> ConsoleLog:
        """Expose console for external log access."""
        return self._console

    @property
    def color_range(self) -> ColorRange:
        """Expose color range widget for external updates."""
        return self._color_range

    def _add_section_label(self, layout: QVBoxLayout, text: str) -> None:
        """Add an uppercase section heading to the layout."""
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; "
            f"font-weight: bold; letter-spacing: 1px; margin-top: 8px;"
        )
        layout.addWidget(lbl)

    def _on_run(self) -> None:
        self._pipeline_ctrl.start()

    def _on_pause(self) -> None:
        if self._state.run_state == RunState.RUNNING:
            self._pipeline_ctrl.pause()
        elif self._state.run_state == RunState.PAUSED:
            self._pipeline_ctrl.resume()

    def _on_stop(self) -> None:
        self._pipeline_ctrl.stop()

    def _on_export(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export Results To")
        if not folder:
            return
        # Export logic will be added when VizController is available (Task 10)
        self._state.log_message.emit(
            f"Export to {folder} -- not yet implemented", "warn"
        )

    def _on_run_state(self, new_state: RunState) -> None:
        """Update button enabled/disabled states on run state change."""
        running = new_state == RunState.RUNNING
        paused = new_state == RunState.PAUSED
        done = new_state == RunState.DONE
        idle = new_state == RunState.IDLE

        self._run_btn.setEnabled(idle or done)
        self._pause_btn.setEnabled(running or paused)
        self._pause_btn.setText("Resume" if paused else "Pause")
        self._stop_btn.setEnabled(running or paused)
        self._export_btn.setEnabled(done)

        if running:
            self._timer.start()
        else:
            self._timer.stop()

        if idle:
            self._progress_bar.setValue(0)
            self._progress_label.setText("Ready")

    def _on_progress(self, fraction: float, message: str) -> None:
        """Update progress bar and label from pipeline progress signal."""
        self._progress_bar.setValue(int(fraction * 1000))
        self._progress_label.setText(message or f"{fraction * 100:.0f}%")

    def _update_elapsed(self) -> None:
        """Refresh elapsed and estimated remaining time labels."""
        elapsed = self._state.elapsed_seconds
        mins, secs = divmod(int(elapsed), 60)
        self._elapsed_label.setText(f"ELAPSED  {mins:02d}:{secs:02d}")

        frac = self._state.progress
        if frac > 0.01:
            estimated_total = elapsed / frac
            remaining = estimated_total - elapsed
            r_mins, r_secs = divmod(int(max(0, remaining)), 60)
            self._remaining_label.setText(f"REMAINING  {r_mins:02d}:{r_secs:02d}")
