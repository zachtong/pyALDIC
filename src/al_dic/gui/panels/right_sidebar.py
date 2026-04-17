"""Right sidebar -- run controls, progress, display settings, console."""

from __future__ import annotations

import re
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.app_state import AppState, RunState
from al_dic.gui.theme import COLORS
from al_dic.gui.widgets.color_range import ColorRange
from al_dic.gui.widgets.console_log import ConsoleLog
from al_dic.gui.widgets.field_selector import FieldSelector
from al_dic.gui.widgets.physical_units_widget import PhysicalUnitsWidget

try:
    from al_dic.gui.icons import icon_download, icon_pause, icon_play, icon_stop
    _HAS_ICONS = True
except ImportError:  # pragma: no cover
    _HAS_ICONS = False


class RightSidebar(QWidget):
    """Right sidebar: run controls, progress, display, console."""

    # Emitted when the user requests the strain post-processing window.
    # MainWindow owns the lazy singleton instance and the slot wiring.
    open_strain_window_requested = Signal()

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
        if _HAS_ICONS:
            self._run_btn.setIcon(icon_play())
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        btn_row = QHBoxLayout()
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedHeight(30)
        self._pause_btn.setEnabled(False)
        if _HAS_ICONS:
            self._pause_btn.setIcon(icon_pause())
        self._pause_btn.clicked.connect(self._on_pause)
        btn_row.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("class", "btn-danger")
        self._stop_btn.setFixedHeight(30)
        self._stop_btn.setEnabled(False)
        if _HAS_ICONS:
            self._stop_btn.setIcon(icon_stop())
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        layout.addLayout(btn_row)

        self._export_btn = QPushButton("Export Results")
        self._export_btn.setFixedHeight(30)
        self._export_btn.setEnabled(False)
        if _HAS_ICONS:
            self._export_btn.setIcon(icon_download())
        self._export_btn.clicked.connect(self._on_export)
        layout.addWidget(self._export_btn)

        self._strain_btn = QPushButton("Open Strain Window")
        self._strain_btn.setFixedHeight(30)
        self._strain_btn.setToolTip(
            "Compute and visualize strain in a separate post-processing "
            "window. Requires displacement results from a completed Run."
        )
        self._strain_btn.clicked.connect(
            self.open_strain_window_requested.emit
        )
        layout.addWidget(self._strain_btn)

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

        # --- Field section ---
        self._add_section_label(layout, "FIELD")
        self._field_selector = FieldSelector()
        layout.addWidget(self._field_selector)

        # Deformed vs reference frame toggle.
        # This controls WHERE the field is plotted (geometry, not styling),
        # so it lives in FIELD rather than VISUALIZATION.
        self._deformed_cb = QCheckBox("Show on deformed frame")
        self._deformed_cb.setChecked(True)
        self._deformed_cb.setToolTip(
            "When checked, overlay results on the deformed (current) frame "
            "instead of the reference frame"
        )
        self._deformed_cb.stateChanged.connect(self._on_deformed_toggled)
        layout.addWidget(self._deformed_cb)

        # --- Visualization section ---
        self._add_section_label(layout, "VISUALIZATION")

        # Colormap selector
        cmap_row = QHBoxLayout()
        cmap_row.setSpacing(4)
        cmap_lbl = QLabel("Colormap")
        cmap_lbl.setFixedWidth(64)
        cmap_row.addWidget(cmap_lbl)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems([
            "jet", "viridis", "turbo", "coolwarm",
            "plasma", "inferno", "RdBu_r", "seismic",
        ])
        self._cmap_combo.setCurrentText(self._state.colormap)
        self._cmap_combo.currentTextChanged.connect(self._state.set_colormap)
        cmap_row.addWidget(self._cmap_combo)
        # Sync combo when active field changes (each field stores its own colormap)
        self._state.display_changed.connect(self._sync_colormap_combo)
        layout.addLayout(cmap_row)

        self._color_range = ColorRange()
        layout.addWidget(self._color_range)

        # Opacity slider (mirrors strain window's StrainVizPanel)
        opacity_row = QHBoxLayout()
        opacity_row.setSpacing(4)
        opacity_lbl = QLabel("Opacity")
        opacity_lbl.setFixedWidth(64)
        opacity_row.addWidget(opacity_lbl)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(int(self._state.overlay_alpha * 100))
        self._opacity_slider.setToolTip("Overlay opacity (0 = transparent, 100 = opaque)")
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)
        layout.addLayout(opacity_row)

        # --- Physical units section ---
        self._add_section_label(layout, "PHYSICAL UNITS")
        self._physical_units = PhysicalUnitsWidget()
        layout.addWidget(self._physical_units)

        # --- Log section ---
        console_header = QHBoxLayout()
        lbl = QLabel("LOG")
        lbl.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; font-weight: bold;"
        )
        console_header.addWidget(lbl)
        console_header.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(56, 20)
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
        # Fatal errors from the pipeline: log + modal dialog
        self._state.fatal_error.connect(self._show_fatal_error)

        # Timer for elapsed display
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_elapsed)
        self._timer.setInterval(1000)

        # Track last known frame string to avoid flickering
        self._last_frame_str: str = ""

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

    def _show_fatal_error(self, title: str, message: str) -> None:
        """Show a modal dialog for pipeline errors that the user must see.

        The console log already has the full error (including tracebacks);
        this modal ensures the user cannot miss critical failures that
        happened below the fold.
        """
        QMessageBox.critical(self, title, message)

    def _on_export(self) -> None:
        if self._state.results is None:
            return

        from al_dic.gui.dialogs.export_dialog import ExportDialog, VizExportHint

        hint = VizExportHint(
            colormap=self._state.colormap,
            auto_range=self._state.color_auto,
            vmin=self._state.color_min,
            vmax=self._state.color_max,
            show_deformed=self._state.show_deformed,
            overlay_alpha=self._state.overlay_alpha,
            use_physical_units=self._state.use_physical_units,
            pixel_size=self._state.pixel_size,
            pixel_unit=self._state.pixel_unit,
            frame_rate=self._state.frame_rate,
        )
        dlg = ExportDialog(
            self._state.results,
            self._state.image_folder,
            hint,
            image_files=self._state.image_files,
            roi_mask=self._state.roi_mask,
            per_frame_rois=self._state.per_frame_rois or None,
            parent=self,
        )
        # All exports happen inside the dialog; exec() blocks until user clicks Close.
        dlg.exec()

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
        elif not paused:
            self._timer.stop()

        if idle:
            self._progress_bar.setValue(0)
            self._progress_label.setText("Ready")
            self._elapsed_label.setText("ELAPSED  --:--")
            self._remaining_label.setText("REMAINING  --:--")
            self._last_frame_str = ""

    def _on_deformed_toggled(self, state: int) -> None:
        """Toggle between reference and deformed frame display."""
        deformed = state == Qt.CheckState.Checked.value
        self._state.show_deformed = deformed
        self._state.display_changed.emit()

    def _on_opacity_changed(self, value: int) -> None:
        """Update overlay opacity from slider (0–100 → 0.0–1.0)."""
        self._state.set_overlay_alpha(value / 100.0)

    def _sync_colormap_combo(self) -> None:
        """Update the colormap combo to reflect the current field's stored colormap."""
        cmap = self._state.colormap
        if self._cmap_combo.currentText() != cmap:
            self._cmap_combo.blockSignals(True)
            idx = self._cmap_combo.findText(cmap)
            if idx >= 0:
                self._cmap_combo.setCurrentIndex(idx)
            self._cmap_combo.blockSignals(False)

    def _on_progress(self, fraction: float, message: str) -> None:
        """Update progress bar and label — show only percentage + frame."""
        self._progress_bar.setValue(int(fraction * 1000))
        # Extract frame info (e.g. "Frame 2/5") from verbose messages
        frame_match = re.search(r"[Ff]rame\s+(\d+/\d+)", message)
        if frame_match:
            self._last_frame_str = frame_match.group(1)
        # Always show last known frame to avoid flickering
        if self._last_frame_str:
            self._progress_label.setText(
                f"{fraction * 100:.0f}%  \u2014  Frame {self._last_frame_str}"
            )
        else:
            self._progress_label.setText(f"{fraction * 100:.0f}%")

    def _update_elapsed(self) -> None:
        """Refresh elapsed and estimated remaining time labels every second."""
        # Compute elapsed from wall clock (not stale state.elapsed_seconds)
        # so the display updates smoothly even between progress callbacks.
        elapsed = self._state.elapsed_seconds
        if self._state.run_state in (RunState.RUNNING, RunState.PAUSED):
            elapsed = time.perf_counter() - self._pipeline_ctrl._start_time
            self._state.elapsed_seconds = elapsed

        mins, secs = divmod(int(elapsed), 60)
        self._elapsed_label.setText(f"ELAPSED  {mins:02d}:{secs:02d}")

        frac = self._state.progress
        if frac > 0.01:
            estimated_total = elapsed / frac
            remaining = estimated_total - elapsed
            r_mins, r_secs = divmod(int(max(0, remaining)), 60)
            self._remaining_label.setText(f"REMAINING  {r_mins:02d}:{r_secs:02d}")
        else:
            self._remaining_label.setText("REMAINING  --:--")
