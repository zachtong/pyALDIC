"""Top-level strain post-processing window.

Independent ``QMainWindow`` that consumes displacement results from
``state.results.result_disp``, runs :class:`StrainController` on demand,
and renders the resulting fields with a *private* ``VizController``.

Decoupling contracts (enforced by tests):

* Owns its own ``_strain_current_frame`` -- never mutates
  ``state.current_frame``.
* Owns a private ``VizController`` cache -- never reads or writes
  ``state.colormap`` / ``state.color_min`` / ``state.color_max`` /
  ``state.display_field``.
* Reads ``state.results.result_disp`` for displacement fields and writes
  back via :func:`dataclasses.replace` through :class:`StrainController`
  only.

Field routing:

* ``disp_u``, ``disp_v``, ``disp_magnitude``, ``velocity`` -- read from
  ``result_disp`` directly; available before Compute Strain.
* ``strain_*``, ``strain_rotation``, ``strain_mean_normal`` -- require a
  completed :meth:`trigger_compute` call.
"""

from __future__ import annotations

import traceback as _tb
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from dataclasses import replace as _dc_replace

from PySide6.QtCore import QEvent, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from al_dic.core.data_structures import PipelineResult, StrainResult
from al_dic.gui.app_state import AppState
from al_dic.gui.controllers.image_controller import ImageController
from al_dic.gui.controllers.strain_controller import (
    StrainComputationCancelled,
    StrainController,
)
from al_dic.gui.controllers.viz_controller import VizController
from al_dic.gui.panels.canvas_area import visible_values
from al_dic.gui.panels.strain_canvas import FieldImage, StrainCanvas
from al_dic.gui.widgets.collapsible_section import CollapsibleSection
from al_dic.gui.widgets.colorbar_overlay import ColorbarOverlay
from al_dic.gui.widgets.console_log import ConsoleLog
from al_dic.gui.widgets.strain_field_selector import (
    DISP_FIELD_NAMES,
    StrainFieldSelector,
)
from al_dic.gui.widgets.strain_param_panel import StrainParamPanel
from al_dic.gui.widgets.strain_navigator import StrainNavigator
from al_dic.gui.widgets.strain_viz_panel import StrainVizPanel
from al_dic.gui.widgets.physical_units_widget import PhysicalUnitsWidget
from al_dic.gui.widgets.strain_field_selector import field_colorbar_label
from al_dic.gui.theme import COLORS

try:
    from al_dic.gui.icons import icon_maximize, icon_zoom_in, icon_zoom_out
    _HAS_ICONS = True
except ImportError:  # pragma: no cover
    _HAS_ICONS = False


# Namespace prefix for the private VizController cache.
_FIELD_NS = "strain_window"


def initial_window_size(avail_width: int, avail_height: int) -> tuple[int, int]:
    """Clamp the preferred 1280x800 default to the usable screen area.

    A fixed 800 px height overflows small laptop screens (1366x768 /
    1280x800 minus menu bar, dock/taskbar and title bar), and on macOS a
    window whose bottom edge opens off-screen cannot be shrunk because
    the resize handle is unreachable and the title bar cannot be dragged
    above the menu bar. The margins keep the whole frame, including the
    resize edges, on screen.
    """
    return (
        max(640, min(1280, avail_width - 40)),
        max(480, min(800, avail_height - 80)),
    )


class _StrainWorker(QThread):
    """Background thread for strain computation.

    Emits ``progress(fraction, message)`` once per frame,
    ``finished(result_list)`` on success, ``cancelled()`` when the user aborts,
    and ``error(message)`` on failure.  Computation happens in the thread;
    state updates happen in the caller's slot (main thread) to avoid
    cross-thread Qt issues.
    """

    progress: Signal = Signal(float, str)
    finished: Signal = Signal(list)
    error: Signal = Signal(str)
    cancelled: Signal = Signal()

    def __init__(
        self,
        strain_ctrl: "StrainController",
        override: dict,
    ) -> None:
        super().__init__()
        self._ctrl = strain_ctrl
        self._override = override
        self._cancel = False

    def cancel(self) -> None:
        """Request cancellation; takes effect at the next frame boundary."""
        self._cancel = True

    def run(self) -> None:
        try:
            results = self._ctrl.compute_all_frames(
                self._override,
                progress_cb=lambda f, m: self.progress.emit(f, m),
                should_stop=lambda: self._cancel,
            )
            self.finished.emit(results)
        except StrainComputationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class StrainWindow(QMainWindow):
    """Independent strain post-processing window."""

    def __init__(
        self,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Strain Post-Processing"))
        # Use the shared app icon so the Strain window is recognisable
        # in the taskbar / Alt-Tab alongside the main window.
        from al_dic.gui.icons import icon_app
        self.setWindowIcon(icon_app())
        # Match the main window's dark OS title bar so the two windows
        # share one visual frame.
        from al_dic.gui.window_chrome import enable_dark_title_bar
        enable_dark_title_bar(self)
        # Default size: 1280×800 clamped to the current screen's usable
        # area so the bottom edge (frame slider, resize handle) never
        # opens off-screen. The right column is wrapped in a QScrollArea
        # so even smaller screens still reach every control.
        screen = self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.resize(*initial_window_size(avail.width(), avail.height()))

        self._state = state
        self._strain_ctrl = StrainController(state)
        self._viz_ctrl = VizController()   # PRIVATE -- isolated from MainWindow
        self._image_ctrl = ImageController(state)
        self._strain_current_frame: int = 0
        # Cache the last auto-computed range so Auto→Manual switch is seamless
        self._last_rendered_vmin: float = 0.0
        self._last_rendered_vmax: float = 1.0
        # Guard flag to prevent save-back loop when we're loading AppState → StrainVizPanel
        self._loading_field_state: bool = False

        # --- Build layout ---
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left pane: zoom toolbar + canvas + colorbar + frame slider
        left = QVBoxLayout()
        left.setSpacing(0)

        # --- Zoom toolbar (matches main window toolbar style) ---
        _zoom_bar = QWidget()
        _zoom_bar.setFixedHeight(36)
        _zoom_bar.setStyleSheet(
            f"background: {COLORS.BG_PANEL}; "
            f"border-bottom: 1px solid {COLORS.BORDER};"
        )
        _zoom_layout = QHBoxLayout(_zoom_bar)
        _zoom_layout.setContentsMargins(8, 2, 8, 2)
        _zoom_layout.setSpacing(4)
        _btn_fit = QPushButton(self.tr("Fit"))
        _btn_fit.setToolTip(self.tr("Fit image to viewport"))
        _btn_fit.setFixedWidth(60)
        if _HAS_ICONS:
            _btn_fit.setIcon(icon_maximize())
            _btn_fit.setText("")
            _btn_fit.setFixedWidth(28)
        _btn_100 = QPushButton(self.tr("100%"))
        _btn_100.setToolTip(self.tr("Zoom to 100% (1:1)"))
        _btn_100.setFixedWidth(60)
        _btn_zin = QPushButton("+")
        _btn_zin.setToolTip(self.tr("Zoom in"))
        _btn_zin.setFixedWidth(28)
        if _HAS_ICONS:
            _btn_zin.setIcon(icon_zoom_in())
            _btn_zin.setText("")
        _btn_zout = QPushButton(self.tr("–"))
        _btn_zout.setToolTip(self.tr("Zoom out"))
        _btn_zout.setFixedWidth(28)
        if _HAS_ICONS:
            _btn_zout.setIcon(icon_zoom_out())
            _btn_zout.setText("")
        _zoom_layout.addWidget(_btn_fit)
        _zoom_layout.addWidget(_btn_100)
        _zoom_layout.addWidget(_btn_zin)
        _zoom_layout.addWidget(_btn_zout)
        _zoom_layout.addStretch()
        left.addWidget(_zoom_bar)

        canvas_row = QHBoxLayout()
        canvas_row.setSpacing(4)
        canvas_row.setContentsMargins(0, 0, 0, 0)
        self._canvas = StrainCanvas()
        canvas_row.addWidget(self._canvas, 1)
        left.addLayout(canvas_row, 1)
        # Colorbar overlaid on the canvas viewport (same pattern as main window)
        self._colorbar = ColorbarOverlay(self._canvas.viewport())
        self._canvas.viewport().installEventFilter(self)

        _btn_fit.clicked.connect(self._canvas.fit_to_view)
        _btn_100.clicked.connect(self._canvas.zoom_to_100)
        _btn_zin.clicked.connect(self._canvas.zoom_in)
        _btn_zout.clicked.connect(self._canvas.zoom_out)

        self._frame_nav = StrainNavigator()
        self._frame_nav.frame_changed.connect(self._on_frame_nav_changed)
        left.addWidget(self._frame_nav)

        root.addLayout(left, 1)

        # --- Right pane: scrollable column with collapsible sections ---
        # Previously a flat QVBoxLayout — on 1366×768 / 1280×800 screens
        # the PHYSICAL UNITS and LOG sections sat below the visible
        # area. Wrapping in QScrollArea + folding low-priority sections
        # by default keeps the most-used controls (parameters, field,
        # viz) on-screen and provides a scrollbar fallback otherwise.
        right_container = QWidget()
        # Ignored horizontal sizePolicy lets the QScrollArea below
        # constrain width to its viewport even when a collapsed section
        # reports an unusual sizeHint — same pattern as left_sidebar.
        right_container.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred,
        )
        right = QVBoxLayout(right_container)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(6)

        # Strain parameters (first thing users tweak — expanded).
        self._params_section = CollapsibleSection(
            self.tr("STRAIN PARAMETERS"), expanded=True,
        )
        self._param_panel = StrainParamPanel()
        self._param_panel.params_dirty.connect(self._on_params_dirty)
        self._params_section.add_widget(self._param_panel)
        right.addWidget(self._params_section)

        # Action buttons stay OUTSIDE collapsible sections so users
        # can trigger Compute / Export even when STRAIN PARAMETERS is
        # folded.
        self._compute_btn = QPushButton(self.tr("Compute Strain"))
        self._compute_btn.setProperty("class", "btn-primary")
        self._compute_btn.setFixedHeight(40)
        self._compute_btn.clicked.connect(self._on_compute_clicked)
        right.addWidget(self._compute_btn)

        # Cancel button: hidden until a compute is running (mirrors the main
        # window's DIC cancel).  Aborts at the next frame boundary and keeps the
        # previous strain result.
        self._cancel_btn = QPushButton(self.tr("Cancel"))
        self._cancel_btn.setProperty("class", "btn-danger")
        self._cancel_btn.setFixedHeight(30)
        self._cancel_btn.setToolTip(
            self.tr("Cancel the running strain computation. "
                    "The previous strain result is kept.")
        )
        from al_dic.gui.icons import icon_stop
        self._cancel_btn.setIcon(icon_stop())
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        right.addWidget(self._cancel_btn)

        self._export_strain_btn = QPushButton(self.tr("Export Results"))
        self._export_strain_btn.setFixedHeight(30)
        self._export_strain_btn.setToolTip(self.tr(
            "Export displacement and strain results to NPZ / MAT / CSV / PNG"
        ))
        self._export_strain_btn.setEnabled(False)
        self._export_strain_btn.clicked.connect(self._on_export_strain)
        right.addWidget(self._export_strain_btn)

        # Progress bar (hidden until compute starts)
        self._strain_progress = QProgressBar()
        self._strain_progress.setRange(0, 1000)
        self._strain_progress.setValue(0)
        self._strain_progress.setTextVisible(False)
        self._strain_progress.setFixedHeight(8)
        self._strain_progress.setVisible(False)
        right.addWidget(self._strain_progress)

        self._strain_progress_label = QLabel("")
        self._strain_progress_label.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px;"
        )
        self._strain_progress_label.setVisible(False)
        right.addWidget(self._strain_progress_label)

        # Worker reference (kept alive while running)
        self._strain_worker: _StrainWorker | None = None

        self._stale_label = QLabel("")
        self._stale_label.setStyleSheet(
            "color: #fbbf24; font-size: 11px; font-style: italic;"
        )
        right.addWidget(self._stale_label)

        # Field selector (switched often — expanded).
        self._field_section = CollapsibleSection(
            self.tr("FIELD"), expanded=True,
        )
        self._field_selector = StrainFieldSelector()
        self._field_selector.field_changed.connect(self._on_field_changed)
        self._field_section.add_widget(self._field_selector)
        right.addWidget(self._field_section)

        # Visualization controls (touched on every render — expanded).
        self._viz_section = CollapsibleSection(
            self.tr("VISUALIZATION"), expanded=True,
        )
        self._viz_panel = StrainVizPanel()
        # Start from the main window's choice, then stay independent -- the
        # same contract as this panel's colormap, opacity and geometry.
        self._viz_panel.set_hidden_bg_color(self._state.hidden_bg_color)
        self._viz_panel.viz_changed.connect(self._on_viz_panel_changed)
        self._viz_panel.auto_disabled.connect(self._on_auto_range_disabled)
        self._viz_section.add_widget(self._viz_panel)
        right.addWidget(self._viz_section)

        # Physical units (rarely changed mid-session — collapsed).
        self._units_section = CollapsibleSection(
            self.tr("PHYSICAL UNITS"), expanded=False,
        )
        self._physical_units = PhysicalUnitsWidget()
        self._units_section.add_widget(self._physical_units)
        right.addWidget(self._units_section)

        # Console / log (diagnostics — collapsed by default; users open
        # it when something looks wrong).
        self._log_section = CollapsibleSection(
            self.tr("LOG"), expanded=False,
        )
        self._console = ConsoleLog()
        self._log_section.add_widget(self._console)
        right.addWidget(self._log_section)

        right.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        right_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setWidget(right_container)
        # 320 content + ~20 for the scrollbar gutter.  Fixed so the
        # canvas keeps its proportional share of the window.
        right_scroll.setFixedWidth(340)
        root.addWidget(right_scroll, 0)

        # The field view becomes the first tab; probe analysis is the second.
        # Wrapping rather than rebuilding keeps this window's behaviour exactly
        # as it was -- the tab is additive.
        from al_dic.gui.panels.analysis_tab import AnalysisTab

        self._tabs = QTabWidget(self)
        self._tabs.addTab(central, self.tr("Strain Field"))
        self._analysis_tab = AnalysisTab(state, self)
        self._tabs.addTab(self._analysis_tab, self.tr("Analysis"))
        self.setCentralWidget(self._tabs)
        # One frame across both tabs; the chart's cursor and a click on it
        # move the field view too.
        self._analysis_tab.frame_requested.connect(self.set_strain_frame)
        self._analysis_tab.set_frame(self._strain_current_frame)
        self._analysis_tab.set_default_field(self._field_selector.current_field())
        self._analysis_tab.set_parameters_provider(self._strain_parameters)
        self._analysis_tab.set_field_renderer(self.reference_field_image)
        # The field view renders only while it is the page on show; a render
        # asked for meanwhile is made once, on return.
        self._render_pending = False
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Track external pipeline runs and shared display settings
        self._state.results_changed.connect(self._on_state_results_changed)
        # display_changed from the main window (e.g. main window changed a field's
        # colormap): reload StrainVizPanel controls and re-render.
        self._state.display_changed.connect(self._on_state_display_changed)
        # Physical units change node values (scaling) → clear Tier-1 grid cache.
        self._state.physical_units_changed.connect(self._viz_ctrl.clear_all)

        # Load default color state for the initially selected field.
        self._load_field_state_into_panel()

        self._sync_slider_range()
        self._render_current()
        # The window can be opened with results already present (session
        # reload, or auto-open when a run finishes) -- results_changed fired
        # before this window existed, so set the Export enabled state now.
        self._refresh_export_enabled()

    # ------------------------------------------------------------------

    def _refresh_export_enabled(self) -> None:
        """Export is available whenever results exist.

        Mirrors the main window (whose Export enables on any completed run):
        :meth:`_on_export_strain` opens the shared export dialog and only needs
        ``results`` -- displacement is exportable even before strain is
        computed -- so gating on strain specifically would leave the button
        wrongly disabled after a displacement-only run or a session reload.
        """
        self._export_strain_btn.setEnabled(self._state.results is not None)
    # Public accessors (used by tests + future integration)
    # ------------------------------------------------------------------

    def strain_current_frame(self) -> int:
        return self._strain_current_frame

    def set_strain_frame(self, idx: int) -> None:
        n = self._strain_frame_count()
        clamped = max(0, min(idx, max(0, n - 1)))
        if clamped == self._strain_current_frame:
            return
        self._strain_current_frame = clamped
        self._frame_nav.set_state(n, clamped)
        self._render_current()
        if hasattr(self, "_analysis_tab"):
            self._analysis_tab.set_frame(clamped)

    def current_field(self) -> str:
        return self._field_selector.current_field()

    def _strain_parameters(self) -> dict[str, str]:
        """The last Compute Strain's settings, for an export header.

        Empty until strain has been computed in this window: after a session
        is reopened the settings that produced the stored strain are unknown,
        and stating the panel's current values would claim something false.
        """
        o = getattr(self, "_last_strain_override", None)
        if not o:
            return {}
        method = o.get("method_to_compute_strain")
        out = {"strain method": {2: "plane fitting", 3: "FEM nodal"}.get(
            method, f"method {method}")}
        if method == 2:
            rad = float(o.get("strain_plane_fit_rad", 0.0))
            out["strain window"] = (
                f"{2 * rad + 1:g} px (VSG as set in pyALDIC; plane-fit radius "
                f"{rad:g} px)")
        out["strain type"] = {
            0: "infinitesimal", 1: "Eulerian-Almansi", 2: "Green-Lagrangian",
        }.get(o.get("strain_type"), str(o.get("strain_type")))
        if o.get("strain_smoothness") is not None:
            out["strain smoothing"] = f"{float(o['strain_smoothness']):g}"
        if o.get("strain_edge_trim_alpha") is not None:
            out["strain edge trim alpha"] = f"{float(o['strain_edge_trim_alpha']):g}"
        return out

    def set_current_field(self, name: str) -> None:
        self._field_selector.set_current_field(name)

    def is_stale(self) -> bool:
        return self._param_panel.is_dirty()

    def param_panel(self) -> StrainParamPanel:
        return self._param_panel

    def viz_panel(self) -> StrainVizPanel:
        return self._viz_panel

    def trigger_compute(self) -> None:
        """Synchronous compute — used by tests (blocking, no progress bar)."""
        if self._state.results is None:
            return
        override = self._param_panel.get_override()
        try:
            self._strain_ctrl.compute_and_store(override=override)
        except Exception as exc:
            from al_dic.i18n import tr_args
            self._log(
                tr_args(
                    self.tr("Strain compute failed: %1: %2"),
                    type(exc).__name__, str(exc),
                ),
                "error",
            )
            return
        self._last_strain_override = dict(override)
        self._param_panel.mark_clean()
        self._stale_label.setText("")
        self._log(self.tr("Strain computation complete."), "success")
        self._export_strain_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_compute_clicked(self) -> None:
        if self._state.results is None:
            self._log(
                self.tr(
                    "Strain window: no displacement results to post-process."
                ),
                "warn",
            )
            return
        # Guard against double-click while running
        if self._strain_worker is not None and self._strain_worker.isRunning():
            return
        self._compute_btn.setEnabled(False)
        self._strain_progress.setValue(0)
        self._strain_progress.setVisible(True)
        self._strain_progress_label.setText(self.tr("Starting…"))
        self._strain_progress_label.setVisible(True)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setVisible(True)

        self._strain_worker = _StrainWorker(
            self._strain_ctrl,
            self._param_panel.get_override(),
        )
        self._strain_worker.progress.connect(self._on_strain_progress)
        self._strain_worker.finished.connect(self._on_strain_finished)
        self._strain_worker.error.connect(self._on_strain_error)
        self._strain_worker.cancelled.connect(self._on_strain_cancelled)
        self._strain_worker.start()

    def _on_cancel_clicked(self) -> None:
        worker = self._strain_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._strain_progress_label.setText(self.tr("Cancelling…"))

    def _on_strain_progress(self, fraction: float, message: str) -> None:
        self._strain_progress.setValue(int(fraction * 1000))
        self._strain_progress_label.setText(message)

    def _on_strain_finished(self, new_strain: list) -> None:
        worker = self._strain_worker
        if worker is not None:
            self._last_strain_override = dict(worker._override)
        current = self._state.results
        self._state.results = _dc_replace(current, result_strain=new_strain)
        self._state.results_changed.emit()

        self._cancel_btn.setVisible(False)
        self._strain_progress.setValue(1000)
        self._strain_progress_label.setText(self.tr("Complete"))
        self._compute_btn.setEnabled(True)
        self._param_panel.mark_clean()
        self._stale_label.setText("")
        self._log(self.tr("Strain computation complete."), "success")
        self._export_strain_btn.setEnabled(True)
        QTimer.singleShot(
            2000,
            lambda: (
                self._strain_progress.setVisible(False),
                self._strain_progress_label.setVisible(False),
            ),
        )

    def _on_strain_cancelled(self) -> None:
        # User aborted mid-run: discard the partial recompute and keep the
        # previous strain result untouched.
        self._cancel_btn.setVisible(False)
        self._strain_progress.setVisible(False)
        self._strain_progress_label.setVisible(False)
        self._compute_btn.setEnabled(True)
        self._log(self.tr("Strain computation cancelled."), "warn")

    def _on_strain_error(self, message: str) -> None:
        from al_dic.i18n import tr_args
        self._cancel_btn.setVisible(False)
        self._strain_progress.setVisible(False)
        self._strain_progress_label.setVisible(False)
        self._compute_btn.setEnabled(True)
        self._log(
            tr_args(self.tr("Strain compute failed: %1"), message),
            "error",
        )
        # Surface the failure in a modal dialog. The LOG panel is collapsed by
        # default, so a log-only error was effectively silent -- the user could
        # unknowingly keep the previous result and export a stale/blank field.
        QMessageBox.warning(
            self, self.tr("Strain Computation Failed"), message,
        )

    def _on_export_strain(self) -> None:
        """Open the export dialog pre-filled with this window's viz settings."""
        if self._state.results is None:
            return

        from al_dic.gui.dialogs.export_dialog import ExportDialog, VizExportHint

        viz = self._viz_panel.get_state()
        hint = VizExportHint(
            colormap=str(viz["colormap"]),
            auto_range=bool(viz["use_percentile"]),
            vmin=float(viz["vmin"]),
            vmax=float(viz["vmax"]),
            show_deformed=bool(viz.get("show_deformed", False)),
            show_background=bool(viz.get("show_background", True)),
            hidden_bg_color=str(viz.get("hidden_bg_color", "white")),
            fill_trimmed_edges=bool(viz.get("fill_trimmed_edges", False)),
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

    def _on_params_dirty(self) -> None:
        self._stale_label.setText(self.tr("⚠ Params changed -- click Compute Strain"))

    def _on_field_changed(self, name: str) -> None:
        """Switch the active display field: restore its remembered color state first."""
        if hasattr(self, "_analysis_tab"):
            self._analysis_tab.set_default_field(name)
        self._load_field_state_into_panel()
        self._viz_ctrl.clear_pixmap_cache()
        self._render_current()

    def _on_viz_panel_changed(self) -> None:
        """User changed a setting in StrainVizPanel — save to AppState, then re-render.

        Saves the current StrainVizPanel state to AppState's per-field store so
        that switching away and back restores the user's settings. The guard flag
        prevents the resulting display_changed signal from causing a second render.
        """
        self._loading_field_state = True
        try:
            self._save_panel_state_to_app()
        finally:
            self._loading_field_state = False
        self._viz_ctrl.clear_pixmap_cache()
        self._render_current()

    def _on_state_display_changed(self) -> None:
        """AppState display settings changed (possibly from the main window).

        If we triggered this ourselves (via _save_panel_state_to_app), skip the
        reload to avoid a redundant render. Otherwise, reload the current field's
        color state so that changes made in the main window are reflected here too
        (e.g. both windows show disp_u → user changes colormap in main window →
        strain window should update to match).
        """
        if self._loading_field_state:
            return
        self._load_field_state_into_panel()
        self._viz_ctrl.clear_pixmap_cache()
        self._render_current()

    def _load_field_state_into_panel(self) -> None:
        """Read AppState's stored color state for the current field and push it
        into StrainVizPanel controls (no signals fired, so no save-back loop)."""
        field = self._field_selector.current_field()
        fs = self._state.get_field_state(field)
        self._viz_panel.load_field_state(
            auto=fs.auto,
            vmin=fs.vmin,
            vmax=fs.vmax,
            colormap=fs.colormap,
        )

    def _save_panel_state_to_app(self) -> None:
        """Persist the current StrainVizPanel state into AppState's per-field store.

        Also emits display_changed so the main window reacts if it is currently
        showing the same field (e.g. disp_u / disp_v shared state).
        """
        field = self._field_selector.current_field()
        viz = self._viz_panel.get_state()
        fs = self._state.get_field_state(field)
        fs.colormap = str(viz["colormap"])
        fs.auto = bool(viz["use_percentile"])
        if not fs.auto:
            fs.vmin = float(viz["vmin"])
            fs.vmax = float(viz["vmax"])
        self._state.display_changed.emit()

    def _on_auto_range_disabled(self) -> None:
        """User switched to manual range: populate spinboxes with the last rendered range.

        Uses the cached vmin/vmax from the most recent _render_current call so
        that the manual spinboxes start at exactly the visible range the user
        sees — including deformed-mask trimming and per-frame ROI clipping.
        """
        self._viz_panel.set_range(self._last_rendered_vmin, self._last_rendered_vmax)

    def _on_state_results_changed(self) -> None:
        self._viz_ctrl.clear_all()
        self._frame_nav.stop_playback()
        self._sync_slider_range()
        self._render_current()
        # Export follows the presence of results (displacement and/or strain),
        # matching the main window and _on_export_strain's requirement.
        self._refresh_export_enabled()

    def _on_frame_nav_changed(self, value: int) -> None:
        self._strain_current_frame = int(value)
        self._render_current()
        self._analysis_tab.set_frame(self._strain_current_frame)

    # ------------------------------------------------------------------
    # Field extraction
    # ------------------------------------------------------------------

    def _get_field_values(
        self,
        field_name: str,
        frame: int,
        result: PipelineResult,
        show_deformed: bool = False,
    ) -> NDArray[np.float64] | None:
        """Extract displayable values for the given field and frame.

        Displacement-family fields (disp_u / disp_v / disp_magnitude /
        velocity) are served from result_disp so they work before Compute
        Strain. All strain fields require result_strain.

        *show_deformed* selects the edge-trim frame: the deformed view uses the
        stored per-frame trim (current-frame crack), the reference view uses
        frame-0 geometry so it matches the main window's displacement overlay.
        """
        if field_name in DISP_FIELD_NAMES:
            return self._get_disp_field(field_name, frame, result)

        # Strain fields — reference frame (0): return zeros by convention
        if frame == 0:
            n = result.dic_mesh.coordinates_fem.shape[0]
            return np.zeros(n, dtype=np.float64)
        strain_idx = frame - 1   # 0-based index into result_strain
        if not result.result_strain or strain_idx >= len(result.result_strain):
            return None
        sr: StrainResult = result.result_strain[strain_idx]

        if field_name == "strain_rotation":
            vals = sr.strain_rotation  # pre-computed from raw F before strain-type conversion
        else:
            vals = getattr(sr, field_name, None)

        # Edge-trim sink: hide low-confidence ROI/hole-edge nodes from the
        # displayed field. ``strain_valid`` is set for plane-fit strain only
        # (None otherwise); the values stored in StrainResult are untouched —
        # we NaN a display copy so the interpolator drops those nodes,
        # producing a transparent boundary band.  The trim frame follows the
        # display frame (see _display_strain_valid).
        valid = self._display_strain_valid(sr, result, show_deformed)
        if vals is not None and valid is not None and len(valid) == len(vals):
            vals = np.asarray(vals, dtype=np.float64).copy()
            vals[~valid] = np.nan
        return vals

    def _display_strain_valid(
        self,
        sr: StrainResult,
        result: PipelineResult,
        show_deformed: bool,
    ) -> NDArray[np.bool_] | None:
        """Edge-trim validity mask to *display*, chosen by the view frame.

        * **Deformed view** — the stored ``sr.strain_valid`` (edge-trim of the
          current frame's crack, warped to the reference position).
        * **Reference view** — recompute the trim from the frame-0 ROI mask so
          the reference strain uses the same frame-0 geometry the main window's
          displacement overlay does; the current frame's (possibly grown) crack
          is not carved into the reference view.  Cached per result.

        Returns ``None`` for FEM nodal strain (no edge-trim) so the caller
        leaves the field dense.
        """
        if sr.strain_valid is None:
            return None
        if show_deformed:
            return sr.strain_valid
        mask0 = self._state.per_frame_rois.get(0)
        if mask0 is None:
            return sr.strain_valid
        rad = float(result.dic_para.strain_plane_fit_rad)
        alpha = float(getattr(result.dic_para, "strain_edge_trim_alpha", 0.0))
        key = (id(result), rad, alpha)
        cached = getattr(self, "_ref_trim_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            from al_dic.strain.comp_def_grad import edge_valid_mask
            valid0 = edge_valid_mask(
                result.dic_mesh.coordinates_fem,
                np.asarray(mask0, dtype=np.float64),
                rad, alpha,
            )
        except Exception:
            return sr.strain_valid
        self._ref_trim_cache = (key, valid0)
        return valid0

    def _get_disp_field(
        self,
        field_name: str,
        frame: int,
        result: PipelineResult,
    ) -> NDArray[np.float64] | None:
        """Serve displacement-family fields from result_disp.

        *frame* is the image-file index (0 = reference frame, 1..N = deformed).
        Returns zeros for the reference frame (index 0) — displacement relative
        to itself is zero by definition.
        """
        n = result.dic_mesh.coordinates_fem.shape[0]
        if frame == 0:
            return np.zeros(n, dtype=np.float64)
        disp_idx = frame - 1   # 0-based index into result_disp
        if disp_idx >= len(result.result_disp):
            return None
        fr = result.result_disp[disp_idx]
        U = fr.U_accum if fr.U_accum is not None else fr.U
        u, v = U[0::2], U[1::2]

        state = self._state
        px = state.pixel_size if (state.use_physical_units and state.pixel_size > 0) else 1.0

        if field_name == "disp_u":
            return u.copy() * px
        if field_name == "disp_v":
            return v.copy() * px
        if field_name == "disp_magnitude":
            return np.sqrt(u ** 2 + v ** 2) * px
        if field_name == "velocity":
            if disp_idx > 0:
                fr_prev = result.result_disp[disp_idx - 1]
                U_prev = fr_prev.U_accum if fr_prev.U_accum is not None else fr_prev.U
                du = u - U_prev[0::2]
                dv = v - U_prev[1::2]
            else:
                du, dv = u, v   # velocity from rest (first deformed frame)
            vel_mag = np.sqrt(du ** 2 + dv ** 2)   # px/frame
            if state.use_physical_units and state.pixel_size > 0 and state.frame_rate > 0:
                return vel_mag * state.pixel_size * state.frame_rate
            return vel_mag
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _strain_frame_count(self) -> int:
        """Total frames including reference (= len(image_files) when loaded)."""
        n_images = len(self._state.image_files)
        if n_images > 0:
            return n_images
        # Fallback when images are not loaded (should not happen in normal use).
        result = self._state.results
        if result is None:
            return 0
        return len(result.result_disp) + 1  # +1 for reference

    def _sync_slider_range(self) -> None:
        n = self._strain_frame_count()
        max_idx = max(0, n - 1)
        if self._strain_current_frame > max_idx:
            self._strain_current_frame = max_idx
        self._frame_nav.set_state(n, self._strain_current_frame)
        if hasattr(self, "_analysis_tab"):
            self._analysis_tab.set_frame(self._strain_current_frame)

    def _try_load_background(self, img_idx: int = 0) -> None:
        """Best-effort background image fetch -- silent on failure."""
        if not self._state.image_files:
            return
        try:
            rgb = self._image_ctrl.read_image_rgb(img_idx)
            self._canvas.set_image(rgb)
        except (IndexError, FileNotFoundError, ValueError):
            pass

    def _blank_background(self, result: PipelineResult, color: str) -> None:
        """Paint the hidden-background fill at the result's image size."""
        shape = tuple(result.dic_para.img_size)
        if shape == (0, 0):
            # Nothing to size the scene with; leave what is showing rather
            # than collapsing it.
            return
        self._canvas.set_blank(shape[0], shape[1], color)

    def _update_trim_readout(
        self, field_name: str, frame: int, result: PipelineResult,
        show_deformed: bool = False,
    ) -> None:
        """Refresh the panel's 'Trimmed: N nodes (M%)' readout for this frame.

        Uses the *displayed* trim mask (frame-0 geometry in the reference view,
        current-frame in the deformed view -- see _display_strain_valid), so
        the readout matches what is actually drawn. Clears for displacement
        fields, the reference frame, or when no validity mask is available.
        """
        sv = None
        if field_name not in DISP_FIELD_NAMES and frame >= 1 and result.result_strain:
            idx = frame - 1
            if idx < len(result.result_strain):
                sv = self._display_strain_valid(
                    result.result_strain[idx], result, show_deformed,
                )
        if sv is None:
            self._param_panel.set_trim_readout(0, 0)
        else:
            self._param_panel.set_trim_readout(
                int(np.count_nonzero(~sv)), int(sv.size),
            )

    def _on_tab_changed(self, _index: int) -> None:
        if self._render_pending and self._tabs.currentWidget() is not self._analysis_tab:
            self._render_pending = False
            self._render_current()

    def _render_current(self) -> None:
        tabs = getattr(self, "_tabs", None)
        if tabs is not None and tabs.currentWidget() is self._analysis_tab:
            self._render_pending = True
            return
        try:
            result = self._state.results
            if result is None:
                self._try_load_background(0)
                self._canvas.clear_overlay()
                self._colorbar.setVisible(False)
                return

            frame = self._strain_current_frame
            field_name = self._field_selector.current_field()

            viz = self._viz_panel.get_state()
            show_deformed = bool(viz.get("show_deformed", False))
            show_background = bool(viz.get("show_background", True))
            hidden_bg = str(viz.get("hidden_bg_color", "white"))

            self._update_trim_readout(field_name, frame, result, show_deformed)

            # Background image: frame is now the image-file index (0=ref, 1..N=deformed).
            # show_deformed → load the current image; otherwise always show
            # reference.  Which frame only matters once one is shown at all.
            if not show_background:
                self._blank_background(result, hidden_bg)
            elif show_deformed and frame >= 1:
                self._try_load_background(frame)
            else:
                self._try_load_background(0)

            image = self._field_image(
                field_name, frame, result,
                deformed=show_deformed,
                cmap=str(viz["colormap"]),
                alpha=float(viz["alpha"]),
                fill_trimmed=bool(viz.get("fill_trimmed_edges", False)),
                range_of=self._resolve_range,
            )
            if image is None:
                self._canvas.clear_overlay()
                self._colorbar.setVisible(False)
                return

            # Cache so that Auto→Manual switch can populate spinboxes with the
            # exact same range that was just rendered (matches visible nodes only).
            self._last_rendered_vmin = image.vmin
            self._last_rendered_vmax = image.vmax

            self._canvas.show_field(image)
            vp = self._canvas.viewport()
            self._colorbar.setGeometry(0, 0, vp.width(), vp.height())
            self._colorbar.update_params(
                image.cmap, image.vmin, image.vmax, image.label)
            self._colorbar.setVisible(True)

        except Exception as exc:  # pragma: no cover
            tb_str = _tb.format_exc()
            print(f"[strain_window._render_current] {exc}\n{tb_str}", flush=True)
            self._log(f"Render error: {type(exc).__name__}: {exc}", "error")
            self._canvas.clear_overlay()
            self._colorbar.setVisible(False)

    def reference_field_image(
        self, field_name: str, frame: int,
    ) -> FieldImage | None:
        """*field_name* at *frame* in the reference configuration.

        What the Analysis tab lays under its probes: probe coordinates are
        frame-0 pixels, so the field is drawn where they are, whatever this
        tab's own view shows. Colormap and range are the field's own (the
        state the Strain Field tab and the main window share); opacity and the
        trimmed-edge fill are this window's.
        """
        result = self._state.results
        if result is None:
            return None
        fs = self._state.get_field_state(field_name)
        viz = self._viz_panel.get_state()

        def range_of(values: NDArray[np.float64]) -> tuple[float, float]:
            if not fs.auto:
                return float(fs.vmin), float(fs.vmax)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return 0.0, 1.0
            return float(finite.min()), float(finite.max())

        return self._field_image(
            field_name, frame, result,
            deformed=False,
            cmap=str(fs.colormap),
            alpha=float(viz["alpha"]),
            fill_trimmed=bool(viz.get("fill_trimmed_edges", False)),
            range_of=range_of,
        )

    def _field_image(
        self,
        field_name: str,
        frame: int,
        result: PipelineResult,
        *,
        deformed: bool,
        cmap: str,
        alpha: float,
        fill_trimmed: bool,
        range_of: Callable[[NDArray[np.float64]], tuple[float, float]],
    ) -> FieldImage | None:
        """Render *field_name* at *frame*: the one path both tabs draw through.

        *deformed* asks for the deformed configuration; the reference is used
        where there is none (frame 0, or no displacement for the frame).
        *range_of* maps the visible values to (vmin, vmax).
        """
        # Trim frame follows the display frame: reference view -> frame-0
        # geometry (matches the main window's displacement), deformed view
        # -> current-frame crack.
        values = self._get_field_values(field_name, frame, result, deformed)
        if values is None:
            return None

        # Use dic_mesh (canonical mesh = result_fe_mesh_each_frame[0]) as
        # reference node positions — mirrors main window's approach.
        ref_nodes = result.dic_mesh.coordinates_fem

        # Deformed rendering: shift node positions by accumulated
        # displacement, then let VizController warp the ROI mask.
        # Matches main window _refresh_overlay exactly:
        #   nodes = ref_nodes + column_stack([u, v])
        #   ref_uv = (u, v)   -- raw pixel displacements for inverse warp
        #   deformed_mask = per_frame_rois.get(frame + 1)
        is_deformed = False
        ref_uv = None
        deformed_mask = None
        nodes = ref_nodes

        # frame is the image-file index (0=reference, 1..N=deformed).
        # result_disp index = frame - 1.
        disp_idx = frame - 1
        if deformed and frame >= 1 and disp_idx < len(result.result_disp):
            fr_d = result.result_disp[disp_idx]
            U = fr_d.U_accum if fr_d.U_accum is not None else fr_d.U
            if U is not None:
                u, v = U[0::2], U[1::2]
                nodes = ref_nodes + np.column_stack([u, v])
                is_deformed = True
                ref_uv = (u, v)
                # Per-frame deformed ROI: use per_frame_rois[frame] (image index)
                deformed_mask = self._state.per_frame_rois.get(frame)

        # Auto-range uses only the nodes visible within the (possibly
        # trimmed) deformed mask -- prevents out-of-view nodes from
        # pulling the colorbar range out of sync with what the user sees.
        vmin, vmax = range_of(visible_values(
            values, nodes, deformed_mask if is_deformed else None,
        ))

        pixmap, xg, yg, out_step = self._viz_ctrl.render_field(
            frame_idx=frame,
            field_name=f"{_FIELD_NS}:{field_name}",
            nodes=nodes,
            values=values,
            img_shape=result.dic_para.img_size,
            mesh_step=result.dic_para.winstepsize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            roi_mask=self._state.per_frame_rois.get(0),
            deformed=is_deformed,
            ref_uv=ref_uv,
            deformed_mask=deformed_mask,
            # Strain fields carry NaN at edge-trimmed / plane-fit-failed
            # nodes; blank those cells so the trim is visible instead of
            # interpolator-backfilled.  Displacement fields are not trimmed.
            # "Fill trimmed edges" (viz panel, off by default) skips the
            # blanking so the interpolator re-fills the band from reliable
            # interior nodes -- display only; data export stays NaN.
            blank_invalid_nodes=(
                field_name not in DISP_FIELD_NAMES and not fill_trimmed
            ),
        )
        has_grid = xg is not None and yg is not None
        return FieldImage(
            pixmap=pixmap,
            x=float(xg.min()) if has_grid else None,
            y=float(yg.min()) if has_grid else None,
            scale=float(out_step),
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label=field_colorbar_label(
                field_name,
                self._state.use_physical_units,
                self._state.pixel_unit,
                self._state.frame_rate,
            ),
        )

    def _resolve_range(
        self, values: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Compute (vmin, vmax) from the local viz panel state."""
        viz = self._viz_panel.get_state()
        if not viz["use_percentile"]:
            return float(viz["vmin"]), float(viz["vmax"])
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return 0.0, 1.0
        return float(valid.min()), float(valid.max())

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str, level: str = "info") -> None:
        """Append a message to the local console."""
        self._console.append_log(message, level)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Reconnect signals: they are disconnected in closeEvent so re-opening
        # after a DIC re-run does not receive stale connections (Bug B fix).
        for sig, slot in [
            (self._state.results_changed, self._on_state_results_changed),
            (self._state.display_changed, self._on_state_display_changed),
            (self._state.physical_units_changed, self._viz_ctrl.clear_all),
        ]:
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
            sig.connect(slot)
        # Clear viz cache so a previous session's overlay never bleeds through
        # when the user re-opens after running new DIC data (Bug B fix).
        self._viz_ctrl.clear_all()
        self._sync_slider_range()
        # Re-render now that the viewport has its real size (Bug A fix).
        self._render_current()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        """Update colorbar geometry when the canvas viewport is resized."""
        if obj is self._canvas.viewport() and event.type() == QEvent.Type.Resize:
            if self._colorbar.isVisible():
                vp = self._canvas.viewport()
                self._colorbar.setGeometry(0, 0, vp.width(), vp.height())
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:  # noqa: N802
        for sig, slot in [
            (self._state.results_changed, self._on_state_results_changed),
            (self._state.display_changed, self._on_state_display_changed),
            (self._state.physical_units_changed, self._viz_ctrl.clear_all),
        ]:
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        super().closeEvent(event)


