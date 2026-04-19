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

import numpy as np
from numpy.typing import NDArray
from dataclasses import replace as _dc_replace

from PySide6.QtCore import QEvent, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from al_dic.core.data_structures import PipelineResult, StrainResult
from al_dic.gui.app_state import AppState
from al_dic.gui.controllers.image_controller import ImageController
from al_dic.gui.controllers.strain_controller import StrainController
from al_dic.gui.controllers.viz_controller import VizController
from al_dic.gui.panels.canvas_area import visible_values
from al_dic.gui.panels.strain_canvas import StrainCanvas
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


class _StrainWorker(QThread):
    """Background thread for strain computation.

    Emits ``progress(fraction, message)`` once per frame and
    ``finished(result_list)`` on success, ``error(message)`` on failure.
    Computation happens in the thread; state updates happen in the caller's
    slot (main thread) to avoid cross-thread Qt issues.
    """

    progress: Signal = Signal(float, str)
    finished: Signal = Signal(list)
    error: Signal = Signal(str)

    def __init__(
        self,
        strain_ctrl: "StrainController",
        override: dict,
    ) -> None:
        super().__init__()
        self._ctrl = strain_ctrl
        self._override = override

    def run(self) -> None:
        try:
            results = self._ctrl.compute_all_frames(
                self._override,
                progress_cb=lambda f, m: self.progress.emit(f, m),
            )
            self.finished.emit(results)
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
        self.setWindowTitle("Strain Post-Processing")
        # Use the shared app icon so the Strain window is recognisable
        # in the taskbar / Alt-Tab alongside the main window.
        from al_dic.gui.icons import icon_app
        self.setWindowIcon(icon_app())
        # Match the main window's dark OS title bar so the two windows
        # share one visual frame.
        from al_dic.gui.window_chrome import enable_dark_title_bar
        enable_dark_title_bar(self)
        # Default size chosen for a roughly square canvas area: 320 px
        # right panel + ~1000 px canvas width, with ~880 px canvas
        # height. Matches the common square-ish DIC image aspect
        # without letterboxing.
        self.resize(1340, 960)

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
        _btn_fit = QPushButton("Fit")
        _btn_fit.setToolTip("Fit image to viewport")
        _btn_fit.setFixedWidth(60)
        if _HAS_ICONS:
            _btn_fit.setIcon(icon_maximize())
            _btn_fit.setText("")
            _btn_fit.setFixedWidth(28)
        _btn_100 = QPushButton("100%")
        _btn_100.setToolTip("Zoom to 100% (1:1)")
        _btn_100.setFixedWidth(60)
        _btn_zin = QPushButton("+")
        _btn_zin.setToolTip("Zoom in")
        _btn_zin.setFixedWidth(28)
        if _HAS_ICONS:
            _btn_zin.setIcon(icon_zoom_in())
            _btn_zin.setText("")
        _btn_zout = QPushButton("\u2013")
        _btn_zout.setToolTip("Zoom out")
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

        # Right pane
        right = QVBoxLayout()
        right.setSpacing(6)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(320)

        # Strain parameters (first: user computes before viewing)
        right.addWidget(_sec_label("STRAIN PARAMETERS"))
        self._param_panel = StrainParamPanel()
        self._param_panel.params_dirty.connect(self._on_params_dirty)
        right.addWidget(self._param_panel)

        # Prominent compute button (matches main window "Run DIC Analysis" style)
        self._compute_btn = QPushButton("Compute Strain")
        self._compute_btn.setProperty("class", "btn-primary")
        self._compute_btn.setFixedHeight(40)
        self._compute_btn.clicked.connect(self._on_compute_clicked)
        right.addWidget(self._compute_btn)

        self._export_strain_btn = QPushButton("Export Results")
        self._export_strain_btn.setFixedHeight(30)
        self._export_strain_btn.setToolTip(
            "Export displacement and strain results to NPZ / MAT / CSV / PNG"
        )
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

        # Field selector (second: choose what to display after compute)
        right.addWidget(_sec_label("FIELD"))
        self._field_selector = StrainFieldSelector()
        self._field_selector.field_changed.connect(self._on_field_changed)
        right.addWidget(self._field_selector)

        # Visualization controls
        right.addWidget(_sec_label("VISUALIZATION"))
        self._viz_panel = StrainVizPanel()
        self._viz_panel.viz_changed.connect(self._on_viz_panel_changed)
        self._viz_panel.auto_disabled.connect(self._on_auto_range_disabled)
        right.addWidget(self._viz_panel)

        # Physical units
        right.addWidget(_sec_label("PHYSICAL UNITS"))
        self._physical_units = PhysicalUnitsWidget()
        right.addWidget(self._physical_units)

        # Console
        right.addWidget(_sec_label("LOG"))
        self._console = ConsoleLog()
        right.addWidget(self._console)

        right.addStretch(1)
        root.addWidget(right_widget, 0)

        self.setCentralWidget(central)

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

    # ------------------------------------------------------------------
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

    def current_field(self) -> str:
        return self._field_selector.current_field()

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
        try:
            self._strain_ctrl.compute_and_store(
                override=self._param_panel.get_override(),
            )
        except Exception as exc:
            self._log(
                f"Strain compute failed: {type(exc).__name__}: {exc}",
                "error",
            )
            return
        self._param_panel.mark_clean()
        self._stale_label.setText("")
        self._log("Strain computation complete.", "success")
        self._export_strain_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_compute_clicked(self) -> None:
        if self._state.results is None:
            self._log(
                "Strain window: no displacement results to post-process.",
                "warn",
            )
            return
        # Guard against double-click while running
        if self._strain_worker is not None and self._strain_worker.isRunning():
            return
        self._compute_btn.setEnabled(False)
        self._strain_progress.setValue(0)
        self._strain_progress.setVisible(True)
        self._strain_progress_label.setText("Starting\u2026")
        self._strain_progress_label.setVisible(True)

        self._strain_worker = _StrainWorker(
            self._strain_ctrl,
            self._param_panel.get_override(),
        )
        self._strain_worker.progress.connect(self._on_strain_progress)
        self._strain_worker.finished.connect(self._on_strain_finished)
        self._strain_worker.error.connect(self._on_strain_error)
        self._strain_worker.start()

    def _on_strain_progress(self, fraction: float, message: str) -> None:
        self._strain_progress.setValue(int(fraction * 1000))
        self._strain_progress_label.setText(message)

    def _on_strain_finished(self, new_strain: list) -> None:
        current = self._state.results
        self._state.results = _dc_replace(current, result_strain=new_strain)
        self._state.results_changed.emit()

        self._strain_progress.setValue(1000)
        self._strain_progress_label.setText("Complete")
        self._compute_btn.setEnabled(True)
        self._param_panel.mark_clean()
        self._stale_label.setText("")
        self._log("Strain computation complete.", "success")
        self._export_strain_btn.setEnabled(True)
        QTimer.singleShot(
            2000,
            lambda: (
                self._strain_progress.setVisible(False),
                self._strain_progress_label.setVisible(False),
            ),
        )

    def _on_strain_error(self, message: str) -> None:
        self._strain_progress.setVisible(False)
        self._strain_progress_label.setVisible(False)
        self._compute_btn.setEnabled(True)
        self._log(f"Strain compute failed: {message}", "error")

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
        self._stale_label.setText("\u26a0 Params changed -- click Compute Strain")

    def _on_field_changed(self, name: str) -> None:
        """Switch the active display field: restore its remembered color state first."""
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
        # Disable export if strain was cleared (e.g. user re-ran DIC)
        has_strain = (
            self._state.results is not None
            and bool(self._state.results.result_strain)
        )
        self._export_strain_btn.setEnabled(has_strain)

    def _on_frame_nav_changed(self, value: int) -> None:
        self._strain_current_frame = int(value)
        self._render_current()

    # ------------------------------------------------------------------
    # Field extraction
    # ------------------------------------------------------------------

    def _get_field_values(
        self,
        field_name: str,
        frame: int,
        result: PipelineResult,
    ) -> NDArray[np.float64] | None:
        """Extract displayable values for the given field and frame.

        Displacement-family fields (disp_u / disp_v / disp_magnitude /
        velocity) are served from result_disp so they work before Compute
        Strain. All strain fields require result_strain.
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
            return sr.strain_rotation  # pre-computed from raw F before strain-type conversion

        return getattr(sr, field_name, None)

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

    def _try_load_background(self, img_idx: int = 0) -> None:
        """Best-effort background image fetch -- silent on failure."""
        if not self._state.image_files:
            return
        try:
            rgb = self._image_ctrl.read_image_rgb(img_idx)
            self._canvas.set_image(rgb)
        except (IndexError, FileNotFoundError, ValueError):
            pass

    def _render_current(self) -> None:
        try:
            result = self._state.results
            if result is None:
                self._try_load_background(0)
                self._canvas.clear_overlay()
                self._colorbar.setVisible(False)
                return

            frame = self._strain_current_frame
            field_name = self._field_selector.current_field()
            values = self._get_field_values(field_name, frame, result)

            viz = self._viz_panel.get_state()
            show_deformed = bool(viz.get("show_deformed", False))

            # Background image: frame is now the image-file index (0=ref, 1..N=deformed).
            # show_deformed → load the current image; otherwise always show reference.
            if show_deformed and frame >= 1:
                self._try_load_background(frame)
            else:
                self._try_load_background(0)

            if values is None:
                self._canvas.clear_overlay()
                self._colorbar.setVisible(False)
                return

            # Use dic_mesh (canonical mesh = result_fe_mesh_each_frame[0]) as
            # reference node positions — mirrors main window's approach.
            ref_nodes = result.dic_mesh.coordinates_fem
            mesh_step = result.dic_para.winstepsize
            img_shape = result.dic_para.img_size
            roi_mask = self._state.per_frame_rois.get(0)

            # Deformed rendering: shift node positions by accumulated
            # displacement, then let VizController warp the ROI mask.
            # Matches main window _refresh_overlay exactly:
            #   nodes = ref_nodes + column_stack([u, v])
            #   ref_uv = (u, v)   -- raw pixel displacements for inverse warp
            #   deformed_mask = per_frame_rois.get(frame + 1)
            deformed = False
            ref_uv = None
            deformed_mask = None
            nodes = ref_nodes

            # frame is the image-file index (0=reference, 1..N=deformed).
            # result_disp index = frame - 1.
            disp_idx = frame - 1
            if show_deformed and frame >= 1 and disp_idx < len(result.result_disp):
                fr_d = result.result_disp[disp_idx]
                U = fr_d.U_accum if fr_d.U_accum is not None else fr_d.U
                if U is not None:
                    u, v = U[0::2], U[1::2]
                    nodes = ref_nodes + np.column_stack([u, v])
                    deformed = True
                    ref_uv = (u, v)
                    # Per-frame deformed ROI: use per_frame_rois[frame] (image index)
                    deformed_mask = self._state.per_frame_rois.get(frame)

            # Auto-range uses only the nodes visible within the (possibly
            # trimmed) deformed mask -- prevents out-of-view nodes from
            # pulling the colorbar range out of sync with what the user sees.
            range_values = visible_values(
                values, nodes, deformed_mask if deformed else None,
            )
            vmin, vmax = self._resolve_range(range_values)
            # Cache so that Auto→Manual switch can populate spinboxes with the
            # exact same range that was just rendered (matches visible nodes only).
            self._last_rendered_vmin = vmin
            self._last_rendered_vmax = vmax

            pixmap, xg, yg, out_step = self._viz_ctrl.render_field(
                frame_idx=frame,
                field_name=f"{_FIELD_NS}:{field_name}",
                nodes=nodes,
                values=values,
                img_shape=img_shape,
                mesh_step=mesh_step,
                cmap=str(viz["colormap"]),
                vmin=vmin,
                vmax=vmax,
                roi_mask=roi_mask,
                deformed=deformed,
                ref_uv=ref_uv,
                deformed_mask=deformed_mask,
            )
            self._canvas.set_overlay_pixmap(pixmap)
            self._canvas.set_overlay_alpha(float(viz["alpha"]))
            self._canvas._overlay_item.setScale(float(out_step))
            if xg is not None and yg is not None:
                self._canvas.set_overlay_pos(
                    float(xg.min()), float(yg.min()),
                )

            cmap_name = str(viz["colormap"])
            colorbar_label = field_colorbar_label(
                field_name,
                self._state.use_physical_units,
                self._state.pixel_unit,
                self._state.frame_rate,
            )
            vp = self._canvas.viewport()
            self._colorbar.setGeometry(0, 0, vp.width(), vp.height())
            self._colorbar.update_params(cmap_name, vmin, vmax, colorbar_label)
            self._colorbar.setVisible(True)

        except Exception as exc:  # pragma: no cover
            tb_str = _tb.format_exc()
            print(f"[strain_window._render_current] {exc}\n{tb_str}", flush=True)
            self._log(f"Render error: {type(exc).__name__}: {exc}", "error")
            self._canvas.clear_overlay()
            self._colorbar.setVisible(False)

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sec_label(text: str) -> QLabel:
    """Small all-caps section header."""
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px; "
        f"font-weight: bold; letter-spacing: 1px; margin-top: 6px;"
    )
    return lbl
