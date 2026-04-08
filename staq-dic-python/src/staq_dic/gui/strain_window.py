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
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from staq_dic.core.data_structures import PipelineResult, StrainResult
from staq_dic.gui.app_state import AppState
from staq_dic.gui.controllers.image_controller import ImageController
from staq_dic.gui.controllers.strain_controller import StrainController
from staq_dic.gui.controllers.viz_controller import VizController
from staq_dic.gui.panels.canvas_area import visible_values
from staq_dic.gui.panels.strain_canvas import StrainCanvas
from staq_dic.gui.widgets.colorbar_overlay import ColorbarOverlay
from staq_dic.gui.widgets.console_log import ConsoleLog
from staq_dic.gui.widgets.strain_field_selector import (
    DISP_FIELD_NAMES,
    StrainFieldSelector,
)
from staq_dic.gui.widgets.strain_param_panel import StrainParamPanel
from staq_dic.gui.widgets.strain_viz_panel import StrainVizPanel
from staq_dic.gui.widgets.velocity_settings import VelocitySettingsWidget
from staq_dic.gui.theme import COLORS


# Namespace prefix for the private VizController cache.
_FIELD_NS = "strain_window"


class StrainWindow(QMainWindow):
    """Independent strain post-processing window."""

    def __init__(
        self,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Strain Post-Processing")
        self.resize(1200, 760)

        self._state = state
        self._strain_ctrl = StrainController(state)
        self._viz_ctrl = VizController()   # PRIVATE -- isolated from MainWindow
        self._image_ctrl = ImageController(state)
        self._strain_current_frame: int = 0

        # --- Build layout ---
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left pane: canvas + colorbar + frame slider
        left = QVBoxLayout()
        left.setSpacing(4)

        canvas_row = QHBoxLayout()
        canvas_row.setSpacing(4)
        self._canvas = StrainCanvas()
        canvas_row.addWidget(self._canvas, 1)
        left.addLayout(canvas_row, 1)
        # Colorbar overlaid on the canvas viewport (same pattern as main window)
        self._colorbar = ColorbarOverlay(self._canvas.viewport())
        self._canvas.viewport().installEventFilter(self)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider)
        left.addWidget(self._frame_slider)

        self._frame_label = QLabel("Frame: 0")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self._frame_label)

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

        # Velocity settings (visible only for velocity field)
        self._velocity_settings = VelocitySettingsWidget()
        self._velocity_settings.setVisible(False)
        self._velocity_settings.settings_changed.connect(self._on_viz_changed)
        right.addWidget(self._velocity_settings)

        # Visualization controls
        right.addWidget(_sec_label("VISUALIZATION"))
        self._viz_panel = StrainVizPanel()
        self._viz_panel.viz_changed.connect(self._on_viz_changed)
        self._viz_panel.auto_disabled.connect(self._on_auto_range_disabled)
        right.addWidget(self._viz_panel)

        # Console
        right.addWidget(_sec_label("LOG"))
        self._console = ConsoleLog()
        right.addWidget(self._console)

        right.addStretch(1)
        root.addWidget(right_widget, 0)

        self.setCentralWidget(central)

        # Track external pipeline runs
        self._state.results_changed.connect(self._on_state_results_changed)

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
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(clamped)
        self._frame_slider.blockSignals(False)
        self._frame_label.setText(f"Frame: {clamped}")
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
        """Programmatic equivalent of clicking 'Compute Strain'."""
        self._on_compute_clicked()

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
        # _on_state_results_changed fires from the controller's emit,
        # clears caches, syncs slider, and re-renders.
        self._param_panel.mark_clean()
        self._stale_label.setText("")
        self._log("Strain computation complete.", "success")

    def _on_params_dirty(self) -> None:
        self._stale_label.setText("\u26a0 Params changed -- click Compute Strain")

    def _on_field_changed(self, name: str) -> None:
        self._velocity_settings.set_visible_for_field(name)
        self._render_current()

    def _on_viz_changed(self) -> None:
        # Colormap / range / opacity / deformed changed -> tier-2 cache dies.
        self._viz_ctrl.clear_pixmap_cache()
        self._render_current()

    def _on_auto_range_disabled(self) -> None:
        """User switched to manual range: populate spinboxes with current field."""
        result = self._state.results
        if result is None:
            return
        values = self._get_field_values(
            self._field_selector.current_field(),
            self._strain_current_frame,
            result,
        )
        if values is None:
            return
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return
        self._viz_panel.set_range(float(valid.min()), float(valid.max()))

    def _on_state_results_changed(self) -> None:
        self._viz_ctrl.clear_all()
        self._sync_slider_range()
        self._render_current()

    def _on_frame_slider(self, value: int) -> None:
        self._strain_current_frame = int(value)
        self._frame_label.setText(f"Frame: {value}")
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

        # Strain fields need a completed computation
        if not result.result_strain or frame >= len(result.result_strain):
            return None
        sr: StrainResult = result.result_strain[frame]

        if field_name == "strain_rotation":
            return sr.strain_rotation  # pre-computed from raw F before strain-type conversion

        return getattr(sr, field_name, None)

    def _get_disp_field(
        self,
        field_name: str,
        frame: int,
        result: PipelineResult,
    ) -> NDArray[np.float64] | None:
        """Serve displacement-family fields from result_disp."""
        if frame >= len(result.result_disp):
            return None
        fr = result.result_disp[frame]
        U = fr.U_accum if fr.U_accum is not None else fr.U
        u, v = U[0::2], U[1::2]

        if field_name == "disp_u":
            return u.copy()
        if field_name == "disp_v":
            return v.copy()
        if field_name == "disp_magnitude":
            return np.sqrt(u ** 2 + v ** 2)
        if field_name == "velocity":
            if frame > 0:
                fr_prev = result.result_disp[frame - 1]
                U_prev = fr_prev.U_accum if fr_prev.U_accum is not None else fr_prev.U
                du = u - U_prev[0::2]
                dv = v - U_prev[1::2]
            else:
                du, dv = u, v   # velocity from rest
            vel_mag = np.sqrt(du ** 2 + dv ** 2)   # px/frame
            converted, _unit = self._velocity_settings.apply_conversion(vel_mag)
            return converted
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _strain_frame_count(self) -> int:
        result = self._state.results
        if result is None:
            return 0
        if result.result_strain:
            return len(result.result_strain)
        return len(result.result_disp)

    def _sync_slider_range(self) -> None:
        n = self._strain_frame_count()
        max_idx = max(0, n - 1)
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, max_idx)
        if self._strain_current_frame > max_idx:
            self._strain_current_frame = max_idx
        self._frame_slider.setValue(self._strain_current_frame)
        self._frame_slider.blockSignals(False)
        self._frame_label.setText(f"Frame: {self._strain_current_frame}")

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

            # Background image: show deformed frame whenever show_deformed is on
            if show_deformed:
                self._try_load_background(frame + 1)
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

            if show_deformed and frame < len(result.result_disp):
                fr_d = result.result_disp[frame]
                U = fr_d.U_accum if fr_d.U_accum is not None else fr_d.U
                if U is not None:
                    u, v = U[0::2], U[1::2]
                    nodes = ref_nodes + np.column_stack([u, v])
                    deformed = True
                    ref_uv = (u, v)
                    # Per-frame deformed ROI — strain frame F corresponds to
                    # main window current_frame F+1, so use per_frame_rois[F+1]
                    deformed_mask = self._state.per_frame_rois.get(frame + 1)

            # Auto-range uses only the nodes visible within the (possibly
            # trimmed) deformed mask -- prevents out-of-view nodes from
            # pulling the colorbar range out of sync with what the user sees.
            range_values = visible_values(
                values, nodes, deformed_mask if deformed else None,
            )
            vmin, vmax = self._resolve_range(range_values)

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
            vp = self._canvas.viewport()
            self._colorbar.setGeometry(0, 0, vp.width(), vp.height())
            self._colorbar.update_params(cmap_name, vmin, vmax)
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
        # Reconnect signal: it was disconnected in closeEvent so re-open after
        # a DIC re-run no longer receives the stale connection (Bug B fix).
        try:
            self._state.results_changed.disconnect(self._on_state_results_changed)
        except (RuntimeError, TypeError):
            pass
        self._state.results_changed.connect(self._on_state_results_changed)
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
        try:
            self._state.results_changed.disconnect(self._on_state_results_changed)
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
