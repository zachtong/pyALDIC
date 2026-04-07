"""Top-level strain post-processing window.

Independent ``QMainWindow`` that consumes the displacement results from
``state.results.result_disp``, runs :class:`StrainController` on demand,
and renders the resulting fields with a *private* ``VizController``.

Decoupling contracts (enforced by tests):

* Owns its own ``_strain_current_frame`` -- never mutates
  ``state.current_frame``.
* Owns a private ``VizController`` cache -- never reads or writes
  ``state.colormap`` / ``state.color_min`` / ``state.color_max`` /
  ``state.display_field``.
* Reads from ``state.results.result_disp`` and writes back via
  :func:`dataclasses.replace` through :class:`StrainController` only.
"""

from __future__ import annotations

import traceback as _tb

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
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

from staq_dic.core.data_structures import StrainResult
from staq_dic.gui.app_state import AppState
from staq_dic.gui.controllers.image_controller import ImageController
from staq_dic.gui.controllers.strain_controller import StrainController
from staq_dic.gui.controllers.viz_controller import VizController
from staq_dic.gui.panels.strain_canvas import StrainCanvas
from staq_dic.gui.widgets.strain_field_selector import StrainFieldSelector
from staq_dic.gui.widgets.strain_param_panel import StrainParamPanel
from staq_dic.gui.widgets.strain_viz_panel import StrainVizPanel


# A namespaced "field bucket" lets the private viz cache distinguish
# strain frames from displacement frames in the main view, even though
# both windows share the same AppState. The strain window's frames live
# in their own keyspace.
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
        self.resize(1100, 720)

        self._state = state
        self._strain_ctrl = StrainController(state)
        self._viz_ctrl = VizController()  # PRIVATE -- isolated from MainWindow
        self._image_ctrl = ImageController(state)
        self._strain_current_frame: int = 0

        # --- Build layout ---
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left pane: canvas + frame slider
        left = QVBoxLayout()
        left.setSpacing(4)
        self._canvas = StrainCanvas()
        left.addWidget(self._canvas, 1)
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider)
        left.addWidget(self._frame_slider)
        self._frame_label = QLabel("Frame: 0")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self._frame_label)
        root.addLayout(left, 1)

        # Right pane: selectors + params + buttons + viz
        right = QVBoxLayout()
        right.setSpacing(8)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(300)

        right.addWidget(QLabel("Field"))
        self._field_selector = StrainFieldSelector()
        self._field_selector.field_changed.connect(self._on_field_changed)
        right.addWidget(self._field_selector)

        right.addWidget(QLabel("Strain parameters"))
        self._param_panel = StrainParamPanel()
        self._param_panel.params_dirty.connect(self._on_params_dirty)
        right.addWidget(self._param_panel)

        self._compute_btn = QPushButton("Compute Strain")
        self._compute_btn.setFixedHeight(32)
        self._compute_btn.clicked.connect(self._on_compute_clicked)
        right.addWidget(self._compute_btn)

        self._stale_label = QLabel("")
        self._stale_label.setStyleSheet(
            "color: #fbbf24; font-size: 11px; font-style: italic;"
        )
        right.addWidget(self._stale_label)

        right.addWidget(QLabel("Visualization"))
        self._viz_panel = StrainVizPanel()
        self._viz_panel.viz_changed.connect(self._on_viz_changed)
        right.addWidget(self._viz_panel)

        right.addStretch(1)
        root.addWidget(right_widget, 0)

        self.setCentralWidget(central)

        # Track external pipeline runs so we can drop our cached overlays
        # whenever MainWindow recomputes the displacement results.
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
            self._state.log_message.emit(
                "Strain window: no displacement results to post-process.",
                "warn",
            )
            return
        try:
            self._strain_ctrl.compute_and_store(
                override=self._param_panel.get_override(),
            )
        except Exception as exc:  # pragma: no cover - defensive UI guard
            self._state.log_message.emit(
                f"Strain compute failed: {type(exc).__name__}: {exc}",
                "error",
            )
            return
        # _on_state_results_changed will fire from the controller's emit;
        # it clears caches, syncs the slider and re-renders. We just need
        # to drop the stale flag here.
        self._param_panel.mark_clean()
        self._stale_label.setText("")

    def _on_params_dirty(self) -> None:
        self._stale_label.setText("Stale (params changed)")

    def _on_field_changed(self, _name: str) -> None:
        self._render_current()

    def _on_viz_changed(self) -> None:
        # Colormap / range / alpha changed -> Tier-2 cache must die.
        self._viz_ctrl.clear_pixmap_cache()
        self._render_current()

    def _on_state_results_changed(self) -> None:
        # External Run (or our own compute) replaced PipelineResult.
        self._viz_ctrl.clear_all()
        self._sync_slider_range()
        self._render_current()

    def _on_frame_slider(self, value: int) -> None:
        self._strain_current_frame = int(value)
        self._frame_label.setText(f"Frame: {value}")
        self._render_current()

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

    def _try_load_background(self) -> None:
        """Best-effort background image fetch -- silent on failure."""
        if not self._state.image_files:
            return
        # Reference frame for the strain view is always the underlying
        # frame index 0 (frame 1 in MATLAB nomenclature).
        try:
            rgb = self._image_ctrl.read_image_rgb(0)
            self._canvas.set_image(rgb)
        except (IndexError, FileNotFoundError, ValueError):
            pass

    def _render_current(self) -> None:
        try:
            self._try_load_background()
            result = self._state.results
            if result is None or not result.result_strain:
                self._canvas.clear_overlay()
                return

            frame = self._strain_current_frame
            if frame < 0 or frame >= len(result.result_strain):
                self._canvas.clear_overlay()
                return

            sr: StrainResult = result.result_strain[frame]
            field_name = self._field_selector.current_field()
            values = getattr(sr, field_name, None)
            if values is None:
                self._canvas.clear_overlay()
                return

            ref_mesh = result.result_fe_mesh_each_frame[0]
            nodes = ref_mesh.coordinates_fem
            mesh_step = result.dic_para.winstepsize
            img_shape = result.dic_para.img_size

            vmin, vmax = self._resolve_range(values)
            viz = self._viz_panel.get_state()

            roi_mask = self._state.per_frame_rois.get(0)

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
            )
            self._canvas.set_overlay_pixmap(pixmap)
            self._canvas.set_overlay_alpha(float(viz["alpha"]))
            self._canvas._overlay_item.setScale(float(out_step))
            if xg is not None and yg is not None:
                self._canvas.set_overlay_pos(
                    float(xg.min()), float(yg.min()),
                )
        except Exception as exc:  # pragma: no cover - defensive UI guard
            tb_str = _tb.format_exc()
            print(f"[strain_window._render_current] {exc}\n{tb_str}", flush=True)
            self._state.log_message.emit(
                f"Strain render error: {type(exc).__name__}: {exc}",
                "error",
            )
            self._canvas.clear_overlay()

    def _resolve_range(
        self, values: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Compute (vmin, vmax) from the *local* viz panel state."""
        viz = self._viz_panel.get_state()
        if not viz["use_percentile"]:
            return float(viz["vmin"]), float(viz["vmax"])
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return 0.0, 1.0
        pct = np.percentile(valid, [5, 95])
        return float(pct[0]), float(pct[1])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        # Disconnect from AppState so the window can be safely re-opened
        # without leaking duplicate slot calls.
        try:
            self._state.results_changed.disconnect(self._on_state_results_changed)
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(event)
