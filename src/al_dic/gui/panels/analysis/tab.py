"""The strain window's Analysis tab: probes on the left, curves on the right.

What a chart shows is one *quantity* -- a field with a statistic, or a gauge
reading such as extensometer strain -- and every visible probe that can
produce it is plotted. Series of one quantity share a y-axis honestly; a
strain and a displacement never do.

Work happens only while the tab is on screen. The strain window is a
singleton that is never destroyed, and the first version extracted curves on
every results change even while the window was closed. Curves are cached per
probe geometry and quantity, so renaming, recolouring, hiding or showing a
probe, and moving the frame cursor, redraw without reading the run again.

The tab is the coordinator: it owns the engine and its caches, the probe set's
edits, frame sync and exports. Its parts draw and report -- the canvas panel
(tools, canvas, field overlay), the chart panel (views, controls, chart) and
the probe table -- and none of them writes application state.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from al_dic.analysis.engine import AnalysisEngine, Kymograph
from al_dic.analysis.probes import Probe, replace
from al_dic.analysis.series import TimeSeries
from al_dic.export.export_line import export_line_csv
from al_dic.export.export_probes import ProbeSeries, export_probe_csv, run_parameters
from al_dic.gui.app_state import AppState
from al_dic.gui.panels.analysis.canvas_panel import AnalysisCanvasPanel
from al_dic.gui.panels.analysis.chart_panel import AnalysisChartPanel, ChartContext
from al_dic.gui.panels.analysis.probe_table import ProbeTable
from al_dic.gui.panels.analysis.quantities import (
    FIELDS,
    GAUGE_TOOLS,
    Quantity,
    format_length,
)
from al_dic.gui.theme import COLORS
from al_dic.i18n import tr_args

logger = logging.getLogger(__name__)


class AnalysisTab(QWidget):
    """Probe placement, the probe list, and the curves they produce."""

    # The user picked a frame here (navigator or a click on the chart).
    frame_requested = Signal(int)

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._selected_id: int | None = None
        self._frame = 0
        self._dirty = True
        self._background_loaded = False
        self._field_chosen = False
        self._engine: AnalysisEngine | None = None
        self._engine_key: tuple | None = None
        self._cache: dict[tuple, TimeSeries] = {}
        self._kymo_cache: dict[tuple, Kymograph] = {}
        self._parameters_provider = None
        self._field_renderer = None
        self._strain_tab_field: str | None = None
        self._armed_tool: str | None = None

        from al_dic.gui.controllers.image_controller import ImageController

        self._image_ctrl = ImageController(state)

        # Field renders wait for the event loop. The renderer's cache is
        # flushed by the strain window's own slots on results and unit
        # changes, and those may run after ours: drawing at once could paint
        # the previous run's values. It also folds a burst of requests into
        # one render.
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(0)
        self._overlay_timer.timeout.connect(self._refresh_overlay)

        self._canvas_panel = AnalysisCanvasPanel()
        self._chart_panel = AnalysisChartPanel()
        self._probe_table = ProbeTable()
        self._alias_parts()
        self._connect_parts()
        self._lay_out()

        # On the whole tab, not the table: a probe picked on the canvas must
        # answer Delete and F2 too. Text fields still get the keys first.
        self._delete_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Delete), self)
        self._delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(self._on_delete)
        self._rename_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F2), self)
        self._rename_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._rename_shortcut.activated.connect(
            lambda: self._on_rename(self._selected_id))

        state.results_changed.connect(self._on_results_changed)
        state.images_changed.connect(self._on_images_changed)
        state.physical_units_changed.connect(self._on_units_changed)
        state.roi_changed.connect(self._drop_engine)
        state.display_changed.connect(self._request_overlay)

        self.retranslate_ui()

    # -- construction -----------------------------------------------------

    def _alias_parts(self) -> None:
        """Short names for the parts' widgets used here (and by tests)."""
        canvas_panel, chart_panel = self._canvas_panel, self._chart_panel
        self._canvas = canvas_panel.canvas
        self._tool_buttons = canvas_panel.tool_buttons
        self._banner = canvas_panel.banner
        self._colorbar = canvas_panel.colorbar
        self._show_field_box = canvas_panel.show_field_box
        self._nav = canvas_panel.nav
        self._chart = chart_panel.chart
        self._view_bar = chart_panel.view_bar
        self._quantity_box = chart_panel.quantity_box
        self._statistic_box = chart_panel.statistic_box
        self._x_box = chart_panel.x_box
        self._unit_box = chart_panel.unit_box
        self._threshold = chart_panel.threshold
        self._other_frames_box = chart_panel.other_frames_box
        self._export_btn = chart_panel.export_btn
        self._table = self._probe_table.table

    def _connect_parts(self) -> None:
        canvas = self._canvas
        canvas.probe_requested.connect(self._on_probe_placed)
        canvas.placement_cancelled.connect(self._on_placement_cancelled)
        canvas.probe_selected.connect(self._on_canvas_selected)
        canvas.probe_edited.connect(self._on_probe_edited)
        canvas.probe_activated.connect(self._on_rename)
        canvas.set_length_format(self._format_length)
        self._canvas_panel.tool_toggled.connect(self._on_tool_clicked)
        self._canvas_panel.show_field_toggled.connect(self._request_overlay)
        self._nav.frame_changed.connect(self._on_nav_frame)

        chart = self._chart_panel
        chart.settings_changed.connect(self._request_refresh)
        chart.quantity_chosen.connect(self._on_quantity_chosen)
        chart.export_csv_requested.connect(self._on_export_csv)
        chart.export_line_requested.connect(self._on_export_line_csv)
        chart.export_chart_requested.connect(self._on_export_chart)
        chart.copy_chart_requested.connect(self._on_copy_chart)
        chart.copy_data_requested.connect(self._on_copy_data)
        self._chart.x_clicked.connect(self._on_chart_clicked)

        table = self._probe_table
        table.selection_changed.connect(self._on_table_selected)
        table.visibility_changed.connect(self._on_visibility_changed)
        table.renamed.connect(self._on_renamed)
        table.recolour_clicked.connect(self._on_recolour)
        table.delete_clicked.connect(self._on_delete)
        table.clear_clicked.connect(self._on_clear)

    def _lay_out(self) -> None:
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._chart_panel)
        right.addWidget(self._probe_table)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 1)
        right.setSizes([500, 6 * self._probe_table.row_height()])
        right.setChildrenCollapsible(False)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._canvas_panel)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)
        root.addWidget(splitter, 1)
        self._status = QLabel()
        self._status.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 11px;")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

    # -- translation ------------------------------------------------------

    def retranslate_ui(self) -> None:
        self._canvas_panel.retranslate_ui(self._selected_probe() is not None)
        self._probe_table.retranslate_ui()
        self._chart_panel.set_frame_rate(self._frame_rate())
        self._chart_panel.retranslate_ui()
        self._show_table()
        self._request_refresh()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)

    # -- wiring from the strain window --------------------------------------

    def set_frame(self, frame: int) -> None:
        """Follow the Strain Field tab's frame (no signal back)."""
        self._frame = max(0, int(frame))
        self._sync_navigator()
        self._follow_frame()

    def set_default_field(self, name: str) -> None:
        """The Strain Field tab's field: where the chart starts until the user
        picks, and what the canvas shows under a gauge reading."""
        self._strain_tab_field = name
        quantity = self._chart_panel.quantity()
        if quantity is not None and quantity.is_gauge:
            self._request_overlay()
        if self._field_chosen or name not in FIELDS:
            return
        target = Quantity("field", name)
        if quantity != target and self._chart_panel.set_quantity(target):
            self._request_refresh()

    def set_parameters_provider(self, provider) -> None:
        """``() -> dict[str, str]`` of the settings of the last Compute Strain."""
        self._parameters_provider = provider

    def set_field_renderer(self, renderer) -> None:
        """``(field, frame) -> FieldImage | None``: the field in the reference
        configuration, as the Strain Field tab would draw it."""
        self._field_renderer = renderer
        self._request_overlay()

    # -- visibility gating ------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._background_loaded:
            self._load_background()
        if self._dirty:
            self._refresh()

    def _request_refresh(self, *_args) -> None:
        if self.isVisible():
            self._refresh()
        else:
            self._dirty = True

    def _request_overlay(self, *_args) -> None:
        if self.isVisible():
            self._overlay_timer.start()
        else:
            self._dirty = True

    def _on_results_changed(self) -> None:
        self._chart_panel.set_frame_rate(self._frame_rate())
        self._drop_engine()

    def _on_units_changed(self) -> None:
        # The frame rate lives with the physical units, and "Time" is only
        # offered while there is one. Cached series are keyed on the pixel
        # size, so the cache needs no flushing.
        self._chart_panel.set_frame_rate(self._frame_rate())
        self._request_refresh()

    def _on_images_changed(self) -> None:
        self._background_loaded = False
        self._drop_engine()
        if self.isVisible():
            self._load_background()

    def _drop_engine(self, *_args) -> None:
        self._engine = None
        self._engine_key = None
        self._cache.clear()
        self._kymo_cache.clear()
        self._request_refresh()

    # -- engine and its caches ----------------------------------------------

    def _ref_mask(self, result):
        mask = self._state.per_frame_rois.get(0)
        if mask is None:
            mask = getattr(result.dic_para, "img_ref_mask", None)
        return mask

    def _engine_for(self, result) -> AnalysisEngine:
        mask = self._ref_mask(result)
        key = (id(result), id(mask))
        if self._engine is None or self._engine_key != key:
            self._engine = AnalysisEngine(result, ref_mask=mask)
            self._engine_key = key
            self._cache.clear()
            self._kymo_cache.clear()
        return self._engine

    def _series(self, probe: Probe, quantity: Quantity, statistic: str,
                *, axes: str = "image") -> TimeSeries:
        engine = self._engine_for(self._state.results)
        pixel = self._pixel_size()
        key = (probe.geometry, probe.kind, quantity, statistic,
               round(self._threshold.value(), 6), pixel, self._length_unit(), axes)
        cached = self._cache.get(key)
        if cached is None:
            cached = engine.series(
                probe,
                None if quantity.is_gauge else quantity.name,
                quantity.name if quantity.is_gauge else statistic,
                pixel_size=pixel,
                length_unit=self._length_unit(),
                min_valid_fraction=self._threshold.value(),
                axes=axes,
            )
            self._cache[key] = cached
        return cached

    def _kymograph(self, probe: Probe, field: str, *, axes: str = "image") -> Kymograph:
        engine = self._engine_for(self._state.results)
        key = (probe.geometry, field, self._pixel_size(), self._length_unit(), axes)
        cached = self._kymo_cache.get(key)
        if cached is None:
            cached = engine.kymograph(
                probe, field, pixel_size=self._pixel_size(),
                length_unit=self._length_unit(), axes=axes)
            self._kymo_cache[key] = cached
        return cached

    # -- refresh ----------------------------------------------------------

    def _chart_context(self) -> ChartContext:
        return ChartContext(
            results=self._state.results,
            probes=tuple(self._state.probes),
            selected=self._selected_probe(),
            frame=self._frame,
            length_unit=self._length_unit(),
            series=self._series,
            kymograph=self._kymograph,
            field_state=self._state.get_field_state,
        )

    def _refresh(self) -> None:
        self._dirty = False
        self._chart_panel.update_controls()
        self._refresh_chart()
        self._show_table()
        self._canvas.set_probes(self._state.probes, self._selected_id)
        self._sync_navigator()
        self._update_banner()
        self._request_overlay()

    def _refresh_chart(self) -> None:
        self._chart_panel.refresh(self._chart_context())
        self._update_actions()

    def _show_table(self) -> None:
        self._probe_table.show_probes(
            self._state.probes, self._chart_panel.notes, self._selected_id)
        self._update_actions()

    def _update_actions(self) -> None:
        self._probe_table.set_actions(
            self._selected_probe() is not None, len(self._state.probes) > 0)
        self._export_btn.setEnabled(self._chart_panel.has_output())

    def _update_banner(self) -> None:
        self._canvas_panel.update_banner(self._selected_probe() is not None)

    def _sync_navigator(self) -> None:
        result = self._state.results
        n = len(result.result_disp) + 1 if result is not None else 0
        self._nav.set_state(n, min(self._frame, max(n - 1, 0)))

    # -- the field under the probes ------------------------------------------

    def _overlay_field(self) -> str | None:
        """The field the canvas shows: the plotted one, or for a gauge
        reading -- which has no field -- the Strain Field tab's."""
        quantity = self._chart_panel.quantity()
        if quantity is None:
            return None
        return self._strain_tab_field if quantity.is_gauge else quantity.name

    def _refresh_overlay(self) -> None:
        if not self.isVisible():
            self._dirty = True
            return
        image = None
        field = self._overlay_field()
        if (field is not None and self._show_field_box.isChecked()
                and self._field_renderer is not None
                and self._state.results is not None):
            try:
                image = self._field_renderer(field, self._frame)
            except Exception as exc:  # the probes stay usable without it
                logger.exception("Analysis canvas: drawing %s failed", field)
                self._say(tr_args(self.tr("Could not draw the field: %1"), exc),
                          "error")
        if image is None:
            self._canvas_panel.clear_field()
            return
        scale, unit = self._chart_panel.display_scale(
            Quantity("field", field), None, self._length_unit())
        self._canvas_panel.show_field(image, scale, unit)

    # -- frame --------------------------------------------------------------

    def _frame_count(self) -> int:
        result = self._state.results
        return len(result.result_disp) + 1 if result is not None else 1

    def _follow_frame(self) -> None:
        """Move the chart to ``self._frame``: a cursor, or a new profile."""
        if self._chart_panel.view() == "profile":
            self._request_refresh()
        else:
            self._chart_panel.set_cursor(self._frame)
            self._request_overlay()

    def _on_nav_frame(self, frame: int) -> None:
        self._frame = int(frame)
        self._follow_frame()
        self.frame_requested.emit(self._frame)

    def _on_chart_clicked(self, x: float) -> None:
        if self._chart_panel.view() == "profile":
            return                      # x is a distance there, not a frame
        frame = self._chart_panel.frame_from_x(x, self._frame_count())
        self._frame = frame
        self._sync_navigator()
        self._follow_frame()
        self.frame_requested.emit(frame)

    def _view_bar_index(self, key: str) -> int:
        return self._chart_panel.view_index(key)

    def _on_quantity_chosen(self) -> None:
        self._field_chosen = True

    # -- probe lifecycle ------------------------------------------------------

    def _on_tool_clicked(self, tool: str, checked: bool) -> None:
        # Remembered here: the canvas reports a gauge tool's line as a plain
        # "line", and what to plot next depends on which tool drew it.
        self._armed_tool = tool if checked else None
        self._canvas.set_tool(tool if checked else "none")  # type: ignore[arg-type]
        self._canvas_panel.sync_tools()
        self._update_banner()

    def _on_placement_cancelled(self) -> None:
        self._armed_tool = None
        self._canvas_panel.sync_tools()
        self._update_banner()

    def _on_probe_placed(self, kind: str, geometry) -> None:
        tool, self._armed_tool = self._armed_tool, None
        probe = self._state.probes.add(kind, geometry)  # type: ignore[arg-type]
        self._selected_id = probe.id
        self._canvas_panel.sync_tools()
        self._adopt_gauge_reading(tool)
        self._refresh()
        self._say(tr_args(self.tr("Added probe '%1'."), probe.label))

    def _adopt_gauge_reading(self, tool: str | None) -> None:
        """Plot what a gauge tool is for, unless the chart already shows a
        reading of that family."""
        if tool not in GAUGE_TOOLS:
            return
        default, family = GAUGE_TOOLS[tool]
        current = self._chart_panel.quantity()
        if current is not None and current.is_gauge and current.name in family:
            return
        if self._chart_panel.set_quantity(Quantity("gauge", default)):
            self._field_chosen = True

    def _selected_probe(self) -> Probe | None:
        if self._selected_id is None:
            return None
        try:
            return self._state.probes.get(self._selected_id)
        except KeyError:
            return None

    def _on_canvas_selected(self, probe_id) -> None:
        self._selected_id = probe_id
        self._probe_table.select(probe_id)
        self._update_banner()
        # Emphasis, or which line a line view reads: series are cached.
        self._refresh_chart()

    def _on_table_selected(self, probe_id) -> None:
        self._selected_id = probe_id
        self._canvas.set_probes(self._state.probes, self._selected_id)
        self._update_banner()
        self._refresh_chart()

    def _on_probe_edited(self, probe_id: int, geometry) -> None:
        try:
            probe = self._state.probes.get(probe_id)
        except KeyError:
            return
        self._state.probes.replace(replace(probe, geometry=geometry))
        self._selected_id = probe_id
        self._refresh()

    def _on_rename(self, probe_id: int | None) -> None:
        """Open the name for editing -- from a double-click or F2."""
        self._probe_table.edit_name(probe_id)

    def _on_visibility_changed(self, probe_id: int, visible: bool) -> None:
        try:
            probe = self._state.probes.get(probe_id)
        except KeyError:
            return
        self._state.probes.replace(replace(probe, visible=visible))
        self._refresh()

    def _on_renamed(self, probe_id: int, label: str) -> None:
        try:
            probe = self._state.probes.get(probe_id)
        except KeyError:
            return
        if label and label != probe.label:
            self._state.probes.replace(replace(probe, label=label))
        self._refresh()

    def _on_recolour(self) -> None:
        probe = self._selected_probe()
        if probe is None:
            return
        colour = QColorDialog.getColor(QColor(probe.color), self)
        if colour.isValid():
            self._state.probes.replace(replace(probe, color=colour.name()))
            self._refresh()

    def _on_delete(self) -> None:
        probe = self._selected_probe()
        if probe is None:
            return
        self._state.probes.remove(probe.id)
        self._selected_id = None
        self._refresh()

    def _on_clear(self) -> None:
        if len(self._state.probes) == 0:
            return
        confirm = QMessageBox.question(
            self, self.tr("Clear All Probes"),
            self.tr("Delete every probe? This cannot be undone."),
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._state.probes.clear()
            self._selected_id = None
            self._refresh()

    # -- background -----------------------------------------------------------

    def _load_background(self) -> None:
        """The reference frame, fitted once per image set -- not per edit.

        Probe coordinates are frame-0 pixels, so the placement canvas shows
        frame 0. Refitting on every probe edit threw away the user's zoom.
        """
        if not self._state.image_files:
            return
        try:
            self._canvas.set_image(self._image_ctrl.read_image_rgb(0))
        except (IndexError, FileNotFoundError, ValueError):
            return
        self._canvas.fit_to_view()
        self._background_loaded = True

    # -- units ------------------------------------------------------------

    def _frame_rate(self) -> float:
        rate = float(getattr(self._state, "frame_rate", 0.0) or 0.0)
        return rate if rate > 0 else 0.0

    def _pixel_size(self) -> float:
        state = self._state
        if state.use_physical_units and state.pixel_size > 0:
            return float(state.pixel_size)
        return 1.0

    def _length_unit(self) -> str:
        state = self._state
        return state.pixel_unit if state.use_physical_units else "px"

    def _format_length(self, px: float) -> str:
        return format_length(px, self._pixel_size(), self._length_unit())

    # -- export -------------------------------------------------------------

    def _say(self, message: str, level: str = "success") -> None:
        self._status.setText(message)
        self._state.log_message.emit(message, level)

    def _export_parameters(self) -> dict[str, str]:
        """Run parameters for an export header, strain settings included."""
        params = run_parameters(self._state.results)
        if self._parameters_provider is not None:
            try:
                params.update(self._parameters_provider() or {})
            except Exception:  # a header is not worth a failed export
                logger.exception("Analysis export: strain parameters unavailable")
        return params

    def _csv_filter(self) -> str:
        return self.tr("CSV Files") + " (*.csv);;" + self.tr("All Files") + " (*)"

    def _on_export_csv(self) -> None:
        plotted = self._chart_panel.plotted
        if not (self._chart.has_data and plotted):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export Probe Data"), "probes.csv", self._csv_filter())
        if not path:
            return
        entries = []
        for probe, quantity, statistic in plotted:
            # Re-read in world axes, as the node export writes v; the chart
            # keeps the screen's convention.
            ts = self._series(probe, quantity, statistic, axes="world")
            entries.append(ProbeSeries(
                probe=probe,
                field=None if quantity.is_gauge else quantity.name,
                reduction=quantity.name if quantity.is_gauge else (
                    "value" if probe.kind == "point" else statistic),
                series=ts,
            ))
        try:
            export_probe_csv(path, entries, frame_rate=self._frame_rate() or None,
                             parameters=self._export_parameters())
        except (OSError, ValueError) as exc:
            self._say(tr_args(self.tr("Probe export failed: %1"), exc), "error")
            return
        self._say(tr_args(self.tr("Probe data written to %1"), path))

    def _on_export_line_csv(self) -> None:
        shown = self._chart_panel.line_shown
        if not (self._chart.has_data and shown):
            return
        probe, field = shown
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export Line Data"), "line_profile.csv", self._csv_filter())
        if not path:
            return
        # World axes, as the node export writes v; the chart keeps the screen's.
        kymo = self._kymograph(probe, field, axes="world")
        xy = self._engine_for(self._state.results).plan(probe.geometry).xy
        try:
            export_line_csv(path, probe, kymo, xy, field=field, axes="world",
                            frame_rate=self._frame_rate() or None,
                            parameters=self._export_parameters())
        except (OSError, ValueError) as exc:
            self._say(tr_args(self.tr("Line export failed: %1"), exc), "error")
            return
        self._say(tr_args(self.tr("Line data written to %1"), path))

    def _image_filters(self) -> list[str]:
        """PNG, SVG, PDF, then anything; names translated, globs literal."""
        return [
            self.tr("PNG Images") + " (*.png)",
            self.tr("SVG Images") + " (*.svg)",
            self.tr("PDF Documents") + " (*.pdf)",
            self.tr("All Files") + " (*)",
        ]

    def _on_export_chart(self) -> None:
        """The chart as a figure for a page: light, 300 dpi, text editable."""
        if not self._chart.has_data:
            return
        filters = self._image_filters()
        path, chosen = QFileDialog.getSaveFileName(
            self, self.tr("Export Chart"), "probe_chart.png", ";;".join(filters))
        if not path:
            return
        out = Path(path)
        if not out.suffix:
            # A bare name takes the format of the filter it was saved under.
            suffixes = dict(zip(filters, (".png", ".svg", ".pdf")))
            out = out.with_suffix(suffixes.get(chosen, ".png"))
        try:
            self._chart.export_figure(out)
        except (OSError, ValueError) as exc:
            self._say(tr_args(self.tr("Chart export failed: %1"), exc), "error")
            return
        self._say(tr_args(self.tr("Chart written to %1"), str(out)))

    def _on_copy_chart(self) -> None:
        if not self._chart.has_data:
            return
        QApplication.clipboard().setImage(self._chart.publication_image())
        self._say(self.tr("Chart copied to the clipboard."))

    def _on_copy_data(self) -> None:
        """What the chart shows, as tab-separated text for a spreadsheet."""
        if not self._chart.has_data:
            return
        rows = self._chart.plotted_table()
        QApplication.clipboard().setText("\n".join("\t".join(r) for r in rows))
        self._say(self.tr("Plotted data copied to the clipboard."))


__all__ = ["AnalysisTab"]
