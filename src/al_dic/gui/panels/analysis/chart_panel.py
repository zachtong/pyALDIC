"""The Analysis tab's right side: the view, what to plot, and the chart.

Three views share the controls: every probe over time; the field along one
line on the current frame, over the other frames in grey; and the same line
on every frame at once as a kymograph. The line views read the selected line
(or the newest one) and one field -- a gauge reading has no profile.

The panel draws from a ``ChartContext`` the tab builds for each refresh; it
reads no application state of its own, so the tab stays the one place that
decides when work is done.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PySide6.QtCore import QCoreApplication, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from al_dic.analysis.engine import DEFAULT_MIN_VALID_FRACTION, AnalysisEngine, SampleStatus
from al_dic.analysis.probes import GAUGE_QUANTITIES, SPATIAL_STATISTICS, Probe
from al_dic.analysis.series import FrameStatus
from al_dic.core.fields import is_strain_field
from al_dic.gui.panels.analysis import text
from al_dic.gui.panels.analysis.quantities import (
    FIELDS,
    VIEWS,
    Quantity,
    display_scale,
    is_strainlike,
    other_frames,
)
from al_dic.gui.widgets.double_spin import LocaleSafeDoubleSpinBox
from al_dic.gui.widgets.mpl_chart import Curve, MplChart, status_label
from al_dic.i18n import tr_args


@dataclass(frozen=True)
class ChartContext:
    """What a chart needs from the tab, read afresh at each refresh."""

    results: object | None
    probes: tuple[Probe, ...]
    selected: Probe | None
    frame: int
    length_unit: str
    series: Callable            # (probe, quantity, statistic) -> TimeSeries
    kymograph: Callable         # (probe, field) -> Kymograph
    field_state: Callable       # field -> FieldColorState


class AnalysisChartPanel(QWidget):
    """View tabs, plot controls, the chart, and what was last drawn."""

    # A control changed what the chart shows.
    settings_changed = Signal()
    # The user picked a quantity (so the Strain Field tab's stops leading).
    quantity_chosen = Signal()
    export_csv_requested = Signal()
    export_line_requested = Signal()
    export_chart_requested = Signal()
    copy_chart_requested = Signal()
    copy_data_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False
        self._rate = 0.0
        # Drawn by the last refresh: (probe, quantity, statistic) per curve.
        self.plotted: list[tuple[Probe, Quantity, str]] = []
        # Why each probe shows gaps, or is not plotted.
        self.notes: dict[int, str] = {}
        # The line and field of the last line view.
        self.line_shown: tuple[Probe, str] | None = None

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        self.view_bar = QTabBar()
        self.view_bar.setExpanding(False)
        self.view_bar.setDrawBase(False)
        for key in VIEWS:
            self.view_bar.setTabData(self.view_bar.addTab(""), key)
        column.addWidget(self.view_bar)
        column.addLayout(self._build_plot_row())
        column.addLayout(self._build_option_row())
        self.chart = MplChart()
        column.addWidget(self.chart, 1)

        self.quantity_box.currentIndexChanged.connect(self._on_quantity_changed)
        for box in (self.statistic_box, self.x_box, self.unit_box):
            box.currentIndexChanged.connect(self._on_setting_changed)
        self.threshold.valueChanged.connect(self._on_setting_changed)
        self.other_frames_box.toggled.connect(self._on_setting_changed)
        # Last: addTab() above already emitted currentChanged, before the
        # controls this slot touches existed.
        self.view_bar.currentChanged.connect(self._on_setting_changed)

    def _build_plot_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self._quantity_label = QLabel()
        self.quantity_box = QComboBox()
        self.quantity_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._statistic_label = QLabel()
        self.statistic_box = QComboBox()
        self.statistic_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for w in (self._quantity_label, self.quantity_box,
                  self._statistic_label, self.statistic_box):
            row.addWidget(w)
        row.addStretch()
        self.export_btn = QToolButton()
        self.export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.export_btn)
        self._export_csv_action = menu.addAction("")
        self._export_csv_action.triggered.connect(self.export_csv_requested)
        self._export_line_action = menu.addAction("")
        self._export_line_action.triggered.connect(self.export_line_requested)
        self._export_chart_action = menu.addAction("")
        self._export_chart_action.triggered.connect(self.export_chart_requested)
        menu.addSeparator()
        self._copy_chart_action = menu.addAction("")
        self._copy_chart_action.triggered.connect(self.copy_chart_requested)
        self._copy_data_action = menu.addAction("")
        self._copy_data_action.triggered.connect(self.copy_data_requested)
        self.export_btn.setMenu(menu)
        row.addWidget(self.export_btn)
        return row

    def _build_option_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self._x_label = QLabel()
        self.x_box = QComboBox()
        self._unit_label = QLabel()
        self.unit_box = QComboBox()
        self._threshold_label = QLabel()
        # Every numeric input in gui/ goes through the locale-safe subclass.
        self.threshold = LocaleSafeDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(DEFAULT_MIN_VALID_FRACTION)
        self.other_frames_box = QCheckBox()
        self.other_frames_box.setChecked(True)
        for w in (self._x_label, self.x_box, self._unit_label, self.unit_box,
                  self._threshold_label, self.threshold, self.other_frames_box):
            row.addWidget(w)
        row.addStretch()
        return row

    # -- text -------------------------------------------------------------

    def retranslate_ui(self) -> None:
        # Spelled out every time: lupdate cannot follow an alias of translate.
        self._quantity_label.setText(QCoreApplication.translate("AnalysisTab", "Plot:"))
        self._statistic_label.setText(
            QCoreApplication.translate("AnalysisTab", "Statistic:"))
        self._x_label.setText(QCoreApplication.translate("AnalysisTab", "X axis:"))
        self._unit_label.setText(QCoreApplication.translate("AnalysisTab", "Strain as:"))
        self._threshold_label.setText(
            QCoreApplication.translate("AnalysisTab", "Min. valid fraction:"))
        self.threshold.setToolTip(QCoreApplication.translate(
            "AnalysisTab",
            "A frame is left blank when fewer than this fraction of a line's "
            "or region's points are reliable. Guards against a curve that "
            "stays smooth while its sample shrinks away."))
        views = {
            "time": (
                QCoreApplication.translate(
                    "AnalysisTab", "Over time", "Chart view: every frame of each probe"),
                QCoreApplication.translate(
                    "AnalysisTab", "Each probe's reading at every frame.")),
            "profile": (
                QCoreApplication.translate(
                    "AnalysisTab", "Along the line", "Chart view: a profile"),
                QCoreApplication.translate(
                    "AnalysisTab",
                    "The field along the selected line at the current frame, "
                    "over the other frames in grey.")),
            "kymograph": (
                QCoreApplication.translate(
                    "AnalysisTab", "Kymograph", "Chart view: distance against frame"),
                QCoreApplication.translate(
                    "AnalysisTab",
                    "The field along the selected line at every frame: "
                    "distance against frame, value as colour.")),
        }
        for index in range(self.view_bar.count()):
            label, tip = views[self.view_bar.tabData(index)]
            self.view_bar.setTabText(index, label)
            self.view_bar.setTabToolTip(index, tip)
        self.other_frames_box.setText(
            QCoreApplication.translate("AnalysisTab", "Other frames"))
        self.other_frames_box.setToolTip(QCoreApplication.translate(
            "AnalysisTab",
            "Draw the other frames' profiles faintly behind the current one "
            "(at most twelve, evenly spaced)."))
        self._export_line_action.setText(
            QCoreApplication.translate("AnalysisTab", "Line data (CSV)…"))
        self.export_btn.setText(QCoreApplication.translate("AnalysisTab", "Export"))
        self._export_csv_action.setText(
            QCoreApplication.translate("AnalysisTab", "Probe data (CSV)…"))
        self._export_chart_action.setText(
            QCoreApplication.translate("AnalysisTab", "Chart image…"))
        self._copy_chart_action.setText(
            QCoreApplication.translate("AnalysisTab", "Copy chart"))
        self._copy_data_action.setText(
            QCoreApplication.translate("AnalysisTab", "Copy plotted data"))
        self._populate_quantities()
        self._populate_statistics()
        self._populate_x_axes()
        self._populate_strain_units()

    def _populate_quantities(self) -> None:
        self._updating = True
        current = self.quantity_box.currentData()
        self.quantity_box.clear()
        for name in FIELDS:
            self.quantity_box.addItem(text.field_title(name), Quantity("field", name).key)
        self.quantity_box.insertSeparator(self.quantity_box.count())
        for name in GAUGE_QUANTITIES:
            self.quantity_box.addItem(text.gauge_title(name), Quantity("gauge", name).key)
        index = self.quantity_box.findData(current) if current else -1
        self.quantity_box.setCurrentIndex(index if index >= 0 else 0)
        self._updating = False

    def _populate_statistics(self) -> None:
        self._updating = True
        current = self.statistic_box.currentData() or "mean"
        self.statistic_box.clear()
        for name in SPATIAL_STATISTICS:
            self.statistic_box.addItem(text.statistic_title(name), name)
        self.statistic_box.setCurrentIndex(max(self.statistic_box.findData(current), 0))
        self._updating = False

    def _populate_x_axes(self) -> None:
        self._updating = True
        current = self.x_box.currentData() or "frame"
        self.x_box.clear()
        self.x_box.addItem(QCoreApplication.translate("AnalysisTab", "Frame"), "frame")
        if self._rate > 0:
            self.x_box.addItem(QCoreApplication.translate("AnalysisTab", "Time (s)"), "time")
        self.x_box.setCurrentIndex(max(self.x_box.findData(current), 0))
        self._updating = False

    def _populate_strain_units(self) -> None:
        self._updating = True
        current = self.unit_box.currentData() or "ratio"
        self.unit_box.clear()
        self.unit_box.addItem(QCoreApplication.translate(
            "AnalysisTab", "ratio", "Strain display unit: plain number"), "ratio")
        self.unit_box.addItem("%", "percent")
        self.unit_box.addItem("µε", "microstrain")
        self.unit_box.setCurrentIndex(max(self.unit_box.findData(current), 0))
        self._updating = False

    # -- state --------------------------------------------------------------

    def view(self) -> str:
        return self.view_bar.tabData(self.view_bar.currentIndex()) or "time"

    def view_index(self, key: str) -> int:
        for index in range(self.view_bar.count()):
            if self.view_bar.tabData(index) == key:
                return index
        raise KeyError(key)

    def quantity(self) -> Quantity | None:
        return Quantity.from_key(self.quantity_box.currentData())

    def set_quantity(self, quantity: Quantity) -> bool:
        """Show *quantity* without reporting a user choice; False if absent."""
        index = self.quantity_box.findData(quantity.key)
        if index < 0:
            return False
        if index != self.quantity_box.currentIndex():
            self._updating = True
            self.quantity_box.setCurrentIndex(index)
            self._updating = False
        self.update_controls()
        return True

    def set_frame_rate(self, rate: float) -> None:
        """"Time" is offered only while the run has a frame rate."""
        self._rate = rate if rate > 0 else 0.0
        self._populate_x_axes()

    def display_scale(self, quantity: Quantity, statistic: str | None,
                      length_unit: str) -> tuple[float, str]:
        return display_scale(quantity, statistic,
                             strain_unit=self.unit_box.currentData() or "ratio",
                             length_unit=length_unit)

    def has_output(self) -> bool:
        """Something is drawn that an export could write."""
        shown = self.plotted if self.view() == "time" else self.line_shown
        return self.chart.has_data and bool(shown)

    def _on_quantity_changed(self) -> None:
        if self._updating:
            return
        self.quantity_chosen.emit()
        self.update_controls()
        self.settings_changed.emit()

    def _on_setting_changed(self, *_args) -> None:
        if self._updating:
            return
        self.update_controls()
        self.settings_changed.emit()

    def update_controls(self) -> None:
        """Show the controls that mean something for this view and quantity."""
        view = self.view()
        over_time = view == "time"
        quantity = self.quantity()
        is_field = quantity is not None and not quantity.is_gauge
        statistic = self.statistic_box.currentData() if over_time else None
        self._statistic_label.setVisible(is_field and over_time)
        self.statistic_box.setVisible(is_field and over_time)
        strainlike = is_strainlike(quantity, statistic)
        self._unit_label.setVisible(strainlike)
        self.unit_box.setVisible(strainlike)
        threshold = (is_field and over_time
                     and self.statistic_box.currentData() != "valid_fraction")
        self._threshold_label.setVisible(threshold)
        self.threshold.setVisible(threshold)
        self._x_label.setVisible(view != "profile")
        self.x_box.setVisible(view != "profile")
        self.other_frames_box.setVisible(view == "profile")
        self._export_csv_action.setVisible(over_time)
        self._export_line_action.setVisible(not over_time)

    # -- x axis -------------------------------------------------------------

    def _time_axis(self) -> bool:
        return self.x_box.currentData() == "time" and self._rate > 0

    def x_values(self, frames: np.ndarray) -> np.ndarray:
        if self._time_axis():
            return frames.astype(float) / self._rate
        return frames.astype(float) + 1.0            # 1-based, as the navigator

    def x_of(self, frame: int) -> float:
        return float(self.x_values(np.array([frame]))[0])

    def x_title(self) -> str:
        if self._time_axis():
            return QCoreApplication.translate("AnalysisTab", "Time (s)")
        return QCoreApplication.translate("AnalysisTab", "Frame")

    def frame_from_x(self, x: float, n_frames: int) -> int:
        frame = int(round(x * self._rate)) if self._time_axis() else int(round(x)) - 1
        return max(0, min(frame, max(n_frames, 1) - 1))

    def set_cursor(self, frame: int) -> None:
        self.chart.set_cursor(self.x_of(frame))

    # -- drawing --------------------------------------------------------------

    def placeholder(self, message: str) -> None:
        self.chart.clear(message)
        self.plotted = []
        self.notes = {}

    def refresh(self, ctx: ChartContext) -> None:
        self.line_shown = None
        if self.view() == "time":
            self._draw_time(ctx)
        else:
            self._draw_line(ctx, self.view())

    @staticmethod
    def _applies(probe: Probe, quantity: Quantity, statistic: str) -> bool:
        if quantity.is_gauge:
            return probe.kind == "line"
        return AnalysisEngine.supports(probe, statistic)

    def _unavailable(self, ctx: ChartContext, quantity: Quantity | None, *,
                     need_probes: bool) -> str | None:
        """Why nothing can be drawn at all, or None."""
        if ctx.results is None:
            return QCoreApplication.translate(
                "AnalysisTab", "Run a DIC analysis to plot probes.")
        if quantity is None:
            return ""
        if need_probes and not ctx.probes:
            return QCoreApplication.translate(
                "AnalysisTab", "Place a probe on the reference image to begin.")
        if (not quantity.is_gauge and is_strain_field(quantity.name)
                and not ctx.results.result_strain):
            return QCoreApplication.translate(
                "AnalysisTab",
                "Strain has not been computed yet. Compute it on the Strain "
                "Field tab, or plot a displacement.")
        return None

    def _draw_time(self, ctx: ChartContext) -> None:
        quantity = self.quantity()
        statistic = self.statistic_box.currentData() or "mean"
        reason = self._unavailable(ctx, quantity, need_probes=True)
        if reason is not None:
            self.placeholder(reason)
            return

        shown = [p for p in ctx.probes
                 if p.visible and self._applies(p, quantity, statistic)]
        notes = {p.id: text.not_applicable_note(p, quantity)
                 for p in ctx.probes if not self._applies(p, quantity, statistic)}
        if not shown:
            message = QCoreApplication.translate(
                "AnalysisTab", "No visible probe can show this quantity.")
            if quantity.is_gauge:
                hint = QCoreApplication.translate(
                    "AnalysisTab", "Gauge quantities need a line probe.")
                message = f"{message} {hint}"
            self.placeholder(message)
            self.notes = notes
            return

        scale, unit = self.display_scale(quantity, statistic, ctx.length_unit)
        curves, plotted = [], []
        selected_id = ctx.selected.id if ctx.selected is not None else None
        for probe in shown:
            try:
                ts = ctx.series(probe, quantity, statistic)
            except ValueError as exc:
                notes[probe.id] = str(exc)
                continue
            note = text.series_note(ts)
            notes[probe.id] = note
            curves.append(Curve(
                label=f"{probe.label} · {note}" if note else probe.label,
                name=probe.label,
                colour=probe.color, x=self.x_values(ts.frames),
                y=ts.values * scale, status=ts.status,
                emphasised=probe.id == selected_id,
            ))
            plotted.append((probe, quantity, statistic))

        points_only = all(p.kind == "point" for p, _, _ in plotted)
        self.chart.plot_curves(
            curves, x_label=self.x_title(),
            y_label=text.y_title(quantity, statistic, unit, points_only),
            integer_x=not self._time_axis(), cursor_x=self.x_of(ctx.frame),
        )
        self.plotted, self.notes = plotted, notes

    @staticmethod
    def line_probe(ctx: ChartContext) -> Probe | None:
        """The selected line, else the newest visible one."""
        selected = ctx.selected
        if selected is not None and selected.kind == "line" and selected.visible:
            return selected
        lines = [p for p in ctx.probes if p.kind == "line" and p.visible]
        return lines[-1] if lines else None

    def _draw_line(self, ctx: ChartContext, view: str) -> None:
        quantity = self.quantity()
        self.plotted, self.notes = [], {}
        reason = self._unavailable(ctx, quantity, need_probes=False)
        if reason is None and quantity.is_gauge:
            reason = QCoreApplication.translate(
                "AnalysisTab", "A line view shows a field. Choose a field to plot.")
        probe = self.line_probe(ctx) if reason is None else None
        if reason is None and probe is None:
            reason = QCoreApplication.translate(
                "AnalysisTab",
                "Place a line probe, or select one, to see the field along it.")
        if reason is not None:
            self.placeholder(reason)
            return

        kymo = ctx.kymograph(probe, quantity.name)
        deformed = slice(1, None) if len(kymo.frames) > 1 else slice(None)
        if not np.isfinite(kymo.values[deformed]).any():
            self.placeholder(text.empty_line_message(probe.label, kymo.status[deformed]))
            return
        scale, unit = self.display_scale(quantity, None, ctx.length_unit)
        values = kymo.values * scale                       # [frame, sample]
        consumed = kymo.status == SampleStatus.CONSUMED
        title = text.field_title(quantity.name)
        value_label = f"{title} ({unit})" if unit else title
        distance_label = tr_args(
            QCoreApplication.translate("AnalysisTab", "Distance along %1 (%2)"),
            probe.label, kymo.distance_unit)
        crack = status_label(FrameStatus.CRACK)
        if view == "profile":
            frame = min(ctx.frame, len(kymo.frames) - 1)
            others = ([values[i] for i in other_frames(len(kymo.frames), frame)]
                      if self.other_frames_box.isChecked() else [])
            self.chart.plot_profile(
                kymo.distance, values[frame], colour=probe.color,
                label=tr_args(QCoreApplication.translate("AnalysisTab", "%1, frame %2"),
                              probe.label, frame + 1),
                x_label=distance_label, y_label=value_label,
                others=others, consumed=consumed[frame], consumed_label=crack,
            )
        else:
            state = ctx.field_state(quantity.name)
            vmin = None if state.auto else float(state.vmin) * scale
            vmax = None if state.auto else float(state.vmax) * scale
            self.chart.plot_kymograph(
                values.T, x=self.x_values(kymo.frames), distance=kymo.distance,
                x_label=self.x_title(), y_label=distance_label,
                value_label=value_label, colormap=state.colormap,
                vmin=vmin, vmax=vmax, consumed=consumed.T, consumed_label=crack,
                integer_x=not self._time_axis(), cursor_x=self.x_of(ctx.frame),
            )
        self.line_shown = (probe, quantity.name)


__all__ = ["AnalysisChartPanel", "ChartContext"]
