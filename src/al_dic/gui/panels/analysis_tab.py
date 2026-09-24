"""The strain window's Analysis tab: probes on the left, curves on the right.

What a chart shows is one *quantity* -- a field with a statistic, or a gauge
reading such as extensometer strain -- and every visible probe that can
produce it is plotted. Series of one quantity share a y-axis honestly; a
strain and a displacement never do. (The first version plotted one probe
*kind* at a time, which forbade the commonest validation plot, an
extensometer against a region average, while protecting nothing the quantity
rule does not.)

Work happens only while the tab is on screen. The strain window is a
singleton that is never destroyed, and the first version extracted curves on
every results change even while the window was closed. Curves are cached per
probe geometry and quantity, so renaming, recolouring, hiding or showing a
probe, and moving the frame cursor, redraw without reading the run again.

The canvas shows the field under the probes, in the reference configuration
their coordinates live in, drawn by the Strain Field tab's own renderer.

Three views share the controls: every probe over time; the field along one
line on the current frame, over the other frames in grey; and the same line
on every frame at once as a kymograph. The line views read the selected line
(or the newest one) and one field -- a gauge reading has no profile.
"""

from __future__ import annotations

import html
import logging
import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from al_dic.analysis.engine import (
    DEFAULT_MIN_VALID_FRACTION,
    AnalysisEngine,
    Kymograph,
    SampleStatus,
)
from al_dic.analysis.probes import (
    GAUGE_QUANTITIES,
    SPATIAL_STATISTICS,
    Probe,
    replace,
)
from al_dic.analysis.series import FrameStatus, TimeSeries
from al_dic.core.fields import is_strain_field
from al_dic.export.export_line import export_line_csv
from al_dic.export.export_probes import (
    ProbeSeries,
    export_probe_csv,
    run_parameters,
)
from al_dic.gui import icons
from al_dic.gui.app_state import AppState
from al_dic.gui.panels.probe_canvas import ProbeCanvas
from al_dic.gui.theme import COLORS
from al_dic.gui.widgets.colorbar_overlay import ColorbarOverlay
from al_dic.gui.widgets.double_spin import LocaleSafeDoubleSpinBox
from al_dic.gui.widgets.mpl_chart import Curve, MplChart, status_label
from al_dic.gui.widgets.strain_navigator import StrainNavigator
from al_dic.i18n import tr_args
from al_dic.utils.locale_format import format_number

logger = logging.getLogger(__name__)

#: Placement tools in toolbar order, with their icons.
_TOOLS = (
    ("point", icons.icon_probe_point),
    ("line", icons.icon_probe_line),
    ("area_rect", icons.icon_probe_rect),
    ("area_circle", icons.icon_probe_circle),
    ("area_polygon", icons.icon_probe_polygon),
    ("extensometer", icons.icon_extensometer),
    ("crack_gauge", icons.icon_crack_gauge),
)

#: What a gauge tool plots once it has placed its line, and the readings that
#: already count as its own -- placing a second extensometer keeps a chart of
#: elongation rather than resetting it to strain.
_GAUGE_TOOLS = {
    "extensometer": ("strain", frozenset({"strain", "true_strain", "elongation"})),
    "crack_gauge": ("cod", frozenset({"cod", "cod_sliding", "cod_magnitude"})),
}

#: The chart's views, in tab order.
_VIEWS = ("time", "profile", "kymograph")

#: A profile draws at most this many other frames: past a dozen grey lines
#: the family stops being readable.
_MAX_OTHER_FRAMES = 12

#: Fields a probe can read, in the Strain Field tab's order.
_FIELDS = (
    "disp_u", "disp_v", "disp_magnitude",
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
)

#: Gauge quantities that are strains (and so take the strain display unit).
_STRAIN_GAUGES = frozenset({"strain", "true_strain"})

#: Display scale and unit for dimensionless strain.
_STRAIN_UNITS = {"ratio": (1.0, ""), "percent": (100.0, "%"),
                 "microstrain": (1e6, "µε")}


@dataclass(frozen=True)
class _Quantity:
    """What the chart is plotting: a field, or a gauge reading."""

    kind: str            # "field" | "gauge"
    name: str            # field name, or gauge quantity

    @property
    def is_gauge(self) -> bool:
        return self.kind == "gauge"

    @property
    def key(self) -> str:
        """Combo item data. A string, because QComboBox.findData compares
        Python objects by identity, so an equal dataclass is never found."""
        return f"{self.kind}:{self.name}"

    @staticmethod
    def from_key(key) -> "_Quantity | None":
        if not isinstance(key, str) or ":" not in key:
            return None
        kind, name = key.split(":", 1)
        return _Quantity(kind, name)


class AnalysisTab(QWidget):
    """Probe placement, the probe list, and the curves they produce."""

    #: The user picked a frame here (navigator or a click on the chart).
    frame_requested = Signal(int)

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._selected_id: int | None = None
        self._updating = False
        self._frame = 0
        self._dirty = True
        self._background_loaded = False
        self._field_chosen = False
        self._engine: AnalysisEngine | None = None
        self._engine_key: tuple | None = None
        self._cache: dict[tuple, TimeSeries] = {}
        self._kymo_cache: dict[tuple, Kymograph] = {}
        self._line_shown: tuple[Probe, str] | None = None
        self._plotted: list[tuple[Probe, _Quantity, str]] = []
        self._notes: dict[int, str] = {}
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

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_canvas_side())
        splitter.addWidget(self._build_chart_side())
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

    def _build_canvas_side(self) -> QWidget:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        bar = QWidget()
        bar.setObjectName("analysisToolBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(4)
        # Icons, named by tooltip: seven text buttons do not fit beside the
        # chart in German. The armed tool's name and instructions show on
        # the canvas while it is armed.
        self._tool_buttons: dict[str, QPushButton] = {}
        for tool, icon in _TOOLS:
            if tool == "extensometer":
                row.addWidget(_separator())
            button = _icon_button(icon())
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, t=tool: self._on_tool_clicked(t, checked)
            )
            row.addWidget(button)
            self._tool_buttons[tool] = button
        row.addWidget(_separator())
        self._fit_btn = _icon_button(icons.icon_maximize())
        self._zoom_100_btn = QPushButton()
        self._zoom_in_btn = _icon_button(icons.icon_zoom_in())
        self._zoom_out_btn = _icon_button(icons.icon_zoom_out())
        for button in (self._fit_btn, self._zoom_100_btn,
                       self._zoom_in_btn, self._zoom_out_btn):
            row.addWidget(button)
        row.addStretch()
        self._show_field_box = QCheckBox()
        self._show_field_box.setChecked(True)
        self._show_field_box.toggled.connect(self._request_overlay)
        row.addWidget(self._show_field_box)
        column.addWidget(bar)

        self._canvas = ProbeCanvas()
        self._canvas.probe_requested.connect(self._on_probe_placed)
        self._canvas.placement_cancelled.connect(self._on_placement_cancelled)
        self._canvas.probe_selected.connect(self._on_canvas_selected)
        self._canvas.probe_edited.connect(self._on_probe_edited)
        self._canvas.probe_activated.connect(self._on_rename)
        self._canvas.set_length_format(self._format_length)
        self._fit_btn.clicked.connect(self._canvas.fit_to_view)
        self._zoom_100_btn.clicked.connect(self._canvas.zoom_to_100)
        self._zoom_in_btn.clicked.connect(self._canvas.zoom_in)
        self._zoom_out_btn.clicked.connect(self._canvas.zoom_out)
        column.addWidget(self._canvas, 1)

        viewport = self._canvas.viewport()
        self._colorbar = ColorbarOverlay(viewport)
        self._banner = QLabel(viewport)
        self._banner.setTextFormat(Qt.TextFormat.RichText)
        self._banner.setWordWrap(True)
        self._banner.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._banner.setStyleSheet(
            f"background: rgba(20, 25, 41, 215); color: {COLORS.TEXT_PRIMARY};"
            " border-radius: 4px; padding: 4px 8px; font-size: 11px;")
        self._banner.hide()
        viewport.installEventFilter(self)

        self._nav = StrainNavigator()
        self._nav.frame_changed.connect(self._on_nav_frame)
        column.addWidget(self._nav)
        return holder

    def _build_chart_side(self) -> QWidget:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        self._view_bar = QTabBar()
        self._view_bar.setExpanding(False)
        self._view_bar.setDrawBase(False)
        for key in _VIEWS:
            self._view_bar.setTabData(self._view_bar.addTab(""), key)
        column.addWidget(self._view_bar)

        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self._quantity_label = QLabel()
        self._quantity_box = QComboBox()
        self._quantity_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._statistic_label = QLabel()
        self._statistic_box = QComboBox()
        self._statistic_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        for w in (self._quantity_label, self._quantity_box,
                  self._statistic_label, self._statistic_box):
            row1.addWidget(w)
        row1.addStretch()
        self._export_btn = QToolButton()
        self._export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._export_menu = QMenu(self._export_btn)
        self._export_csv_action = self._export_menu.addAction("")
        self._export_csv_action.triggered.connect(self._on_export_csv)
        self._export_line_action = self._export_menu.addAction("")
        self._export_line_action.triggered.connect(self._on_export_line_csv)
        self._export_chart_action = self._export_menu.addAction("")
        self._export_chart_action.triggered.connect(self._on_export_chart)
        self._export_btn.setMenu(self._export_menu)
        row1.addWidget(self._export_btn)
        column.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self._x_label = QLabel()
        self._x_box = QComboBox()
        self._unit_label = QLabel()
        self._unit_box = QComboBox()
        self._threshold_label = QLabel()
        # Every numeric input in gui/ goes through the locale-safe subclass.
        self._threshold = LocaleSafeDoubleSpinBox()
        self._threshold.setRange(0.0, 1.0)
        self._threshold.setSingleStep(0.05)
        self._threshold.setValue(DEFAULT_MIN_VALID_FRACTION)
        self._other_frames_box = QCheckBox()
        self._other_frames_box.setChecked(True)
        for w in (self._x_label, self._x_box, self._unit_label, self._unit_box,
                  self._threshold_label, self._threshold, self._other_frames_box):
            row2.addWidget(w)
        row2.addStretch()
        column.addLayout(row2)

        split = QSplitter(Qt.Orientation.Vertical)
        self._chart = MplChart()
        self._chart.x_clicked.connect(self._on_chart_clicked)
        split.addWidget(self._chart)
        probe_panel = self._build_probe_panel()
        # Four rows and the buttons stay visible: at a laptop's 728 px the
        # stretch factors alone squeezed the list to a single row.
        rows = self._table.fontMetrics().height() + 12
        probe_panel.setMinimumHeight(4 * rows + 2 * rows)
        split.addWidget(probe_panel)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 6 * rows])
        split.setChildrenCollapsible(False)
        column.addWidget(split, 1)

        self._quantity_box.currentIndexChanged.connect(self._on_quantity_changed)
        self._statistic_box.currentIndexChanged.connect(self._request_refresh)
        self._x_box.currentIndexChanged.connect(self._request_refresh)
        self._unit_box.currentIndexChanged.connect(self._request_refresh)
        self._threshold.valueChanged.connect(self._request_refresh)
        self._other_frames_box.toggled.connect(self._request_refresh)
        # Last: addTab() above already emitted currentChanged, before the
        # controls this slot touches existed.
        self._view_bar.currentChanged.connect(self._on_view_changed)
        return holder

    def _build_probe_panel(self) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        self._table = QTableWidget(0, 5)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed)
        self._table.verticalHeader().setVisible(False)
        self._table.setIconSize(QSize(14, 14))
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)
        column.addWidget(self._table, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        self._colour_btn = QPushButton()
        self._colour_btn.clicked.connect(self._on_recolour)
        self._delete_btn = QPushButton()
        self._delete_btn.clicked.connect(self._on_delete)
        self._clear_btn = QPushButton()
        self._clear_btn.clicked.connect(self._on_clear)
        for b in (self._colour_btn, self._delete_btn, self._clear_btn):
            buttons.addWidget(b)
        buttons.addStretch()
        column.addLayout(buttons)
        return panel

    # -- translation ------------------------------------------------------

    def _tool_name(self, tool: str) -> str:
        names = {
            "point": self.tr("Point", "Placement tool: a single location"),
            "line": self.tr("Line", "Placement tool: a two-point gauge"),
            "area_rect": self.tr("Rectangle", "Placement tool"),
            "area_circle": self.tr("Circle", "Placement tool"),
            "area_polygon": self.tr("Polygon", "Placement tool"),
            "extensometer": self.tr("Virtual extensometer", "Placement tool"),
            "crack_gauge": self.tr(
                "Crack gauge", "Placement tool: a line across a crack"),
        }
        return names.get(tool, tool)

    def _tool_instruction(self, tool: str) -> str:
        instructions = {
            "point": self.tr("Click once to place a point probe."),
            "line": self.tr(
                "Click twice: start and end. A line is also a virtual "
                "extensometer and a crack-opening gauge."
            ),
            "area_rect": self.tr("Click twice: opposite corners."),
            "area_circle": self.tr("Click twice: centre, then the edge."),
            "area_polygon": self.tr(
                "Click each vertex, then double-click to close."
            ),
            "extensometer": self.tr(
                "Click the two gauge points. The chart then shows the strain "
                "between them."
            ),
            "crack_gauge": self.tr(
                "Click one point on each side of the crack. The chart then "
                "shows how far it opens."
            ),
        }
        return instructions.get(tool, "")

    def retranslate_ui(self) -> None:
        for tool, button in self._tool_buttons.items():
            # Name and instructions are two sentences, one per line.
            button.setToolTip(
                f"{self._tool_name(tool)}\n{self._tool_instruction(tool)}")
            if button.icon().isNull():          # no SVG support: say it
                button.setText(self._tool_name(tool))
        for button, text in (
            (self._fit_btn, self.tr("Fit", "Zoom button: fit the image to the view")),
            (self._zoom_in_btn, "+"), (self._zoom_out_btn, "–"),
        ):
            if button.icon().isNull():
                button.setText(text)
        self._fit_btn.setToolTip(self.tr("Fit image to viewport"))
        self._zoom_100_btn.setText(self.tr("100%", "Zoom button: one image pixel per screen pixel"))
        self._zoom_100_btn.setToolTip(self.tr("Zoom to 100% (1:1)"))
        self._zoom_in_btn.setToolTip(self.tr("Zoom in"))
        self._zoom_out_btn.setToolTip(self.tr("Zoom out"))
        self._show_field_box.setText(
            self.tr("Show field", "Analysis canvas: colour the image by the field"))
        self._show_field_box.setToolTip(self.tr(
            "Colour the reference image with the plotted field at the current "
            "frame. For a gauge reading, the Strain Field tab's field is shown."
        ))
        self._update_banner()

        self._table.setHorizontalHeaderLabels([
            self.tr("Show", "Probe list column: visibility checkbox"),
            self.tr("Name", "Probe list column: the probe's label"),
            self.tr("Type", "Probe list column: point, line or region"),
            self.tr("Colour", "Probe list column: colour swatch"),
            self.tr("Note", "Probe list column: why a probe shows gaps"),
        ])
        self._colour_btn.setText(self.tr("Colour…"))
        self._delete_btn.setText(self.tr("Delete", "Button: delete the selected probe"))
        self._clear_btn.setText(self.tr("Clear All"))
        self._quantity_label.setText(self.tr("Plot:"))
        self._statistic_label.setText(self.tr("Statistic:"))
        self._x_label.setText(self.tr("X axis:"))
        self._unit_label.setText(self.tr("Strain as:"))
        self._threshold_label.setText(self.tr("Min. valid fraction:"))
        self._threshold.setToolTip(self.tr(
            "A frame is left blank when fewer than this fraction of a line's "
            "or region's points are reliable. Guards against a curve that "
            "stays smooth while its sample shrinks away."
        ))
        views = {
            "time": (self.tr("Over time", "Chart view: every frame of each probe"),
                     self.tr("Each probe's reading at every frame.")),
            "profile": (self.tr("Along the line", "Chart view: a profile"),
                        self.tr("The field along the selected line at the "
                                "current frame, over the other frames in grey.")),
            "kymograph": (self.tr("Kymograph", "Chart view: distance against frame"),
                          self.tr("The field along the selected line at every "
                                  "frame: distance against frame, value as colour.")),
        }
        for index in range(self._view_bar.count()):
            text, tip = views[self._view_bar.tabData(index)]
            self._view_bar.setTabText(index, text)
            self._view_bar.setTabToolTip(index, tip)
        self._other_frames_box.setText(self.tr("Other frames"))
        self._other_frames_box.setToolTip(self.tr(
            "Draw the other frames' profiles faintly behind the current one "
            "(at most twelve, evenly spaced)."
        ))
        self._export_line_action.setText(self.tr("Line data (CSV)…"))
        self._export_btn.setText(self.tr("Export"))
        self._export_csv_action.setText(self.tr("Probe data (CSV)…"))
        self._export_chart_action.setText(self.tr("Chart image…"))

        self._populate_quantities()
        self._populate_statistics()
        self._populate_x_axes()
        self._populate_strain_units()
        self._refresh_table()
        self._request_refresh()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)

    def _field_title(self, name: str) -> str:
        titles = {
            "disp_u": self.tr("Displacement U"),
            "disp_v": self.tr("Displacement V"),
            "disp_magnitude": self.tr("Displacement magnitude"),
            "strain_exx": "εxx",
            "strain_eyy": "εyy",
            "strain_exy": "εxy",
            "strain_principal_max": "ε₁",
            "strain_principal_min": "ε₂",
            "strain_maxshear": "γ max",
            "strain_von_mises": "von Mises",
            "strain_rotation": "ω rot",
        }
        return titles.get(name, name)

    def _gauge_title(self, name: str) -> str:
        titles = {
            "strain": self.tr("Extensometer strain"),
            "true_strain": self.tr("Extensometer true strain"),
            "elongation": self.tr("Elongation ΔL"),
            "cod": self.tr("Crack opening"),
            "cod_sliding": self.tr("Crack sliding"),
            "cod_magnitude": self.tr("Crack opening magnitude"),
        }
        return titles.get(name, name)

    def _statistic_title(self, name: str) -> str:
        titles = {
            "mean": self.tr("Mean", "Statistic: arithmetic mean"),
            "median": self.tr("Median", "Statistic"),
            "max": self.tr("Maximum", "Statistic"),
            "min": self.tr("Minimum", "Statistic"),
            "std": self.tr("Standard deviation"),
            "valid_fraction": self.tr("Valid fraction"),
        }
        return titles.get(name, name)

    def _populate_quantities(self) -> None:
        self._updating = True
        current = self._quantity_box.currentData()
        self._quantity_box.clear()
        for name in _FIELDS:
            self._quantity_box.addItem(self._field_title(name),
                                       _Quantity("field", name).key)
        self._quantity_box.insertSeparator(self._quantity_box.count())
        for name in GAUGE_QUANTITIES:
            self._quantity_box.addItem(self._gauge_title(name),
                                       _Quantity("gauge", name).key)
        index = self._quantity_box.findData(current) if current else -1
        self._quantity_box.setCurrentIndex(index if index >= 0 else 0)
        self._updating = False

    def _populate_statistics(self) -> None:
        self._updating = True
        current = self._statistic_box.currentData() or "mean"
        self._statistic_box.clear()
        for name in SPATIAL_STATISTICS:
            self._statistic_box.addItem(self._statistic_title(name), name)
        self._statistic_box.setCurrentIndex(
            max(self._statistic_box.findData(current), 0))
        self._updating = False

    def _populate_x_axes(self) -> None:
        self._updating = True
        current = self._x_box.currentData() or "frame"
        self._x_box.clear()
        self._x_box.addItem(self.tr("Frame"), "frame")
        if self._frame_rate() > 0:
            self._x_box.addItem(self.tr("Time (s)"), "time")
        self._x_box.setCurrentIndex(max(self._x_box.findData(current), 0))
        self._updating = False

    def _populate_strain_units(self) -> None:
        self._updating = True
        current = self._unit_box.currentData() or "ratio"
        self._unit_box.clear()
        self._unit_box.addItem(self.tr("ratio", "Strain display unit: plain number"),
                               "ratio")
        self._unit_box.addItem("%", "percent")
        self._unit_box.addItem("µε", "microstrain")
        self._unit_box.setCurrentIndex(max(self._unit_box.findData(current), 0))
        self._updating = False

    # -- wiring from the strain window --------------------------------------

    def set_frame(self, frame: int) -> None:
        """Follow the Strain Field tab's frame (no signal back)."""
        self._frame = max(0, int(frame))
        self._sync_navigator()
        self._follow_frame()

    def _follow_frame(self) -> None:
        """Move the chart to ``self._frame``: a cursor, or a new profile."""
        if self._view() == "profile":
            self._request_refresh()
        else:
            self._chart.set_cursor(self._x_of(self._frame))
            self._request_overlay()

    def set_default_field(self, name: str) -> None:
        """The Strain Field tab's field: where the chart starts until the user
        picks, and what the canvas shows under a gauge reading."""
        self._strain_tab_field = name
        quantity = self._current_quantity()
        if quantity is not None and quantity.is_gauge:
            self._request_overlay()
        if self._field_chosen or name not in _FIELDS:
            return
        index = self._quantity_box.findData(_Quantity("field", name).key)
        if index >= 0 and index != self._quantity_box.currentIndex():
            self._updating = True
            self._quantity_box.setCurrentIndex(index)
            self._updating = False
            self._update_control_visibility()
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
        if self._updating:
            return
        if self.isVisible():
            self._refresh()
        else:
            self._dirty = True

    def _request_overlay(self, *_args) -> None:
        if self._updating:
            return
        if self.isVisible():
            self._overlay_timer.start()
        else:
            self._dirty = True

    def _on_results_changed(self) -> None:
        self._populate_x_axes()
        self._drop_engine()

    def _on_units_changed(self) -> None:
        # The frame rate lives with the physical units, and "Time" is only
        # offered while there is one. Cached series are keyed on the pixel
        # size, so the cache needs no flushing.
        self._populate_x_axes()
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

    # -- engine and series ----------------------------------------------------

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

    def _series(self, probe: Probe, quantity: _Quantity, statistic: str,
                *, axes: str = "image") -> TimeSeries:
        engine = self._engine_for(self._state.results)
        pixel = self._pixel_size()
        key = (probe.geometry, probe.kind, quantity, statistic,
               round(self._threshold.value(), 6), pixel, self._length_unit(),
               axes)
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

    def _applies(self, probe: Probe, quantity: _Quantity, statistic: str) -> bool:
        if quantity.is_gauge:
            return probe.kind == "line"
        return AnalysisEngine.supports(probe, statistic)

    # -- refresh ----------------------------------------------------------

    def _refresh(self) -> None:
        self._dirty = False
        self._update_control_visibility()
        self._refresh_chart()
        self._refresh_table()
        self._canvas.set_probes(self._state.probes, self._selected_id)
        self._sync_navigator()
        self._update_actions()
        self._update_banner()
        self._request_overlay()

    # -- the field under the probes ------------------------------------------

    def _overlay_field(self) -> str | None:
        """The field the canvas shows: the plotted one, or for a gauge
        reading -- which has no field -- the Strain Field tab's."""
        quantity = self._current_quantity()
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
            self._canvas.clear_overlay()
            self._colorbar.setVisible(False)
            return
        self._canvas.show_field(image)
        viewport = self._canvas.viewport()
        self._colorbar.setGeometry(0, 0, viewport.width(), viewport.height())
        # Strain in the chart's unit (%, µε): the colorbar is only a legend,
        # so its numbers can be rescaled without touching the picture.
        scale, unit = self._display_scale(_Quantity("field", field), None)
        label = f"{image.label} ({unit})" if scale != 1.0 else image.label
        self._colorbar.update_params(
            image.cmap, image.vmin * scale, image.vmax * scale, label)
        self._colorbar.setVisible(True)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self._canvas.viewport() and event.type() == QEvent.Type.Resize:
            viewport = self._canvas.viewport()
            self._colorbar.setGeometry(0, 0, viewport.width(), viewport.height())
            self._place_banner()
        return super().eventFilter(obj, event)

    # -- the banner over the canvas -------------------------------------------

    def _update_banner(self) -> None:
        """Say what the canvas is waiting for: the armed tool's clicks, or how
        to edit the selected probe."""
        tool = self._canvas.tool
        if tool != "none":
            # Not inside the f-string below: lupdate cannot see a tr() call
            # there, and dropped the translation as obsolete.
            cancel = self.tr("Esc cancels placement")
            # Three sentences, a line each; nothing is spliced into another.
            text = "<br>".join((
                f"<b>{html.escape(self._tool_name(tool))}</b>",
                html.escape(self._tool_instruction(tool)),
                f"<span style='color:{COLORS.TEXT_MUTED}'>"
                f"{html.escape(cancel)}</span>",
            ))
        elif self._selected_probe() is not None:
            text = html.escape(self.tr(
                "Drag to move the probe, or drag a handle to reshape it. "
                "Delete removes it; F2 renames it."
            ))
        else:
            text = ""
        self._banner.setText(text)
        self._banner.setVisible(bool(text))
        self._place_banner()

    def _place_banner(self) -> None:
        if self._banner.isHidden():
            return
        viewport = self._canvas.viewport()
        # Clear of the colorbar on the right.
        width = max(160, min(460, viewport.width() - 130))
        self._banner.setGeometry(8, 8, width, self._banner.heightForWidth(width))
        self._banner.raise_()

    def _sync_navigator(self) -> None:
        result = self._state.results
        n = len(result.result_disp) + 1 if result is not None else 0
        self._nav.set_state(n, min(self._frame, max(n - 1, 0)))

    def _update_control_visibility(self) -> None:
        view = self._view()
        over_time = view == "time"
        quantity = self._current_quantity()
        is_field = quantity is not None and not quantity.is_gauge
        statistic = self._statistic_box.currentData() if over_time else None
        self._statistic_label.setVisible(is_field and over_time)
        self._statistic_box.setVisible(is_field and over_time)
        strainlike = self._is_strainlike(quantity, statistic)
        self._unit_label.setVisible(strainlike)
        self._unit_box.setVisible(strainlike)
        threshold = (is_field and over_time
                     and self._statistic_box.currentData() != "valid_fraction")
        self._threshold_label.setVisible(threshold)
        self._threshold.setVisible(threshold)
        self._x_label.setVisible(view != "profile")
        self._x_box.setVisible(view != "profile")
        self._other_frames_box.setVisible(view == "profile")
        self._export_csv_action.setVisible(over_time)
        self._export_line_action.setVisible(not over_time)

    @staticmethod
    def _is_strainlike(quantity: _Quantity | None, statistic: str | None) -> bool:
        if quantity is None:
            return False
        if quantity.is_gauge:
            return quantity.name in _STRAIN_GAUGES
        return (is_strain_field(quantity.name)
                and quantity.name != "strain_rotation"
                and statistic != "valid_fraction")

    def _placeholder(self, message: str) -> None:
        self._chart.clear(message)
        self._plotted = []
        self._notes = {}

    def _refresh_chart(self) -> None:
        self._line_shown = None
        if self._view() == "time":
            self._refresh_time_chart()
        else:
            self._refresh_line_chart(self._view())

    def _refresh_time_chart(self) -> None:
        result = self._state.results
        quantity = self._current_quantity()
        statistic = self._statistic_box.currentData() or "mean"
        if result is None:
            self._placeholder(self.tr("Run a DIC analysis to plot probes."))
            return
        if quantity is None:
            self._placeholder("")
            return
        probes = list(self._state.probes)
        if not probes:
            self._placeholder(
                self.tr("Place a probe on the reference image to begin."))
            return
        if (not quantity.is_gauge and is_strain_field(quantity.name)
                and not result.result_strain):
            self._placeholder(self.tr(
                "Strain has not been computed yet. Compute it on the Strain "
                "Field tab, or plot a displacement."))
            return

        shown = [p for p in probes
                 if p.visible and self._applies(p, quantity, statistic)]
        self._notes = {p.id: self._not_applicable_note(p, quantity, statistic)
                       for p in probes
                       if not self._applies(p, quantity, statistic)}
        if not shown:
            hint = (self.tr("Gauge quantities need a line probe.")
                    if quantity.is_gauge else "")
            self._placeholder(
                (self.tr("No visible probe can show this quantity.")
                 + (" " + hint if hint else "")))
            self._notes = {p.id: self._not_applicable_note(p, quantity, statistic)
                           for p in probes
                           if not self._applies(p, quantity, statistic)}
            return

        scale, unit = self._display_scale(quantity, statistic)
        curves, plotted = [], []
        for probe in shown:
            try:
                ts = self._series(probe, quantity, statistic)
            except ValueError as exc:
                self._notes[probe.id] = str(exc)
                continue
            note = self._series_note(ts)
            self._notes[probe.id] = note
            label = f"{probe.label} · {note}" if note else probe.label
            curves.append(Curve(
                label=label, colour=probe.color,
                x=self._x_values(ts.frames), y=ts.values * scale,
                status=ts.status, emphasised=probe.id == self._selected_id,
            ))
            plotted.append((probe, quantity, statistic))

        points_only = all(p.kind == "point" for p, _, _ in plotted)
        self._chart.plot_curves(
            curves,
            x_label=self._x_title(),
            y_label=self._y_title(quantity, statistic, unit, points_only),
            integer_x=self._x_box.currentData() != "time",
            cursor_x=self._x_of(self._frame),
        )
        self._plotted = plotted

    # -- line views -----------------------------------------------------------

    def _view(self) -> str:
        return self._view_bar.tabData(self._view_bar.currentIndex()) or "time"

    def _view_bar_index(self, key: str) -> int:
        for index in range(self._view_bar.count()):
            if self._view_bar.tabData(index) == key:
                return index
        raise KeyError(key)

    def _on_view_changed(self, _index: int) -> None:
        self._update_control_visibility()
        self._request_refresh()

    def _line_probe(self) -> Probe | None:
        """The selected line, else the newest visible one."""
        selected = self._selected_probe()
        if selected is not None and selected.kind == "line" and selected.visible:
            return selected
        lines = [p for p in self._state.probes if p.kind == "line" and p.visible]
        return lines[-1] if lines else None

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

    def _empty_line_message(self, probe: Probe, status: np.ndarray) -> str:
        """Why a line has nothing to show on any deformed frame.

        The reason named is the one the user can act on, not the commonest:
        a line over a hole and a trimmed ligament is mostly off the material,
        but only the trim can be changed.
        """
        if (status == SampleStatus.UNRELIABLE).any():
            text = self.tr(
                "Nothing valid along %1: its strain is trimmed as low-confidence "
                "near an edge or a hole. Plot a displacement, or trim less on "
                "the Strain Field tab.")
        elif (status == SampleStatus.CONSUMED).any():
            text = self.tr(
                "Nothing valid along %1: a crack has consumed the material "
                "under it.")
        else:
            text = self.tr("Nothing valid along %1: it lies off the measured area.")
        return tr_args(text, probe.label)

    @staticmethod
    def _other_frames(n: int, current: int) -> list[int]:
        others = [i for i in range(n) if i != current]
        if len(others) > _MAX_OTHER_FRAMES:
            picks = np.linspace(0, len(others) - 1, _MAX_OTHER_FRAMES)
            others = [others[i] for i in sorted(set(np.round(picks).astype(int)))]
        return others

    def _refresh_line_chart(self, view: str) -> None:
        result = self._state.results
        quantity = self._current_quantity()
        self._plotted, self._notes = [], {}
        if result is None:
            self._placeholder(self.tr("Run a DIC analysis to plot probes."))
            return
        if quantity is None:
            self._placeholder("")
            return
        if quantity.is_gauge:
            self._placeholder(self.tr(
                "A line view shows a field. Choose a field to plot."))
            return
        if is_strain_field(quantity.name) and not result.result_strain:
            self._placeholder(self.tr(
                "Strain has not been computed yet. Compute it on the Strain "
                "Field tab, or plot a displacement."))
            return
        probe = self._line_probe()
        if probe is None:
            self._placeholder(self.tr(
                "Place a line probe, or select one, to see the field along it."))
            return

        kymo = self._kymograph(probe, quantity.name)
        deformed = slice(1, None) if len(kymo.frames) > 1 else slice(None)
        if not np.isfinite(kymo.values[deformed]).any():
            self._placeholder(self._empty_line_message(probe, kymo.status[deformed]))
            return
        scale, unit = self._display_scale(quantity, None)
        values = kymo.values * scale                       # [frame, sample]
        consumed = kymo.status == SampleStatus.CONSUMED
        title = self._field_title(quantity.name)
        value_label = f"{title} ({unit})" if unit else title
        distance_label = tr_args(self.tr("Distance along %1 (%2)"),
                                 probe.label, kymo.distance_unit)
        crack = status_label(FrameStatus.CRACK)
        if view == "profile":
            frame = min(self._frame, len(kymo.frames) - 1)
            others = ([values[i] for i in self._other_frames(len(kymo.frames), frame)]
                      if self._other_frames_box.isChecked() else [])
            self._chart.plot_profile(
                kymo.distance, values[frame], colour=probe.color,
                label=tr_args(self.tr("%1, frame %2"), probe.label, frame + 1),
                x_label=distance_label, y_label=value_label,
                others=others, consumed=consumed[frame], consumed_label=crack,
            )
        else:
            state = self._state.get_field_state(quantity.name)
            vmin = None if state.auto else float(state.vmin) * scale
            vmax = None if state.auto else float(state.vmax) * scale
            self._chart.plot_kymograph(
                values.T, x=self._x_values(kymo.frames), distance=kymo.distance,
                x_label=self._x_title(), y_label=distance_label,
                value_label=value_label, colormap=state.colormap,
                vmin=vmin, vmax=vmax, consumed=consumed.T, consumed_label=crack,
                integer_x=self._x_box.currentData() != "time",
                cursor_x=self._x_of(self._frame),
            )
        self._line_shown = (probe, quantity.name)

    def _display_scale(self, quantity: _Quantity, statistic: str | None) -> tuple[float, str]:
        if self._is_strainlike(quantity, statistic):
            return _STRAIN_UNITS[self._unit_box.currentData() or "ratio"]
        if quantity.is_gauge:
            return 1.0, self._length_unit()
        if statistic == "valid_fraction":
            return 1.0, ""
        if quantity.name == "strain_rotation":
            return 1.0, "°"
        if is_strain_field(quantity.name):
            return 1.0, ""
        return 1.0, self._length_unit()

    def _y_title(self, quantity: _Quantity, statistic: str, unit: str,
                 points_only: bool) -> str:
        if quantity.is_gauge:
            text = self._gauge_title(quantity.name)
        elif points_only and statistic in ("mean", "median", "max", "min"):
            # A point has one value; calling it a mean would be noise.
            text = self._field_title(quantity.name)
        else:
            text = (f"{self._field_title(quantity.name)} — "
                    f"{self._statistic_title(statistic)}")
        return f"{text} ({unit})" if unit else text

    def _series_note(self, ts: TimeSeries) -> str:
        """Why a curve has gaps or marks, for the legend and the probe list."""
        # Nothing to measure even on the reference frame: the probe was put
        # off the specimen. Say that, rather than blame the run.
        first = ts.status[0] if len(ts.status) else None
        if first is FrameStatus.NO_DATA:
            return self.tr("not plotted: off the measured area")
        if first is FrameStatus.ENDPOINT_LOST:
            return self.tr("not plotted: a gauge end is off the measured area")
        # Frame 0 is the reference: a displacement is 0 there by definition,
        # so it says nothing about whether the probe ever measures.
        values = ts.values[1:] if len(ts.values) > 1 else ts.values
        statuses = list(ts.status[1:]) if len(ts.status) > 1 else list(ts.status)
        if not np.isfinite(values).any():
            reasons = [st for st in statuses if st is not FrameStatus.OK]
            if reasons:
                common = max(set(reasons), key=reasons.count)
                return tr_args(self.tr("no valid data: %1"), status_label(common))
        crack = ts.first_frame(FrameStatus.CRACK)
        if crack is not None:
            return tr_args(self.tr("crack from frame %1"), crack + 1)
        lost = ts.first_frame(FrameStatus.ENDPOINT_LOST)
        if lost is not None:
            return tr_args(self.tr("endpoint lost from frame %1"), lost + 1)
        if any(st is FrameStatus.BELOW_THRESHOLD for st in ts.status):
            return self.tr("gaps: too few valid points")
        if any(st is FrameStatus.UNRELIABLE for st in ts.status):
            return self.tr("gaps: unreliable strain")
        return ""

    def _not_applicable_note(self, probe: Probe, quantity: _Quantity,
                             statistic: str) -> str:
        if quantity.is_gauge:
            return self.tr("not plotted: gauges need a line")
        if probe.kind == "point":
            return self.tr("not plotted: one point has no spread or coverage")
        return self.tr("not plotted")

    # -- x axis -----------------------------------------------------------

    def _frame_rate(self) -> float:
        rate = float(getattr(self._state, "frame_rate", 0.0) or 0.0)
        return rate if rate > 0 else 0.0

    def _x_values(self, frames: np.ndarray) -> np.ndarray:
        if self._x_box.currentData() == "time" and self._frame_rate() > 0:
            return frames.astype(float) / self._frame_rate()
        return frames.astype(float) + 1.0            # 1-based, as the navigator

    def _x_of(self, frame: int) -> float:
        return float(self._x_values(np.array([frame]))[0])

    def _x_title(self) -> str:
        if self._x_box.currentData() == "time" and self._frame_rate() > 0:
            return self.tr("Time (s)")
        return self.tr("Frame")

    def _frame_from_x(self, x: float) -> int:
        if self._x_box.currentData() == "time" and self._frame_rate() > 0:
            frame = int(round(x * self._frame_rate()))
        else:
            frame = int(round(x)) - 1
        result = self._state.results
        n = len(result.result_disp) + 1 if result is not None else 1
        return max(0, min(frame, n - 1))

    # -- frame --------------------------------------------------------------

    def _on_nav_frame(self, frame: int) -> None:
        self._frame = int(frame)
        self._follow_frame()
        self.frame_requested.emit(self._frame)

    def _on_chart_clicked(self, x: float) -> None:
        if self._view() == "profile":
            return                      # x is a distance there, not a frame
        frame = self._frame_from_x(x)
        self._frame = frame
        self._sync_navigator()
        self._follow_frame()
        self.frame_requested.emit(frame)

    # -- quantity -----------------------------------------------------------

    def _current_quantity(self) -> _Quantity | None:
        return _Quantity.from_key(self._quantity_box.currentData())

    def _on_quantity_changed(self) -> None:
        if self._updating:
            return
        self._field_chosen = True
        self._update_control_visibility()
        self._request_refresh()

    # -- probe lifecycle ------------------------------------------------------

    def _on_tool_clicked(self, tool: str, checked: bool) -> None:
        # Remembered here: the canvas reports a gauge tool's line as a plain
        # "line", and what to plot next depends on which tool drew it.
        self._armed_tool = tool if checked else None
        self._canvas.set_tool(tool if checked else "none")  # type: ignore[arg-type]
        self._sync_tool_buttons()
        self._update_banner()

    def _on_placement_cancelled(self) -> None:
        self._armed_tool = None
        self._sync_tool_buttons()
        self._update_banner()

    def _sync_tool_buttons(self) -> None:
        active = self._canvas.tool
        for tool, button in self._tool_buttons.items():
            button.setChecked(tool == active)

    def _on_probe_placed(self, kind: str, geometry) -> None:
        tool, self._armed_tool = self._armed_tool, None
        probe = self._state.probes.add(kind, geometry)  # type: ignore[arg-type]
        self._selected_id = probe.id
        self._sync_tool_buttons()
        self._adopt_gauge_reading(tool)
        self._refresh()
        self._say(tr_args(self.tr("Added probe '%1'."), probe.label))

    def _adopt_gauge_reading(self, tool: str | None) -> None:
        """Plot what a gauge tool is for, unless the chart already shows a
        reading of that family."""
        if tool not in _GAUGE_TOOLS:
            return
        default, family = _GAUGE_TOOLS[tool]
        current = self._current_quantity()
        if current is not None and current.is_gauge and current.name in family:
            return
        index = self._quantity_box.findData(_Quantity("gauge", default).key)
        if index < 0:
            return
        self._updating = True
        self._quantity_box.setCurrentIndex(index)
        self._updating = False
        self._field_chosen = True
        self._update_control_visibility()

    def _on_canvas_selected(self, probe_id) -> None:
        self._selected_id = probe_id
        self._select_row(probe_id)
        self._update_actions()
        self._update_banner()
        # Emphasis only: every series is cached, so this is a redraw.
        self._refresh_chart()

    def _select_row(self, probe_id: int | None) -> None:
        self._updating = True
        self._table.clearSelection()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 1)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == probe_id:
                self._table.selectRow(row)
                self._table.scrollToItem(item)
                break
        self._updating = False

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
        if probe_id is None:
            return
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 1)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == probe_id:
                self._table.setCurrentItem(item)
                self._table.editItem(item)
                return

    def _selected_probe(self) -> Probe | None:
        if self._selected_id is None:
            return None
        try:
            return self._state.probes.get(self._selected_id)
        except KeyError:
            return None

    def _on_selection_changed(self) -> None:
        if self._updating:
            return
        rows = self._table.selectionModel().selectedRows()
        self._selected_id = (
            self._table.item(rows[0].row(), 1).data(Qt.ItemDataRole.UserRole)
            if rows else None
        )
        self._canvas.set_probes(self._state.probes, self._selected_id)
        self._update_actions()
        self._update_banner()
        # Emphasis only: every series is cached, so this is a redraw.
        self._refresh_chart()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        probe_id = self._table.item(item.row(), 1).data(Qt.ItemDataRole.UserRole)
        try:
            probe = self._state.probes.get(probe_id)
        except KeyError:
            return
        if item.column() == 0:
            visible = item.checkState() == Qt.CheckState.Checked
            self._state.probes.replace(replace(probe, visible=visible))
        elif item.column() == 1:
            text = " ".join(item.text().split())
            if text and text != probe.label:
                self._state.probes.replace(replace(probe, label=text))
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

    def _update_actions(self) -> None:
        has_selection = self._selected_probe() is not None
        self._colour_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)
        self._clear_btn.setEnabled(len(self._state.probes) > 0)
        shown = self._plotted if self._view() == "time" else self._line_shown
        self._export_btn.setEnabled(self._chart.has_data and bool(shown))

    # -- table ----------------------------------------------------------------

    def _refresh_table(self) -> None:
        self._updating = True
        kinds = {
            "point": self.tr("Point", "Probe type"),
            "line": self.tr("Line", "Probe type"),
            "area": self.tr("Region", "Probe type: an enclosed area"),
        }
        probes = list(self._state.probes)
        self._table.setRowCount(len(probes))
        for row, probe in enumerate(probes):
            show = QTableWidgetItem()
            show.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            show.setCheckState(
                Qt.CheckState.Checked if probe.visible else Qt.CheckState.Unchecked)
            self._table.setItem(row, 0, show)

            name = QTableWidgetItem(probe.label)
            name.setData(Qt.ItemDataRole.UserRole, probe.id)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                          | Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 1, name)

            kind = QTableWidgetItem(kinds.get(probe.kind, probe.kind))
            kind.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._table.setItem(row, 2, kind)

            # An icon, not a background: a selected row's highlight painted
            # over the background and hid the swatch.
            swatch = QTableWidgetItem()
            pixmap = QPixmap(14, 14)
            pixmap.fill(QColor(probe.color))
            swatch.setIcon(QIcon(pixmap))
            swatch.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._table.setItem(row, 3, swatch)

            note = QTableWidgetItem(self._notes.get(probe.id, ""))
            note.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            note.setForeground(QColor(COLORS.TEXT_MUTED))
            self._table.setItem(row, 4, note)

            if probe.id == self._selected_id:
                self._table.selectRow(row)
        self._updating = False
        self._update_actions()

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

    def _pixel_size(self) -> float:
        state = self._state
        if state.use_physical_units and state.pixel_size > 0:
            return float(state.pixel_size)
        return 1.0

    def _length_unit(self) -> str:
        state = self._state
        return state.pixel_unit if state.use_physical_units else "px"

    def _format_length(self, px: float) -> str:
        """A gauge length for its label: three significant figures, at least
        one decimal, in the display unit."""
        value = px * self._pixel_size()
        decimals = 1
        if value != 0.0 and math.isfinite(value):
            decimals = max(1, 2 - int(math.floor(math.log10(abs(value)))))
        return f"{format_number(value, decimals)} {self._length_unit()}"

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

    def _on_export_csv(self) -> None:
        if not (self._chart.has_data and self._plotted):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export Probe Data"), "probes.csv",
            self.tr("CSV Files") + " (*.csv);;" + self.tr("All Files") + " (*)",
        )
        if not path:
            return
        entries = []
        for probe, quantity, statistic in self._plotted:
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
        rate = self._frame_rate() or None
        try:
            export_probe_csv(path, entries, frame_rate=rate,
                             parameters=self._export_parameters())
        except (OSError, ValueError) as exc:
            self._say(tr_args(self.tr("Probe export failed: %1"), exc), "error")
            return
        self._say(tr_args(self.tr("Probe data written to %1"), path))

    def _on_export_line_csv(self) -> None:
        if not (self._chart.has_data and self._line_shown):
            return
        probe, field = self._line_shown
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export Line Data"), "line_profile.csv",
            self.tr("CSV Files") + " (*.csv);;" + self.tr("All Files") + " (*)",
        )
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

    def _on_export_chart(self) -> None:
        if not self._chart.has_data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export Chart"), "probe_chart.png",
            self.tr("PNG Images") + " (*.png);;"
            + self.tr("PDF Documents") + " (*.pdf);;"
            + self.tr("All Files") + " (*)",
        )
        if not path:
            return
        try:
            self._chart.save_figure(path)
        except (OSError, ValueError) as exc:
            self._say(tr_args(self.tr("Chart export failed: %1"), exc), "error")
            return
        self._say(tr_args(self.tr("Chart written to %1"), path))


def _icon_button(icon: QIcon) -> QPushButton:
    button = QPushButton()
    button.setIcon(icon)
    button.setIconSize(QSize(18, 18))
    if not icon.isNull():
        button.setFixedWidth(32)        # an icon alone has nothing to translate
    return button


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setStyleSheet(f"color: {COLORS.BORDER};")
    return line


__all__ = ["AnalysisTab"]
