"""The strain window's Analysis tab: probes on the left, curves on the right.

Probes are placed on the reference image and read across the sequence. What
appears here is the same data the field view shows, read through the same
crack-aware sampler, so a curve and a picture cannot disagree.

Only probes of one kind are charted at a time. Their reductions share a Y axis,
and a strain, a displacement and a crack opening do not belong on one -- the
constraint is what makes the comparison honest rather than a limitation to work
around.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from al_dic.analysis.extract import DEFAULT_MIN_VALID_FRACTION, extract_series
from al_dic.analysis.probes import Probe, allowed_reductions, replace
from al_dic.core.fields import DISP_FIELDS, STRAIN_FIELDS
from al_dic.export.export_probes import ProbeSeries, export_probe_csv
from al_dic.gui.app_state import AppState
from al_dic.gui.panels.probe_canvas import ProbeCanvas
from al_dic.gui.theme import COLORS
from al_dic.gui.widgets.double_spin import LocaleSafeDoubleSpinBox
from al_dic.gui.widgets.mpl_chart import MplChart

#: Tool button label, tool token, and the probe kind it produces.
_TOOLS = (
    ("point", "point"),
    ("line", "line"),
    ("area_rect", "area"),
    ("area_circle", "area"),
    ("area_polygon", "area"),
)


class AnalysisTab(QWidget):
    """Probe placement, the probe list, and the curves they produce."""

    frame_requested = Signal(int)

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._selected_id: int | None = None
        self._updating = False
        self._last_entries: list[ProbeSeries] = []
        # Reuses the main image path, so this canvas and the field view agree
        # about orientation, bit depth and normalisation.
        from al_dic.gui.controllers.image_controller import ImageController

        self._image_ctrl = ImageController(state)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_canvas_side())
        splitter.addWidget(self._build_chart_side())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(splitter)

        state.results_changed.connect(self._refresh_all)
        state.images_changed.connect(self._refresh_background)
        state.physical_units_changed.connect(self._refresh_chart)

        self.retranslate_ui()
        self._refresh_all()

    # -- construction -----------------------------------------------------

    def _build_canvas_side(self) -> QWidget:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet(
            f"background: {COLORS.BG_PANEL}; "
            f"border-bottom: 1px solid {COLORS.BORDER};"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(4)
        self._tool_buttons: dict[str, QPushButton] = {}
        for tool, _kind in _TOOLS:
            button = QPushButton()
            button.setCheckable(True)
            button.setFixedHeight(26)
            button.clicked.connect(
                lambda checked, t=tool: self._on_tool_clicked(t, checked)
            )
            row.addWidget(button)
            self._tool_buttons[tool] = button
        row.addStretch()
        self._hint = QLabel()
        self._hint.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 11px;")
        row.addWidget(self._hint)
        column.addWidget(bar)

        self._canvas = ProbeCanvas()
        self._canvas.probe_requested.connect(self._on_probe_placed)
        self._canvas.placement_cancelled.connect(self._sync_tool_buttons)
        column.addWidget(self._canvas, 1)
        return holder

    def _build_chart_side(self) -> QWidget:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        self._table = QTableWidget(0, 4)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setMaximumHeight(170)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)
        column.addWidget(self._table)

        actions = QHBoxLayout()
        actions.setSpacing(4)
        self._colour_btn = QPushButton()
        self._colour_btn.clicked.connect(self._on_recolour)
        self._delete_btn = QPushButton()
        self._delete_btn.clicked.connect(self._on_delete)
        self._clear_btn = QPushButton()
        self._clear_btn.clicked.connect(self._on_clear)
        for b in (self._colour_btn, self._delete_btn, self._clear_btn):
            actions.addWidget(b)
        actions.addStretch()
        column.addLayout(actions)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self._kind_label = QLabel()
        self._kind_box = QComboBox()
        self._field_label = QLabel()
        self._field_box = QComboBox()
        self._reduction_label = QLabel()
        self._reduction_box = QComboBox()
        for widget in (
            self._kind_label, self._kind_box, self._field_label,
            self._field_box, self._reduction_label, self._reduction_box,
        ):
            controls.addWidget(widget)
        controls.addStretch()
        column.addLayout(controls)

        threshold_row = QHBoxLayout()
        threshold_row.setSpacing(6)
        self._threshold_label = QLabel()
        # Every numeric input in gui/ goes through the locale-safe
        # subclass; a raw QDoubleSpinBox reintroduces the OS-locale
        # parsing bug fixed in 0.7.x, and a test guards against it.
        self._threshold = LocaleSafeDoubleSpinBox()
        self._threshold.setRange(0.0, 1.0)
        self._threshold.setSingleStep(0.05)
        self._threshold.setValue(DEFAULT_MIN_VALID_FRACTION)
        self._threshold.valueChanged.connect(self._refresh_chart)
        threshold_row.addWidget(self._threshold_label)
        threshold_row.addWidget(self._threshold)
        threshold_row.addStretch()
        self._export_csv_btn = QPushButton()
        self._export_csv_btn.clicked.connect(self._on_export_csv)
        self._export_png_btn = QPushButton()
        self._export_png_btn.clicked.connect(self._on_export_chart)
        threshold_row.addWidget(self._export_csv_btn)
        threshold_row.addWidget(self._export_png_btn)
        column.addLayout(threshold_row)

        self._chart = MplChart()
        column.addWidget(self._chart, 1)

        self._kind_box.currentIndexChanged.connect(self._on_kind_changed)
        self._field_box.currentIndexChanged.connect(self._refresh_chart)
        self._reduction_box.currentIndexChanged.connect(self._refresh_chart)
        return holder

    # -- translation ------------------------------------------------------

    def retranslate_ui(self) -> None:
        labels = {
            "point": self.tr("Point", "Placement tool: a single location"),
            "line": self.tr("Line", "Placement tool: a two-point gauge"),
            "area_rect": self.tr("Rectangle", "Placement tool"),
            "area_circle": self.tr("Circle", "Placement tool"),
            "area_polygon": self.tr("Polygon", "Placement tool"),
        }
        tips = {
            "point": self.tr("Click once to place a point probe."),
            "line": self.tr("Click twice: start and end of the gauge."),
            "area_rect": self.tr("Click twice: opposite corners."),
            "area_circle": self.tr("Click twice: centre, then the edge."),
            "area_polygon": self.tr(
                "Click each vertex, then double-click to close."
            ),
        }
        for tool, button in self._tool_buttons.items():
            button.setText(labels[tool])
            button.setToolTip(tips[tool])

        self._table.setHorizontalHeaderLabels([
            self.tr("Show", "Probe list column: visibility checkbox"),
            self.tr("Name", "Probe list column: the probe's label"),
            self.tr("Type", "Probe list column: point, line or region"),
            self.tr("Colour", "Probe list column: colour swatch"),
        ])
        self._colour_btn.setText(self.tr("Colour…"))
        self._delete_btn.setText(self.tr("Delete", "Button: delete the selected probe"))
        self._clear_btn.setText(self.tr("Clear All"))
        self._kind_label.setText(self.tr("Compare:"))
        self._field_label.setText(self.tr("Field:"))
        self._reduction_label.setText(self.tr("Statistic:"))
        self._threshold_label.setText(self.tr("Minimum valid fraction:"))
        self._threshold.setToolTip(self.tr(
            "A frame is left blank when fewer than this fraction of the "
            "probe's points are reliable. Guards against a curve that stays "
            "smooth while its sample shrinks away."
        ))
        self._export_csv_btn.setText(self.tr("Export CSV…"))
        self._export_png_btn.setText(self.tr("Export Chart…"))
        self._hint.setText(self.tr("Esc cancels placement"))
        self._populate_kind_box()
        self._populate_field_box()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
            self._refresh_chart()
        super().changeEvent(event)

    def _populate_kind_box(self) -> None:
        self._updating = True
        current = self._kind_box.currentData()
        self._kind_box.clear()
        for kind, text in (
            ("point", self.tr("Point probes")),
            ("line", self.tr("Line probes")),
            ("area", self.tr("Region probes")),
        ):
            self._kind_box.addItem(text, kind)
        if current is not None:
            index = self._kind_box.findData(current)
            if index >= 0:
                self._kind_box.setCurrentIndex(index)
        self._updating = False

    def _populate_field_box(self) -> None:
        self._updating = True
        current = self._field_box.currentData()
        self._field_box.clear()
        for name in sorted(DISP_FIELDS) + sorted(STRAIN_FIELDS):
            self._field_box.addItem(_field_title(name), name)
        if current is not None:
            index = self._field_box.findData(current)
            if index >= 0:
                self._field_box.setCurrentIndex(index)
        self._updating = False

    def _populate_reduction_box(self) -> None:
        self._updating = True
        kind = self._kind_box.currentData() or "point"
        current = self._reduction_box.currentData()
        self._reduction_box.clear()
        titles = {
            "value": self.tr("Value", "Statistic: the sample itself, for a point probe"),
            "mean": self.tr("Mean", "Statistic: arithmetic mean"),
            "median": self.tr("Median", "Statistic"),
            "max": self.tr("Maximum", "Statistic"),
            "min": self.tr("Minimum", "Statistic"),
            "std": self.tr("Standard deviation"),
            "valid_fraction": self.tr("Valid fraction"),
            "strain": self.tr("Engineering strain"),
            "cod": self.tr("Crack opening"),
        }
        for name in sorted(allowed_reductions(kind)):
            self._reduction_box.addItem(titles.get(name, name), name)
        if current is not None:
            index = self._reduction_box.findData(current)
            if index >= 0:
                self._reduction_box.setCurrentIndex(index)
        self._updating = False

    # -- probe lifecycle --------------------------------------------------

    def _on_tool_clicked(self, tool: str, checked: bool) -> None:
        self._canvas.set_tool(tool if checked else "none")  # type: ignore[arg-type]
        self._sync_tool_buttons()

    def _sync_tool_buttons(self) -> None:
        active = self._canvas.tool
        for tool, button in self._tool_buttons.items():
            button.setChecked(tool == active)

    def _on_probe_placed(self, kind: str, geometry) -> None:
        probe = self._state.probes.add(kind, geometry)  # type: ignore[arg-type]
        self._selected_id = probe.id
        index = self._kind_box.findData(kind)
        if index >= 0:
            self._kind_box.setCurrentIndex(index)
        self._sync_tool_buttons()
        self._refresh_all()
        self._state.log_message.emit(
            self.tr("Added probe '%1'.").replace("%1", probe.label), "success"
        )

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
            text = item.text().strip()
            if text and text != probe.label:
                self._state.probes.replace(replace(probe, label=text))
        self._refresh_all()

    def _on_recolour(self) -> None:
        probe = self._selected_probe()
        if probe is None:
            return
        colour = QColorDialog.getColor(QColor(probe.color), self)
        if colour.isValid():
            self._state.probes.replace(replace(probe, color=colour.name()))
            self._refresh_all()

    def _on_delete(self) -> None:
        probe = self._selected_probe()
        if probe is None:
            return
        self._state.probes.remove(probe.id)
        self._selected_id = None
        self._refresh_all()

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
            self._refresh_all()

    def _on_kind_changed(self) -> None:
        if self._updating:
            return
        self._populate_reduction_box()
        self._refresh_chart()

    # -- refresh ----------------------------------------------------------

    def _refresh_all(self) -> None:
        self._refresh_background()
        self._refresh_table()
        self._canvas.set_probes(self._state.probes, self._selected_id)
        self._populate_reduction_box()
        self._refresh_chart()

    def _refresh_background(self) -> None:
        """Always the reference frame.

        Probe coordinates are frame-0 image pixels, so a marker drawn over a
        deformed frame would sit beside the material it measures. The field
        itself lives on the Strain Field tab; this canvas is for placement.
        """
        if not self._state.image_files:
            return
        try:
            self._canvas.set_image(self._image_ctrl.read_image_rgb(0))
        except (IndexError, FileNotFoundError, ValueError):
            return
        self._canvas.fit_to_view()

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
            show.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
            )
            show.setCheckState(
                Qt.CheckState.Checked if probe.visible else Qt.CheckState.Unchecked
            )
            self._table.setItem(row, 0, show)

            name = QTableWidgetItem(probe.label)
            name.setData(Qt.ItemDataRole.UserRole, probe.id)
            self._table.setItem(row, 1, name)

            kind = QTableWidgetItem(kinds.get(probe.kind, probe.kind))
            kind.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._table.setItem(row, 2, kind)

            swatch = QTableWidgetItem()
            swatch.setBackground(QColor(probe.color))
            swatch.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._table.setItem(row, 3, swatch)

            if probe.id == self._selected_id:
                self._table.selectRow(row)
        self._updating = False

    def _refresh_chart(self) -> None:
        if self._updating:
            return
        result = self._state.results
        kind = self._kind_box.currentData()
        field = self._field_box.currentData()
        reduction = self._reduction_box.currentData()
        if result is None:
            self._chart.clear(self.tr("Run a DIC analysis to plot probes."))
            return
        if not kind or not field or not reduction:
            self._chart.clear()
            return

        probes = [
            p for p in self._state.probes.of_kind(kind) if p.visible
        ]
        if not probes:
            self._chart.clear(
                self.tr("Place a probe on the reference image to begin.")
            )
            return

        entries = []
        for probe in probes:
            try:
                series = extract_series(
                    result, probe, field, reduction,
                    # Holes and notches present from frame 0. Cracks need no
                    # mask: consumed material has NaN U_accum.
                    ref_mask=self._state.per_frame_rois.get(0),
                    pixel_size=self._pixel_size(),
                    length_unit=self._length_unit(),
                    min_valid_fraction=self._threshold.value(),
                )
            except ValueError:
                continue
            entries.append((probe, series))

        if not entries:
            self._chart.clear(self.tr("This statistic does not apply here."))
            return

        unit = entries[0][1].unit
        self._chart.plot_series(
            [(p.label, p.color, s) for p, s in entries],
            y_label=_axis_label(
                self._field_box.currentText(),
                self._reduction_box.currentText(),
                unit,
            ),
            current_frame=self._state.current_frame,
        )
        self._last_entries = [
            ProbeSeries(probe=p, field=field, reduction=reduction, series=s)
            for p, s in entries
        ]

    # -- units ------------------------------------------------------------

    def _pixel_size(self) -> float:
        state = self._state
        if state.use_physical_units and state.pixel_size > 0:
            return float(state.pixel_size)
        return 1.0

    def _length_unit(self) -> str:
        state = self._state
        return state.pixel_unit if state.use_physical_units else "px"

    # -- export -----------------------------------------------------------

    def _on_export_csv(self) -> None:
        entries = getattr(self, "_last_entries", [])
        if not entries:
            QMessageBox.information(
                self, self.tr("Export Probe Data"),
                self.tr("There is nothing to export yet."),
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export Probe Data"), "probes.csv",
            self.tr("CSV Files") + " (*.csv);;" + self.tr("All Files") + " (*)",
        )
        if not path:
            return
        state = self._state
        rate = (
            float(state.frame_rate)
            if state.use_physical_units and state.frame_rate > 0 else None
        )
        try:
            export_probe_csv(path, entries, frame_rate=rate)
        except (OSError, ValueError) as exc:
            state.log_message.emit(
                self.tr("Probe export failed: %1").replace("%1", str(exc)),
                "error",
            )
            return
        state.log_message.emit(
            self.tr("Probe data written to %1").replace("%1", path), "success"
        )

    def _on_export_chart(self) -> None:
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
            self._state.log_message.emit(
                self.tr("Chart export failed: %1").replace("%1", str(exc)),
                "error",
            )
            return
        self._state.log_message.emit(
            self.tr("Chart written to %1").replace("%1", path), "success"
        )


def _field_title(name: str) -> str:
    """Human name for a canonical field, symbols kept literal."""
    from PySide6.QtCore import QCoreApplication

    ctx = "AnalysisTab"
    titles = {
        "disp_u": QCoreApplication.translate(ctx, "Displacement u"),
        "disp_v": QCoreApplication.translate(ctx, "Displacement v"),
        "disp_magnitude": QCoreApplication.translate(ctx, "Displacement magnitude"),
        "strain_exx": "εxx",
        "strain_eyy": "εyy",
        "strain_exy": "εxy",
        "strain_principal_max": "ε₁",
        "strain_principal_min": "ε₂",
        "strain_maxshear": QCoreApplication.translate(ctx, "γ max"),
        "strain_von_mises": "von Mises",
        "strain_rotation": QCoreApplication.translate(ctx, "ω rot"),
    }
    return titles.get(name, name)


def _axis_label(field_title: str, reduction_title: str, unit: str) -> str:
    label = f"{field_title} — {reduction_title}"
    return f"{label} ({unit})" if unit else label


__all__ = ["AnalysisTab"]
