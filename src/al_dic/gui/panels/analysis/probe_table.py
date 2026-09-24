"""The Analysis tab's probe list: show, name, type, colour and a note per probe.

It edits nothing itself. Ticking a box or renaming a row reports the change;
the tab applies it to the probe set and redraws, so the set has one writer.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from PySide6.QtCore import QCoreApplication, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from al_dic.analysis.probes import Probe
from al_dic.gui.theme import COLORS

_SHOW, _NAME, _TYPE, _COLOUR, _NOTE = range(5)


class ProbeTable(QWidget):
    """One row per probe, and buttons for the selected one."""

    # The user selected a row, or cleared the selection: id or None.
    selection_changed = Signal(object)
    # A row's Show box was ticked or cleared: (id, visible).
    visibility_changed = Signal(int, bool)
    # A row was renamed: (id, new label), whitespace already collapsed.
    renamed = Signal(int, str)
    recolour_clicked = Signal()
    delete_clicked = Signal()
    clear_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        self.table = QTableWidget(0, 5)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.verticalHeader().setVisible(False)
        self.table.setIconSize(QSize(14, 14))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(_NAME, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_NOTE, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemChanged.connect(self._on_item_changed)
        column.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        self.colour_btn = QPushButton()
        self.colour_btn.clicked.connect(self.recolour_clicked)
        self.delete_btn = QPushButton()
        self.delete_btn.clicked.connect(self.delete_clicked)
        self.clear_btn = QPushButton()
        self.clear_btn.clicked.connect(self.clear_clicked)
        for b in (self.colour_btn, self.delete_btn, self.clear_btn):
            buttons.addWidget(b)
        buttons.addStretch()
        column.addLayout(buttons)

        # Four rows and the buttons stay visible: at a laptop's 728 px the
        # splitter's stretch factors alone squeezed the list to a single row.
        self.setMinimumHeight(6 * self.row_height())

    def row_height(self) -> int:
        return self.table.fontMetrics().height() + 12

    def retranslate_ui(self) -> None:
        self.table.setHorizontalHeaderLabels([
            QCoreApplication.translate(
                "AnalysisTab", "Show", "Probe list column: visibility checkbox"),
            QCoreApplication.translate(
                "AnalysisTab", "Name", "Probe list column: the probe's label"),
            QCoreApplication.translate(
                "AnalysisTab", "Type", "Probe list column: point, line or region"),
            QCoreApplication.translate(
                "AnalysisTab", "Colour", "Probe list column: colour swatch"),
            QCoreApplication.translate(
                "AnalysisTab", "Note", "Probe list column: why a probe shows gaps"),
        ])
        self.colour_btn.setText(QCoreApplication.translate("AnalysisTab", "Colour…"))
        self.delete_btn.setText(QCoreApplication.translate(
            "AnalysisTab", "Delete", "Button: delete the selected probe"))
        self.clear_btn.setText(QCoreApplication.translate("AnalysisTab", "Clear All"))

    # -- content --------------------------------------------------------------

    def show_probes(self, probes: Iterable[Probe], notes: Mapping[int, str],
                    selected_id: int | None) -> None:
        kinds = {
            "point": QCoreApplication.translate("AnalysisTab", "Point", "Probe type"),
            "line": QCoreApplication.translate("AnalysisTab", "Line", "Probe type"),
            "area": QCoreApplication.translate(
                "AnalysisTab", "Region", "Probe type: an enclosed area"),
        }
        probes = list(probes)
        self._updating = True
        self.table.setRowCount(len(probes))
        for row, probe in enumerate(probes):
            show = QTableWidgetItem()
            show.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            show.setCheckState(
                Qt.CheckState.Checked if probe.visible else Qt.CheckState.Unchecked)
            self.table.setItem(row, _SHOW, show)

            name = QTableWidgetItem(probe.label)
            name.setData(Qt.ItemDataRole.UserRole, probe.id)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                          | Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, _NAME, name)

            kind = QTableWidgetItem(kinds.get(probe.kind, probe.kind))
            kind.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, _TYPE, kind)

            # An icon, not a background: a selected row's highlight painted
            # over the background and hid the swatch.
            swatch = QTableWidgetItem()
            pixmap = QPixmap(14, 14)
            pixmap.fill(QColor(probe.color))
            swatch.setIcon(QIcon(pixmap))
            swatch.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, _COLOUR, swatch)

            note = QTableWidgetItem(notes.get(probe.id, ""))
            note.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            note.setForeground(QColor(COLORS.TEXT_MUTED))
            self.table.setItem(row, _NOTE, note)

            if probe.id == selected_id:
                self.table.selectRow(row)
        self._updating = False

    def set_actions(self, has_selection: bool, has_probes: bool) -> None:
        self.colour_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        self.clear_btn.setEnabled(has_probes)

    def _name_item(self, probe_id: int | None) -> QTableWidgetItem | None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, _NAME)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == probe_id:
                return item
        return None

    def select(self, probe_id: int | None) -> None:
        """Select *probe_id*'s row (or none) without reporting it back."""
        self._updating = True
        self.table.clearSelection()
        item = self._name_item(probe_id)
        if item is not None:
            self.table.selectRow(item.row())
            self.table.scrollToItem(item)
        self._updating = False

    def edit_name(self, probe_id: int | None) -> None:
        """Open *probe_id*'s name for editing -- from a double-click or F2."""
        item = self._name_item(probe_id)
        if item is not None:
            self.table.setCurrentItem(item)
            self.table.editItem(item)

    # -- user edits ---------------------------------------------------------

    def _on_selection_changed(self) -> None:
        if self._updating:
            return
        rows = self.table.selectionModel().selectedRows()
        self.selection_changed.emit(
            self.table.item(rows[0].row(), _NAME).data(Qt.ItemDataRole.UserRole)
            if rows else None)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        probe_id = self.table.item(item.row(), _NAME).data(Qt.ItemDataRole.UserRole)
        if item.column() == _SHOW:
            self.visibility_changed.emit(
                probe_id, item.checkState() == Qt.CheckState.Checked)
        elif item.column() == _NAME:
            self.renamed.emit(probe_id, " ".join(item.text().split()))


__all__ = ["ProbeTable"]
