"""Import a testing machine's record for the Analysis tab.

Choose the CSV; say which column is the load and in what unit, and how its
rows meet the frames -- by frame number, or by time with an offset; and give
the initial cross-section if stress is wanted. A summary line says how many
frames the record covers while the mapping changes, so a wrong offset shows
at once rather than as a strange curve later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from al_dic.analysis.load_data import LoadData, LoadSync, LoadTable, read_load_table
from al_dic.gui.theme import COLORS
from al_dic.gui.widgets.double_spin import LocaleSafeDoubleSpinBox
from al_dic.gui.window_geometry import fit_dialog_to_screen
from al_dic.i18n import tr_args

_PREVIEW_ROWS = 8

# Words that name a column, in the languages machines are set up in.
_LOAD_WORDS = ("load", "force", "kraft", "charge", "carga", "载荷", "載荷",
               "荷重", "하중", "力")
_TIME_WORDS = ("time", "zeit", "temps", "tiempo", "时间", "時間", "시간")
_FRAME_WORDS = ("frame", "image", "bild", "img", "帧", "影格", "フレーム", "프레임")


@dataclass(frozen=True)
class Mapping:
    """A first guess at which column is which."""

    mode: str
    load_column: str
    load_unit: str
    time_column: str
    frame_column: str

    @property
    def key_column(self) -> str:
        return self.time_column if self.mode == "time" else self.frame_column


def _find(names: tuple[str, ...], words: tuple[str, ...]) -> str | None:
    for name in names:
        low = name.lower()
        if any(word in low for word in words):
            return name
    return None


def guess_mapping(table: LoadTable, frame_rate: float) -> Mapping:
    """Load, time and frame columns by name; by time when there is a rate."""
    names = table.names
    load = _find(names, _LOAD_WORDS) or names[-1]
    others = [n for n in names if n != load] or list(names)
    time_column = _find(names, _TIME_WORDS)
    frame_column = _find(names, _FRAME_WORDS)
    if frame_rate > 0 and (time_column or not frame_column):
        mode = "time"
    else:
        mode = "frame"
    return Mapping(
        mode=mode,
        load_column=load,
        load_unit="kN" if "kn" in load.lower() else "N",
        time_column=time_column or others[0],
        frame_column=frame_column or others[0],
    )


class LoadDataDialog(QDialog):
    """Choose a machine CSV and map it to the frames."""

    def __init__(self, current: LoadData | None = None, n_frames: int = 0,
                 frame_rate: float = 0.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_frames = int(n_frames)
        self._rate = float(frame_rate) if frame_rate > 0 else 0.0
        self._table: LoadTable | None = None
        self._removed = False
        self._fitted = False
        self.setWindowTitle(self.tr("Load Data"))
        self._build(current is not None)
        if current is not None:
            self._show_current(current)
        self._update()

    # -- construction -----------------------------------------------------

    def _build(self, editing: bool) -> None:
        root = QVBoxLayout(self)

        file_row = QHBoxLayout()
        self._file_label = QLabel(self.tr("No file chosen."))
        self._file_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED};")
        choose = QPushButton(self.tr("Choose file…"))
        choose.clicked.connect(self._on_choose)
        file_row.addWidget(self._file_label, 1)
        file_row.addWidget(choose)
        root.addLayout(file_row)

        self._preview = QTableWidget(0, 0)
        self._preview.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._preview.verticalHeader().setVisible(False)
        self._preview.setMinimumHeight(140)
        root.addWidget(self._preview, 1)

        form = QFormLayout()
        load_row = QHBoxLayout()
        self._load_col = QComboBox()
        self._unit = QComboBox()
        self._unit.addItem("N", "N")
        self._unit.addItem("kN", "kN")
        load_row.addWidget(self._load_col, 1)
        load_row.addWidget(self._unit)
        form.addRow(self.tr("Load column:"), load_row)

        self._by_time = QRadioButton(self.tr("By time"))
        self._by_frame = QRadioButton(self.tr("By frame number"))
        group = QButtonGroup(self)
        group.addButton(self._by_time)
        group.addButton(self._by_frame)
        form.addRow(self.tr("Match rows to frames:"), self._by_time)

        self._time_col = QComboBox()
        form.addRow(self.tr("Time column:"), self._time_col)
        self._offset = LocaleSafeDoubleSpinBox()
        self._offset.setRange(-1e6, 1e6)
        self._offset.setDecimals(3)
        self._offset.setSuffix(" s")
        self._offset.setToolTip(self.tr(
            "The machine's time at the reference image. If the camera started "
            "2 s after the machine, enter 2."))
        form.addRow(self.tr("Offset:"), self._offset)
        self._rate_note = QLabel()
        self._rate_note.setWordWrap(True)
        self._rate_note.setStyleSheet(f"color: {COLORS.TEXT_MUTED};")
        form.addRow("", self._rate_note)

        form.addRow("", self._by_frame)
        self._frame_col = QComboBox()
        form.addRow(self.tr("Frame column:"), self._frame_col)
        self._frame_base = QComboBox()
        self._frame_base.addItem("1", 1)
        self._frame_base.addItem("0", 0)
        form.addRow(self.tr("The first image is numbered:"), self._frame_base)

        self._area = LocaleSafeDoubleSpinBox()
        self._area.setRange(0.0, 1e9)
        self._area.setDecimals(4)
        self._area.setSuffix(" mm²")
        self._area.setToolTip(self.tr(
            "Initial cross-section, for engineering stress F / A0 in MPa. "
            "Leave at 0 for load only."))
        form.addRow(self.tr("Cross-section A0:"), self._area)
        root.addLayout(form)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)
        self._error = QLabel()
        self._error.setWordWrap(True)
        self._error.setStyleSheet(f"color: {COLORS.DANGER};")
        self._error.hide()
        root.addWidget(self._error)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._remove_btn = QPushButton(self.tr("Remove Load Data"))
        self._remove_btn.clicked.connect(self._on_remove)
        self._buttons.addButton(self._remove_btn, QDialogButtonBox.ButtonRole.ResetRole)
        # After addButton, which shows what it adds.
        self._remove_btn.setVisible(editing)
        root.addWidget(self._buttons)

        if self._rate > 0:
            self._rate_note.setText(tr_args(self.tr(
                "Camera frame rate: %1 fps, from Physical Units."), f"{self._rate:g}"))
        else:
            self._rate_note.setText(self.tr(
                "Set the camera frame rate under Physical Units to match by time."))
            self._by_time.setEnabled(False)
        self._by_frame.setChecked(True)

        for box in (self._load_col, self._unit, self._time_col, self._frame_col,
                    self._frame_base):
            box.currentIndexChanged.connect(self._update)
        for button in (self._by_time, self._by_frame):
            button.toggled.connect(self._update)
        self._offset.valueChanged.connect(self._update)
        self._area.valueChanged.connect(self._update)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._fitted:
            self._fitted = True
            fit_dialog_to_screen(self, self.parentWidget())

    # -- content ------------------------------------------------------------

    def set_table(self, table: LoadTable) -> None:
        """Show *table* and guess its mapping."""
        self._table = table
        self._file_label.setText(table.source or self.tr("(unnamed)"))
        for box in (self._load_col, self._time_col, self._frame_col):
            box.blockSignals(True)
            box.clear()
            box.addItems(list(table.names))
            box.blockSignals(False)
        guess = guess_mapping(table, self._rate)
        self._select(self._load_col, guess.load_column)
        self._select(self._time_col, guess.time_column)
        self._select(self._frame_col, guess.frame_column)
        self._unit.setCurrentIndex(max(self._unit.findData(guess.load_unit), 0))
        (self._by_time if guess.mode == "time" else self._by_frame).setChecked(True)
        self._fill_preview(table)
        self._update()

    def _show_current(self, current: LoadData) -> None:
        self.set_table(current.table)
        sync = current.sync
        self._file_label.setText(current.source or self.tr("(unnamed)"))
        self._select(self._load_col, sync.load_column)
        self._unit.setCurrentIndex(max(self._unit.findData(sync.load_unit), 0))
        if sync.mode == "time" and sync.time_column:
            self._select(self._time_col, sync.time_column)
            self._offset.setValue(sync.offset_s)
            if self._by_time.isEnabled():
                self._by_time.setChecked(True)
        elif sync.frame_column:
            self._select(self._frame_col, sync.frame_column)
            self._frame_base.setCurrentIndex(max(self._frame_base.findData(sync.frame_base), 0))
            self._by_frame.setChecked(True)
        self._area.setValue(current.area_mm2 or 0.0)

    @staticmethod
    def _select(box: QComboBox, text: str) -> None:
        index = box.findText(text)
        if index >= 0:
            box.setCurrentIndex(index)

    def _fill_preview(self, table: LoadTable) -> None:
        rows = min(table.n_rows, _PREVIEW_ROWS)
        self._preview.setColumnCount(len(table.names))
        self._preview.setRowCount(rows)
        self._preview.setHorizontalHeaderLabels(list(table.names))
        for j, column in enumerate(table.columns):
            for i in range(rows):
                value = column[i]
                item = QTableWidgetItem("" if not np.isfinite(value) else f"{value:g}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                self._preview.setItem(i, j, item)

    # -- the mapping ----------------------------------------------------------

    def _build_data(self) -> LoadData:
        by_time = self._by_time.isChecked()
        sync = LoadSync(
            mode="time" if by_time else "frame",
            load_column=self._load_col.currentText(),
            time_column=self._time_col.currentText() if by_time else None,
            frame_column=None if by_time else self._frame_col.currentText(),
            offset_s=self._offset.value() if by_time else 0.0,
            load_unit=self._unit.currentData(),
            frame_base=1 if by_time else int(self._frame_base.currentData()),
        )
        area = self._area.value()
        return LoadData(self._table, sync, area_mm2=area if area > 0 else None)

    def load_data(self) -> LoadData | None:
        """The record as mapped, or None: none chosen, or removed."""
        if self._removed or self._table is None:
            return None
        try:
            return self._build_data()
        except ValueError:
            return None

    def _ok_button(self) -> QPushButton:
        return self._buttons.button(QDialogButtonBox.StandardButton.Ok)

    def _update(self, *_args) -> None:
        by_time = self._by_time.isChecked()
        self._time_col.setEnabled(by_time)
        self._offset.setEnabled(by_time)
        self._frame_col.setEnabled(not by_time)
        self._frame_base.setEnabled(not by_time)
        self._ok_button().setEnabled(self._table is not None)
        if self._table is None:
            self._summary.setText("")
            return
        try:
            load = self._build_data().load_n(self._n_frames, self._rate)
        except ValueError as exc:
            self._summary.setText(str(exc))
            return
        covered = int(np.count_nonzero(np.isfinite(load)))
        if covered == 0 and self._n_frames > 0:
            self._summary.setText(self.tr(
                "No frame falls within the record: check the columns and the offset."))
        else:
            # No plural to agree with: "%1 of %2" reads right for any count.
            self._summary.setText(tr_args(
                self.tr("Frames with a load: %1 of %2."), covered, self._n_frames))

    # -- actions ----------------------------------------------------------------

    def _on_choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Open Load Data"), "",
            self.tr("CSV Files") + " (*.csv *.txt *.dat);;" + self.tr("All Files") + " (*)")
        if not path:
            return
        try:
            table = read_load_table(path)
        except (OSError, ValueError) as exc:
            self._show_error(tr_args(self.tr("Could not read %1: %2"), path, exc))
            return
        self._error.hide()
        self.set_table(table)

    def _show_error(self, message: str) -> None:
        self._error.setText(message)
        self._error.show()

    def _on_remove(self) -> None:
        self._removed = True
        super().accept()

    def accept(self) -> None:  # noqa: D102 - validate before closing
        if self._table is None:
            return
        try:
            self._build_data()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        super().accept()


__all__ = ["LoadDataDialog", "Mapping", "guess_mapping"]
