"""Export dialog for DIC results.

Two-tab layout:
  - Data Tab:   NPZ / MAT / CSV format selection, component checkboxes.
  - Images Tab: Per-field colormap / range / alpha settings, PNG batch export.

Entry points:
  - MainWindow "Export Results" button  (right_sidebar._on_export)
  - StrainWindow "Export Strain" button (strain_window._on_export_strain)

Both pass a VizExportHint built from the window's current visualisation state
so that Image Tab settings are pre-filled (WYSIWYG).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from staq_dic.core.data_structures import PipelineResult
from staq_dic.gui.theme import COLORS


# ---------------------------------------------------------------------------
# Pure-data configuration types (no Qt, fully testable)
# ---------------------------------------------------------------------------

@dataclass
class VizExportHint:
    """Snapshot of the current window's visualisation settings.

    Passed to ExportDialog so Image Tab controls are pre-filled with the
    same colormap / range the user already sees on screen (WYSIWYG).
    """
    colormap: str = "jet"
    auto_range: bool = True
    vmin: float = 0.0
    vmax: float = 1.0
    show_deformed: bool = False


@dataclass
class FieldImageConfig:
    """Per-field image export settings."""
    field_name: str
    enabled: bool
    colormap: str
    auto_range: bool
    vmin: float
    vmax: float
    bg_alpha: float = 0.7


@dataclass
class ExportConfig:
    """Aggregated export settings collected from the dialog."""
    # Shared
    dest_dir: Path
    prefix: str
    timestamp: str

    # Data Tab
    export_npz: bool = True
    export_mat: bool = True
    export_csv: bool = True
    include_disp: bool = True
    include_strain: bool = True
    npz_per_frame: bool = False

    # Image Tab
    export_images: bool = False
    image_fields: list[FieldImageConfig] = field(default_factory=list)
    image_format: str = "png"
    image_dpi: int = 150
    show_deformed: bool = False
    show_background: bool = True
    frame_start: int = 0
    frame_end: int = -1   # -1 = all frames


# ---------------------------------------------------------------------------
# Known exportable fields and their default colormaps
# ---------------------------------------------------------------------------

_DISP_FIELDS = [
    ("disp_u",         "jet"),
    ("disp_v",         "jet"),
    ("disp_magnitude", "jet"),
]

_STRAIN_FIELDS = [
    ("strain_exx",           "RdBu_r"),
    ("strain_eyy",           "RdBu_r"),
    ("strain_exy",           "RdBu_r"),
    ("strain_principal_max", "jet"),
    ("strain_principal_min", "jet"),
    ("strain_maxshear",      "jet"),
    ("strain_von_mises",     "jet"),
    ("strain_rotation",      "RdBu_r"),
]

_ALL_FIELDS = _DISP_FIELDS + _STRAIN_FIELDS

_DEFAULT_ENABLED = {
    "disp_u", "disp_v", "strain_exx", "strain_eyy", "strain_exy",
}


# ---------------------------------------------------------------------------
# Image export worker (QThread)
# ---------------------------------------------------------------------------

class ExportImagesWorker(QThread):
    """Background worker that renders and writes PNG images.

    Signals:
        progress(done, total)  -- emitted after each frame.
        finished(paths)        -- emitted on successful completion.
        error(message)         -- emitted if an exception occurs.
    """

    progress = Signal(int, int)
    finished = Signal(list)
    error = Signal(str)

    def __init__(
        self,
        config: ExportConfig,
        results: PipelineResult,
        ref_image: NDArray | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._results = results
        self._ref_image = ref_image
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        """Signal the worker to stop after the current frame."""
        self._stop_event.set()

    def run(self) -> None:
        try:
            from staq_dic.export.export_png import export_png

            frame_end = self._config.frame_end
            if frame_end < 0:
                frame_end = len(self._results.result_disp) - 1

            paths = export_png(
                dest_dir=self._config.dest_dir,
                prefix=self._config.prefix,
                timestamp=self._config.timestamp,
                results=self._results,
                configs=self._config.image_fields,
                ref_image=self._ref_image,
                dpi=self._config.image_dpi,
                show_deformed=self._config.show_deformed,
                show_background=self._config.show_background,
                frame_start=self._config.frame_start,
                frame_end=frame_end,
                stop_event=self._stop_event,
                progress_callback=lambda d, t: self.progress.emit(d, t),
            )
            self.finished.emit(paths)
        except Exception as exc:  # pragma: no cover
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Per-field row widget (Image Tab)
# ---------------------------------------------------------------------------

class _FieldRow(QWidget):
    """One row in the Image Tab field list."""

    def __init__(
        self,
        field_name: str,
        default_cmap: str,
        hint: VizExportHint,
        has_data: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._field_name = field_name
        enabled_default = has_data and (field_name in _DEFAULT_ENABLED)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)

        self._check = QCheckBox()
        self._check.setChecked(enabled_default)
        self._check.setEnabled(has_data)
        row.addWidget(self._check)

        name_lbl = QLabel(field_name)
        name_lbl.setFixedWidth(160)
        name_lbl.setStyleSheet(
            f"color: {'#e2e8f0' if has_data else COLORS.TEXT_MUTED}; font-size: 11px;"
        )
        row.addWidget(name_lbl)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems([
            "jet", "viridis", "turbo", "coolwarm",
            "plasma", "RdBu_r", "seismic", "inferno",
        ])
        self._cmap_combo.setCurrentText(hint.colormap if field_name in _DEFAULT_ENABLED else default_cmap)
        self._cmap_combo.setEnabled(has_data)
        self._cmap_combo.setFixedWidth(90)
        row.addWidget(self._cmap_combo)

        self._auto_check = QCheckBox("Auto")
        self._auto_check.setChecked(hint.auto_range)
        self._auto_check.setEnabled(has_data)
        self._auto_check.stateChanged.connect(self._on_auto_changed)
        row.addWidget(self._auto_check)

        self._vmin_spin = QDoubleSpinBox()
        self._vmin_spin.setRange(-1e9, 1e9)
        self._vmin_spin.setValue(hint.vmin)
        self._vmin_spin.setDecimals(4)
        self._vmin_spin.setFixedWidth(70)
        self._vmin_spin.setEnabled(has_data and not hint.auto_range)
        row.addWidget(self._vmin_spin)

        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(-1e9, 1e9)
        self._vmax_spin.setValue(hint.vmax)
        self._vmax_spin.setDecimals(4)
        self._vmax_spin.setFixedWidth(70)
        self._vmax_spin.setEnabled(has_data and not hint.auto_range)
        row.addWidget(self._vmax_spin)

        alpha_lbl = QLabel("α")
        row.addWidget(alpha_lbl)
        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0, 1.0)
        self._alpha_spin.setSingleStep(0.05)
        self._alpha_spin.setValue(0.7)
        self._alpha_spin.setDecimals(2)
        self._alpha_spin.setFixedWidth(55)
        self._alpha_spin.setEnabled(has_data)
        row.addWidget(self._alpha_spin)

        row.addStretch()

    def _on_auto_changed(self, state: int) -> None:
        auto = state == Qt.CheckState.Checked.value
        self._vmin_spin.setEnabled(not auto)
        self._vmax_spin.setEnabled(not auto)

    def get_config(self) -> FieldImageConfig:
        return FieldImageConfig(
            field_name=self._field_name,
            enabled=self._check.isChecked(),
            colormap=self._cmap_combo.currentText(),
            auto_range=self._auto_check.isChecked(),
            vmin=self._vmin_spin.value(),
            vmax=self._vmax_spin.value(),
            bg_alpha=self._alpha_spin.value(),
        )


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class ExportDialog(QDialog):
    """Two-tab export dialog: Data and Images.

    Args:
        results:      Full pipeline results.
        image_folder: Source image folder (used to derive export prefix).
        hint:         Current viz settings for WYSIWYG Image Tab pre-fill.
        parent:       Optional Qt parent widget.
    """

    def __init__(
        self,
        results: PipelineResult,
        image_folder: Path | None,
        hint: VizExportHint,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setMinimumWidth(640)

        from staq_dic.export.export_utils import make_prefix, make_timestamp

        self._results = results
        self._hint = hint
        self._prefix = make_prefix(image_folder)
        self._timestamp = make_timestamp()
        self._has_strain = len(results.result_strain) > 0
        self._worker: ExportImagesWorker | None = None

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Shared folder row
        folder_row = QHBoxLayout()
        folder_lbl = QLabel("OUTPUT FOLDER")
        folder_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; font-weight: bold;"
        )
        folder_row.addWidget(folder_lbl)
        root.addLayout(folder_row)

        path_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select output folder…")
        self._folder_edit.textChanged.connect(self._on_folder_changed)
        path_row.addWidget(self._folder_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(browse_btn)
        root.addLayout(path_row)

        # Tab widget
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        self._data_tab = self._build_data_tab()
        self._tabs.addTab(self._data_tab, "Data")

        self._images_tab = self._build_images_tab()
        self._tabs.addTab(self._images_tab, "Images")

        # Button box
        self._btn_box = QDialogButtonBox()
        self._export_data_btn = self._btn_box.addButton(
            "Export Data", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._export_data_btn.setEnabled(False)
        self._btn_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._btn_box.accepted.connect(self.accept)
        self._btn_box.rejected.connect(self.reject)
        root.addWidget(self._btn_box)

        # Image export progress (hidden initially)
        self._img_progress = QProgressBar()
        self._img_progress.setRange(0, 100)
        self._img_progress.setValue(0)
        self._img_progress.setVisible(False)
        root.addWidget(self._img_progress)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # FORMAT group
        fmt_group = QGroupBox("FORMAT")
        fmt_layout = QVBoxLayout(fmt_group)
        self._npz_check = QCheckBox("NumPy Archive (.npz)")
        self._npz_check.setChecked(True)
        self._mat_check = QCheckBox("MATLAB (.mat)")
        self._mat_check.setChecked(True)
        self._csv_check = QCheckBox("CSV (per frame)")
        self._csv_check.setChecked(True)
        fmt_layout.addWidget(self._npz_check)
        fmt_layout.addWidget(self._mat_check)
        fmt_layout.addWidget(self._csv_check)
        layout.addWidget(fmt_group)

        # COMPONENTS group
        comp_group = QGroupBox("COMPONENTS")
        comp_layout = QVBoxLayout(comp_group)
        self._disp_check = QCheckBox("Displacement  (U, V, magnitude, cumulative)")
        self._disp_check.setChecked(True)
        self._strain_check = QCheckBox(
            "Strain  (exx, eyy, exy, principals, maxshear, von_mises, rotation)"
        )
        self._strain_check.setChecked(self._has_strain)
        self._strain_check.setEnabled(self._has_strain)
        if not self._has_strain:
            self._strain_check.setToolTip("No strain results available. Run Compute Strain first.")
        comp_layout.addWidget(self._disp_check)
        comp_layout.addWidget(self._strain_check)
        layout.addWidget(comp_group)

        # OPTIONS group
        opt_group = QGroupBox("OPTIONS")
        opt_layout = QVBoxLayout(opt_group)
        self._npz_per_frame_check = QCheckBox("NPZ per-frame files (default: single merged file)")
        self._npz_per_frame_check.setChecked(False)
        opt_layout.addWidget(self._npz_per_frame_check)
        params_note = QLabel("✓ Parameters file (JSON) always exported")
        params_note.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 11px;")
        opt_layout.addWidget(params_note)
        layout.addWidget(opt_group)

        layout.addStretch()
        return w

    def _build_images_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        # Field list header
        header_row = QHBoxLayout()
        for text, width in [
            ("Export", 50), ("Field", 160), ("Colormap", 90),
            ("Auto", 45), ("Min", 70), ("Max", 70), ("α(bg)", 55),
        ]:
            lbl = QLabel(text)
            lbl.setFixedWidth(width)
            lbl.setStyleSheet(
                f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px; font-weight: bold;"
            )
            header_row.addWidget(lbl)
        header_row.addStretch()
        layout.addLayout(header_row)

        # Scrollable field rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(220)
        field_container = QWidget()
        field_layout = QVBoxLayout(field_container)
        field_layout.setSpacing(0)
        field_layout.setContentsMargins(0, 0, 0, 0)

        self._field_rows: list[_FieldRow] = []
        for fname, default_cmap in _ALL_FIELDS:
            # Strain fields only have data if strain was computed
            is_strain = fname.startswith("strain_")
            has_data = True if not is_strain else self._has_strain
            row_w = _FieldRow(fname, default_cmap, self._hint, has_data)
            field_layout.addWidget(row_w)
            self._field_rows.append(row_w)

        scroll.setWidget(field_container)
        layout.addWidget(scroll)

        # IMAGE SETTINGS group
        img_group = QGroupBox("IMAGE SETTINGS")
        img_layout = QFormLayout(img_group)
        img_layout.setSpacing(6)

        fmt_row = QHBoxLayout()
        self._img_fmt_combo = QComboBox()
        self._img_fmt_combo.addItems(["PNG", "JPEG", "TIFF"])
        fmt_row.addWidget(self._img_fmt_combo)
        fmt_row.addStretch()
        img_layout.addRow("Format", fmt_row)

        dpi_row = QHBoxLayout()
        self._img_dpi_spin = QSpinBox()
        self._img_dpi_spin.setRange(72, 600)
        self._img_dpi_spin.setValue(150)
        self._img_dpi_spin.setFixedWidth(70)
        dpi_row.addWidget(self._img_dpi_spin)
        dpi_row.addStretch()
        img_layout.addRow("DPI", dpi_row)

        self._img_deformed_check = QCheckBox("Show on deformed frame")
        self._img_deformed_check.setChecked(self._hint.show_deformed)
        img_layout.addRow("", self._img_deformed_check)

        self._img_bg_check = QCheckBox("Overlay reference image (background)")
        self._img_bg_check.setChecked(True)
        img_layout.addRow("", self._img_bg_check)
        layout.addWidget(img_group)

        # FRAME RANGE group
        range_group = QGroupBox("FRAME RANGE")
        range_layout = QVBoxLayout(range_group)
        self._all_frames_check = QCheckBox("All frames")
        self._all_frames_check.setChecked(True)
        self._all_frames_check.stateChanged.connect(self._on_all_frames_changed)
        range_layout.addWidget(self._all_frames_check)

        frame_range_row = QHBoxLayout()
        frame_range_row.addWidget(QLabel("From"))
        self._frame_start_spin = QSpinBox()
        self._frame_start_spin.setRange(0, max(0, len(self._results.result_disp) - 1))
        self._frame_start_spin.setValue(0)
        self._frame_start_spin.setEnabled(False)
        frame_range_row.addWidget(self._frame_start_spin)
        frame_range_row.addWidget(QLabel("to"))
        self._frame_end_spin = QSpinBox()
        self._frame_end_spin.setRange(0, max(0, len(self._results.result_disp) - 1))
        self._frame_end_spin.setValue(max(0, len(self._results.result_disp) - 1))
        self._frame_end_spin.setEnabled(False)
        frame_range_row.addWidget(self._frame_end_spin)
        frame_range_row.addStretch()
        range_layout.addLayout(frame_range_row)
        layout.addWidget(range_group)

        # Export Images button + progress
        img_btn_row = QHBoxLayout()
        self._cancel_img_btn = QPushButton("Cancel Export")
        self._cancel_img_btn.setVisible(False)
        self._cancel_img_btn.clicked.connect(self._on_cancel_images)
        img_btn_row.addWidget(self._cancel_img_btn)
        self._export_img_btn = QPushButton("Export Images")
        self._export_img_btn.setEnabled(False)
        self._export_img_btn.clicked.connect(self._on_export_images)
        img_btn_row.addWidget(self._export_img_btn)
        layout.addLayout(img_btn_row)

        return w

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._folder_edit.setText(folder)

    def _on_folder_changed(self, text: str) -> None:
        has_folder = bool(text.strip())
        self._export_data_btn.setEnabled(has_folder)
        self._export_img_btn.setEnabled(has_folder)

    def _on_all_frames_changed(self, state: int) -> None:
        all_frames = state == Qt.CheckState.Checked.value
        self._frame_start_spin.setEnabled(not all_frames)
        self._frame_end_spin.setEnabled(not all_frames)

    def _on_export_images(self) -> None:
        folder = self._folder_edit.text().strip()
        if not folder:
            return

        cfg = self.get_config()
        if not cfg.export_images:
            cfg.export_images = True

        self._export_img_btn.setEnabled(False)
        self._cancel_img_btn.setVisible(True)
        self._img_progress.setVisible(True)
        self._img_progress.setValue(0)

        self._worker = ExportImagesWorker(cfg, self._results, None, self)
        self._worker.progress.connect(self._on_img_progress)
        self._worker.finished.connect(self._on_img_finished)
        self._worker.error.connect(self._on_img_error)
        self._worker.start()

    def _on_cancel_images(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
        self._reset_img_controls()

    def _on_img_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._img_progress.setValue(int(done / total * 100))

    def _on_img_finished(self, paths: list) -> None:
        self._reset_img_controls()
        self._img_progress.setValue(100)

    def _on_img_error(self, msg: str) -> None:
        self._reset_img_controls()

    def _reset_img_controls(self) -> None:
        folder = self._folder_edit.text().strip()
        self._export_img_btn.setEnabled(bool(folder))
        self._cancel_img_btn.setVisible(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> ExportConfig:
        """Collect all dialog settings into an ExportConfig."""
        all_frames = self._all_frames_check.isChecked()
        frame_end = -1 if all_frames else self._frame_end_spin.value()

        return ExportConfig(
            dest_dir=Path(self._folder_edit.text().strip()),
            prefix=self._prefix,
            timestamp=self._timestamp,
            # Data
            export_npz=self._npz_check.isChecked(),
            export_mat=self._mat_check.isChecked(),
            export_csv=self._csv_check.isChecked(),
            include_disp=self._disp_check.isChecked(),
            include_strain=self._strain_check.isChecked(),
            npz_per_frame=self._npz_per_frame_check.isChecked(),
            # Images
            export_images=False,   # set True only in _on_export_images
            image_fields=[r.get_config() for r in self._field_rows],
            image_format=self._img_fmt_combo.currentText().lower(),
            image_dpi=self._img_dpi_spin.value(),
            show_deformed=self._img_deformed_check.isChecked(),
            show_background=self._img_bg_check.isChecked(),
            frame_start=self._frame_start_spin.value() if not all_frames else 0,
            frame_end=frame_end,
        )
