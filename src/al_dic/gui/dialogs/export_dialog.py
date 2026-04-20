"""Export dialog for DIC results — 4-tab layout.

Tabs:
  - Data Tab:      Per-variable checkboxes → NPZ / MAT / CSV.
  - Images Tab:    Per-field rows → PNG batch export.
  - Animation Tab: Per-field rows → GIF / MP4 animation export.
  - Report Tab:    Field statistics + sample images → self-contained HTML.

UX design:
  - All export actions are QPushButton (non-AcceptRole); dialog stays open.
  - Timestamp is refreshed on every export click → no folder overwrite.
  - "Open Folder" button appears after first successful export.
  - Last-used folder persisted via QSettings.
  - 1-based frame numbers in spinboxes.
  - Progress bar + status label inside each tab.
  - Field labels match the main window / strain window naming (U, V, εxx, …).
  - Image/animation export: background image always shown (WYSIWYG).
    User chooses background source: "Reference frame" (frame 1) or "Current frame".
    Optional "Show on deformed configuration" plots field at deformed node positions.

Entry points:
  - MainWindow "Export Results" button  (right_sidebar._on_export)
  - StrainWindow "Export Results" button (strain_window._on_export_strain)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import (
    QCoreApplication, QSettings, QThread, QUrl, Qt, Signal,
)
from PySide6.QtGui import QDesktopServices
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
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from al_dic.core.data_structures import PipelineResult
from al_dic.gui.app_state import AppState
from al_dic.gui.theme import COLORS

_SETTINGS_ORG = "AL-DIC"
_SETTINGS_APP = "ExportDialog"
_SETTINGS_KEY_FOLDER = "last_folder"


# ---------------------------------------------------------------------------
# Friendly UI labels — match the main window / strain window naming
# ---------------------------------------------------------------------------

_EXPORT_LABELS: dict[str, str] = {
    "disp_u":               "U",
    "disp_v":               "V",
    "disp_magnitude":       "Magnitude",
    "strain_exx":           "\u03b5xx",          # εxx
    "strain_eyy":           "\u03b5yy",
    "strain_exy":           "\u03b5xy",
    "strain_principal_max": "\u03b5\u2081",      # ε₁
    "strain_principal_min": "\u03b5\u2082",      # ε₂
    "strain_maxshear":      "\u03b3 max",        # γ max
    "strain_von_mises":     "von Mises",
    "strain_rotation":      "\u03c9 rot",        # ω rot
}


def _label(key: str) -> str:
    """Return the user-facing label for a canonical field name."""
    return _EXPORT_LABELS.get(key, key)


# ---------------------------------------------------------------------------
# Pure-data configuration types (no Qt, fully testable)
# ---------------------------------------------------------------------------

@dataclass
class VizExportHint:
    """Snapshot of the current window's visualisation settings."""
    colormap: str = "jet"
    auto_range: bool = True
    vmin: float = 0.0
    vmax: float = 1.0
    show_deformed: bool = False
    overlay_alpha: float = 0.7
    use_physical_units: bool = False
    pixel_size: float = 1.0
    pixel_unit: str = "mm"
    frame_rate: float = 1.0


@dataclass
class FieldImageConfig:
    """Per-field image/animation export settings."""
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
    data_fields: list[str] = field(default_factory=list)
    npz_per_frame: bool = False

    # Images Tab
    export_images: bool = False
    image_fields: list[FieldImageConfig] = field(default_factory=list)
    image_format: str = "png"
    image_dpi: int = 150
    show_deformed: bool = False     # render field at deformed node positions
    bg_mode: str = "ref_frame"      # "ref_frame" | "current_frame"
    frame_start: int = 0
    frame_end: int = -1

    # Animation Tab
    export_animation: bool = False
    anim_fields: list[FieldImageConfig] = field(default_factory=list)
    anim_format: str = "mp4"
    anim_fps: int = 10
    anim_show_deformed: bool = False
    anim_bg_mode: str = "ref_frame"  # "ref_frame" | "current_frame"
    anim_frame_start: int = 0
    anim_frame_end: int = -1

    # Report Tab
    export_report: bool = False
    report_fields: list[str] = field(default_factory=list)
    report_sample_every: int = 5

    # Visual export settings (shared by Images + Animation)
    # Default ON: users almost always want a colorbar with exported
    # images / animations so the field values are actually readable.
    # The old default (off) forced them to toggle it every time.
    img_include_colorbar: bool = True
    anim_include_colorbar: bool = True
    use_physical_units: bool = False
    pixel_size: float = 1.0
    pixel_unit: str = "mm"
    frame_rate: float = 1.0


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

_DATA_DEFAULT_SELECTED = {
    "disp_u", "disp_v", "disp_magnitude",
    "strain_exx", "strain_eyy", "strain_exy",
}


# ---------------------------------------------------------------------------
# Image export worker
# ---------------------------------------------------------------------------

class ExportImagesWorker(QThread):
    """Background worker that renders and writes PNG images."""

    progress = Signal(int, int, str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, config: ExportConfig, results: PipelineResult,
                 image_files: list[str], roi_mask,
                 per_frame_rois: dict | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._results = results
        self._image_files = image_files
        self._roi_mask = roi_mask
        self._per_frame_rois = per_frame_rois
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            from al_dic.export.export_png import export_png

            frame_end = self._config.frame_end
            if frame_end < 0:
                frame_end = len(self._results.result_disp) - 1

            paths = export_png(
                dest_dir=self._config.dest_dir,
                prefix=self._config.prefix,
                timestamp=self._config.timestamp,
                results=self._results,
                configs=self._config.image_fields,
                image_files=self._image_files,
                bg_mode=self._config.bg_mode,
                roi_mask=self._roi_mask,
                dpi=self._config.image_dpi,
                show_deformed=self._config.show_deformed,
                frame_start=self._config.frame_start,
                frame_end=frame_end,
                stop_event=self._stop_event,
                progress_callback=lambda d, t: self.progress.emit(d, t, ""),
                per_frame_rois=self._per_frame_rois,
                include_colorbar=self._config.img_include_colorbar,
                use_physical_units=self._config.use_physical_units,
                pixel_size=self._config.pixel_size,
                pixel_unit=self._config.pixel_unit,
            )
            self.finished.emit(paths)
        except Exception as exc:  # pragma: no cover
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Animation export worker
# ---------------------------------------------------------------------------

class ExportAnimationWorker(QThread):
    """Background worker that renders and writes animation files."""

    progress = Signal(int, int, str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, config: ExportConfig, results: PipelineResult,
                 image_files: list[str], roi_mask,
                 per_frame_rois: dict | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._results = results
        self._image_files = image_files
        self._roi_mask = roi_mask
        self._per_frame_rois = per_frame_rois
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            from al_dic.export.export_animation import export_animation

            frame_end = self._config.anim_frame_end
            if frame_end < 0:
                frame_end = len(self._results.result_disp) - 1

            paths = export_animation(
                dest_dir=self._config.dest_dir,
                prefix=self._config.prefix,
                timestamp=self._config.timestamp,
                results=self._results,
                configs=self._config.anim_fields,
                image_files=self._image_files,
                bg_mode=self._config.anim_bg_mode,
                roi_mask=self._roi_mask,
                fmt=self._config.anim_format,
                fps=self._config.anim_fps,
                show_deformed=self._config.anim_show_deformed,
                frame_start=self._config.anim_frame_start,
                frame_end=frame_end,
                stop_event=self._stop_event,
                progress_callback=lambda d, t, f: self.progress.emit(d, t, f),
                per_frame_rois=self._per_frame_rois,
                include_colorbar=self._config.anim_include_colorbar,
                use_physical_units=self._config.use_physical_units,
                pixel_size=self._config.pixel_size,
                pixel_unit=self._config.pixel_unit,
            )
            self.finished.emit(paths)
        except Exception as exc:  # pragma: no cover
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Per-field row widget (Images Tab + Animation Tab)
# ---------------------------------------------------------------------------

class _FieldRow(QWidget):
    """One row in the field list (Images or Animation tab)."""

    def __init__(self, field_name: str, default_cmap: str, hint: VizExportHint,
                 has_data: bool, parent: QWidget | None = None) -> None:
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

        # Show user-facing label, not internal variable name
        name_lbl = QLabel(_label(field_name))
        name_lbl.setFixedWidth(100)
        name_lbl.setToolTip(field_name)   # internal name visible on hover
        name_lbl.setStyleSheet(
            f"color: {'#e2e8f0' if has_data else COLORS.TEXT_MUTED}; font-size: 11px;"
        )
        row.addWidget(name_lbl)

        # Read per-field color state from AppState (WYSIWYG: export matches the live view).
        _fs = AppState.instance().get_field_state(field_name)
        _field_cmap = _fs.colormap
        _field_auto = _fs.auto
        _field_vmin = _fs.vmin
        _field_vmax = _fs.vmax

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems([
            "jet", "viridis", "turbo", "coolwarm",
            "plasma", "RdBu_r", "seismic", "inferno",
        ])
        self._cmap_combo.setCurrentText(_field_cmap)
        self._cmap_combo.setEnabled(has_data)
        self._cmap_combo.setFixedWidth(90)
        row.addWidget(self._cmap_combo)

        self._auto_check = QCheckBox("Auto")
        self._auto_check.setChecked(_field_auto)
        self._auto_check.setEnabled(has_data)
        self._auto_check.stateChanged.connect(self._on_auto_changed)
        row.addWidget(self._auto_check)

        self._vmin_spin = QDoubleSpinBox()
        self._vmin_spin.setRange(-1e9, 1e9)
        self._vmin_spin.setValue(_field_vmin)
        self._vmin_spin.setDecimals(4)
        self._vmin_spin.setFixedWidth(85)
        self._vmin_spin.setEnabled(has_data and not _field_auto)
        row.addWidget(self._vmin_spin)

        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(-1e9, 1e9)
        self._vmax_spin.setValue(_field_vmax)
        self._vmax_spin.setDecimals(4)
        self._vmax_spin.setFixedWidth(85)
        self._vmax_spin.setEnabled(has_data and not _field_auto)
        row.addWidget(self._vmax_spin)

        opacity_lbl = QLabel("Opacity")
        opacity_lbl.setToolTip("Field opacity (0 = transparent, 1 = fully opaque)")
        row.addWidget(opacity_lbl)
        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0, 1.0)
        self._alpha_spin.setSingleStep(0.05)
        self._alpha_spin.setValue(hint.overlay_alpha)
        self._alpha_spin.setDecimals(2)
        self._alpha_spin.setFixedWidth(65)
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
# Helper: build scrollable field-row list
# ---------------------------------------------------------------------------

def _build_field_rows_scroll(
    fields: list[tuple[str, str]],
    hint: VizExportHint,
    has_strain: bool,
) -> tuple[QScrollArea, list[_FieldRow]]:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setMaximumHeight(220)

    container = QWidget()
    c_layout = QVBoxLayout(container)
    c_layout.setSpacing(0)
    c_layout.setContentsMargins(0, 0, 0, 0)

    rows: list[_FieldRow] = []
    for fname, default_cmap in fields:
        is_strain = fname.startswith("strain_")
        has_data = not is_strain or has_strain
        row_w = _FieldRow(fname, default_cmap, hint, has_data)
        c_layout.addWidget(row_w)
        rows.append(row_w)

    scroll.setWidget(container)
    return scroll, rows


# ---------------------------------------------------------------------------
# Shared "Select: All / None" helper
# ---------------------------------------------------------------------------

def _make_select_all_none(
    checks: dict[str, QCheckBox],
) -> tuple[QPushButton, QPushButton]:
    """Return (All button, None button) wired to the given checkbox dict."""
    all_btn = QPushButton(QCoreApplication.translate("ExportDialog", "All"))
    all_btn.setFixedWidth(60)
    none_btn = QPushButton(QCoreApplication.translate("ExportDialog", "None"))
    none_btn.setFixedWidth(60)
    all_btn.clicked.connect(
        lambda: [c.setChecked(True) for c in checks.values() if c.isEnabled()]
    )
    none_btn.clicked.connect(
        lambda: [c.setChecked(False) for c in checks.values()]
    )
    return all_btn, none_btn


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class ExportDialog(QDialog):
    """Four-tab export dialog: Data, Images, Animation, Report.

    Timestamp is refreshed on every export action so repeated clicks
    produce distinct output folders rather than overwriting.
    """

    def __init__(self, results: PipelineResult, image_folder: Path | None,
                 hint: VizExportHint,
                 image_files: list[str] | None = None,
                 roi_mask=None,
                 per_frame_rois: dict | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export Results"))
        self.setMinimumWidth(780)

        from al_dic.export.export_utils import make_prefix, make_timestamp

        self._results = results
        self._hint = hint
        self._prefix = make_prefix(image_folder)
        self._image_folder = image_folder
        self._image_files: list[str] = image_files or []
        self._roi_mask = roi_mask
        self._per_frame_rois = per_frame_rois
        self._has_strain = len(results.result_strain) > 0
        self._img_worker: ExportImagesWorker | None = None
        self._anim_worker: ExportAnimationWorker | None = None

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # --- Shared folder row ---
        folder_lbl = QLabel(self.tr("OUTPUT FOLDER"))
        folder_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; font-weight: bold;"
        )
        root.addWidget(folder_lbl)

        path_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText(self.tr("Select output folder…"))
        self._folder_edit.textChanged.connect(self._on_folder_changed)
        path_row.addWidget(self._folder_edit, 1)

        browse_btn = QPushButton(self.tr("Browse…"))
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(browse_btn)

        self._open_folder_btn = QPushButton(self.tr("Open Folder"))
        self._open_folder_btn.setFixedWidth(90)
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        path_row.addWidget(self._open_folder_btn)
        root.addLayout(path_row)

        # --- Physical units (shared by Images + Animation) ---
        units_group = QGroupBox(self.tr("PHYSICAL UNITS"))
        units_form = QFormLayout(units_group)
        units_form.setSpacing(4)

        self._phys_units_check = QCheckBox(self.tr("Enable physical units"))
        self._phys_units_check.setChecked(hint.use_physical_units)
        self._phys_units_check.setToolTip(self.tr(
            "Scale displacement values by pixel size and show physical units "
            "on colorbar labels. Strain is dimensionless and unaffected."
        ))
        units_form.addRow(self._phys_units_check)

        phys_row = QHBoxLayout()
        self._pixel_size_spin = QDoubleSpinBox()
        self._pixel_size_spin.setRange(1e-6, 1e6)
        self._pixel_size_spin.setValue(hint.pixel_size)
        self._pixel_size_spin.setDecimals(4)
        self._pixel_size_spin.setFixedWidth(100)
        phys_row.addWidget(self._pixel_size_spin)
        self._pixel_unit_combo = QComboBox()
        self._pixel_unit_combo.addItems(["nm", "μm", "mm", "cm", "m", "inch"])
        self._pixel_unit_combo.setCurrentText(hint.pixel_unit)
        self._pixel_unit_combo.setFixedWidth(70)
        phys_row.addWidget(self._pixel_unit_combo)
        phys_row.addWidget(QLabel(self.tr("/ pixel")))
        phys_row.addStretch()
        units_form.addRow(self.tr("Pixel size"), phys_row)

        fr_row = QHBoxLayout()
        self._frame_rate_spin = QDoubleSpinBox()
        self._frame_rate_spin.setRange(1e-3, 1e6)
        self._frame_rate_spin.setValue(hint.frame_rate)
        self._frame_rate_spin.setDecimals(2)
        self._frame_rate_spin.setFixedWidth(100)
        fr_row.addWidget(self._frame_rate_spin)
        fr_row.addWidget(QLabel(self.tr("fps")))
        fr_row.addStretch()
        units_form.addRow(self.tr("Frame rate"), fr_row)

        root.addWidget(units_group)

        # --- Tabs ---
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_data_tab(), self.tr("Data"))
        self._tabs.addTab(self._build_images_tab(), self.tr("Images"))
        self._tabs.addTab(self._build_animation_tab(), self.tr("Animation"))
        self._tabs.addTab(self._build_report_tab(), self.tr("Report"))

        # Sync button enabled state now that all tabs (and buttons) are built
        self._on_folder_changed(self._folder_edit.text())

        # --- Close button ---
        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        root.addWidget(close_box)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # FORMAT group
        fmt_group = QGroupBox(self.tr("FORMAT"))
        fmt_layout = QVBoxLayout(fmt_group)
        self._npz_check = QCheckBox(self.tr("NumPy Archive (.npz)"))
        self._npz_check.setChecked(True)
        self._mat_check = QCheckBox(self.tr("MATLAB (.mat)"))
        self._mat_check.setChecked(True)
        self._csv_check = QCheckBox(self.tr("CSV (per frame)"))
        self._csv_check.setChecked(True)
        self._npz_per_frame_check = QCheckBox(self.tr(
            "NPZ: one file per frame (default: single merged file)"))
        self._npz_per_frame_check.setChecked(False)
        for chk in (self._npz_check, self._mat_check, self._csv_check,
                    self._npz_per_frame_check):
            fmt_layout.addWidget(chk)
        layout.addWidget(fmt_group)

        # DISPLACEMENT group
        disp_group = QGroupBox(self.tr("DISPLACEMENT"))
        disp_vl = QVBoxLayout(disp_group)
        self._data_disp_checks: dict[str, QCheckBox] = {}
        for key, _ in _DISP_FIELDS:
            chk = QCheckBox(_label(key))
            chk.setChecked(key in _DATA_DEFAULT_SELECTED)
            self._data_disp_checks[key] = chk

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel(self.tr("Select:")))
        all_btn, none_btn = _make_select_all_none(self._data_disp_checks)
        sel_row.addWidget(all_btn)
        sel_row.addWidget(none_btn)
        sel_row.addStretch()
        disp_vl.addLayout(sel_row)

        chk_row = QHBoxLayout()
        for chk in self._data_disp_checks.values():
            chk_row.addWidget(chk)
        chk_row.addStretch()
        disp_vl.addLayout(chk_row)
        layout.addWidget(disp_group)

        # STRAIN group
        strain_group = QGroupBox(self.tr("STRAIN"))
        strain_group.setEnabled(self._has_strain)
        if not self._has_strain:
            strain_group.setToolTip(self.tr("Run Compute Strain first."))
        strain_vl = QVBoxLayout(strain_group)
        self._data_strain_checks: dict[str, QCheckBox] = {}
        for key, _ in _STRAIN_FIELDS:
            chk = QCheckBox(_label(key))
            chk.setChecked(key in _DATA_DEFAULT_SELECTED)
            self._data_strain_checks[key] = chk

        ssel_row = QHBoxLayout()
        ssel_row.addWidget(QLabel(self.tr("Select:")))
        sall_btn, snone_btn = _make_select_all_none(self._data_strain_checks)
        ssel_row.addWidget(sall_btn)
        ssel_row.addWidget(snone_btn)
        ssel_row.addStretch()
        strain_vl.addLayout(ssel_row)

        strain_r1 = QHBoxLayout()
        strain_r2 = QHBoxLayout()
        for i, key in enumerate(self._data_strain_checks):
            (strain_r1 if i < 4 else strain_r2).addWidget(self._data_strain_checks[key])
        strain_r1.addStretch()
        strain_r2.addStretch()
        strain_vl.addLayout(strain_r1)
        strain_vl.addLayout(strain_r2)
        layout.addWidget(strain_group)

        params_note = QLabel(self.tr(
            "✓ Parameters file (JSON) always exported"))
        params_note.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(params_note)

        data_btn_row = QHBoxLayout()
        data_btn_row.addStretch()
        self._export_data_btn = QPushButton(self.tr("Export Data"))
        self._export_data_btn.setEnabled(False)
        self._export_data_btn.clicked.connect(self._on_export_data)
        data_btn_row.addWidget(self._export_data_btn)
        layout.addLayout(data_btn_row)

        self._data_status_lbl = QLabel("")
        self._data_status_lbl.setWordWrap(True)
        layout.addWidget(self._data_status_lbl)

        layout.addStretch()
        return w

    def _build_images_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        # Field list header
        header_row = QHBoxLayout()
        for text, width in [
            ("Export", 50), ("Field", 100), ("Colormap", 90),
            ("Auto", 45), ("Min", 70), ("Max", 70), ("α", 55),
        ]:
            lbl = QLabel(text)
            lbl.setFixedWidth(width)
            lbl.setStyleSheet(
                f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px; font-weight: bold;"
            )
            header_row.addWidget(lbl)
        header_row.addStretch()
        layout.addLayout(header_row)

        scroll, self._img_field_rows = _build_field_rows_scroll(
            _ALL_FIELDS, self._hint, self._has_strain
        )
        layout.addWidget(scroll)

        # IMAGE SETTINGS group
        img_group = QGroupBox("IMAGE SETTINGS")
        img_form = QFormLayout(img_group)
        img_form.setSpacing(6)

        fmt_row = QHBoxLayout()
        self._img_fmt_combo = QComboBox()
        self._img_fmt_combo.addItems(["PNG", "JPEG", "TIFF"])
        fmt_row.addWidget(self._img_fmt_combo)
        fmt_row.addStretch()
        img_form.addRow("Format", fmt_row)

        dpi_row = QHBoxLayout()
        self._img_dpi_spin = QSpinBox()
        self._img_dpi_spin.setRange(72, 600)
        self._img_dpi_spin.setValue(150)
        self._img_dpi_spin.setFixedWidth(70)
        dpi_row.addWidget(self._img_dpi_spin)
        dpi_row.addStretch()
        img_form.addRow("DPI", dpi_row)

        self._img_colorbar_check = QCheckBox("Include colorbar")
        self._img_colorbar_check.setChecked(True)
        self._img_colorbar_check.setToolTip(
            "Append a vertical colorbar strip to the right of each image.\n"
            "Tick labels update per frame when Auto range is enabled."
        )
        img_form.addRow(self._img_colorbar_check)

        # Reference vs. deformed: couples background image + field node positions
        config_row = QHBoxLayout()
        self._img_ref_rb = QRadioButton("Original (frame 1 background)")
        self._img_ref_rb.setChecked(not self._hint.show_deformed)
        self._img_ref_rb.setToolTip(
            "Field is drawn at the original (undeformed) node positions.\n"
            "Background image is always the first frame."
        )
        self._img_def_rb = QRadioButton("Deformed (current frame background)")
        self._img_def_rb.setChecked(self._hint.show_deformed)
        self._img_def_rb.setToolTip(
            "Field is drawn at the displaced node positions (reference + displacement).\n"
            "Background image follows each frame's own photo."
        )
        config_row.addWidget(self._img_ref_rb)
        config_row.addWidget(self._img_def_rb)
        config_row.addStretch()
        img_form.addRow("Render as", config_row)
        layout.addWidget(img_group)

        # FRAME RANGE
        layout.addWidget(self._build_frame_range_widget("img"))

        # Progress + status
        self._img_progress = QProgressBar()
        self._img_progress.setRange(0, 100)
        self._img_progress.setValue(0)
        self._img_progress.setVisible(False)
        layout.addWidget(self._img_progress)

        self._img_progress_lbl = QLabel("")
        self._img_progress_lbl.setVisible(False)
        layout.addWidget(self._img_progress_lbl)

        img_btn_row = QHBoxLayout()
        self._cancel_img_btn = QPushButton("Cancel Export")
        self._cancel_img_btn.setVisible(False)
        self._cancel_img_btn.clicked.connect(self._on_cancel_images)
        img_btn_row.addWidget(self._cancel_img_btn)
        img_btn_row.addStretch()
        self._export_img_btn = QPushButton("Export Images")
        self._export_img_btn.setEnabled(False)
        self._export_img_btn.clicked.connect(self._on_export_images)
        img_btn_row.addWidget(self._export_img_btn)
        layout.addLayout(img_btn_row)

        self._img_status_lbl = QLabel("")
        self._img_status_lbl.setWordWrap(True)
        layout.addWidget(self._img_status_lbl)

        return w

    def _build_animation_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        for text, width in [
            ("Export", 50), ("Field", 100), ("Colormap", 90),
            ("Auto", 45), ("Min", 70), ("Max", 70), ("α", 55),
        ]:
            lbl = QLabel(text)
            lbl.setFixedWidth(width)
            lbl.setStyleSheet(
                f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px; font-weight: bold;"
            )
            header_row.addWidget(lbl)
        header_row.addStretch()
        layout.addLayout(header_row)

        scroll, self._anim_field_rows = _build_field_rows_scroll(
            _ALL_FIELDS, self._hint, self._has_strain
        )
        layout.addWidget(scroll)

        # ANIMATION SETTINGS group
        anim_group = QGroupBox("ANIMATION SETTINGS")
        anim_form = QFormLayout(anim_group)
        anim_form.setSpacing(6)

        fmt_row = QHBoxLayout()
        self._anim_fmt_combo = QComboBox()
        self._anim_fmt_combo.addItems(["MP4", "GIF"])
        fmt_row.addWidget(self._anim_fmt_combo)
        fmt_row.addStretch()
        anim_form.addRow("Format", fmt_row)

        fps_row = QHBoxLayout()
        self._anim_fps_spin = QSpinBox()
        self._anim_fps_spin.setRange(1, 60)
        self._anim_fps_spin.setValue(10)
        self._anim_fps_spin.setFixedWidth(60)
        fps_row.addWidget(self._anim_fps_spin)
        fps_row.addStretch()
        anim_form.addRow("FPS", fps_row)

        self._anim_colorbar_check = QCheckBox("Include colorbar")
        self._anim_colorbar_check.setChecked(True)
        self._anim_colorbar_check.setToolTip(
            "Append a vertical colorbar strip to the right of each frame.\n"
            "Tick labels update per frame when Auto range is enabled."
        )
        anim_form.addRow(self._anim_colorbar_check)

        # Reference vs. deformed: couples background image + field node positions
        anim_config_row = QHBoxLayout()
        self._anim_ref_rb = QRadioButton("Original (frame 1 background)")
        self._anim_ref_rb.setChecked(not self._hint.show_deformed)
        self._anim_ref_rb.setToolTip(
            "Field is drawn at the original (undeformed) node positions.\n"
            "Background image is always the first frame."
        )
        self._anim_def_rb = QRadioButton("Deformed (current frame background)")
        self._anim_def_rb.setChecked(self._hint.show_deformed)
        self._anim_def_rb.setToolTip(
            "Field is drawn at the displaced node positions (reference + displacement).\n"
            "Background image follows each frame's own photo."
        )
        anim_config_row.addWidget(self._anim_ref_rb)
        anim_config_row.addWidget(self._anim_def_rb)
        anim_config_row.addStretch()
        anim_form.addRow("Render as", anim_config_row)
        layout.addWidget(anim_group)

        layout.addWidget(self._build_frame_range_widget("anim"))

        self._anim_progress = QProgressBar()
        self._anim_progress.setRange(0, 100)
        self._anim_progress.setValue(0)
        self._anim_progress.setVisible(False)
        layout.addWidget(self._anim_progress)

        self._anim_progress_lbl = QLabel("")
        self._anim_progress_lbl.setVisible(False)
        layout.addWidget(self._anim_progress_lbl)

        anim_btn_row = QHBoxLayout()
        self._cancel_anim_btn = QPushButton("Cancel Export")
        self._cancel_anim_btn.setVisible(False)
        self._cancel_anim_btn.clicked.connect(self._on_cancel_animation)
        anim_btn_row.addWidget(self._cancel_anim_btn)
        anim_btn_row.addStretch()
        self._export_anim_btn = QPushButton("Export Animation")
        self._export_anim_btn.setEnabled(False)
        self._export_anim_btn.clicked.connect(self._on_export_animation)
        anim_btn_row.addWidget(self._export_anim_btn)
        layout.addLayout(anim_btn_row)

        self._anim_status_lbl = QLabel("")
        self._anim_status_lbl.setWordWrap(True)
        layout.addWidget(self._anim_status_lbl)

        return w

    def _build_report_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        content_group = QGroupBox("CONTENT")
        content_vl = QVBoxLayout(content_group)
        self._report_params_check = QCheckBox("Parameter summary table")
        self._report_params_check.setChecked(True)
        self._report_stats_check = QCheckBox("Field statistics (min/max/mean/std per frame)")
        self._report_stats_check.setChecked(True)
        self._report_images_check = QCheckBox("Sample field images")
        self._report_images_check.setChecked(True)
        for chk in (self._report_params_check, self._report_stats_check,
                    self._report_images_check):
            content_vl.addWidget(chk)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Sample every"))
        self._report_sample_spin = QSpinBox()
        self._report_sample_spin.setRange(1, 100)
        self._report_sample_spin.setValue(5)
        self._report_sample_spin.setFixedWidth(60)
        sample_row.addWidget(self._report_sample_spin)
        sample_row.addWidget(QLabel("frames"))
        sample_row.addStretch()
        content_vl.addLayout(sample_row)
        layout.addWidget(content_group)

        # FIELDS group
        fields_group = QGroupBox("FIELDS")
        fields_vl = QVBoxLayout(fields_group)

        disp_lbl = QLabel("Displacement:")
        disp_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; font-weight: bold;"
        )
        fields_vl.addWidget(disp_lbl)
        self._report_disp_checks: dict[str, QCheckBox] = {}
        disp_row = QHBoxLayout()
        for key, _ in _DISP_FIELDS:
            chk = QCheckBox(_label(key))
            chk.setChecked(key in _DATA_DEFAULT_SELECTED)
            self._report_disp_checks[key] = chk
            disp_row.addWidget(chk)
        disp_row.addStretch()
        fields_vl.addLayout(disp_row)

        strain_lbl = QLabel("Strain:")
        strain_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; font-weight: bold;"
        )
        fields_vl.addWidget(strain_lbl)
        self._report_strain_checks: dict[str, QCheckBox] = {}
        strain_r1 = QHBoxLayout()
        strain_r2 = QHBoxLayout()
        for i, (key, _) in enumerate(_STRAIN_FIELDS):
            chk = QCheckBox(_label(key))
            chk.setChecked(key in _DATA_DEFAULT_SELECTED)
            if not self._has_strain:
                chk.setEnabled(False)
            self._report_strain_checks[key] = chk
            (strain_r1 if i < 4 else strain_r2).addWidget(chk)
        strain_r1.addStretch()
        strain_r2.addStretch()
        fields_vl.addLayout(strain_r1)
        fields_vl.addLayout(strain_r2)
        layout.addWidget(fields_group)

        format_note = QLabel("Format: HTML (self-contained, view in any browser)")
        format_note.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(format_note)

        report_btn_row = QHBoxLayout()
        report_btn_row.addStretch()
        self._generate_report_btn = QPushButton("Generate Report")
        self._generate_report_btn.setEnabled(False)
        self._generate_report_btn.clicked.connect(self._on_generate_report)
        report_btn_row.addWidget(self._generate_report_btn)
        layout.addLayout(report_btn_row)

        self._report_status_lbl = QLabel("")
        self._report_status_lbl.setWordWrap(True)
        layout.addWidget(self._report_status_lbl)

        layout.addStretch()
        return w

    def _build_frame_range_widget(self, prefix: str) -> QGroupBox:
        group = QGroupBox("FRAME RANGE")
        vl = QVBoxLayout(group)

        all_chk = QCheckBox("All frames")
        all_chk.setChecked(True)
        vl.addWidget(all_chk)

        n = len(self._results.result_disp)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("From"))

        start_spin = QSpinBox()
        start_spin.setRange(1, max(1, n))
        start_spin.setValue(1)
        start_spin.setEnabled(False)
        range_row.addWidget(start_spin)

        range_row.addWidget(QLabel("to"))

        end_spin = QSpinBox()
        end_spin.setRange(1, max(1, n))
        end_spin.setValue(max(1, n))
        end_spin.setEnabled(False)
        range_row.addWidget(end_spin)
        range_row.addStretch()
        vl.addLayout(range_row)

        def on_all_changed(state: int) -> None:
            all_on = state == Qt.CheckState.Checked.value
            start_spin.setEnabled(not all_on)
            end_spin.setEnabled(not all_on)

        all_chk.stateChanged.connect(on_all_changed)
        setattr(self, f"_{prefix}_all_frames_check", all_chk)
        setattr(self, f"_{prefix}_frame_start_spin", start_spin)
        setattr(self, f"_{prefix}_frame_end_spin", end_spin)
        return group

    # ------------------------------------------------------------------
    # Slots — folder
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        # Start from the current text, or fall back to the image folder.
        start = self._folder_edit.text().strip()
        if not start and self._image_folder:
            start = str(self._image_folder)
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", start
        )
        if folder:
            self._folder_edit.setText(folder)
            QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(
                _SETTINGS_KEY_FOLDER, folder
            )

    def _on_folder_changed(self, text: str) -> None:
        # Guard: tabs may not be built yet when the signal fires during __init__
        if not hasattr(self, "_export_data_btn"):
            return
        has_folder = bool(text.strip())
        for btn in (self._export_data_btn, self._export_img_btn,
                    self._export_anim_btn, self._generate_report_btn):
            btn.setEnabled(has_folder)

    def _on_open_folder(self) -> None:
        folder = self._folder_edit.text().strip()
        if folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    # ------------------------------------------------------------------
    # Slots — data export
    # ------------------------------------------------------------------

    def _on_export_data(self) -> None:
        from al_dic.export.export_utils import make_timestamp
        self._timestamp = make_timestamp()   # fresh timestamp per click

        folder = self._folder_edit.text().strip()
        if not folder:
            return
        cfg = self.get_config()
        exported: list[str] = []
        try:
            from al_dic.export.export_params import export_params
            from al_dic.export.export_npz import export_npz
            from al_dic.export.export_mat import export_mat
            from al_dic.export.export_csv import export_csv

            export_params(cfg.dest_dir, cfg.prefix, cfg.timestamp, self._results)
            exported.append("params.json")
            if cfg.export_npz:
                export_npz(cfg.dest_dir, cfg.prefix, cfg.timestamp,
                           self._results, cfg.data_fields, cfg.npz_per_frame)
                exported.append(".npz")
            if cfg.export_mat:
                export_mat(cfg.dest_dir, cfg.prefix, cfg.timestamp,
                           self._results, cfg.data_fields)
                exported.append(".mat")
            if cfg.export_csv:
                export_csv(cfg.dest_dir, cfg.prefix, cfg.timestamp,
                           self._results, cfg.data_fields)
                exported.append("csv/")

            self._data_status_lbl.setText(
                f"Exported {len(exported)} files → {cfg.dest_dir}"
            )
            self._data_status_lbl.setStyleSheet("color: #4ade80; font-size: 11px;")
            self._open_folder_btn.setEnabled(True)
        except Exception as exc:
            self._data_status_lbl.setText(f"Error: {exc}")
            self._data_status_lbl.setStyleSheet(
                f"color: {COLORS.ERROR}; font-size: 11px;"
            )

    # ------------------------------------------------------------------
    # Slots — image export
    # ------------------------------------------------------------------

    def _on_export_images(self) -> None:
        from al_dic.export.export_utils import make_timestamp
        self._timestamp = make_timestamp()   # fresh timestamp per click

        folder = self._folder_edit.text().strip()
        if not folder:
            return
        cfg = self.get_config()
        cfg.export_images = True

        self._export_img_btn.setEnabled(False)
        self._cancel_img_btn.setVisible(True)
        self._img_progress.setVisible(True)
        self._img_progress.setValue(0)
        self._img_progress_lbl.setVisible(True)
        self._img_progress_lbl.setText("Starting…")

        self._img_worker = ExportImagesWorker(
            cfg, self._results, self._image_files, self._roi_mask,
            per_frame_rois=self._per_frame_rois, parent=self,
        )
        self._img_worker.progress.connect(self._on_img_progress)
        self._img_worker.finished.connect(self._on_img_finished)
        self._img_worker.error.connect(self._on_img_error)
        self._img_worker.start()

    def _on_cancel_images(self) -> None:
        if self._img_worker is not None:
            self._img_worker.request_stop()
        self._reset_img_controls()

    def _on_img_progress(self, done: int, total: int, field: str) -> None:
        if total > 0:
            self._img_progress.setValue(int(done / total * 100))
        self._img_progress_lbl.setText(
            f"Rendering {field} ({done}/{total})" if field else f"Frame {done}/{total}"
        )

    def _on_img_finished(self, paths: list) -> None:
        self._reset_img_controls()
        self._img_progress.setValue(100)
        folder = self._folder_edit.text().strip()
        self._img_status_lbl.setText(f"Exported {len(paths)} images → {folder}")
        self._img_status_lbl.setStyleSheet("color: #4ade80; font-size: 11px;")
        self._open_folder_btn.setEnabled(True)

    def _on_img_error(self, msg: str) -> None:
        self._reset_img_controls()
        self._img_status_lbl.setText(f"Error: {msg}")
        self._img_status_lbl.setStyleSheet(f"color: {COLORS.ERROR}; font-size: 11px;")

    def _reset_img_controls(self) -> None:
        self._export_img_btn.setEnabled(bool(self._folder_edit.text().strip()))
        self._cancel_img_btn.setVisible(False)
        self._img_progress_lbl.setVisible(False)

    # ------------------------------------------------------------------
    # Slots — animation export
    # ------------------------------------------------------------------

    def _on_export_animation(self) -> None:
        from al_dic.export.export_utils import make_timestamp
        self._timestamp = make_timestamp()   # fresh timestamp per click

        folder = self._folder_edit.text().strip()
        if not folder:
            return
        cfg = self.get_config()
        cfg.export_animation = True

        self._export_anim_btn.setEnabled(False)
        self._cancel_anim_btn.setVisible(True)
        self._anim_progress.setVisible(True)
        self._anim_progress.setValue(0)
        self._anim_progress_lbl.setVisible(True)
        self._anim_progress_lbl.setText("Starting…")

        self._anim_worker = ExportAnimationWorker(
            cfg, self._results, self._image_files, self._roi_mask,
            per_frame_rois=self._per_frame_rois, parent=self,
        )
        self._anim_worker.progress.connect(self._on_anim_progress)
        self._anim_worker.finished.connect(self._on_anim_finished)
        self._anim_worker.error.connect(self._on_anim_error)
        self._anim_worker.start()

    def _on_cancel_animation(self) -> None:
        if self._anim_worker is not None:
            self._anim_worker.request_stop()
        self._reset_anim_controls()

    def _on_anim_progress(self, done: int, total: int, field: str) -> None:
        if total > 0:
            self._anim_progress.setValue(int(done / total * 100))
        self._anim_progress_lbl.setText(
            f"Rendering {field} ({done}/{total})" if field else f"Frame {done}/{total}"
        )

    def _on_anim_finished(self, paths: list) -> None:
        self._reset_anim_controls()
        self._anim_progress.setValue(100)
        folder = self._folder_edit.text().strip()
        self._anim_status_lbl.setText(
            f"Exported {len(paths)} animation(s) → {folder}"
        )
        self._anim_status_lbl.setStyleSheet("color: #4ade80; font-size: 11px;")
        self._open_folder_btn.setEnabled(True)

    def _on_anim_error(self, msg: str) -> None:
        self._reset_anim_controls()
        self._anim_status_lbl.setText(f"Error: {msg}")
        self._anim_status_lbl.setStyleSheet(f"color: {COLORS.ERROR}; font-size: 11px;")

    def _reset_anim_controls(self) -> None:
        self._export_anim_btn.setEnabled(bool(self._folder_edit.text().strip()))
        self._cancel_anim_btn.setVisible(False)
        self._anim_progress_lbl.setVisible(False)

    # ------------------------------------------------------------------
    # Slots — report
    # ------------------------------------------------------------------

    def _on_generate_report(self) -> None:
        from al_dic.export.export_utils import make_timestamp
        self._timestamp = make_timestamp()

        folder = self._folder_edit.text().strip()
        if not folder:
            return
        cfg = self.get_config()
        try:
            from al_dic.export.export_report import export_html_report
            p = export_html_report(
                dest_dir=cfg.dest_dir,
                prefix=cfg.prefix,
                timestamp=cfg.timestamp,
                results=self._results,
                fields=cfg.report_fields,
                image_configs=cfg.anim_fields,
                sample_every=cfg.report_sample_every,
                ref_image=None,
            )
            self._report_status_lbl.setText(f"Report saved → {p}")
            self._report_status_lbl.setStyleSheet("color: #4ade80; font-size: 11px;")
            self._open_folder_btn.setEnabled(True)
        except Exception as exc:
            self._report_status_lbl.setText(f"Error: {exc}")
            self._report_status_lbl.setStyleSheet(
                f"color: {COLORS.ERROR}; font-size: 11px;"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> ExportConfig:
        """Collect all dialog settings into an ExportConfig.

        Timestamp is NOT refreshed here — callers that need a fresh
        timestamp must set self._timestamp before calling get_config().
        """
        folder = self._folder_edit.text().strip()
        timestamp = getattr(self, "_timestamp", "")
        if not timestamp:
            from al_dic.export.export_utils import make_timestamp
            timestamp = make_timestamp()
            self._timestamp = timestamp

        img_all = self._img_all_frames_check.isChecked()
        img_start = 0 if img_all else self._img_frame_start_spin.value() - 1
        img_end = -1 if img_all else self._img_frame_end_spin.value() - 1

        anim_all = self._anim_all_frames_check.isChecked()
        anim_start = 0 if anim_all else self._anim_frame_start_spin.value() - 1
        anim_end = -1 if anim_all else self._anim_frame_end_spin.value() - 1

        data_fields = [
            k for k, chk in {**self._data_disp_checks,
                              **self._data_strain_checks}.items()
            if chk.isChecked()
        ]
        report_fields = [
            k for k, chk in {**self._report_disp_checks,
                              **self._report_strain_checks}.items()
            if chk.isChecked()
        ]

        return ExportConfig(
            dest_dir=Path(folder) if folder else Path("."),
            prefix=self._prefix,
            timestamp=timestamp,
            # Data
            export_npz=self._npz_check.isChecked(),
            export_mat=self._mat_check.isChecked(),
            export_csv=self._csv_check.isChecked(),
            data_fields=data_fields,
            npz_per_frame=self._npz_per_frame_check.isChecked(),
            # Images — "Deformed frame" couples bg_mode + show_deformed together
            export_images=False,
            image_fields=[r.get_config() for r in self._img_field_rows],
            image_format=self._img_fmt_combo.currentText().lower(),
            image_dpi=self._img_dpi_spin.value(),
            show_deformed=self._img_def_rb.isChecked(),
            bg_mode="current_frame" if self._img_def_rb.isChecked() else "ref_frame",
            frame_start=img_start,
            frame_end=img_end,
            # Animation — same coupling
            export_animation=False,
            anim_fields=[r.get_config() for r in self._anim_field_rows],
            anim_format=self._anim_fmt_combo.currentText().lower(),
            anim_fps=self._anim_fps_spin.value(),
            anim_show_deformed=self._anim_def_rb.isChecked(),
            anim_bg_mode="current_frame" if self._anim_def_rb.isChecked() else "ref_frame",
            anim_frame_start=anim_start,
            anim_frame_end=anim_end,
            # Report
            export_report=False,
            report_fields=report_fields,
            report_sample_every=self._report_sample_spin.value(),
            # Visual export settings
            img_include_colorbar=self._img_colorbar_check.isChecked(),
            anim_include_colorbar=self._anim_colorbar_check.isChecked(),
            use_physical_units=self._phys_units_check.isChecked(),
            pixel_size=self._pixel_size_spin.value(),
            pixel_unit=self._pixel_unit_combo.currentText(),
            frame_rate=self._frame_rate_spin.value(),
        )
