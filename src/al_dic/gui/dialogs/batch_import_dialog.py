"""Batch ROI mask import dialog.

Two-panel layout for assigning mask image files to frames:
  - Left panel: available mask images from a selected folder
  - Right panel: frame assignments (frame index, image filename, mask filename)

Supports auto-match by filename number, sequential assignment, and
manual selected-to-selected pairing.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from al_dic.gui.theme import COLORS

# Image file extensions accepted as mask files
_MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".jp2", ".webp"}


class BatchImportDialog(QDialog):
    """Dialog for batch-importing mask files and assigning them to frames."""

    def __init__(self, image_files: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Batch Import Region of Interest Masks"))
        self.setMinimumSize(700, 500)

        self._image_files = list(image_files)
        self._mask_folder = ""
        self._mask_files: list[str] = []
        self._assignments: dict[int, str] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Folder browser row ---
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel(self.tr("Mask Folder:")))
        self._folder_label = QLabel(self.tr("(none)"))
        self._folder_label.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-style: italic;"
        )
        folder_row.addWidget(self._folder_label, stretch=1)
        browse_btn = QPushButton(self.tr("Browse..."))
        browse_btn.clicked.connect(self._on_browse)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        # --- Two-panel area ---
        panels = QHBoxLayout()

        # Left panel: available mask files
        left = QVBoxLayout()
        left.addWidget(QLabel(self.tr("Available Masks")))
        self._mask_list = QListWidget()
        self._mask_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        left.addWidget(self._mask_list)

        auto_btn = QPushButton(self.tr("Auto-Match by Name"))
        auto_btn.setToolTip(self.tr(
            "Match mask files to frames by number in filename"))
        auto_btn.clicked.connect(self._auto_match)
        left.addWidget(auto_btn)

        seq_btn = QPushButton(self.tr("Assign Sequential"))
        seq_btn.setToolTip(self.tr(
            "Assign masks to frames in order starting from frame 0"))
        seq_btn.clicked.connect(self._assign_sequential)
        left.addWidget(seq_btn)

        panels.addLayout(left)

        # Right panel: frame assignments
        right = QVBoxLayout()
        right.addWidget(QLabel(self.tr("Frame Assignments")))
        self._assign_tree = QTreeWidget()
        self._assign_tree.setHeaderLabels([
            self.tr("Frame"), self.tr("Image"), self.tr("Mask")])
        self._assign_tree.setColumnCount(3)
        self._assign_tree.setRootIsDecorated(False)
        header = self._assign_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._assign_tree.setColumnWidth(0, 50)
        right.addWidget(self._assign_tree)

        assign_btn = QPushButton(self.tr("Assign Selected ->"))
        assign_btn.setToolTip(self.tr(
            "Pair selected mask(s) with selected frame(s)"))
        assign_btn.clicked.connect(self._assign_selected)
        right.addWidget(assign_btn)

        clear_btn = QPushButton(self.tr("Clear All"))
        clear_btn.clicked.connect(self._clear_assignments)
        right.addWidget(clear_btn)

        panels.addLayout(right)
        layout.addLayout(panels)

        # --- OK / Cancel ---
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._populate_frames()

    def _populate_frames(self) -> None:
        """Fill the assignment tree with one row per image frame."""
        self._assign_tree.clear()
        for i, fpath in enumerate(self._image_files):
            fname = Path(fpath).name
            item = QTreeWidgetItem([f"{i:02d}", fname, ""])
            item.setData(0, Qt.ItemDataRole.UserRole, i)
            self._assign_tree.addTopLevelItem(item)

    # ------------------------------------------------------------------
    # Folder browsing and mask loading
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Mask Folder")
        if not folder:
            return
        self._mask_folder = folder
        self._folder_label.setText(folder)
        self._load_mask_files(folder)

    def _load_mask_files(self, folder: str) -> None:
        """Scan *folder* for image files and populate the left list."""
        p = Path(folder)
        self._mask_files = sorted(
            str(f)
            for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in _MASK_EXTENSIONS
        )
        self._mask_list.clear()
        for mf in self._mask_files:
            item = QListWidgetItem(Path(mf).name)
            item.setData(Qt.ItemDataRole.UserRole, mf)
            self._mask_list.addItem(item)

    # ------------------------------------------------------------------
    # Assignment strategies
    # ------------------------------------------------------------------

    def _auto_match(self) -> None:
        """Match masks to frames by extracting the last number in filename."""
        self._assignments.clear()
        for mf in self._mask_files:
            numbers = re.findall(r"\d+", Path(mf).stem)
            if numbers:
                idx = int(numbers[-1])
                if 0 <= idx < len(self._image_files):
                    self._assignments[idx] = mf
        self._refresh_display()

    def _assign_sequential(self) -> None:
        """Assign masks to frames 0, 1, 2, ... in sorted order."""
        self._assignments.clear()
        for i, mf in enumerate(self._mask_files):
            if i < len(self._image_files):
                self._assignments[i] = mf
        self._refresh_display()

    def _assign_selected(self) -> None:
        """Pair each selected mask with the corresponding selected frame row."""
        mask_items = self._mask_list.selectedItems()
        frame_items = self._assign_tree.selectedItems()
        if not mask_items or not frame_items:
            return
        for mi, fi in zip(mask_items, frame_items):
            idx = fi.data(0, Qt.ItemDataRole.UserRole)
            path = mi.data(Qt.ItemDataRole.UserRole)
            if idx is not None and path:
                self._assignments[idx] = path
        self._refresh_display()

    def _clear_assignments(self) -> None:
        """Remove all mask-to-frame assignments."""
        self._assignments.clear()
        self._refresh_display()

    # ------------------------------------------------------------------
    # Display refresh
    # ------------------------------------------------------------------

    def _refresh_display(self) -> None:
        """Update the Mask column in the assignment tree."""
        for i in range(self._assign_tree.topLevelItemCount()):
            item = self._assign_tree.topLevelItem(i)
            idx = item.data(0, Qt.ItemDataRole.UserRole)
            mask_path = self._assignments.get(idx, "")
            name = Path(mask_path).name if mask_path else ""
            item.setText(2, name)
            color = QColor(COLORS.SUCCESS) if name else QColor(COLORS.TEXT_MUTED)
            item.setForeground(2, color)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_assignments(self) -> dict[int, str]:
        """Return the current frame-index-to-mask-path mapping."""
        return dict(self._assignments)

    def load_masks(self, img_shape: tuple[int, int]) -> dict[int, np.ndarray]:
        """Read and threshold all assigned mask files.

        Supports all bit depths (uint8, uint16, float) and common
        formats (tif, png, bmp, jpg, jp2, webp).

        Args:
            img_shape: (height, width) to resize masks to if needed.

        Returns:
            Mapping of frame_index -> boolean mask array.
        """
        from al_dic.io.io_utils import read_mask_as_bool

        result: dict[int, np.ndarray] = {}
        for idx, path in self._assignments.items():
            try:
                result[idx] = read_mask_as_bool(path, target_shape=img_shape)
            except (FileNotFoundError, IOError):
                continue
        return result
