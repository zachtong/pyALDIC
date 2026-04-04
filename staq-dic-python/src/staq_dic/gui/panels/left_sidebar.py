"""Left sidebar panel — image loading, ROI tools, parameters.

Fixed width 220px.  Contains:
1. IMAGES section with count badge and drop zone
2. ImageList (scrollable file list)
3. ROI toolbar (rect / polygon / circle / import / cut / clear)
4. PARAMETERS section (subset size, step, search range, tracking mode)
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from staq_dic.gui.app_state import AppState
from staq_dic.gui.controllers.image_controller import ImageController
from staq_dic.gui.theme import COLORS
from staq_dic.gui.widgets.image_list import ImageList
from staq_dic.gui.widgets.param_panel import ParamPanel
from staq_dic.gui.widgets.roi_toolbar import ROIToolbar


class _SectionHeader(QWidget):
    """Compact section header with title and optional badge."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 4)
        layout.setSpacing(6)

        label = QLabel(title)
        label.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; "
            f"font-weight: bold; letter-spacing: 1px;"
        )
        layout.addWidget(label)

        self._badge = QLabel("")
        self._badge.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 10px; "
            f"background: {COLORS.BG_INPUT}; border-radius: 7px; "
            f"padding: 1px 6px;"
        )
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.hide()
        layout.addWidget(self._badge)

        layout.addStretch()

    def set_badge(self, text: str) -> None:
        """Show a badge with the given text (e.g., image count)."""
        if text:
            self._badge.setText(text)
            self._badge.show()
        else:
            self._badge.hide()


class _DropZone(QWidget):
    """Drop zone accepting folder drops, with a Browse button."""

    def __init__(
        self,
        image_ctrl: ImageController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_ctrl = image_ctrl
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.setMinimumHeight(72)
        self.setStyleSheet(
            f"""
            _DropZone {{
                background: {COLORS.BG_PANEL};
                border: 1px dashed {COLORS.BORDER};
                border-radius: 6px;
                margin: 4px 8px;
            }}
            _DropZone:hover {{
                border-color: {COLORS.ACCENT};
                background: {COLORS.BG_INPUT};
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(4)

        icon_label = QLabel("\U0001f4c2")  # folder emoji
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 20px; background: transparent;")
        layout.addWidget(icon_label)

        text_label = QLabel("Drop image folder\nor Browse")
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 11px; background: transparent;"
        )
        layout.addWidget(text_label)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Open folder dialog on click."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", ""
        )
        if folder:
            self._image_ctrl.load_folder(folder)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        """Accept drag if it contains URLs (folders)."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        """Load the first dropped folder."""
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if os.path.isdir(path):
            self._image_ctrl.load_folder(path)
        elif os.path.isfile(path):
            # If a file was dropped, use its parent directory
            self._image_ctrl.load_folder(str(Path(path).parent))


class LeftSidebar(QWidget):
    """Left sidebar: image loading, ROI tools, parameters."""

    def __init__(
        self,
        image_ctrl: ImageController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_ctrl = image_ctrl
        self._state = image_ctrl._state

        self.setObjectName("leftSidebar")
        self.setFixedWidth(270)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- IMAGES section ---
        self._images_header = _SectionHeader("IMAGES")
        layout.addWidget(self._images_header)

        self._drop_zone = _DropZone(image_ctrl)
        layout.addWidget(self._drop_zone)

        # Natural sort checkbox
        self._natural_sort_cb = QCheckBox("Natural Sort (1, 2, …, 10)")
        self._natural_sort_cb.setChecked(False)
        self._natural_sort_cb.setToolTip(
            "Sort by embedded numbers: image1, image2, …, image10\n"
            "Default (unchecked): lexicographic — best for zero-padded names"
        )
        self._natural_sort_cb.setStyleSheet(
            f"QCheckBox {{ color: {COLORS.TEXT_SECONDARY}; font-size: 11px; "
            f"margin: 2px 12px; }}"
        )
        self._natural_sort_cb.toggled.connect(self._image_ctrl.set_natural_sort)
        layout.addWidget(self._natural_sort_cb)

        self._image_list = ImageList(self._state, image_ctrl)
        layout.addWidget(self._image_list, stretch=1)

        # --- ROI section ---
        roi_header = _SectionHeader("REGION OF INTEREST")
        layout.addWidget(roi_header)
        self._roi_toolbar = ROIToolbar()
        layout.addWidget(self._roi_toolbar)

        # --- PARAMETERS section ---
        params_header = _SectionHeader("PARAMETERS")
        layout.addWidget(params_header)
        self._param_panel = ParamPanel()
        layout.addWidget(self._param_panel)

        # Connect state changes to update badge
        self._state.images_changed.connect(self._update_badge)

    @property
    def roi_toolbar(self) -> ROIToolbar:
        """Access the ROI toolbar widget."""
        return self._roi_toolbar

    def _update_badge(self) -> None:
        """Update the IMAGES section badge with current count."""
        count = len(self._state.image_files)
        self._images_header.set_badge(str(count) if count > 0 else "")
