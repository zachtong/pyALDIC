"""Scrollable image list widget showing loaded frames.

Each item displays:
- Frame number (01, 02, ...)
- Filename
- "Reference" label for frame 1, "Frame N" for others
- Image dimensions (e.g., "2048x2048")
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from staq_dic.gui.app_state import AppState
from staq_dic.gui.controllers.image_controller import ImageController
from staq_dic.gui.theme import COLORS


class ImageListItem(QWidget):
    """Custom widget for a single image entry in the list."""

    def __init__(
        self,
        index: int,
        filename: str,
        dimensions: tuple[int, int] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Frame number badge
        frame_num = f"{index + 1:02d}"
        badge = QLabel(frame_num)
        badge.setFixedWidth(28)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 11px; font-weight: bold;"
        )
        layout.addWidget(badge)

        # File info column
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(1)

        name_label = QLabel(filename)
        name_label.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 12px;")
        info_layout.addWidget(name_label)

        # Role label + dimensions
        role = "Reference" if index == 0 else f"Frame {index + 1}"
        dim_text = ""
        if dimensions is not None:
            dim_text = f"  {dimensions[1]}\u00d7{dimensions[0]}"
        detail = QLabel(f"{role}{dim_text}")
        detail.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px;")
        info_layout.addWidget(detail)

        layout.addLayout(info_layout, stretch=1)


class ImageList(QWidget):
    """Scrollable list of loaded image files."""

    def __init__(
        self,
        state: AppState,
        image_ctrl: ImageController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._image_ctrl = image_ctrl

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._list_widget = QListWidget()
        self._list_widget.setStyleSheet(
            f"""
            QListWidget {{
                background: {COLORS.BG_SIDEBAR};
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                border-bottom: 1px solid {COLORS.BORDER};
                padding: 2px 0px;
            }}
            QListWidget::item:selected {{
                background: {COLORS.BG_HOVER};
            }}
            QListWidget::item:hover:!selected {{
                background: {COLORS.BG_INPUT};
            }}
            """
        )
        layout.addWidget(self._list_widget)

        # Connect signals
        self._state.images_changed.connect(self._rebuild_list)
        self._state.current_frame_changed.connect(self._on_frame_changed)
        self._list_widget.currentRowChanged.connect(self._on_row_selected)

    def _rebuild_list(self) -> None:
        """Rebuild the list from current state."""
        self._list_widget.blockSignals(True)
        self._list_widget.clear()

        for i, filepath in enumerate(self._state.image_files):
            filename = Path(filepath).name
            # Try to get dimensions (lazy — only read if needed)
            dims: tuple[int, int] | None = None
            try:
                dims = self._image_ctrl.image_dimensions(i)
            except Exception:
                pass

            item_widget = ImageListItem(i, filename, dims)
            item = QListWidgetItem()
            item.setSizeHint(item_widget.sizeHint())
            self._list_widget.addItem(item)
            self._list_widget.setItemWidget(item, item_widget)

        # Select frame 0 if images are loaded
        if self._state.image_files:
            self._list_widget.setCurrentRow(0)

        self._list_widget.blockSignals(False)

    def _on_row_selected(self, row: int) -> None:
        """Handle user clicking a list item."""
        if row >= 0:
            self._state.set_current_frame(row)

    def _on_frame_changed(self, idx: int) -> None:
        """Sync selection when frame is changed externally."""
        self._list_widget.blockSignals(True)
        self._list_widget.setCurrentRow(idx)
        self._list_widget.blockSignals(False)
