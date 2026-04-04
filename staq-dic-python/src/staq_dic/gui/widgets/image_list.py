"""Multi-column image list with per-frame ROI buttons and display override.

QTreeWidget with 4 columns:
1. # (Frame number, "00", "01", ...) — fixed 32px
2. Filename — stretch
3. ROI — clickable QPushButton (50px): red "Need" / green "Edit" / gray "Add"
4. Disp — QCheckBox (40px): display override for deformed configuration

Supports:
- Extended selection (Ctrl+click, Shift+click)
- Right-click context menu: Delete Selected
- ROI button indicators based on tracking mode / ref-frame status
- Display override checkboxes per frame
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from staq_dic.gui.app_state import AppState
from staq_dic.gui.controllers.image_controller import ImageController
from staq_dic.gui.theme import COLORS

# Column indices
_COL_FRAME = 0
_COL_FILENAME = 1
_COL_ROI = 2
_COL_DISP = 3

# ROI button style templates
_STYLE_ROI_HAS = (
    f"background: {COLORS.SUCCESS}; color: #fff; border: none; "
    f"border-radius: 3px; font-size: 10px; font-weight: bold;"
)
_STYLE_ROI_NEED = (
    f"background: {COLORS.DANGER}; color: #fff; border: none; "
    f"border-radius: 3px; font-size: 10px; font-weight: bold;"
)
_STYLE_ROI_ADD = (
    f"background: {COLORS.BG_INPUT}; color: {COLORS.TEXT_MUTED}; "
    f"border: none; border-radius: 3px; font-size: 10px;"
)


class ImageList(QWidget):
    """Scrollable multi-column image list with ROI indicators and display override."""

    # Emitted when images are removed (indices list)
    images_removed = Signal(list)

    # Emitted when user clicks an ROI button to edit that frame's ROI
    roi_edit_requested = Signal(int)

    def __init__(
        self,
        state: AppState,
        image_ctrl: ImageController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._image_ctrl = image_ctrl

        # Lookup maps for per-row widgets
        self._roi_buttons: dict[int, QPushButton] = {}
        self._disp_checks: dict[int, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["#", "Filename", "ROI", "Disp"])
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._tree.setRootIsDecorated(False)
        self._tree.setIndentation(0)

        # Column sizing
        header = self._tree.header()
        header.setMinimumSectionSize(10)
        header.resizeSection(_COL_FRAME, 32)
        header.resizeSection(_COL_ROI, 50)
        header.resizeSection(_COL_DISP, 40)
        header.setSectionResizeMode(_COL_FRAME, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(_COL_FILENAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_ROI, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(_COL_DISP, QHeaderView.ResizeMode.Fixed)

        # Styling
        self._tree.setStyleSheet(
            f"""
            QTreeWidget {{
                background: {COLORS.BG_SIDEBAR};
                border: none;
                outline: none;
            }}
            QTreeWidget::item {{
                border-bottom: 1px solid {COLORS.BORDER};
                padding: 2px 0px;
            }}
            QTreeWidget::item:selected {{
                background: {COLORS.BG_HOVER};
            }}
            QTreeWidget::item:hover:!selected {{
                background: {COLORS.BG_INPUT};
            }}
            QHeaderView::section {{
                background: {COLORS.BG_PANEL};
                color: {COLORS.TEXT_MUTED};
                border: none;
                border-bottom: 1px solid {COLORS.BORDER};
                border-right: 1px solid {COLORS.BORDER};
                padding: 3px 4px;
                font-size: 10px;
                font-weight: bold;
            }}
            """
        )
        layout.addWidget(self._tree)

        # Connect state signals
        self._state.images_changed.connect(self._rebuild_list)
        self._state.current_frame_changed.connect(self._sync_selection)
        self._state.roi_changed.connect(self._update_roi_indicators)
        self._state.params_changed.connect(self._update_roi_indicators)

        # Tree selection -> state update
        self._tree.currentItemChanged.connect(self._on_item_changed)

    def _rebuild_list(self) -> None:
        """Rebuild the tree from current state."""
        self._tree.blockSignals(True)
        self._tree.clear()
        self._roi_buttons.clear()
        self._disp_checks.clear()

        for i, filepath in enumerate(self._state.image_files):
            filename = Path(filepath).name
            item = QTreeWidgetItem()
            item.setText(_COL_FRAME, f"{i:02d}")
            item.setTextAlignment(
                _COL_FRAME, Qt.AlignmentFlag.AlignCenter
            )
            item.setText(_COL_FILENAME, filename)
            item.setData(
                _COL_FRAME, Qt.ItemDataRole.UserRole, i
            )
            # Disable default editing
            item.setFlags(
                item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            self._tree.addTopLevelItem(item)

            # ROI button
            roi_btn = QPushButton("Add")
            roi_btn.setFixedHeight(20)
            roi_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            roi_btn.setStyleSheet(_STYLE_ROI_ADD)
            roi_btn.clicked.connect(
                self._make_roi_click_handler(i)
            )
            self._tree.setItemWidget(item, _COL_ROI, roi_btn)
            self._roi_buttons[i] = roi_btn

            # Display checkbox
            disp_container = QWidget()
            disp_layout = QHBoxLayout(disp_container)
            disp_layout.setContentsMargins(0, 0, 0, 0)
            disp_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            disp_cb = QCheckBox()
            disp_cb.setChecked(
                self._state.display_roi_enabled.get(i, False)
            )
            disp_cb.toggled.connect(
                self._make_disp_toggle_handler(i)
            )
            disp_layout.addWidget(disp_cb)
            self._tree.setItemWidget(item, _COL_DISP, disp_container)
            self._disp_checks[i] = disp_cb

        # Select frame 0 if images are loaded
        if self._state.image_files:
            first_item = self._tree.topLevelItem(0)
            if first_item is not None:
                self._tree.setCurrentItem(first_item)

        self._tree.blockSignals(False)

        # Update ROI indicators after rebuild
        self._update_roi_indicators()

    def _make_roi_click_handler(self, frame: int):
        """Create a closure for the ROI button click of a specific frame."""

        def handler() -> None:
            self.roi_edit_requested.emit(frame)

        return handler

    def _make_disp_toggle_handler(self, frame: int):
        """Create a closure for the display checkbox toggle of a specific frame."""

        def handler(checked: bool) -> None:
            self._state.display_roi_enabled[frame] = checked
            self._state.display_changed.emit()

        return handler

    def _update_roi_indicators(self) -> None:
        """Update ROI button text/style and ref-frame row highlighting."""
        n = len(self._state.image_files)
        if n == 0:
            return

        ref_frames = self._get_ref_frames()

        for i in range(n):
            item = self._tree.topLevelItem(i)
            btn = self._roi_buttons.get(i)
            if item is None or btn is None:
                continue

            has_roi = i in self._state.per_frame_rois
            is_ref = i in ref_frames

            if has_roi:
                btn.setText("Edit")
                btn.setStyleSheet(_STYLE_ROI_HAS)
            elif is_ref:
                btn.setText("Need")
                btn.setStyleSheet(_STYLE_ROI_NEED)
            else:
                btn.setText("Add")
                btn.setStyleSheet(_STYLE_ROI_ADD)

            # Ref-frame row highlight
            if is_ref:
                highlight = QColor(COLORS.ACCENT)
                highlight = highlight.lighter(180)
                for col in range(_COL_DISP + 1):
                    item.setBackground(col, highlight)
            else:
                for col in range(_COL_DISP + 1):
                    item.setData(col, Qt.ItemDataRole.BackgroundRole, None)

    def _get_ref_frames(self) -> set[int]:
        """Compute which frames are reference frames based on tracking settings."""
        n = len(self._state.image_files)
        if n == 0:
            return set()

        mode = self._state.tracking_mode

        if mode == "accumulative":
            return {0}

        # Incremental modes
        sub_mode = self._state.inc_ref_mode

        if sub_mode == "every_frame":
            # Every frame except the last is a reference
            return set(range(n - 1))

        if sub_mode == "every_n":
            interval = max(1, self._state.inc_ref_interval)
            return {i for i in range(0, n - 1, interval)}

        if sub_mode == "custom":
            custom = set(self._state.inc_custom_refs) | {0}
            return {i for i in custom if i < n - 1}

        # Fallback
        return {0}

    def _sync_selection(self, idx: int) -> None:
        """Sync tree selection when frame is changed externally."""
        self._tree.blockSignals(True)
        item = self._tree.topLevelItem(idx)
        if item is not None:
            self._tree.setCurrentItem(item)
        self._tree.blockSignals(False)

    def _on_item_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        """Handle user clicking a tree item."""
        if current is None:
            return
        frame = current.data(_COL_FRAME, Qt.ItemDataRole.UserRole)
        if frame is not None:
            self._state.set_current_frame(frame)

    def _show_context_menu(self, pos) -> None:
        """Show right-click context menu with delete option."""
        selected = self._tree.selectedItems()
        if not selected:
            return

        menu = QMenu(self)
        count = len(selected)
        label = f"Delete {count} image{'s' if count > 1 else ''}"
        delete_action = QAction(label, self)
        delete_action.triggered.connect(self._delete_selected)
        menu.addAction(delete_action)
        menu.popup(self._tree.mapToGlobal(pos))

    def _delete_selected(self) -> None:
        """Remove selected images from the list and update state."""
        selected_items = self._tree.selectedItems()
        if not selected_items:
            return

        # Collect frame indices to delete (high indices first)
        rows_to_delete = sorted(
            {
                item.data(_COL_FRAME, Qt.ItemDataRole.UserRole)
                for item in selected_items
                if item.data(_COL_FRAME, Qt.ItemDataRole.UserRole) is not None
            },
            reverse=True,
        )
        if not rows_to_delete:
            return

        # Remove from file list (high indices first to preserve ordering)
        files = list(self._state.image_files)
        for row in rows_to_delete:
            if 0 <= row < len(files):
                del files[row]
                # Clean up per-frame ROI and display data
                self._state.per_frame_rois.pop(row, None)
                self._state.display_roi_enabled.pop(row, None)

        if not files:
            self._state.set_image_files([])
            return

        # Clear image caches for removed images
        self._image_ctrl.clear_cache()

        # Update state (triggers _rebuild_list via images_changed signal)
        self._state.image_files = files
        self._state.current_frame = min(
            self._state.current_frame, len(files) - 1
        )
        self._state.images_changed.emit()
