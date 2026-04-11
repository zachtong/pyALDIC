"""Multi-column image list with per-frame ROI buttons.

QTreeWidget with 3 columns:
1. # (Frame number, "00", "01", ...) — fixed 32px
2. Filename — stretch
3. ROI — clickable QPushButton (50px): red "Need" / green "Edit" / gray "Add"

Ref frames get bold + accent-colored filenames for visibility.

Supports:
- Extended selection (Ctrl+click, Shift+click)
- Right-click context menu: Import/Clear ROI, Delete Selected
- ROI button indicators based on tracking mode / ref-frame status
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (
    QHeaderView,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.app_state import AppState
from al_dic.gui.controllers.image_controller import ImageController
from al_dic.gui.theme import COLORS

# Column indices
_COL_FRAME = 0
_COL_FILENAME = 1
_COL_ROI = 2

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
    """Scrollable multi-column image list with ROI indicators."""

    # Emitted when images are removed (indices list)
    images_removed = Signal(list)

    # Emitted when user clicks an ROI button to edit that frame's ROI
    roi_edit_requested = Signal(int)

    # Emitted when user imports ROI masks for specific frames via context menu
    # Carries dict {frame_idx: file_path}
    roi_import_for_frames = Signal(object)

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

        # Suppress _sync_selection when frame change originates from this widget
        self._suppress_sync = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["#", "Filename", "ROI"])
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._tree.setRootIsDecorated(False)
        self._tree.setIndentation(0)
        # Disable Qt's built-in autoScroll: when an item-widget QPushButton
        # (the ROI Add/Edit/Need button) receives a hover/focus event, Qt
        # otherwise calls ensureVisible() on the owning row, which scrolls
        # the bottommost visible row fully into view -- producing the
        # "hovering Add scrolls list to last frame" bug.  We re-add an
        # explicit scrollToItem in _sync_selection so keyboard navigation
        # and external frame changes still keep the current row visible.
        self._tree.setAutoScroll(False)

        # Column sizing
        header = self._tree.header()
        header.setMinimumSectionSize(10)
        header.resizeSection(_COL_FRAME, 32)
        header.resizeSection(_COL_ROI, 50)
        header.setSectionResizeMode(_COL_FRAME, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(_COL_FILENAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_ROI, QHeaderView.ResizeMode.Fixed)

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

        # Tree selection -> state update (keyboard: eventFilter, mouse: itemClicked)
        self._tree.installEventFilter(self)
        self._tree.itemClicked.connect(self._on_item_clicked)

    def _rebuild_list(self) -> None:
        """Rebuild the tree from current state."""
        self._tree.blockSignals(True)
        self._tree.clear()
        self._roi_buttons.clear()

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
            # NoFocus prevents the button from stealing focus on hover/click,
            # which would otherwise let Qt's view scroll-on-focus behaviour
            # drag the bottommost row into view (in addition to the
            # tree-level autoScroll guard set in __init__).
            roi_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            roi_btn.setStyleSheet(_STYLE_ROI_ADD)
            roi_btn.clicked.connect(
                self._make_roi_click_handler(i)
            )
            self._tree.setItemWidget(item, _COL_ROI, roi_btn)
            self._roi_buttons[i] = roi_btn

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

            # Ref-frame highlighting: subtle row background + bold accent filename
            n_cols = 3  # frame, filename, ROI
            if is_ref:
                highlight = QColor(COLORS.ACCENT)
                highlight = highlight.lighter(180)
                for col in range(n_cols):
                    item.setBackground(col, highlight)
                # Bold + accent color on filename for ref frames
                bold_font = QFont()
                bold_font.setBold(True)
                item.setFont(_COL_FILENAME, bold_font)
                item.setForeground(_COL_FILENAME, QColor(COLORS.ACCENT))
            else:
                for col in range(n_cols):
                    item.setData(col, Qt.ItemDataRole.BackgroundRole, None)
                # Reset filename styling
                item.setFont(_COL_FILENAME, QFont())
                item.setData(
                    _COL_FILENAME, Qt.ItemDataRole.ForegroundRole, None
                )

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
        if self._suppress_sync:
            return
        self._tree.blockSignals(True)
        item = self._tree.topLevelItem(idx)
        if item is not None:
            self._tree.setCurrentItem(item)
            # autoScroll is disabled tree-wide to stop hover-on-button from
            # scrolling the list, so we have to do explicit scrolling here
            # to keep the externally-selected frame visible (default hint
            # is EnsureVisible).
            self._tree.scrollToItem(item)
        self._tree.blockSignals(False)

    def eventFilter(self, obj, event):
        """Handle keyboard navigation in the tree — switch frame on arrow keys."""
        if obj is self._tree and event.type() == QEvent.Type.KeyRelease:
            if event.key() in (
                Qt.Key.Key_Up, Qt.Key.Key_Down,
                Qt.Key.Key_PageUp, Qt.Key.Key_PageDown,
                Qt.Key.Key_Home, Qt.Key.Key_End,
            ):
                current = self._tree.currentItem()
                if current is not None:
                    frame = current.data(_COL_FRAME, Qt.ItemDataRole.UserRole)
                    if frame is not None:
                        self._suppress_sync = True
                        self._state.set_current_frame(frame)
                        self._suppress_sync = False
                        # autoScroll is disabled tree-wide; scroll
                        # explicitly so keyboard nav still keeps the
                        # newly current row visible.
                        self._tree.scrollToItem(current)
        return super().eventFilter(obj, event)

    def _on_item_clicked(
        self,
        item: QTreeWidgetItem,
        column: int,
    ) -> None:
        """Handle mouse click — only switch frame for frame#/filename columns."""
        if column > _COL_FILENAME:
            return
        frame = item.data(_COL_FRAME, Qt.ItemDataRole.UserRole)
        if frame is not None:
            self._suppress_sync = True
            self._state.set_current_frame(frame)
            self._suppress_sync = False

    def _show_context_menu(self, pos) -> None:
        """Show right-click context menu with ROI and delete options."""
        menu = QMenu(self)
        selected = self._tree.selectedItems()
        sel_frames = self._get_selected_frames()
        n_sel = len(sel_frames)

        # --- ROI batch operations (when frames are selected) ---
        if n_sel > 0:
            import_action = QAction(
                f"Import ROI for {n_sel} frame{'s' if n_sel > 1 else ''}",
                self,
            )
            import_action.triggered.connect(self._import_roi_selected)
            menu.addAction(import_action)

            n_with_roi = sum(
                1 for f in sel_frames if f in self._state.per_frame_rois
            )
            clear_label = (
                f"Clear ROI ({n_with_roi} with ROI)"
                if n_with_roi
                else "Clear ROI"
            )
            clear_action = QAction(clear_label, self)
            clear_action.setEnabled(n_with_roi > 0)
            clear_action.triggered.connect(self._clear_roi_selected)
            menu.addAction(clear_action)

            menu.addSeparator()

        # --- Delete option ---
        if selected:
            count = len(selected)
            label = f"Delete {count} image{'s' if count > 1 else ''}"
            delete_action = QAction(label, self)
            delete_action.triggered.connect(self._delete_selected)
            menu.addAction(delete_action)

        if not menu.actions():
            return
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
        original_count = len(files)
        for row in rows_to_delete:
            if 0 <= row < len(files):
                del files[row]

        if not files:
            self._state.per_frame_rois.clear()
            self._state.set_image_files([])
            return

        # Re-key per_frame_rois after deletion.
        # Build old->new index mapping for surviving frames.
        deleted_set = set(rows_to_delete)
        old_to_new: dict[int, int] = {}
        new_idx = 0
        for old_idx in range(original_count):
            if old_idx not in deleted_set:
                old_to_new[old_idx] = new_idx
                new_idx += 1

        new_rois: dict[int, object] = {}
        for old_key, mask in list(self._state.per_frame_rois.items()):
            if old_key in old_to_new:
                new_rois[old_to_new[old_key]] = mask
        self._state.per_frame_rois = new_rois  # type: ignore[assignment]

        # Q6: image-list mutation invalidates pipeline results
        # (frame count / indices changed).
        from al_dic.gui.app_state import RunState
        had_results = self._state.results is not None
        if had_results:
            self._state.results = None
            self._state.deformed_masks = None
            self._state.run_state = RunState.IDLE
            self._state.show_deformed = False

        # Clear image caches for removed images
        self._image_ctrl.clear_cache()

        # Update state (triggers _rebuild_list via images_changed signal)
        self._state.image_files = files
        self._state.current_frame = min(
            self._state.current_frame, len(files) - 1
        )
        if had_results:
            self._state.results_changed.emit()
            self._state.run_state_changed.emit(RunState.IDLE)
        self._state.images_changed.emit()

    def _get_selected_frames(self) -> set[int]:
        """Extract frame indices from currently selected tree items."""
        frames = set()
        for item in self._tree.selectedItems():
            frame = item.data(_COL_FRAME, Qt.ItemDataRole.UserRole)
            if frame is not None:
                frames.add(frame)
        return frames

    def _import_roi_selected(self) -> None:
        """Open file dialog and import ROI masks for selected frames."""
        frames = sorted(self._get_selected_frames())
        if not frames:
            return
        from PySide6.QtWidgets import QFileDialog

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Select {len(frames)} Mask File{'s' if len(frames) > 1 else ''}",
            "",
            "Images (*.png *.bmp *.tif *.tiff *.jpg *.jpeg);;All Files (*)",
        )
        if not paths:
            return
        if len(paths) != len(frames):
            self._state.log_message.emit(
                f"Selected {len(paths)} files for {len(frames)} frames"
                " — count must match",
                "warn",
            )
            return
        self.roi_import_for_frames.emit(dict(zip(frames, paths)))

    def _clear_roi_selected(self) -> None:
        """Clear ROI masks for all selected frames.

        If frame 0 is among the cleared frames, also drop the brush
        refinement mask -- it lives only on frame 0 and is meaningless
        without an ROI to scope it (Brush#5).
        """
        frames = self._get_selected_frames()
        changed = False
        for f in frames:
            if f in self._state.per_frame_rois:
                del self._state.per_frame_rois[f]
                changed = True
        if 0 in frames and self._state.refine_brush_mask is not None:
            self._state.set_refine_brush_mask(None)
            changed = True
        if changed:
            self._state.roi_changed.emit()
