"""ROI toolbar — shape drawing tools with add/cut mode toggle.

Layout (2 rows x 3 columns):
    Row 1: [Rectangle] [Polygon] [Circle]   — exclusive shape toggles
    Row 2: [Import]    [Cut]     [Clear]     — actions

Rectangle / Polygon / Circle: selecting one activates that drawing tool.
Cut: toggles between "add" (blue) and "cut" (red) drawing modes.
Import: opens a file dialog to load a mask PNG.
Clear: resets the mask to empty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QPushButton,
    QWidget,
)

from staq_dic.gui.theme import COLORS

if TYPE_CHECKING:
    from staq_dic.gui.controllers.roi_controller import ROIController
    from staq_dic.gui.panels.canvas_area import ImageCanvas


class ROIToolbar(QWidget):
    """Six-button toolbar for ROI shape drawing and management."""

    # Emitted when active tool changes: "select", "rect", "polygon", "circle"
    tool_changed = Signal(str)
    # Emitted when drawing mode changes: "add" or "cut"
    mode_changed = Signal(str)
    # Emitted when clear is clicked
    clear_requested = Signal()
    # Emitted when a mask file is imported (path)
    import_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._active_tool: str = "select"
        self._drawing_mode: str = "add"

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # --- Row 1: shape tools (exclusive) ---
        self._btn_rect = QPushButton("Rect")
        self._btn_rect.setCheckable(True)
        self._btn_rect.setToolTip("Draw rectangle ROI")

        self._btn_polygon = QPushButton("Poly")
        self._btn_polygon.setCheckable(True)
        self._btn_polygon.setToolTip("Draw polygon ROI")

        self._btn_circle = QPushButton("Circle")
        self._btn_circle.setCheckable(True)
        self._btn_circle.setToolTip("Draw circle ROI")

        layout.addWidget(self._btn_rect, 0, 0)
        layout.addWidget(self._btn_polygon, 0, 1)
        layout.addWidget(self._btn_circle, 0, 2)

        self._shape_buttons = [
            self._btn_rect,
            self._btn_polygon,
            self._btn_circle,
        ]

        # --- Row 2: actions ---
        self._btn_import = QPushButton("Import")
        self._btn_import.setToolTip("Import mask from PNG file")

        self._btn_cut = QPushButton("Add")
        self._btn_cut.setCheckable(True)
        self._btn_cut.setToolTip("Toggle between Add / Cut mode")

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setToolTip("Clear all ROI masks")

        layout.addWidget(self._btn_import, 1, 0)
        layout.addWidget(self._btn_cut, 1, 1)
        layout.addWidget(self._btn_clear, 1, 2)

        # --- Connections ---
        self._btn_rect.clicked.connect(lambda: self._set_shape("rect"))
        self._btn_polygon.clicked.connect(lambda: self._set_shape("polygon"))
        self._btn_circle.clicked.connect(lambda: self._set_shape("circle"))
        self._btn_cut.clicked.connect(self._toggle_mode)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_import.clicked.connect(self._on_import)

        # Style the Cut/Add button
        self._update_cut_button_style()

    def _set_shape(self, tool: str) -> None:
        """Activate a shape tool, deactivating others."""
        # If the same tool is clicked again, deactivate it
        if self._active_tool == tool:
            self._active_tool = "select"
        else:
            self._active_tool = tool

        # Update button checked state
        tool_map = {
            "rect": self._btn_rect,
            "polygon": self._btn_polygon,
            "circle": self._btn_circle,
        }
        for name, btn in tool_map.items():
            btn.setChecked(name == self._active_tool)

        self.tool_changed.emit(self._active_tool)

    def _toggle_mode(self) -> None:
        """Toggle between add and cut drawing modes."""
        if self._drawing_mode == "add":
            self._drawing_mode = "cut"
        else:
            self._drawing_mode = "add"
        self._btn_cut.setChecked(self._drawing_mode == "cut")
        self._update_cut_button_style()
        self.mode_changed.emit(self._drawing_mode)

    def _update_cut_button_style(self) -> None:
        """Style the mode button to reflect current state."""
        if self._drawing_mode == "cut":
            self._btn_cut.setText("Cut")
            self._btn_cut.setStyleSheet(
                f"QPushButton {{ background: {COLORS.DANGER}; "
                f"color: #ffffff; border: 1px solid {COLORS.DANGER}; "
                f"border-radius: 4px; }}"
            )
        else:
            self._btn_cut.setText("Add")
            self._btn_cut.setStyleSheet(
                f"QPushButton {{ background: {COLORS.ACCENT}; "
                f"color: #ffffff; border: 1px solid {COLORS.ACCENT}; "
                f"border-radius: 4px; }}"
            )

    def _on_clear(self) -> None:
        """Emit clear signal."""
        self.clear_requested.emit()

    def _on_import(self) -> None:
        """Open file dialog and emit import signal with selected path."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Mask Image",
            "",
            "Images (*.png *.bmp *.tif *.tiff *.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self.import_requested.emit(path)
