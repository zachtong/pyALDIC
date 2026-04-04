"""ROI toolbar — Add/Cut dropdown buttons with shape popup menus.

Layout:
    Row 1: [+ Add ^]  [scissors Cut ^]    — dropdown shape selectors
    Row 2: [Import] [Save] [Invert] [Clear] — utility buttons

Each Add/Cut click opens a popup menu (Polygon / Rectangle / Circle).
Selecting a shape activates one-shot drawing mode — the tool auto-resets
to "select" after completing one shape.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QWidget,
)

from staq_dic.gui.theme import COLORS


class ROIToolbar(QWidget):
    """Add/Cut dropdown toolbar with Import/Save/Invert/Clear utilities."""

    # Emitted when user selects a shape from Add or Cut menu: (shape, mode)
    # shape: "rect", "polygon", "circle"
    # mode: "add" or "cut"
    draw_requested = Signal(str, str)

    # Emitted when clear is clicked
    clear_requested = Signal()

    # Emitted when a mask file is imported (path)
    import_requested = Signal(str)

    # Emitted when save is clicked
    save_requested = Signal()

    # Emitted when batch import is requested
    batch_import_requested = Signal()

    # Emitted when invert is clicked
    invert_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._active_mode: str | None = None  # "add" or "cut" while drawing

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # --- Row 1: Add / Cut dropdown buttons ---
        self._btn_add = QPushButton("+ Add  \u25b4")
        self._btn_add.setCheckable(True)
        self._btn_add.setToolTip("Add region to ROI (Polygon / Rectangle / Circle)")
        self._btn_add.setFixedHeight(30)

        self._btn_cut = QPushButton("\u2702 Cut  \u25b4")
        self._btn_cut.setCheckable(True)
        self._btn_cut.setToolTip("Cut region from ROI (Polygon / Rectangle / Circle)")
        self._btn_cut.setFixedHeight(30)

        layout.addWidget(self._btn_add, 0, 0)
        layout.addWidget(self._btn_cut, 0, 1)

        # Build popup menus
        self._add_menu = self._build_shape_menu("add")
        self._cut_menu = self._build_shape_menu("cut")

        self._btn_add.clicked.connect(self._show_add_menu)
        self._btn_cut.clicked.connect(self._show_cut_menu)

        # --- Row 2: Import / Save / Invert / Clear ---
        util_row = QHBoxLayout()
        util_row.setSpacing(4)

        self._btn_import = QPushButton("Import")
        self._btn_import.setToolTip("Import mask from image file")
        self._btn_import.setFixedHeight(26)
        self._btn_import.clicked.connect(self._on_import)
        util_row.addWidget(self._btn_import)

        self._btn_batch = QPushButton("Batch")
        self._btn_batch.setToolTip("Batch import mask files for multiple frames")
        self._btn_batch.setFixedHeight(26)
        self._btn_batch.clicked.connect(
            lambda: self.batch_import_requested.emit()
        )
        util_row.addWidget(self._btn_batch)

        self._btn_save = QPushButton("Save")
        self._btn_save.setToolTip("Save current mask to PNG file")
        self._btn_save.setFixedHeight(26)
        self._btn_save.clicked.connect(lambda: self.save_requested.emit())
        util_row.addWidget(self._btn_save)

        self._btn_invert = QPushButton("Invert")
        self._btn_invert.setToolTip("Invert the ROI mask")
        self._btn_invert.setFixedHeight(26)
        self._btn_invert.clicked.connect(lambda: self.invert_requested.emit())
        util_row.addWidget(self._btn_invert)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setToolTip("Clear all ROI masks")
        self._btn_clear.setFixedHeight(26)
        self._btn_clear.clicked.connect(lambda: self.clear_requested.emit())
        util_row.addWidget(self._btn_clear)

        layout.addLayout(util_row, 1, 0, 1, 2)

        # Apply initial styling
        self._update_button_styles()

    def _build_shape_menu(self, mode: str) -> QMenu:
        """Create a popup menu with Polygon / Rectangle / Circle actions."""
        menu = QMenu(self)
        menu.addAction(
            "\u2b1f  Polygon", lambda: self._on_shape_selected("polygon", mode)
        )
        menu.addAction(
            "\u25a1  Rectangle", lambda: self._on_shape_selected("rect", mode)
        )
        menu.addAction(
            "\u25cb  Circle", lambda: self._on_shape_selected("circle", mode)
        )
        return menu

    def _show_add_menu(self) -> None:
        """Show the Add shape popup above the Add button."""
        # Uncheck immediately — the checked state is controlled by _on_shape_selected
        self._btn_add.setChecked(False)
        # Position menu above the button
        pos = self._btn_add.mapToGlobal(self._btn_add.rect().topLeft())
        menu_height = self._add_menu.sizeHint().height()
        pos.setY(pos.y() - menu_height)
        self._add_menu.popup(pos)

    def _show_cut_menu(self) -> None:
        """Show the Cut shape popup above the Cut button."""
        self._btn_cut.setChecked(False)
        pos = self._btn_cut.mapToGlobal(self._btn_cut.rect().topLeft())
        menu_height = self._cut_menu.sizeHint().height()
        pos.setY(pos.y() - menu_height)
        self._cut_menu.popup(pos)

    def _on_shape_selected(self, shape: str, mode: str) -> None:
        """Handle shape selection from popup menu."""
        self._active_mode = mode
        self._update_button_styles()
        self.draw_requested.emit(shape, mode)

    def deactivate(self) -> None:
        """Reset the active state — called when drawing finishes or is canceled."""
        self._active_mode = None
        self._update_button_styles()

    def _update_button_styles(self) -> None:
        """Update Add/Cut button highlight based on active mode."""
        if self._active_mode == "add":
            self._btn_add.setChecked(True)
            self._btn_cut.setChecked(False)
            self._btn_cut.setStyleSheet("")
            self._btn_add.setStyleSheet(
                f"QPushButton {{ background: {COLORS.ACCENT}; "
                f"color: #ffffff; border: 1px solid {COLORS.ACCENT}; "
                f"border-radius: 4px; }}"
            )
        elif self._active_mode == "cut":
            self._btn_cut.setChecked(True)
            self._btn_add.setChecked(False)
            self._btn_add.setStyleSheet("")
            self._btn_cut.setStyleSheet(
                f"QPushButton {{ background: {COLORS.DANGER}; "
                f"color: #ffffff; border: 1px solid {COLORS.DANGER}; "
                f"border-radius: 4px; }}"
            )
        else:
            self._btn_add.setChecked(False)
            self._btn_cut.setChecked(False)
            self._btn_add.setStyleSheet("")
            self._btn_cut.setStyleSheet("")

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
