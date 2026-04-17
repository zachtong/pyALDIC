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
    QLabel,
    QMenu,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from al_dic.gui.theme import COLORS


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

    # Brush refinement signals
    # (mode, radius_px) — mode is "paint" or "erase"
    brush_requested = Signal(str, int)
    brush_clear_requested = Signal()
    # Emitted live whenever the radius spinbox changes, so the canvas
    # can update its active brush radius without re-clicking Paint/Erase.
    brush_radius_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._active_mode: str | None = None  # "add" or "cut" while drawing

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # --- Row 1: Add / Cut dropdown buttons ---
        self._btn_add = QPushButton("+ Add  \u25b4")
        self._btn_add.setToolTip(
            "Add region to the Region of Interest "
            "(Polygon / Rectangle / Circle)"
        )
        self._btn_add.setFixedHeight(30)

        self._btn_cut = QPushButton("\u2702 Cut  \u25b4")
        self._btn_cut.setToolTip(
            "Cut region from the Region of Interest "
            "(Polygon / Rectangle / Circle)"
        )
        self._btn_cut.setFixedHeight(30)

        self._btn_refine = QPushButton("+ Refine  \u25b4")
        # The enabled tooltip is also the default; when the button is
        # disabled (non-frame-0) we swap in a short explanation instead.
        self._refine_enabled_tooltip = (
            "Paint extra mesh-refinement zones with a brush\n"
            "(only on frame 1 — material points auto-warped to later frames)"
        )
        self._refine_disabled_tooltip = (
            "Refine brush is only available on frame 1. Switch to frame 1 "
            "to paint refinement zones; they are automatically warped to "
            "later frames."
        )
        self._btn_refine.setToolTip(self._refine_enabled_tooltip)
        self._btn_refine.setFixedHeight(30)

        layout.addWidget(self._btn_add, 0, 0)
        layout.addWidget(self._btn_cut, 0, 1)
        layout.addWidget(self._btn_refine, 0, 2)

        # Build popup menus
        self._add_menu = self._build_shape_menu("add")
        self._cut_menu = self._build_shape_menu("cut")
        self._brush_menu = self._build_brush_menu()

        self._btn_add.clicked.connect(self._show_add_menu)
        self._btn_cut.clicked.connect(self._show_cut_menu)
        self._btn_refine.clicked.connect(self._show_brush_menu)

        # --- Row 2: Import (single / batch) ---
        import_row = QHBoxLayout()
        import_row.setSpacing(4)

        self._btn_import = QPushButton("Import")
        self._btn_import.setToolTip("Import mask from image file")
        self._btn_import.setFixedHeight(26)
        self._btn_import.clicked.connect(self._on_import)
        import_row.addWidget(self._btn_import)

        self._btn_batch = QPushButton("Batch Import")
        self._btn_batch.setToolTip("Batch import mask files for multiple frames")
        self._btn_batch.setFixedHeight(26)
        self._btn_batch.clicked.connect(
            lambda: self.batch_import_requested.emit()
        )
        import_row.addWidget(self._btn_batch)

        layout.addLayout(import_row, 1, 0, 1, 3)

        # --- Row 3: Save / Invert / Clear ---
        action_row = QHBoxLayout()
        action_row.setSpacing(4)

        self._btn_save = QPushButton("Save")
        self._btn_save.setToolTip("Save current mask to PNG file")
        self._btn_save.setFixedHeight(26)
        self._btn_save.clicked.connect(lambda: self.save_requested.emit())
        action_row.addWidget(self._btn_save)

        self._btn_invert = QPushButton("Invert")
        self._btn_invert.setToolTip("Invert the Region of Interest mask")
        self._btn_invert.setFixedHeight(26)
        self._btn_invert.clicked.connect(lambda: self.invert_requested.emit())
        action_row.addWidget(self._btn_invert)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setToolTip("Clear all Region of Interest masks")
        self._btn_clear.setFixedHeight(26)
        self._btn_clear.clicked.connect(lambda: self.clear_requested.emit())
        action_row.addWidget(self._btn_clear)

        layout.addLayout(action_row, 2, 0, 1, 3)

        # Lock all utility button geometry to prevent layout jitter
        # when stylesheets change on the row-0 buttons.
        _util_style = (
            f"QPushButton {{ background: {COLORS.BG_INPUT}; "
            f"color: {COLORS.TEXT_PRIMARY}; border: 1px solid {COLORS.BORDER}; "
            f"{self._GEOM} }}"
        )
        for btn in (
            self._btn_import, self._btn_batch,
            self._btn_save, self._btn_invert, self._btn_clear,
        ):
            btn.setStyleSheet(_util_style)

        # Apply initial styling
        self._update_button_styles()

    def _build_brush_menu(self) -> QMenu:
        """Create the Refine brush popup menu (radius + Paint/Erase + Clear)."""
        menu = QMenu(self)

        # Radius row: label + spinbox embedded via QWidgetAction
        radius_widget = QWidget()
        radius_layout = QHBoxLayout(radius_widget)
        radius_layout.setContentsMargins(8, 4, 8, 4)
        radius_layout.setSpacing(6)
        radius_layout.addWidget(QLabel("Radius"))
        self._brush_radius_spin = QSpinBox()
        self._brush_radius_spin.setRange(2, 500)
        self._brush_radius_spin.setValue(16)
        self._brush_radius_spin.setSuffix(" px")
        # Live-update the canvas radius so the user does not have to
        # re-click Paint/Erase after changing the spinbox.  Applies
        # equally to paint and erase modes.
        self._brush_radius_spin.valueChanged.connect(
            lambda v: self.brush_radius_changed.emit(int(v))
        )
        radius_layout.addWidget(self._brush_radius_spin)
        radius_action = QWidgetAction(menu)
        radius_action.setDefaultWidget(radius_widget)
        menu.addAction(radius_action)

        menu.addSeparator()
        menu.addAction(
            "\u270E  Paint", lambda: self._on_brush_selected(
                "paint", self._brush_radius_spin.value()
            )
        )
        menu.addAction(
            "\u2716  Erase", lambda: self._on_brush_selected(
                "erase", self._brush_radius_spin.value()
            )
        )
        menu.addSeparator()

        # Clear button as QWidgetAction so we can keep a public ref for tests
        clear_widget = QWidget()
        clear_layout = QVBoxLayout(clear_widget)
        clear_layout.setContentsMargins(8, 4, 8, 4)
        self._brush_clear_btn = QPushButton("Clear Brush")
        self._brush_clear_btn.clicked.connect(
            lambda: self.brush_clear_requested.emit()
        )
        clear_layout.addWidget(self._brush_clear_btn)
        clear_action = QWidgetAction(menu)
        clear_action.setDefaultWidget(clear_widget)
        menu.addAction(clear_action)

        return menu

    def _show_brush_menu(self) -> None:
        """Show the Refine brush popup above the Refine button."""
        pos = self._btn_refine.mapToGlobal(self._btn_refine.rect().topLeft())
        menu_height = self._brush_menu.sizeHint().height()
        pos.setY(pos.y() - menu_height)
        self._brush_menu.popup(pos)

    def _on_brush_selected(self, mode: str, radius: int) -> None:
        """Handle Paint / Erase selection from the brush popup."""
        self._active_mode = "brush_paint" if mode == "paint" else "brush_erase"
        self._update_button_styles()
        self.brush_requested.emit(mode, int(radius))

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
        # Position menu above the button
        pos = self._btn_add.mapToGlobal(self._btn_add.rect().topLeft())
        menu_height = self._add_menu.sizeHint().height()
        pos.setY(pos.y() - menu_height)
        self._add_menu.popup(pos)

    def _show_cut_menu(self) -> None:
        """Show the Cut shape popup above the Cut button."""
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

    # Shared geometry tokens — keep padding/margin identical across all
    # states so that Qt never recalculates sizeHint on style switches.
    _GEOM = "border-radius: 4px; padding: 2px 4px; margin: 0px;"

    # The :disabled rule is critical — without it, our custom QPushButton
    # selector shadows Qt's default gray-out, leaving disabled buttons
    # indistinguishable from enabled ones. Refine-brush gating relies
    # on this disabled appearance to signal "switch to frame 1".
    _BASE_STYLE = (
        f"QPushButton {{ background: {COLORS.BG_INPUT}; "
        f"color: {COLORS.TEXT_PRIMARY}; border: 1px solid {COLORS.BORDER}; "
        f"{_GEOM} }} "
        f"QPushButton:disabled {{ background: {COLORS.BG_DARKEST}; "
        f"color: {COLORS.TEXT_MUTED}; "
        f"border: 1px dashed {COLORS.TEXT_MUTED}; }}"
    )

    def _make_active_style(self, bg: str, fg: str = "#ffffff") -> str:
        return (
            f"QPushButton {{ background: {bg}; color: {fg}; "
            f"border: 1px solid {bg}; {self._GEOM} }}"
        )

    def _update_button_styles(self) -> None:
        """Update Add/Cut/Refine button highlight based on active mode."""
        # Reset all to a consistent base style (never use empty string —
        # that triggers Qt style-engine fallback and causes layout jitter).
        self._btn_add.setStyleSheet(self._BASE_STYLE)
        self._btn_cut.setStyleSheet(self._BASE_STYLE)
        self._btn_refine.setStyleSheet(self._BASE_STYLE)

        if self._active_mode == "add":
            self._btn_add.setStyleSheet(
                self._make_active_style(COLORS.ACCENT)
            )
        elif self._active_mode == "cut":
            self._btn_cut.setStyleSheet(
                self._make_active_style(COLORS.DANGER)
            )
        elif self._active_mode in ("brush_paint", "brush_erase"):
            color = "#14dcc8" if self._active_mode == "brush_paint" else "#0a8a7a"
            self._btn_refine.setStyleSheet(
                self._make_active_style(color, "#001417")
            )

    def set_refine_brush_enabled(self, enabled: bool) -> None:
        """Enable or disable the Refine brush button.

        Refine strokes are warped forward from frame 1; painting on any
        other frame produces visually surprising results (strokes in the
        deformed coordinate system rather than the reference one), so
        the button is gated to frame 1.
        """
        self._btn_refine.setEnabled(enabled)
        self._btn_refine.setToolTip(
            self._refine_enabled_tooltip if enabled
            else self._refine_disabled_tooltip
        )

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
