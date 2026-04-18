"""Left sidebar panel — image loading, ROI tools, parameters.

Fixed width 270px.  Contains:
1. IMAGES section with count badge and drop zone (always visible)
2. ImageList (scrollable file list)
3. REGION OF INTEREST — collapsible
4. PARAMETERS — collapsible
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.app_state import AppState
from al_dic.gui.controllers.image_controller import ImageController
from al_dic.gui.theme import COLORS
from al_dic.gui.widgets.collapsible_section import CollapsibleSection
from al_dic.gui.widgets.image_list import ImageList
from al_dic.gui.widgets.init_guess_widget import InitGuessWidget
from al_dic.gui.widgets.mesh_appearance_widget import MeshAppearanceWidget
from al_dic.gui.widgets.param_panel import ParamPanel
from al_dic.gui.widgets.roi_hint import ROIHint
from al_dic.gui.widgets.roi_toolbar import ROIToolbar
from al_dic.gui.widgets.workflow_type_panel import WorkflowTypePanel


class _SectionHeader(QWidget):
    """Compact section header with title and optional badge (for IMAGES)."""

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
        # Extra 18 px accounts for the vertical scrollbar in the settings
        # scroll area below so the viewport content width stays ≥ 270 px
        # whether or not the scrollbar is visible.
        self.setFixedWidth(288)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- IMAGES section (always visible, not collapsible) ---
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

        # Image list: fixed height range — internal QTreeWidget handles its own
        # scrolling, so capping the height here is safe.  No stretch: the
        # settings scroll area below it will absorb all remaining space.
        self._image_list = ImageList(self._state, image_ctrl)
        self._image_list.setMinimumHeight(80)
        self._image_list.setMaximumHeight(240)
        layout.addWidget(self._image_list)

        # --- Collapsible settings sections inside a QScrollArea ---
        # Putting ROI + PARAMETERS + ADVANCED in a scroll area means that
        # expanding any section grows the content inside the scroll area
        # (showing a scrollbar) rather than pushing the image list upward.
        self._settings_scroll = QScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._settings_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        # Use setFrameShape instead of CSS border so the physical QFrame
        # border is truly removed (CSS "border: none" only hides the
        # CSS-drawn border; the underlying QFrame still paints its frame).
        self._settings_scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Plain QWidget — NO stylesheet on the container.
        # Setting any stylesheet (even just "background: transparent") on
        # a QWidget triggers Qt's stylesheet engine cascade which strips
        # native drawing from all descendant buttons and inputs, causing
        # unexpected style and margin changes.
        settings_container = QWidget()
        # Force the container to follow the scroll-area viewport width.
        # Without this, child widgets' minimumSizeHint (e.g. 343 px from
        # the collapsed ADVANCED section's default 640 px content widget)
        # prevents setWidgetResizable(True) from constraining the container
        # to the 288 px viewport, causing horizontal overflow.
        from PySide6.QtWidgets import QSizePolicy
        settings_container.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(0)

        # Workflow Type comes BEFORE Region of Interest because the tracking
        # mode and reference-update policy determine which frames need a
        # Region of Interest. Picking that first avoids drawing regions on
        # the wrong frames.
        self._workflow_section = CollapsibleSection(
            "WORKFLOW TYPE", expanded=True,
        )
        self._workflow_panel = WorkflowTypePanel()
        self._workflow_section.add_widget(self._workflow_panel)
        settings_layout.addWidget(self._workflow_section)

        self._roi_section = CollapsibleSection("REGION OF INTEREST", expanded=True)
        # Dynamic hint: lives above the toolbar and updates whenever the
        # workflow-type panel changes tracking mode / ref-update policy.
        self._roi_hint = ROIHint()
        self._roi_section.add_widget(self._roi_hint)
        self._roi_toolbar = ROIToolbar()
        self._roi_section.add_widget(self._roi_toolbar)
        settings_layout.addWidget(self._roi_section)

        self._params_section = CollapsibleSection("PARAMETERS", expanded=True)
        self._param_panel = ParamPanel()
        self._params_section.add_widget(self._param_panel)
        settings_layout.addWidget(self._params_section)

        self._advanced_section = CollapsibleSection("ADVANCED", expanded=False)
        self._mesh_appearance = MeshAppearanceWidget()
        self._advanced_section.add_widget(self._mesh_appearance)
        self._init_guess_widget = InitGuessWidget()
        self._advanced_section.add_widget(self._init_guess_widget)
        settings_layout.addWidget(self._advanced_section)

        # Pushes sections to the top when the scroll area is taller than content
        settings_layout.addStretch()

        self._settings_scroll.setWidget(settings_container)
        layout.addWidget(self._settings_scroll, stretch=1)

        # Connect state changes to update badge
        self._state.images_changed.connect(self._update_badge)

        # Gate Refine brush to frame 1 only (material points warp forward
        # from frame 0, so brushing on later frames is misleading).
        self._state.current_frame_changed.connect(
            self._on_current_frame_changed
        )
        self._on_current_frame_changed(self._state.current_frame)

    @property
    def roi_toolbar(self) -> ROIToolbar:
        """Access the ROI toolbar widget."""
        return self._roi_toolbar

    @property
    def init_guess_widget(self) -> InitGuessWidget:
        """Access the Initial-Guess panel (for top-level controller wiring)."""
        return self._init_guess_widget

    def _update_badge(self) -> None:
        """Update the IMAGES section badge with current count."""
        count = len(self._state.image_files)
        self._images_header.set_badge(str(count) if count > 0 else "")

    def _on_current_frame_changed(self, frame: int) -> None:
        """Gate Refine brush to frame 1 (index 0)."""
        self._roi_toolbar.set_refine_brush_enabled(frame == 0)
