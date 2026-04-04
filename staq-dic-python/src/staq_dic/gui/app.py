"""STAQ-DIC GUI application entry point."""

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget

from staq_dic.gui.app_state import AppState
from staq_dic.gui.controllers.image_controller import ImageController
from staq_dic.gui.controllers.pipeline_controller import PipelineController
from staq_dic.gui.controllers.roi_controller import ROIController
from staq_dic.gui.panels.canvas_area import CanvasArea
from staq_dic.gui.panels.left_sidebar import LeftSidebar
from staq_dic.gui.panels.right_sidebar import RightSidebar
from staq_dic.gui.theme import COLORS, build_stylesheet


class MainWindow(QMainWindow):
    """Three-column main window: left sidebar | canvas | right sidebar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STAQ-DIC v0.1")
        self.setMinimumSize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # State and controllers
        self._state = AppState.instance()
        self._image_ctrl = ImageController(self._state)
        self._roi_ctrl: ROIController | None = None

        # Left sidebar — image loading + ROI toolbar
        self._left_sidebar = LeftSidebar(self._image_ctrl)
        layout.addWidget(self._left_sidebar, stretch=0)

        # Center canvas
        self._canvas_area = CanvasArea(self._image_ctrl)
        layout.addWidget(self._canvas_area, stretch=1)

        # Pipeline controller
        self._pipeline_ctrl = PipelineController(self._state, self._image_ctrl)

        # Right sidebar -- run controls, progress, display, console
        self._right_sidebar = RightSidebar(self._pipeline_ctrl)
        layout.addWidget(self._right_sidebar, stretch=0)

        # Wire ROI toolbar signals to canvas
        roi_tb = self._left_sidebar.roi_toolbar
        roi_tb.tool_changed.connect(self._canvas_area.canvas.set_tool)
        roi_tb.mode_changed.connect(self._canvas_area.canvas.set_drawing_mode)
        roi_tb.clear_requested.connect(self._on_roi_clear)
        roi_tb.import_requested.connect(self._on_roi_import)

        # Initialize ROI controller when images are loaded
        self._state.images_changed.connect(self._init_roi_controller)

    def _init_roi_controller(self) -> None:
        """Create a ROI controller matching the loaded image dimensions."""
        if not self._state.image_files:
            return
        try:
            rgb = self._image_ctrl.read_image_rgb(0)
            h, w = rgb.shape[:2]
            self._roi_ctrl = ROIController((h, w))
            self._canvas_area.canvas.set_roi_controller(self._roi_ctrl)
        except (IndexError, FileNotFoundError, ValueError):
            pass

    def _on_roi_clear(self) -> None:
        """Clear the ROI mask."""
        if self._roi_ctrl is not None:
            self._roi_ctrl.clear()
            self._canvas_area.canvas.update_roi_overlay()
            self._state.set_roi_mask(None)

    def _on_roi_import(self, path: str) -> None:
        """Import a mask file into the ROI controller."""
        if self._roi_ctrl is None:
            return
        try:
            self._roi_ctrl.import_mask(path)
            self._canvas_area.canvas.update_roi_overlay()
            self._state.set_roi_mask(self._roi_ctrl.mask.copy())
        except IOError:
            pass


def main() -> None:
    """Launch the GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # required for QSS to work correctly
    app.setStyleSheet(build_stylesheet())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
