"""STAQ-DIC GUI application entry point."""

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget

from staq_dic.gui.app_state import AppState
from staq_dic.gui.controllers.image_controller import ImageController
from staq_dic.gui.panels.left_sidebar import LeftSidebar
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

        # Left sidebar — image loading
        self._left_sidebar = LeftSidebar(self._image_ctrl)
        layout.addWidget(self._left_sidebar, stretch=0)

        # Center placeholder (canvas — Task 4)
        center = QWidget()
        center.setStyleSheet(f"background: {COLORS.BG_CANVAS};")
        layout.addWidget(center, stretch=1)


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
