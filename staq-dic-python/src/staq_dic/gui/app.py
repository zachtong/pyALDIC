"""STAQ-DIC GUI application entry point."""

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget

from staq_dic.gui.theme import build_stylesheet


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

        # Placeholder panels will be added in later tasks


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
