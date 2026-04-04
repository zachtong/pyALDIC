"""Console log widget -- read-only timestamped message log."""

from PySide6.QtCore import QTime
from PySide6.QtWidgets import QTextEdit

from staq_dic.gui.theme import COLORS


class ConsoleLog(QTextEdit):
    """Read-only log viewer with colored, timestamped messages."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(100)
        self.setMaximumHeight(200)
        self.setStyleSheet(
            f"background: {COLORS.BG_DARKEST}; border: 1px solid {COLORS.BORDER}; "
            f"border-radius: 4px; padding: 4px; font-family: 'Consolas', monospace; "
            f"font-size: 11px;"
        )

    def append_log(self, message: str, level: str = "info") -> None:
        """Append a timestamped, color-coded message to the log."""
        color = {
            "info": COLORS.TEXT_SECONDARY,
            "warn": COLORS.WARNING,
            "error": COLORS.DANGER,
            "success": COLORS.SUCCESS,
        }.get(level, COLORS.TEXT_SECONDARY)
        timestamp = QTime.currentTime().toString("HH:mm:ss")
        self.append(f'<span style="color:{color}">{timestamp} {message}</span>')
        # Auto-scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear_log(self) -> None:
        """Clear all log entries."""
        self.clear()
