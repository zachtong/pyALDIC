"""Console log widget -- read-only timestamped message log."""

from PySide6.QtCore import QTime
from PySide6.QtWidgets import QTextEdit

from al_dic.gui.theme import COLORS
from al_dic.gui.theme_manager import connect_theme, themed


class ConsoleLog(QTextEdit):
    """Read-only log viewer with colored, timestamped messages."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(100)
        self.setMaximumHeight(200)
        themed(self,
            "background: {BG_DARKEST}; border: 1px solid {BORDER}; "
            "border-radius: 4px; padding: 4px; font-family: 'Consolas', monospace; "
            "font-size: 11px;"
        )
        # (timestamp, message, level) of every line, so a theme change can
        # colour the lines already written: the colours sit in their HTML.
        self._entries: list[tuple[str, str, str]] = []
        connect_theme(self._on_theme_changed)

    @staticmethod
    def _line(timestamp: str, message: str, level: str) -> str:
        color = {
            "info": COLORS.TEXT_SECONDARY,
            "warn": COLORS.WARNING,
            "error": COLORS.DANGER,
            "success": COLORS.SUCCESS,
        }.get(level, COLORS.TEXT_SECONDARY)
        return f'<span style="color:{color}">{timestamp} {message}</span>'

    def append_log(self, message: str, level: str = "info") -> None:
        """Append a timestamped, color-coded message to the log."""
        timestamp = QTime.currentTime().toString("HH:mm:ss")
        self._entries.append((timestamp, message, level))
        self.append(self._line(timestamp, message, level))
        # Auto-scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear_log(self) -> None:
        """Clear all log entries."""
        self._entries.clear()
        self.clear()

    def _on_theme_changed(self, _name: str) -> None:
        """Write the log again in the new theme's colours, keeping the view
        at the bottom if that is where it was."""
        bar = self.verticalScrollBar()
        at_end = bar.value() >= bar.maximum()
        self.clear()
        for entry in self._entries:
            self.append(self._line(*entry))
        if at_end:
            bar.setValue(bar.maximum())
