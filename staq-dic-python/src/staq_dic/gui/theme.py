"""Dark navy theme with indigo accent for STAQ-DIC GUI.

Provides a frozen Colors dataclass and a build_stylesheet() function
that returns a complete QSS string for PySide6 widgets.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Colors:
    """Dark navy color palette with indigo accent."""

    # Backgrounds
    BG_DARKEST: str = "#0b0f1a"
    BG_SIDEBAR: str = "#0f1322"
    BG_PANEL: str = "#141929"
    BG_INPUT: str = "#1a1f33"
    BG_HOVER: str = "#1e2440"
    BG_CANVAS: str = "#0c0d12"

    # Accent
    ACCENT: str = "#6366f1"
    ACCENT_HOVER: str = "#818cf8"
    ACCENT_PRESSED: str = "#4f46e5"

    # Text
    TEXT_PRIMARY: str = "#e2e8f0"
    TEXT_SECONDARY: str = "#94a3b8"
    TEXT_MUTED: str = "#64748b"

    # Borders
    BORDER: str = "#1e293b"
    BORDER_FOCUS: str = "#6366f1"

    # Semantic
    DANGER: str = "#ef4444"
    DANGER_HOVER: str = "#f87171"
    SUCCESS: str = "#22c55e"
    WARNING: str = "#eab308"


# Singleton instance used throughout the application
COLORS = Colors()


def build_stylesheet() -> str:
    """Return a complete QSS stylesheet for the dark navy theme."""
    c = COLORS
    return f"""
/* ============================================================
   Global
   ============================================================ */
* {{
    color: {c.TEXT_PRIMARY};
    font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}

QMainWindow,
QWidget {{
    background: {c.BG_DARKEST};
}}

/* ============================================================
   Sidebar panels
   ============================================================ */
QWidget#leftSidebar,
QWidget#rightSidebar {{
    background: {c.BG_SIDEBAR};
    border-right: 1px solid {c.BORDER};
}}

QWidget#rightSidebar {{
    border-right: none;
    border-left: 1px solid {c.BORDER};
}}

/* ============================================================
   Panel / Card sections
   ============================================================ */
QFrame[frameShape="1"],
QGroupBox {{
    background: {c.BG_PANEL};
    border: 1px solid {c.BORDER};
    border-radius: 6px;
    padding: 10px;
    margin: 4px 0px;
}}

QGroupBox::title {{
    color: {c.TEXT_SECONDARY};
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    font-weight: bold;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ============================================================
   Labels
   ============================================================ */
QLabel {{
    background: transparent;
    border: none;
    color: {c.TEXT_PRIMARY};
}}

QLabel[class="secondary"] {{
    color: {c.TEXT_SECONDARY};
}}

QLabel[class="muted"] {{
    color: {c.TEXT_MUTED};
    font-size: 11px;
}}

QLabel[class="heading"] {{
    font-weight: bold;
    font-size: 14px;
}}

/* ============================================================
   Input fields: QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox
   ============================================================ */
QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {c.TEXT_PRIMARY};
    selection-background-color: {c.ACCENT};
    min-height: 20px;
}}

QLineEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus {{
    border-color: {c.BORDER_FOCUS};
}}

QLineEdit:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QComboBox:hover {{
    background: {c.BG_HOVER};
}}

QLineEdit:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QComboBox:disabled {{
    color: {c.TEXT_MUTED};
    background: {c.BG_PANEL};
}}

/* SpinBox buttons */
QSpinBox::up-button,
QDoubleSpinBox::up-button,
QSpinBox::down-button,
QDoubleSpinBox::down-button {{
    background: {c.BG_HOVER};
    border: none;
    width: 16px;
}}

QSpinBox::up-button:hover,
QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover,
QDoubleSpinBox::down-button:hover {{
    background: {c.ACCENT};
}}

/* ComboBox dropdown */
QComboBox::drop-down {{
    border: none;
    width: 24px;
    background: transparent;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {c.TEXT_SECONDARY};
    margin-right: 6px;
}}

QComboBox QAbstractItemView {{
    background: {c.BG_PANEL};
    border: 1px solid {c.BORDER};
    selection-background-color: {c.ACCENT};
    selection-color: {c.TEXT_PRIMARY};
    outline: none;
}}

/* ============================================================
   Buttons — default
   ============================================================ */
QPushButton {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    color: {c.TEXT_PRIMARY};
    font-weight: 500;
    min-height: 22px;
}}

QPushButton:hover {{
    background: {c.BG_HOVER};
    border-color: {c.TEXT_MUTED};
}}

QPushButton:pressed {{
    background: {c.BG_PANEL};
}}

QPushButton:disabled {{
    color: {c.TEXT_MUTED};
    background: {c.BG_PANEL};
    border-color: {c.BORDER};
}}

/* Primary action button */
QPushButton[class="btn-primary"] {{
    background: {c.ACCENT};
    border: 1px solid {c.ACCENT};
    color: #ffffff;
    font-weight: bold;
}}

QPushButton[class="btn-primary"]:hover {{
    background: {c.ACCENT_HOVER};
    border-color: {c.ACCENT_HOVER};
}}

QPushButton[class="btn-primary"]:pressed {{
    background: {c.ACCENT_PRESSED};
    border-color: {c.ACCENT_PRESSED};
}}

QPushButton[class="btn-primary"]:disabled {{
    background: {c.BG_HOVER};
    border-color: {c.BORDER};
    color: {c.TEXT_MUTED};
}}

/* Danger button */
QPushButton[class="btn-danger"] {{
    background: transparent;
    border: 1px solid {c.DANGER};
    color: {c.DANGER};
}}

QPushButton[class="btn-danger"]:hover {{
    background: {c.DANGER};
    color: #ffffff;
}}

QPushButton[class="btn-danger"]:pressed {{
    background: #dc2626;
    border-color: #dc2626;
}}

/* ============================================================
   Toggle buttons (checkable QPushButton)
   ============================================================ */
QPushButton:checkable {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
}}

QPushButton:checkable:checked {{
    background: {c.ACCENT};
    border-color: {c.ACCENT};
    color: #ffffff;
}}

QPushButton:checkable:checked:hover {{
    background: {c.ACCENT_HOVER};
}}

/* ============================================================
   QCheckBox / QRadioButton
   ============================================================ */
QCheckBox,
QRadioButton {{
    background: transparent;
    spacing: 6px;
}}

QCheckBox::indicator,
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {c.BORDER};
    background: {c.BG_INPUT};
}}

QCheckBox::indicator {{
    border-radius: 3px;
}}

QRadioButton::indicator {{
    border-radius: 8px;
}}

QCheckBox::indicator:checked,
QRadioButton::indicator:checked {{
    background: {c.ACCENT};
    border-color: {c.ACCENT};
}}

QCheckBox::indicator:hover,
QRadioButton::indicator:hover {{
    border-color: {c.TEXT_MUTED};
}}

/* ============================================================
   Progress bar
   ============================================================ */
QProgressBar {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    text-align: center;
    color: {c.TEXT_PRIMARY};
    min-height: 18px;
    font-size: 11px;
}}

QProgressBar::chunk {{
    background: {c.ACCENT};
    border-radius: 3px;
}}

/* ============================================================
   Scroll bars
   ============================================================ */
QScrollBar:vertical {{
    background: {c.BG_DARKEST};
    width: 8px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:vertical {{
    background: {c.BG_HOVER};
    min-height: 30px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: {c.TEXT_MUTED};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
    border: none;
}}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background: {c.BG_DARKEST};
    height: 8px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background: {c.BG_HOVER};
    min-width: 30px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {c.TEXT_MUTED};
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
    border: none;
}}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* ============================================================
   Sliders
   ============================================================ */
QSlider::groove:horizontal {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    height: 4px;
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {c.ACCENT};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {c.ACCENT};
    border: 2px solid {c.ACCENT_HOVER};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {c.ACCENT_HOVER};
    border-color: #a5b4fc;
}}

QSlider::groove:vertical {{
    background: {c.BG_INPUT};
    border: 1px solid {c.BORDER};
    width: 4px;
    border-radius: 2px;
}}

QSlider::sub-page:vertical {{
    background: {c.ACCENT};
    border-radius: 2px;
}}

QSlider::handle:vertical {{
    background: {c.ACCENT};
    border: 2px solid {c.ACCENT_HOVER};
    width: 14px;
    height: 14px;
    margin: 0 -6px;
    border-radius: 8px;
}}

/* ============================================================
   Tab widget
   ============================================================ */
QTabWidget::pane {{
    background: {c.BG_PANEL};
    border: 1px solid {c.BORDER};
    border-top: none;
}}

QTabBar::tab {{
    background: {c.BG_SIDEBAR};
    border: 1px solid {c.BORDER};
    border-bottom: none;
    padding: 6px 14px;
    color: {c.TEXT_SECONDARY};
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background: {c.BG_PANEL};
    color: {c.TEXT_PRIMARY};
    border-bottom: 2px solid {c.ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background: {c.BG_HOVER};
    color: {c.TEXT_PRIMARY};
}}

/* ============================================================
   Console / log text
   ============================================================ */
QPlainTextEdit[class="console"],
QTextEdit[class="console"] {{
    background: {c.BG_DARKEST};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    color: {c.TEXT_SECONDARY};
    font-family: "Consolas", "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
    padding: 6px;
    selection-background-color: {c.ACCENT};
}}

/* ============================================================
   Tooltips
   ============================================================ */
QToolTip {{
    background: {c.BG_PANEL};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    color: {c.TEXT_PRIMARY};
    padding: 4px 8px;
    font-size: 12px;
}}

/* ============================================================
   Splitter handle
   ============================================================ */
QSplitter::handle {{
    background: {c.BORDER};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

/* ============================================================
   Menu bar and menus
   ============================================================ */
QMenuBar {{
    background: {c.BG_SIDEBAR};
    border-bottom: 1px solid {c.BORDER};
    padding: 2px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: 3px;
}}

QMenuBar::item:selected {{
    background: {c.BG_HOVER};
}}

QMenu {{
    background: {c.BG_PANEL};
    border: 1px solid {c.BORDER};
    border-radius: 4px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 28px 6px 12px;
    border-radius: 3px;
}}

QMenu::item:selected {{
    background: {c.ACCENT};
    color: #ffffff;
}}

QMenu::separator {{
    height: 1px;
    background: {c.BORDER};
    margin: 4px 8px;
}}

/* ============================================================
   Status bar
   ============================================================ */
QStatusBar {{
    background: {c.BG_SIDEBAR};
    border-top: 1px solid {c.BORDER};
    color: {c.TEXT_MUTED};
    font-size: 11px;
}}
"""
