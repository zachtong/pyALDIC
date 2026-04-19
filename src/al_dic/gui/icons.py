"""Inline SVG icons for AL-DIC GUI buttons.

Minimal Lucide-style stroke icons (24x24 viewBox, stroke-width 2, no fill).
Each public function returns a QIcon. If PySide6.QtSvg is unavailable the
module still imports but every icon function returns an empty QIcon.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap

try:
    from PySide6.QtSvg import QSvgRenderer

    _HAS_SVG = True
except ImportError:  # pragma: no cover
    _HAS_SVG = False

# Stroke color that works on the dark theme (#e2e8f0 ≈ slate-200)
_STROKE = "#e2e8f0"


def _svg_to_icon(svg_str: str, size: int = 16) -> QIcon:
    """Convert an SVG string to a *size x size* QIcon."""
    if not _HAS_SVG:
        return QIcon()
    renderer = QSvgRenderer(QByteArray(svg_str.encode()))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


# -- Run controls -----------------------------------------------------------

def icon_play() -> QIcon:
    """Solid-ish play triangle."""
    return _svg_to_icon(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<polygon points="5 3 19 12 5 21 5 3"/></svg>'
    )


def icon_pause() -> QIcon:
    """Two vertical bars."""
    return _svg_to_icon(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<rect x="6" y="4" width="4" height="16"/>'
        f'<rect x="14" y="4" width="4" height="16"/></svg>'
    )


def icon_stop() -> QIcon:
    """Rounded square."""
    return _svg_to_icon(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<rect x="4" y="4" width="16" height="16" rx="2"/></svg>'
    )


def icon_download() -> QIcon:
    """Download arrow."""
    return _svg_to_icon(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        f'<polyline points="7 10 12 15 17 10"/>'
        f'<line x1="12" y1="15" x2="12" y2="3"/></svg>'
    )


# -- Navigation -------------------------------------------------------------

def icon_folder() -> QIcon:
    """Folder outline."""
    return _svg_to_icon(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 '
        f'1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>'
    )


def icon_chevron_left() -> QIcon:
    """Left chevron."""
    return _svg_to_icon(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<polyline points="15 18 9 12 15 6"/></svg>'
    )


def icon_chevron_right() -> QIcon:
    """Right chevron."""
    return _svg_to_icon(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<polyline points="9 18 15 12 9 6"/></svg>'
    )


# -- Zoom / viewport --------------------------------------------------------

def icon_zoom_in() -> QIcon:
    """Magnifying glass with plus."""
    return _svg_to_icon(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<circle cx="11" cy="11" r="8"/>'
        f'<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        f'<line x1="11" y1="8" x2="11" y2="14"/>'
        f'<line x1="8" y1="11" x2="14" y2="11"/></svg>'
    )


def icon_zoom_out() -> QIcon:
    """Magnifying glass with minus."""
    return _svg_to_icon(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<circle cx="11" cy="11" r="8"/>'
        f'<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        f'<line x1="8" y1="11" x2="14" y2="11"/></svg>'
    )


def icon_maximize() -> QIcon:
    """Expand-to-fit / maximize corners."""
    return _svg_to_icon(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3'
        f'm0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>'
    )


# -- Application icon -------------------------------------------------------

def icon_app() -> QIcon:
    """Load the packaged pyALDIC application icon.

    Loads ``gui/assets/icon/pyALDIC.ico`` — a multi-resolution ICO that
    Qt scales into whichever size the OS asks for (window chrome, task
    bar, Alt-Tab, etc.). Falls back to the 256-px PNG if the ICO is
    missing and finally to an empty QIcon so a broken asset never
    crashes startup.
    """
    from pathlib import Path

    asset_dir = Path(__file__).parent / "assets" / "icon"
    for name in ("pyALDIC.ico", "pyALDIC-256.png", "pyALDIC.svg"):
        path = asset_dir / name
        if path.is_file():
            icon = QIcon(str(path))
            if not icon.isNull():
                return icon
    return QIcon()
