"""Helpers for matching our dark UI theme to the OS-native window chrome.

On Windows 10/11 the title-bar / caption / min-max-close buttons are
drawn by the Desktop Window Manager, not Qt. Flipping the
``DWMWA_USE_IMMERSIVE_DARK_MODE`` attribute tells DWM to re-paint them
in the dark variant so the whole window frame matches our dark
stylesheet instead of the default white.

Usage::

    from al_dic.gui.window_chrome import enable_dark_title_bar

    class MyWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            enable_dark_title_bar(self)

No-op on platforms that do not expose DWM (macOS, Linux).
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QWidget


def enable_dark_title_bar(window: QWidget) -> None:
    """Turn on Windows' immersive dark title bar for ``window``.

    Silently no-ops on non-Windows platforms and on Windows 10 builds
    too old to support the attribute. The window must already have a
    valid native handle (call this in ``__init__`` after the widget
    has been realised, e.g. after ``super().__init__()``).
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(window.winId())
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(1)),
            ctypes.sizeof(ctypes.c_int),
        )
    except Exception:  # pragma: no cover
        pass  # non-critical — fall back to default title bar
