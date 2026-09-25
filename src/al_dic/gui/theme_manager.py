"""Switching the colour theme of the running application (*Settings > Theme*).

One :class:`ThemeManager` per ``QApplication`` (:func:`theme_manager`) applies
a theme: it makes the palette current (:func:`al_dic.gui.theme.set_current_theme`),
re-applies the application stylesheet, renders again what carries colours of its
own -- widgets styled with :func:`themed`, buttons whose icon was set with
:func:`set_icon`, Windows title bars -- stores the choice in ``QSettings`` and
emits :attr:`ThemeManager.theme_changed`. Canvases and widgets whose styles
depend on their state redraw on that signal (:func:`connect_theme`), reading
their colours when they draw, so nothing waits for a restart. ``main()`` applies
the saved theme before the first window is built.

Ported from pyALDVC, whose palette pyALDIC shares.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Callable, TypeVar

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import QApplication, QWidget

from al_dic.gui.theme import (
    DEFAULT_THEME,
    THEMES,
    build_stylesheet,
    current_colors,
    current_theme,
    set_current_theme,
    style_text,
)
from al_dic.gui.window_chrome import set_title_bar_theme

logger = logging.getLogger(__name__)

# In QSettings() beside the language (i18n.SETTINGS_KEY): the application's
# organisation and name are set in main(), and tests redirect the default
# format to a temporary folder.
SETTINGS_KEY = "ui/theme"
ENV_THEME = "PYALDIC_THEME"  # overrides the saved choice: scripts capturing one theme
STYLE_PROPERTY = "pyaldicStyle"  # the style sheet template of a widget styled with themed()
ICON_PROPERTY = "pyaldicIcon"  # the icons.py factory of a button's icon, set with set_icon()

W = TypeVar("W", bound=QWidget)

__all__ = [
    "ENV_THEME",
    "SETTINGS_KEY",
    "ThemeManager",
    "apply_title_bar",
    "connect_theme",
    "qcolor",
    "restyle",
    "set_icon",
    "startup_theme",
    "theme_manager",
    "themed",
]


def startup_theme() -> str:
    """The theme to start with: ``$PYALDIC_THEME`` when it names one, else the
    choice saved last time, else the dark default."""
    forced = os.environ.get(ENV_THEME, "")
    if forced in THEMES:
        return forced
    name = QSettings().value(SETTINGS_KEY, DEFAULT_THEME, type=str)
    return name if name in THEMES else DEFAULT_THEME


class ThemeManager(QObject):
    """Applies a theme to the whole application; :attr:`theme_changed` tells
    the canvases and state-dependent widgets to redraw."""

    theme_changed = Signal(str)

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._app = app

    @property
    def name(self) -> str:
        return current_theme()

    def set_theme(self, name: str, persist: bool = True) -> None:
        """Switch to ``name`` (``"dark"`` or ``"light"``) now; ``persist``
        remembers it for the next start."""
        if name not in THEMES:
            raise ValueError(f"unknown theme {name!r}; available: {sorted(THEMES)}")
        changed = name != current_theme()
        set_current_theme(name)
        self._app.setStyleSheet(build_stylesheet())
        if persist:
            QSettings().setValue(SETTINGS_KEY, name)
        if not changed:
            return
        for window in self._app.topLevelWidgets():
            restyle(window)
        logger.info("theme: %s", name)
        self.theme_changed.emit(name)


def theme_manager(app: QApplication | None = None) -> ThemeManager:
    """The application's theme manager, created on first use."""
    app = app if app is not None else QApplication.instance()
    if app is None:
        raise RuntimeError("the theme manager needs a QApplication")
    manager = getattr(app, "_pyaldic_theme_manager", None)
    if manager is None:
        manager = ThemeManager(app)
        app._pyaldic_theme_manager = manager  # type: ignore[attr-defined]
    return manager


def connect_theme(slot: Callable[[str], None]) -> None:
    """Call ``slot(name)`` after every theme change. ``slot`` should be a
    method of a ``QObject``, so the connection ends with the object."""
    theme_manager().theme_changed.connect(slot)


def themed(widget: W, template: str) -> W:
    """Style ``widget`` with ``template``: a style sheet whose ``{FIELD}``
    placeholders name colours of :class:`~al_dic.gui.theme.Colors` (literal
    braces doubled). It is filled again after every theme change."""
    widget.setProperty(STYLE_PROPERTY, template)
    widget.setStyleSheet(style_text(template))
    return widget


def set_icon(widget: QWidget, factory: Callable[[], QIcon]) -> None:
    """Give ``widget`` the icon ``factory`` draws (a function of
    :mod:`al_dic.gui.icons`), and draw it again after a theme change: the
    icons are stroked in the theme's text colour."""
    widget.setProperty(ICON_PROPERTY, factory.__name__)
    widget.setIcon(factory())


def apply_title_bar(window: QWidget) -> None:
    """The Windows title bar of ``window`` in the theme's variant."""
    set_title_bar_theme(window, current_colors().IS_DARK)


def restyle(root: QWidget) -> None:
    """Render ``root`` and its children again in the current theme: themed
    style sheets, remembered icons and, for a window already on screen, the
    title bar."""
    from al_dic.gui import icons

    for widget in (root, *root.findChildren(QWidget)):
        template = widget.property(STYLE_PROPERTY)
        if template:
            widget.setStyleSheet(style_text(str(template)))
        name = widget.property(ICON_PROPERTY)
        if name and hasattr(widget, "setIcon"):
            factory = getattr(icons, str(name), None)
            if factory is not None:
                widget.setIcon(factory())
    backend = sys.modules.get("matplotlib.backends.backend_qt")
    if backend is not None:  # no toolbar can exist before the backend is loaded
        for toolbar in root.findChildren(backend.NavigationToolbar2QT):
            refresh_toolbar_icons(toolbar)
    if root.isWindow() and root.internalWinId():
        apply_title_bar(root)


def refresh_toolbar_icons(toolbar) -> None:
    """Choose a matplotlib toolbar's icons again. Before 3.11 matplotlib draws
    them black, or in the text colour on a dark background, from the palette
    the toolbar has when they are made -- before a theme change; from 3.11 it
    reads the palette at every paint and this only makes them anew. Either way
    the palette's background must be opaque (see ``_light_details`` in
    :mod:`al_dic.gui.theme`)."""
    toolbar.ensurePolished()  # the stylesheet's colours reach the palette first
    try:
        for text, _tip, image, callback in toolbar.toolitems:
            if text is not None and callback in toolbar._actions:
                toolbar._actions[callback].setIcon(toolbar._icon(image + ".png"))
    except (AttributeError, TypeError, ValueError) as exc:  # another matplotlib
        logger.debug("matplotlib toolbar icons not refreshed: %s", exc)


_RGBA = re.compile(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")


def qcolor(css: str) -> QColor:
    """A ``QColor`` from a palette value: ``#rrggbb`` or ``rgba(r, g, b, a)``
    (alpha 0-255), the two forms the palette holds for style sheets."""
    match = _RGBA.fullmatch(css.strip())
    if match:
        return QColor(*(int(v) for v in match.groups()))
    return QColor(css)
