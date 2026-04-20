"""Runtime language management for pyALDIC.

Installs ``QTranslator`` instances against the currently running
``QApplication`` so every ``tr()`` / ``QCoreApplication.translate()``
call returns the correct localized string. Language choice is
persisted via ``QSettings`` and applied at the next startup.

Typical call site (from ``gui/app.py``, right after ``QApplication``
is created and before the main window is shown):

    from al_dic.i18n import LanguageManager

    lang_mgr = LanguageManager(app)
    lang_mgr.load(LanguageManager.saved_preference())

Switching at runtime:

    lang_mgr.load("zh_CN")

Any widget that wants to react to language changes should override
``changeEvent`` and rebuild its ``tr()``-wrapped strings on
``QEvent.Type.LanguageChange`` — see the pattern documented in
``CLAUDE.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from PySide6.QtCore import QLibraryInfo, QLocale, QObject, QSettings, Signal
from PySide6.QtCore import QTranslator

logger = logging.getLogger(__name__)

SETTINGS_KEY: Final[str] = "pyaldic/language"

# code -> human-readable name (as it should appear in a language picker).
# Add entries here AND register them in tools/i18n.py LANGUAGES.
SUPPORTED_LANGUAGES: Final[dict[str, str]] = {
    "en": "English",
    "zh_CN": "简体中文",
    "zh_TW": "繁體中文",
    "ja": "日本語",
}

_COMPILED_DIR = Path(__file__).parent / "compiled"


class LanguageManager(QObject):
    """Thin wrapper around ``QTranslator`` for pyALDIC-specific catalogs.

    Holds two translators per session:
        * ``_app_translator`` — pyALDIC-owned strings, shipped via
          ``src/al_dic/i18n/compiled/al_dic_<lang>.qm``.
        * ``_qt_translator`` — Qt's built-in widget strings (OK, Cancel,
          Apply, standard dialog captions, etc.), loaded from the
          installed PySide6 translations directory.

    Both are removed and reinstalled on every ``load()`` call, so
    switching languages at runtime is a single operation.
    """

    language_changed = Signal(str)  # emitted after a successful load()

    def __init__(self, app) -> None:  # app: QApplication
        super().__init__()
        self._app = app
        self._app_translator = QTranslator(self)
        self._qt_translator = QTranslator(self)
        self._current: str = "en"

    # -- Public API --------------------------------------------------------

    @property
    def current(self) -> str:
        return self._current

    def load(self, lang_code: str) -> bool:
        """Install translators for ``lang_code``; return True on success.

        ``lang_code == "en"`` removes any active translators (English is
        the source language — no catalog needed).
        """
        self._app.removeTranslator(self._app_translator)
        self._app.removeTranslator(self._qt_translator)

        if lang_code == "en" or not lang_code:
            self._current = "en"
            self._persist("en")
            self.language_changed.emit("en")
            logger.info("Language set to English (source).")
            return True

        if lang_code not in SUPPORTED_LANGUAGES:
            logger.warning(
                "Unsupported language %r; falling back to English.",
                lang_code,
            )
            return self.load("en")

        qm_file = _COMPILED_DIR / f"al_dic_{lang_code}.qm"
        if not qm_file.is_file():
            logger.warning(
                "Translation catalog missing for %r at %s; "
                "falling back to English.",
                lang_code, qm_file,
            )
            return self.load("en")

        if self._app_translator.load(str(qm_file)):
            self._app.installTranslator(self._app_translator)
        else:
            logger.warning("Failed to load %s.", qm_file)
            return self.load("en")

        # Qt's own translations for built-in widget strings (OK, Cancel, …).
        # Failure here is not fatal — the built-in strings just stay English.
        qt_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        if self._qt_translator.load(
            QLocale(lang_code), "qtbase", "_", qt_dir
        ):
            self._app.installTranslator(self._qt_translator)

        self._current = lang_code
        self._persist(lang_code)
        self.language_changed.emit(lang_code)
        logger.info("Language set to %s.", lang_code)
        return True

    # -- Preference persistence -------------------------------------------

    @staticmethod
    def saved_preference() -> str:
        value = QSettings().value(SETTINGS_KEY, "en")
        return str(value) if value else "en"

    @staticmethod
    def _persist(lang_code: str) -> None:
        QSettings().setValue(SETTINGS_KEY, lang_code)


__all__ = ["LanguageManager", "SUPPORTED_LANGUAGES"]
