"""Runtime language management for pyALDIC.

Installs ``QTranslator`` instances against the currently running
``QApplication`` so every ``tr()`` / ``QCoreApplication.translate()``
call returns the correct localized string. Language choice is
persisted via ``QSettings`` and applied at the next startup.

Typical call site (from ``gui/app.py``, right after ``QApplication``
is created and before the main window is shown):

    from al_dic.i18n import LanguageManager

    lang_mgr = LanguageManager(app)
    lang_mgr.load(LanguageManager.resolve_language())

Switching at runtime:

    lang_mgr.load("zh_CN")

Any widget that wants to react to language changes should override
``changeEvent`` and rebuild its ``tr()``-wrapped strings on
``QEvent.Type.LanguageChange`` — see the pattern documented in
``CLAUDE.md``.

Pseudo-locale
-------------
Passing ``"pseudo"`` as the language code wraps every translated string
with ASCII markers and padding (``⟦…~~~⟧``). Use it during development
to (a) visually confirm which strings still lack ``tr()`` wrappers
(they stay plain English) and (b) stress-test widget layouts against
~30% text expansion before German / Japanese translations land.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from PySide6.QtCore import (
    QLibraryInfo,
    QLocale,
    QObject,
    QSettings,
    QTranslator,
    Signal,
)

logger = logging.getLogger(__name__)

SETTINGS_KEY: Final[str] = "pyaldic/language"
_COMPILED_DIR = Path(__file__).parent / "compiled"


# ---------- Language registry ---------------------------------------------

# code -> human-readable name (as it should appear in a language picker).
# Add entries here AND register them in tools/i18n.py LANGUAGES.
SUPPORTED_LANGUAGES: Final[dict[str, str]] = {
    "en":    "English",
    "zh_CN": "简体中文",
    "zh_TW": "繁體中文",
    "ja":    "日本語",
    "ko":    "한국어",
    "de":    "Deutsch",
    "fr":    "Français",
    "es":    "Español",
}

# Variants we don't ship dedicated catalogs for but still want to serve
# sensibly when an end-user's OS locale matches one of them. First entry
# that has a shipped catalog wins; "en" is the final fallback.
FALLBACK_CHAIN: Final[dict[str, tuple[str, ...]]] = {
    # Chinese variants
    "zh_HK":   ("zh_TW", "zh_CN", "en"),
    "zh_MO":   ("zh_TW", "zh_CN", "en"),
    "zh_SG":   ("zh_CN", "en"),
    "zh":      ("zh_CN", "en"),
    # Portuguese / Spanish
    "pt_PT":   ("es", "en"),
    "pt_BR":   ("es", "en"),
    "pt":      ("es", "en"),
    # English regional
    "en_GB":   ("en",),
    "en_US":   ("en",),
    # German / French / Japanese / Korean bare codes
    "de_DE":   ("de", "en"),
    "de_AT":   ("de", "en"),
    "de_CH":   ("de", "en"),
    "fr_FR":   ("fr", "en"),
    "fr_CA":   ("fr", "en"),
    "es_ES":   ("es", "en"),
    "es_MX":   ("es", "en"),
    "ja_JP":   ("ja", "en"),
    "ko_KR":   ("ko", "en"),
}

# Dev-only fake locale — not a real language, never shipped to users.
PSEUDO_LOCALE: Final[str] = "pseudo"


# ---------- Pseudo-translator --------------------------------------------

class _PseudoTranslator(QTranslator):
    """Wraps every translation in ⟦…⟧ + padding for dev-time validation.

    Any string that is NOT wrapped in ``tr()`` passes through untouched
    (because ``translate`` is never called for it) — those strings show
    up as plain English on screen, making the audit trivial: anything
    not in ⟦brackets⟧ is a missing ``tr()``.
    """

    def translate(self, context, text, disambiguation=None, n=-1):
        if not text:
            return text
        # Expand by ~30% to simulate German / Finnish length.
        pad_count = max(1, len(text) // 3)
        return "\u27e6" + text + "~" * pad_count + "\u27e7"


# ---------- LanguageManager ----------------------------------------------

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
        self._app_translator: QTranslator = QTranslator(self)
        self._qt_translator: QTranslator = QTranslator(self)
        self._current: str = "en"

    # -- Public API --------------------------------------------------------

    @property
    def current(self) -> str:
        return self._current

    def load(self, lang_code: str) -> bool:
        """Install translators for ``lang_code``; return True on success."""
        self._app.removeTranslator(self._app_translator)
        self._app.removeTranslator(self._qt_translator)

        # Pseudo-locale: swap translator class, no .qm needed.
        if lang_code == PSEUDO_LOCALE:
            self._app_translator = _PseudoTranslator(self)
            self._app.installTranslator(self._app_translator)
            self._current = PSEUDO_LOCALE
            self._persist(PSEUDO_LOCALE)
            self.language_changed.emit(PSEUDO_LOCALE)
            logger.info("Language set to pseudo-locale (dev only).")
            return True

        # English = source language, no catalog required.
        if lang_code == "en" or not lang_code:
            self._app_translator = QTranslator(self)  # reset
            self._qt_translator = QTranslator(self)
            self._current = "en"
            self._persist("en")
            self.language_changed.emit("en")
            logger.info("Language set to English (source).")
            return True

        # Resolve variants via fallback chain.
        resolved = self._resolve(lang_code)
        if resolved != lang_code:
            logger.info("Resolving %r -> %r via fallback chain.",
                        lang_code, resolved)

        if resolved == "en":
            return self.load("en")

        qm_file = _COMPILED_DIR / f"al_dic_{resolved}.qm"
        if not qm_file.is_file():
            logger.warning(
                "Translation catalog missing for %r at %s; "
                "falling back to English.",
                resolved, qm_file,
            )
            return self.load("en")

        self._app_translator = QTranslator(self)
        if self._app_translator.load(str(qm_file)):
            self._app.installTranslator(self._app_translator)
        else:
            logger.warning("Failed to load %s.", qm_file)
            return self.load("en")

        # Qt's own translations for built-in widget strings (OK, Cancel …).
        # Failure here is not fatal — built-in strings just stay English.
        qt_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        self._qt_translator = QTranslator(self)
        if self._qt_translator.load(
            QLocale(resolved), "qtbase", "_", qt_dir
        ):
            self._app.installTranslator(self._qt_translator)

        self._current = resolved
        self._persist(resolved)
        self.language_changed.emit(resolved)
        logger.info("Language set to %s.", resolved)
        return True

    # -- Resolution --------------------------------------------------------

    @staticmethod
    def _resolve(lang_code: str) -> str:
        """Map an arbitrary locale code to a shipped language or 'en'.

        The dev-only pseudo locale is preserved as-is so it can be
        restored from QSettings across restarts during development.
        """
        if lang_code == PSEUDO_LOCALE:
            return PSEUDO_LOCALE
        if lang_code in SUPPORTED_LANGUAGES:
            return lang_code
        for candidate in FALLBACK_CHAIN.get(lang_code, ("en",)):
            if candidate in SUPPORTED_LANGUAGES:
                return candidate
        return "en"

    # -- Preference persistence -------------------------------------------

    @classmethod
    def resolve_language(cls) -> str:
        """Return the language to load at app start.

        Priority: QSettings (previous user choice) > system locale > 'en'.
        """
        saved = QSettings().value(SETTINGS_KEY)
        if saved:
            return cls._resolve(str(saved))
        system = QLocale.system().name()  # e.g. "en_US", "zh_CN"
        return cls._resolve(system)

    @staticmethod
    def saved_preference() -> str:
        """Read the saved preference verbatim (no fallback resolution)."""
        value = QSettings().value(SETTINGS_KEY, "en")
        return str(value) if value else "en"

    @staticmethod
    def _persist(lang_code: str) -> None:
        QSettings().setValue(SETTINGS_KEY, lang_code)


__all__ = [
    "LanguageManager",
    "SUPPORTED_LANGUAGES",
    "FALLBACK_CHAIN",
    "PSEUDO_LOCALE",
]
