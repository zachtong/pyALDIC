"""Locale-aware formatting for dates, numbers, and units.

All user-facing number / date / currency strings in the GUI should
flow through this module rather than through Python's ``str.format`` or
``datetime.strftime`` with hard-coded patterns. This keeps output
consistent with the user's chosen GUI language:

    2026-04-20  →  "2026/4/20"  (ja)  /  "20.04.2026"  (de)  /  "20/4/2026"  (es)
    3.14        →  "3,14"       (de/fr/es)
    1000000     →  "1 000 000"  (fr)  /  "1.000.000"  (de)  /  "1,000,000"  (en)

The helpers here respect the *currently loaded* language (the one
``LanguageManager`` last installed), so they automatically follow
runtime language switches.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Union

from PySide6.QtCore import QDate, QDateTime, QLocale


Number = Union[int, float]


def _locale() -> QLocale:
    """Locale matching the GUI's currently active language."""
    # Lazy import to avoid cycle (locale_format is used throughout gui/).
    from al_dic.i18n import LanguageManager  # noqa: PLC0415

    # No QApplication-scoped lookup for the manager exists yet; pull the
    # persisted preference. If the GUI is live, QLocale.system() honours
    # any QLocale.setDefault() the LanguageManager installed.
    try:
        current = LanguageManager.saved_preference()
    except Exception:
        current = "en"
    if current in ("en", "pseudo", ""):
        return QLocale("en_US")
    return QLocale(current)


def format_number(value: Number, precision: int | None = None) -> str:
    """Format a number in the current locale.

    ``precision=None`` uses the locale default (typically 6 digits for
    ``float``, none for ``int``). Pass an explicit precision when you
    want deterministic output (e.g. for displaying px values).
    """
    if isinstance(value, int):
        return _locale().toString(value)
    if precision is None:
        return _locale().toString(float(value), "g")
    return _locale().toString(float(value), "f", precision)


def format_date(value: date | datetime | QDate | QDateTime,
                long_format: bool = False) -> str:
    """Format a date/datetime in the current locale."""
    loc = _locale()
    fmt = QLocale.FormatType.LongFormat if long_format \
        else QLocale.FormatType.ShortFormat

    if isinstance(value, datetime):
        return loc.toString(QDateTime(value), fmt)
    if isinstance(value, date):
        return loc.toString(QDate(value), fmt)
    if isinstance(value, (QDate, QDateTime)):
        return loc.toString(value, fmt)
    raise TypeError(f"Unsupported date type: {type(value).__name__}")


def format_bytes(n_bytes: int) -> str:
    """Human-readable file size in the current locale (e.g. '12.4 MB')."""
    loc = _locale()
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n_bytes)
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{loc.toString(size, 'f', 1)} {units[idx]}"


def format_duration_seconds(seconds: float) -> str:
    """Format a duration as 'HH:MM:SS' using the locale's numeric system.

    Kept position-based (H:M:S) rather than locale-specific ("3 时 25 分")
    because colon-separated times are universally recognized and fit
    narrow progress widgets.
    """
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


__all__ = [
    "format_number",
    "format_date",
    "format_bytes",
    "format_duration_seconds",
]
