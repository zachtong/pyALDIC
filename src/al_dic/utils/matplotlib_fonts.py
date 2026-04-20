"""Matplotlib font fallback configuration.

pyALDIC embeds matplotlib ``Figure`` widgets inside the Qt GUI for the
canvas overlay, colorbar overlay, and viz controller. Matplotlib's
default sans-serif stack (``DejaVu Sans``) does **not** include CJK
glyphs — any Chinese / Japanese / Korean axis label or title rendered
inside the GUI would come out as ``□□□`` (tofu blocks).

This module configures matplotlib's ``font.sans-serif`` search order so
that whichever system font covers the active locale's script is picked
up automatically.

Call ``configure_matplotlib_fonts()`` once during application startup,
after matplotlib is imported but before any Figure is drawn.
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

# Per-platform fallback chains. Matplotlib walks the list and picks the
# first font it finds actually installed on the system. We lead with
# the platform's native sans-serif (so Latin text looks native), then
# append a CJK coverage list, then DejaVu as a last-resort for symbols.

_FONT_FALLBACK_CHAIN: Final[list[str]] = [
    # Windows
    "Segoe UI",
    "Microsoft YaHei",   # Simplified Chinese
    "Microsoft JhengHei",  # Traditional Chinese
    "Yu Gothic UI",      # Japanese
    "Malgun Gothic",     # Korean
    # macOS
    "Helvetica Neue",
    "PingFang SC",       # Simplified Chinese
    "PingFang TC",       # Traditional Chinese
    "Hiragino Sans",     # Japanese
    "Apple SD Gothic Neo",  # Korean
    # Linux / cross-platform
    "Noto Sans",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Noto Sans CJK KR",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",       # final fallback (no CJK coverage but universal)
]


def configure_matplotlib_fonts() -> None:
    """Install the CJK-aware font fallback chain on matplotlib rcParams."""
    try:
        import matplotlib  # noqa: PLC0415
    except ImportError:
        logger.debug("matplotlib not installed; skipping font setup.")
        return

    matplotlib.rcParams["font.sans-serif"] = list(_FONT_FALLBACK_CHAIN)
    matplotlib.rcParams["font.family"] = "sans-serif"

    # With CJK fonts active, matplotlib's default Unicode minus sign
    # (U+2212) renders as a tofu block in some fonts. Force ASCII '-'.
    matplotlib.rcParams["axes.unicode_minus"] = False

    logger.info(
        "matplotlib font fallback installed (%d fonts in chain)",
        len(_FONT_FALLBACK_CHAIN),
    )


__all__ = ["configure_matplotlib_fonts"]
