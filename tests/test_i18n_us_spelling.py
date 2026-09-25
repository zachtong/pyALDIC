"""The interface is written in American English.

Every English source string the translators see -- all user-visible GUI
text, and the comments that tell translators what it means -- is checked for
British spellings: color, not colour; center, not centre; canceled, not
cancelled. One catalog suffices: all locales share the same sources.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "src/al_dic/i18n/source/al_dic_de.ts"

_ISE = ("optimis|normalis|visualis|initialis|organis|recognis|summaris|customis"
        "|minimis|maximis|analys")
# Whole British words with their inflections; the American form in the comment.
BRITISH = re.compile(
    r"\b("
    r"colour(s|ed|ing|ful)?"                       # color
    r"|centre(s|d)?"                               # center
    r"|grey(s|ed|ish)?"                            # gray
    r"|behaviour(s|al)?"                           # behavior
    r"|neighbour(s|hood|ing)?"                     # neighbor
    r"|honour(s|ed)?|favourite(s)?"                # honor, favorite
    r"|cancell(ed|ing)"                            # canceled, canceling
    r"|labell(ed|ing)|modelling|travelling"        # labeled, modeling, traveling
    r"|(milli|centi)?metre(s)?|licence(s)?|catalogue(s)?|artefact(s)?"
    rf"|({_ISE})(e|ed|es|ing|ation|ations|er|ers)"  # -ize, -yze
    r")\b",
    re.IGNORECASE,
)
# Fine in American English though it matches: the plural noun of "analysis".
ALLOWED = {"analyses"}


def _sources() -> list[tuple[str, str, str]]:
    root = ET.parse(CATALOG).getroot()
    return [(ctx.findtext("name") or "", msg.findtext("source") or "",
             msg.findtext("comment") or "")
            for ctx in root.iter("context") for msg in ctx.iter("message")]


def test_the_catalog_has_the_interface_strings():
    assert len(_sources()) > 500


def test_the_pattern_catches_british_and_spares_american():
    for british in ("Colour", "colours", "centred", "greyed", "cancelled",
                    "Cancelling", "neighbours", "optimised", "normalisation"):
        assert BRITISH.search(british), british
    for american in ("color", "center", "central", "gray", "canceled", "neighbors",
                     "optimized", "metric", "analysis", "analyses", "labeled"):
        match = BRITISH.search(american)
        assert match is None or match.group(0).lower() in ALLOWED, american


def test_interface_text_uses_american_spelling():
    found = []
    for context, source, comment in _sources():
        for text in (source, comment):
            words = {m.group(0) for m in BRITISH.finditer(text)
                     if m.group(0).lower() not in ALLOWED}
            if words:
                found.append(f"[{context}] {sorted(words)} in {text[:70]!r}")
    assert not found, "British spelling in interface text:\n  " + "\n  ".join(found)
