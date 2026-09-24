"""The translation catalogs contain translations, not fragments of them.

Coverage and extract-drift checks count a message as done once it has any
finished translation. Seventy messages shipped with a single character -- the
last character of the intended text, because they were filed in the plural
table and ``forms[-1]`` of a string is its last letter. The German tabs of the
strain window read "d" and "e"; the fatal-error dialog's title read "n".

Two guards: the tables must be well formed (the cause), and no catalog may
hold a one-character translation of a word (the symptom, whatever causes it
next time).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TS_DIR = ROOT / "src" / "al_dic" / "i18n" / "source"

#: Latin-script locales: a word never translates to one character.
_LATIN = ("de", "fr", "es")
#: ...except English function words whose translation genuinely is one
#: letter: "to" is "à" in French and "a" in Spanish, as in "de 3 à 7".
#: Allowed by SOURCE, never by translation: the corruption this guards against
#: turned "Line" into "a" (the last letter of "Línea"), and allowing the
#: letter "a" would have let exactly that through.
_FUNCTION_WORDS = frozenset({"to", "and", "or"})
#: CJK locales legitimately use one character for a short word ("点" for
#: "Point"); only a long source rendered as one character is implausible.
_CJK = ("zh_CN", "zh_TW", "ja", "ko")
_CJK_MIN_SOURCE = 8


def test_translation_tables_are_well_formed():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import fill_translations
    finally:
        sys.path.pop(0)
    fill_translations.check_tables()


def _finished(lang: str):
    tree = ET.parse(TS_DIR / f"al_dic_{lang}.ts")
    for message in tree.getroot().iter("message"):
        source = message.findtext("source") or ""
        tr = message.find("translation")
        if tr is None or tr.get("type") in ("unfinished", "obsolete", "vanished"):
            continue
        forms = [f.text or "" for f in tr.findall("numerusform")] or [tr.text or ""]
        for text in forms:
            yield source, text


@pytest.mark.parametrize("lang", _LATIN + _CJK)
def test_no_word_is_translated_to_a_single_character(lang):
    threshold = 2 if lang in _LATIN else _CJK_MIN_SOURCE
    bad = [
        (src, text) for src, text in _finished(lang)
        if len(text.strip()) == 1 and len(src.strip()) >= threshold
        and src.strip().lower() not in _FUNCTION_WORDS
    ]
    assert not bad, (
        f"{len(bad)} {lang} translation(s) are a single character, e.g. "
        + ", ".join(f"{s!r} -> {t!r}" for s, t in bad[:5])
    )
