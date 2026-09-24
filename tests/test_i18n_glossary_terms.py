"""The glossary's terms read the same everywhere, in the glossary and the UI.

docs/i18n/glossary.md fixes one rendering per term and language. By
2026-09 both the glossary and the catalogs had drifted: the zh_TW table gave
"frame" as 影格 in one row and 幀 in the next, and the catalogs followed
whichever row a translator happened to read -- German said "Frame" in 39
messages and "Bild" in 91, zh_TW said 幀 in 64 and 影格 in 42, and the Load
Data dialog sent users to "Kamera-Framerate" under a panel labelled
"Bildrate".

Two guards. The glossary agrees with itself: a row that contains a base term
("Reference frame") renders it as the base row does. And the variants the
unification removed stay removed: where the English source means the term,
the translation must not use the old rendering.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TS_DIR = ROOT / "src" / "al_dic" / "i18n" / "source"
GLOSSARY = ROOT / "docs" / "i18n" / "glossary.md"
LANGS = ("zh_CN", "zh_TW", "ja", "ko", "de", "fr", "es")

#: Base terms and their renderings. Every glossary row whose English contains
#: the word must contain the rendering (case-insensitively).
_BASES = {
    "frame": ("帧", "影格", "フレーム", "프레임", "bild", "image", "fotograma"),
    "strain": ("应变", "應變", "ひずみ", "변형률", "dehnung", "déformation",
               "deformación"),
    "displacement": ("位移", "位移", "変位", "변위", "verschiebung",
                     "déplacement", "desplazamiento"),
    "mesh": ("网格", "網格", "メッシュ", "메시", "netz", "maillage", "malla"),
    "subset": ("子集", "子集", "サブセット", "서브셋", "subset", "imagette",
               "subconjunto"),
    "crack": ("裂纹", "裂紋", "き裂", "균열", "riss", "fissure", "grieta"),
}

#: (id, language, English source pattern, removed variant, glossary
#: rendering). Source patterns match case-insensitively; variants
#: case-sensitively.
_REMOVED = [
    ("frame", "zh_TW", r"\bframes?\b", r"幀", "影格"),
    ("mask", "zh_TW", r"\bmasks?\b", r"掩模|掩膜", "遮罩"),
    ("field", "zh_TW", r"\bfields?\b", r"欄位|場變量", "場 / 場變數"),
    ("mask", "zh_CN", r"\bmasks?\b", r"掩码|蒙版|掩膜|遮罩", "掩模"),
    ("field", "zh_CN", r"\bfields?\b", r"字段", "场 / 场变量"),
    ("fem-nodal", "zh_CN", r"\bFEM nodal\b", r"有限元节点", "FEM 节点"),
    ("reference-frame", "ko", r"\breference frames?\b", r"기준 프레임", "참조 프레임"),
    ("frame", "de", r"\bframes?\b", r"\bFrames?\b|Frame-|-Frame|[a-zäöü]frames?\b",
     "Bild"),
    ("roi", "de", r"\bregions? of interest\b", r"Interessenbereich|INTERESSENBEREICH",
     "Region of Interest"),
    ("frame-rate", "de", r"\bframe rate\b", r"Framerate", "Bildrate"),
    ("frame-rate", "fr", r"\bframe rate\b", r"\b[Cc]adence\b", "Fréquence d'images"),
    ("subset", "fr", r"\bsubsets?\b", r"\bsubsets?\b|sous-ensemble", "imagette"),
    ("frame-rate", "es", r"\bframe rate\b", r"[Ff]otogramas por segundo",
     "Velocidad de fotogramas"),
    ("subset", "es", r"\bsubsets?\b", r"\bsubsets?\b", "subconjunto"),
    ("incremental", "ko", r"\bincremental\b", r"점진적", "증분"),
    ("session", "zh_TW", r"\bsessions?\b", r"會話", "工作階段"),
    ("run", "zh_TW", r"\bruns?\b|\brunning\b", r"運行", "執行"),
]

#: key=value dumps keep their keys in every locale (CLAUDE.md whitelist:
#: "subset=40, step=16, mode=accumulative").
_KEY_VALUE = re.compile(r"\b([a-z][a-z_]*)=%\d")
#: Rows of the do-not-translate table whose tokens are names or acronyms. The
#: symbol, strain-component and colormap rows are not: "max" there is part of
#: "γ max", while "min/max" in running text is translated.
_TOKEN_ROWS = ("Brand / method", "Algorithm", "File formats",
               "Tech abbreviations", "Library names")
#: Do-not-translate tokens a translation may legitimately replace: ADMM is
#: shown to users as AL-DIC (the glossary's display-facing exception).
_TOKEN_EXCEPTIONS = {"ADMM"}


def _glossary_rows() -> list[list[str]]:
    rows = []
    for line in GLOSSARY.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[0] != "English" and set(cells[0]) - set("-"):
            rows.append(cells)
    return rows


def _term(english: str) -> str:
    """The term itself, without a parenthesised note on its sense."""
    return re.sub(r"\s*\(.*?\)", "", english)


def _do_not_translate_tokens() -> set[str]:
    """The names and acronyms of the glossary's "Do NOT translate" table."""
    tokens = set()
    for line in GLOSSARY.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] in _TOKEN_ROWS:
            tokens |= set(re.findall(r"`([^`]+)`", cells[1]))
    assert len(tokens) > 20, "the do-not-translate table changed shape"
    return tokens - _TOKEN_EXCEPTIONS


def _finished(lang: str):
    tree = ET.parse(TS_DIR / f"al_dic_{lang}.ts")
    for context in tree.getroot().iter("context"):
        name = context.findtext("name")
        for message in context.iter("message"):
            tr = message.find("translation")
            if tr is None or tr.get("type") in ("unfinished", "obsolete", "vanished"):
                continue
            forms = [f.text or "" for f in tr.findall("numerusform")] or [tr.text or ""]
            for text in forms:
                yield name, message.findtext("source") or "", text


def test_the_glossary_has_the_rows_this_test_reads():
    english = {row[0] for row in _glossary_rows()}
    assert {"Frame", "Strain", "Displacement", "Mesh", "Subset", "Crack"} <= english


@pytest.mark.parametrize("word", sorted(_BASES))
def test_glossary_rows_render_a_base_term_as_its_own_row_does(word):
    renderings = dict(zip(LANGS, _BASES[word]))
    base = next(row for row in _glossary_rows() if row[0].lower() == word)
    assert [cell.lower() for cell in base[1:]] == list(_BASES[word]), (
        f"the {word!r} row changed; update _BASES to match it"
    )
    pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
    bad = [
        f"{row[0]!r} [{lang}] = {cell!r}"
        for row in _glossary_rows() if pattern.search(_term(row[0]))
        for lang, cell in zip(LANGS, row[1:])
        if renderings[lang] not in cell.lower()
    ]
    assert not bad, f"rows disagree with {word!r}: " + "; ".join(bad)


@pytest.mark.parametrize(
    "lang, source_pattern, variant, rendering",
    [rule[1:] for rule in _REMOVED],
    ids=[f"{rule[1]}-{rule[0]}" for rule in _REMOVED],
)
def test_a_removed_variant_does_not_come_back(lang, source_pattern, variant,
                                              rendering):
    source_rx = re.compile(source_pattern, re.IGNORECASE)
    variant_rx = re.compile(variant)
    bad = [
        f"{context}: {source[:60]!r} -> {text[:60]!r}"
        for context, source, text in _finished(lang)
        if source_rx.search(source) and variant_rx.search(text)
    ]
    assert not bad, (
        f"{len(bad)} {lang} translation(s) use {variant!r} where the glossary "
        f"says {rendering!r}:\n  " + "\n  ".join(bad)
    )


@pytest.mark.parametrize("lang", LANGS)
def test_key_value_dumps_keep_their_keys(lang):
    bad = [
        f"{context}: {source!r} -> {text!r}"
        for context, source, text in _finished(lang)
        for key in _KEY_VALUE.findall(source)
        if f"{key}=" not in text
    ]
    assert not bad, f"{lang} translated a key of a key=value dump:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("lang", LANGS)
def test_do_not_translate_tokens_survive_translation(lang):
    tokens = _do_not_translate_tokens()
    bad = [
        f"{context}: {token} in {source[:60]!r} -> {text[:60]!r}"
        for context, source, text in _finished(lang)
        for token in tokens
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", source)
        and token not in text
    ]
    assert not bad, f"{lang} translated a do-not-translate token:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("lang", LANGS)
def test_the_refine_brush_menu_items_read_differently(lang):
    """Paint / Erase / Clear Brush share one menu; es once read Borrar twice."""
    items = {source: text for context, source, text in _finished(lang)
             if context == "ROIToolbar" and source in ("Paint", "Erase", "Clear Brush")}
    assert len(items) == 3 and len(set(items.values())) == 3, items
