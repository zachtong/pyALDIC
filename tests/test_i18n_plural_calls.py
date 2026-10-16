"""Counted messages are written the one way lupdate extracts as plurals.

pyside6-lupdate (6.11.1) marks a message numerus only for a ``tr()`` call
with a count: ``self.tr(source, "", n)`` in a ``QObject`` subclass, the
context being the class name. Other spellings run, and substitute the
count, yet ship English:

- ``QCoreApplication.translate(context, source, "", n)`` is extracted as a
  plain message with one translation. Qt looks up the plural form the
  language's rule asks for, finds none, and falls back to the English
  source: German, French and Spanish read "Delete 2 image(s)".
- ``QCoreApplication.translate`` with the count written as an expression
  (``len(x)``) or as a keyword (``n=n``) is not extracted at all: English
  in every language ("Select %n Mask File(s)" was never in a catalog).
- ``self.tr(source, n=n)`` is extracted as a plain message.

``tr()`` with an expression count is extracted as numerus today, but the
house rule (CLAUDE.md R2) is one form for every count -- a plain local
name passed positionally -- so no count is one edit away from the forms
above. The catalog side is guarded by ``test_every_count_message_is_numerus``
in ``tests/test_smoke_i18n_languages.py``; this guard names the line.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "src" / "al_dic" / "gui"

#: Receivers whose ``translate`` is Qt's static translation function.
_APPLICATIONS = frozenset({"QCoreApplication", "QGuiApplication", "QApplication"})
#: Positional arguments of ``translate`` without a count:
#: (context, source, disambiguation).
_TRANSLATE_MAX_ARGS = 3
#: Position of the count in ``tr(source, disambiguation, n)``.
_TR_COUNT_ARG = 2

_TRANSLATE_COUNT = (
    "QCoreApplication.translate() with a count is never extracted as a "
    "plural; call self.tr(source, \"\", n) on a QObject instead"
)
_KEYWORD_COUNT = (
    "count passed as keyword n=; pass a local name positionally: "
    "tr(source, \"\", n)"
)
_EXPRESSION_COUNT = (
    "count is an expression; bind it to a local name first: "
    "n = ...; tr(source, \"\", n)"
)


def _receiver(func: ast.Attribute) -> str:
    value = func.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return ""


def _problem(call: ast.Call) -> str | None:
    """Why lupdate mis-extracts ``call``, or ``None`` when it does not."""
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    keyword_count = any(k.arg == "n" for k in call.keywords)
    if (name == "translate" and isinstance(func, ast.Attribute)
            and _receiver(func) in _APPLICATIONS):
        if len(call.args) > _TRANSLATE_MAX_ARGS or keyword_count:
            return _TRANSLATE_COUNT
        return None
    if name != "tr":
        return None
    if keyword_count:
        return _KEYWORD_COUNT
    if len(call.args) > _TR_COUNT_ARG and not isinstance(
            call.args[_TR_COUNT_ARG], (ast.Name, ast.Constant)):
        return _EXPRESSION_COUNT
    return None


def _problems(source: str, filename: str = "<snippet>") -> list[str]:
    tree = ast.parse(source, filename)
    return [
        f"{filename}:{node.lineno}: {problem}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (problem := _problem(node)) is not None
    ]


def test_every_counted_message_is_extractable_as_a_plural():
    hits = []
    for path in sorted(GUI.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        hits += _problems(path.read_text(encoding="utf-8"), rel)
    assert not hits, (
        "counted messages lupdate does not extract as plurals:\n"
        + "\n".join(hits)
    )


@pytest.mark.parametrize("snippet", [
    'QCoreApplication.translate("Ctx", "%n file(s)", "", n)',
    'QCoreApplication.translate("Ctx", "%n file(s)", "", len(files))',
    'QCoreApplication.translate("Ctx", "%n file(s)", n=n)',
    'QtCore.QCoreApplication.translate("Ctx", "%n file(s)", "", n)',
    'QApplication.translate("Ctx", "%n file(s)", None, n)',
    'self.tr("%n file(s)", "", len(files))',
    'self.tr("%n file(s)", "", int(n))',
    'self.tr("%n file(s)", "", n + 0)',
    'self.tr("%n file(s)", "", self._count)',
    'self.tr("%n file(s)", n=n)',
    'Messages.tr("%n file(s)", "", n=n)',
])
def test_guard_flags_what_lupdate_mis_extracts(snippet):
    assert _problems(snippet), snippet


@pytest.mark.parametrize("snippet", [
    'self.tr("%n file(s)", "", n)',
    'self.tr("%n file(s)", None, n)',
    'self.tr("%n file(s)", "", 1)',
    'Messages.tr("%n file(s)", "", n)',
    'self.tr("Open", "File menu")',
    'QCoreApplication.translate("Ctx", "Open")',
    'QCoreApplication.translate("Ctx", "Open", "File menu")',
    'painter.translate(10, 20, 30, 40)',
    '"abc".translate(table)',
])
def test_guard_accepts_what_lupdate_extracts(snippet):
    assert not _problems(snippet), snippet
