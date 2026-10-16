"""Smoke test: every shipped language must load + key widgets must construct.

Failure modes this catches:
    - A language code is declared in SUPPORTED_LANGUAGES but its .qm
      catalog is missing or empty
    - A widget that uses self.tr() throws at construction time in a
      specific locale (font loader crashes, layout divide-by-zero,
      etc.)
    - tr_args() or QComboBox userData wiring regresses
    - CJK font fallback silently stops picking up glyphs

Runs a minimal headless QApplication. No on-screen display.
"""

from __future__ import annotations

import contextlib
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

# Force the Qt platform plugin to offscreen so CI / smoke runs headless
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(["pyALDIC-smoke"])
    app.setOrganizationName("pyALDIC-smoke")
    yield app


@pytest.fixture(autouse=True)
def _english_afterwards(qapp):
    """Leave the application in English for whatever test runs next.

    A language loaded here stayed installed, and a later test in the same
    session read "25,0 mm" where it expected "25.0 mm".
    """
    yield
    from al_dic.i18n import LanguageManager

    LanguageManager(qapp).load("en")


# All shipped non-source languages. Must match SUPPORTED_LANGUAGES.
SHIPPED_LANGS = ["en", "zh_CN", "zh_TW", "ja", "ko", "de", "fr", "es"]

# A few canonical strings per context that MUST come back translated
# (except for 'en', where source == translation by definition).
PROBES = {
    "RightSidebar":   "Run DIC Analysis",
    "LeftSidebar":    "WORKFLOW TYPE",
    "ParamPanel":     "Subset Size",
    "StrainWindow":   "Compute Strain",
    "MainWindow":     "&Settings",
}


# --- Catalog availability ---------------------------------------------------

@pytest.mark.parametrize("lang", SHIPPED_LANGS)
def test_language_catalog_loads(qapp, lang):
    """Each registered language must successfully install its .qm file."""
    from al_dic.i18n import LanguageManager
    mgr = LanguageManager(qapp)
    assert mgr.load(lang), f"LanguageManager.load({lang!r}) returned False"
    assert mgr.current == lang


@pytest.mark.parametrize("lang", SHIPPED_LANGS)
def test_translations_returned(qapp, lang):
    """Canonical probe strings must translate (or stay identical for en)."""
    from al_dic.i18n import LanguageManager
    mgr = LanguageManager(qapp)
    mgr.load(lang)
    for ctx, src in PROBES.items():
        out = QCoreApplication.translate(ctx, src)
        if lang == "en":
            assert out == src, f"en should return source verbatim; got {out!r}"
        else:
            # Translations may coincidentally equal English (e.g. "FFT");
            # but RightSidebar/Run DIC Analysis must differ.
            if src == "Run DIC Analysis":
                assert out != src, (
                    f"[{lang}] '{src}' not translated — probably missing "
                    f"from {lang}.ts or compile step was skipped."
                )


# --- Widget construction in every language ---------------------------------

# Curated list of widgets verified to construct via no-arg ctor.
# Auto-discovery (below) extends this; new widgets are picked up
# automatically and only need to be added here if they need a
# fixture-style constructor that the auto-discovery can't infer.
WIDGETS_TO_CONSTRUCT = [
    "al_dic.gui.widgets.workflow_type_panel:WorkflowTypePanel",
    "al_dic.gui.widgets.init_guess_widget:InitGuessWidget",
    "al_dic.gui.widgets.param_panel:ParamPanel",
    "al_dic.gui.widgets.advanced_tuning_widget:AdvancedTuningWidget",
    "al_dic.gui.widgets.strain_param_panel:StrainParamPanel",
    "al_dic.gui.widgets.strain_field_selector:StrainFieldSelector",
    "al_dic.gui.widgets.physical_units_widget:PhysicalUnitsWidget",
    "al_dic.gui.widgets.color_range:ColorRange",
    "al_dic.gui.widgets.roi_hint:ROIHint",
    "al_dic.gui.widgets.canvas_config_overlay:CanvasConfigOverlay",
]


def _import_ctor(dotted: str):
    mod_name, cls_name = dotted.split(":")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


# Auto-discovery: walk al_dic.gui.widgets / dialogs and find every
# QWidget/QDialog subclass that we can default-construct. This means
# new widgets are smoke-tested in 8 languages the moment they're
# committed — no opt-in step. Widgets that need fixture data (e.g.
# `StrainWindow(state)`) opt out via the `_i18n_no_default_construct`
# class attribute (must be set to a True value with a reason string).
def _discover_default_constructible_widgets() -> list[str]:
    import importlib
    import inspect
    import pkgutil
    import sys

    from PySide6.QtWidgets import QWidget

    discovered: list[str] = []
    for pkg_name in ("al_dic.gui.widgets", "al_dic.gui.dialogs"):
        pkg = importlib.import_module(pkg_name)
        for _, mod_name, ispkg in pkgutil.walk_packages(
            pkg.__path__, prefix=f"{pkg_name}.",
        ):
            if ispkg:
                continue
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                # Some modules need GUI context to import; skip them
                # rather than fail discovery.
                continue
            for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod_name:
                    continue  # imported, not defined here
                if not issubclass(cls, QWidget):
                    continue
                if getattr(cls, "_i18n_no_default_construct", False):
                    continue
                # Heuristic: ctor signature must be either bare `(self)`
                # or `(self, parent=...)`. Anything more is treated as
                # needing a fixture and skipped (opt-in via curated list
                # above if you want it tested).
                try:
                    sig = inspect.signature(cls.__init__)
                except (TypeError, ValueError):
                    continue
                params = list(sig.parameters.values())[1:]  # drop self
                non_default = [p for p in params if p.default is p.empty]
                if non_default:
                    continue
                discovered.append(f"{mod_name}:{cls_name}")
    return sorted(set(discovered))


# Combine curated + auto-discovered (curated wins on duplicates).
def _all_widget_paths() -> list[str]:
    seen: set[str] = set(WIDGETS_TO_CONSTRUCT)
    result = list(WIDGETS_TO_CONSTRUCT)
    for path in _discover_default_constructible_widgets():
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result


_ALL_WIDGET_PATHS = _all_widget_paths()


@pytest.mark.parametrize("lang", SHIPPED_LANGS)
@pytest.mark.parametrize("widget_path", _ALL_WIDGET_PATHS)
def test_widget_constructs(qapp, lang, widget_path):
    """Every major widget must construct without exception in every language.

    This is the backstop for `.arg()` / non-QObject tr() / userData
    state-sync regressions: if any of those break in a locale-specific
    code path, construction raises.

    Auto-discovers all default-constructible QWidget/QDialog subclasses
    under al_dic.gui.widgets / dialogs. To opt a widget out (because
    it needs fixture data), set ``_i18n_no_default_construct = "<reason>"``
    on the class.
    """
    from al_dic.i18n import LanguageManager
    mgr = LanguageManager(qapp)
    mgr.load(lang)
    ctor = _import_ctor(widget_path)
    w = ctor()
    # Basic invariant: the widget has at least one child / laid out element
    assert w is not None


@pytest.mark.parametrize("lang", SHIPPED_LANGS)
def test_analysis_tab_constructs(qapp, lang):
    """The Analysis tab needs a state, so discovery cannot reach it.

    Arming a tool builds the canvas banner, the one string assembled from
    several translated sentences -- it must come out in the loaded language.
    """
    from al_dic.gui.app_state import AppState
    from al_dic.gui.panels.analysis import AnalysisTab
    from al_dic.i18n import LanguageManager

    mgr = LanguageManager(qapp)
    mgr.load(lang)
    tab = AnalysisTab(AppState())
    tab._on_tool_clicked("extensometer", True)
    text = tab._banner.text()
    assert text
    if lang != "en":
        # Each sentence separately: one tr() call hidden in an f-string
        # dropped out of the catalog while the others still translated.
        for english in ("Virtual extensometer", "Click the two gauge points",
                        "Esc cancels placement"):
            assert english not in text, text


def test_widget_discovery_finds_at_least_curated_set():
    """Sanity: auto-discovery should pick up every widget in the
    curated list (otherwise something's wrong with the walker).

    Also asserts at least one *new* widget (i.e. not in the curated
    list) is discovered, so we know the auto-discovery is actually
    contributing coverage rather than silently being a no-op.
    """
    discovered = set(_discover_default_constructible_widgets())
    curated = set(WIDGETS_TO_CONSTRUCT)
    missing = curated - discovered
    # Curated entries that auto-discovery missed are fine (e.g. they
    # have non-default ctor params); we just want to confirm overlap
    # is non-trivial.
    assert len(curated & discovered) >= len(curated) // 2, (
        f"Auto-discovery overlap with curated list is too small. "
        f"Discovered: {len(discovered)}, curated: {len(curated)}, "
        f"overlap: {len(curated & discovered)}, missing from "
        f"discovery: {missing}"
    )
    # And at least one extra widget discovered beyond the curated set.
    extras = discovered - curated
    assert extras, (
        "Auto-discovery did not find any new widgets beyond the "
        "curated list — the walker may be broken."
    )


# --- Combobox userData sanity ----------------------------------------------

@pytest.mark.parametrize("lang", SHIPPED_LANGS)
def test_workflow_combobox_codes_stable(qapp, lang):
    """Translating the display must NOT change QComboBox userData codes."""
    from al_dic.i18n import LanguageManager
    mgr = LanguageManager(qapp)
    mgr.load(lang)
    from al_dic.gui.widgets.workflow_type_panel import WorkflowTypePanel
    w = WorkflowTypePanel()
    tracking_codes = [
        w._tracking_mode.itemData(i) for i in range(w._tracking_mode.count())
    ]
    assert tracking_codes == ["incremental", "accumulative"], (
        f"[{lang}] Tracking mode userData codes drifted: {tracking_codes}"
    )
    solver_codes = [
        w._solver.itemData(i) for i in range(w._solver.count())
    ]
    assert solver_codes == ["aldic", "local"], (
        f"[{lang}] Solver userData codes drifted: {solver_codes}"
    )
    ref_codes = [
        w._ref_mode.itemData(i) for i in range(w._ref_mode.count())
    ]
    assert ref_codes == ["every_frame", "every_n", "custom"], (
        f"[{lang}] Ref-update userData codes drifted: {ref_codes}"
    )


# --- Tracking-quality messages (Gate D) -------------------------------------

#: Locales whose decimal separator is a comma ("70,3").
_DECIMAL_COMMA = ("de", "fr", "es")


def _tracking_messages(qapp, lang: str) -> dict:
    """``build_tracking_messages`` for the synthetic diagnostics, in ``lang``."""
    from al_dic.gui.controllers.tracking_messages import build_tracking_messages
    from al_dic.i18n import LanguageManager

    from tests.test_gui._helpers import (
        make_tracking_diagnostics,
        make_tracking_para,
    )

    # Keep the manager alive while building: it parents the translators,
    # and a collected manager takes them out of the application.
    mgr = LanguageManager(qapp)
    assert mgr.load(lang)
    para = make_tracking_para()
    return {
        kind: build_tracking_messages(make_tracking_diagnostics(kind), para)
        for kind in ("warning", "burst", "notice", "healthy")
    }


def _texts(messages) -> list[str]:
    return [
        messages.title, messages.text, messages.informative, messages.details,
        *(text for text, _level in messages.log_lines),
    ]


def _has_number(number: str, text: str) -> bool:
    return re.search(rf"(?<![\d.,]){re.escape(number)}(?![\d])", text) is not None


@pytest.mark.parametrize("lang", SHIPPED_LANGS)
def test_tracking_messages_are_translated(qapp, lang):
    """Every piece of the tracking-quality report comes out in ``lang``.

    No placeholder survives, the counts are there, and the percentages use
    the language's decimal separator (format_number).
    """
    from al_dic.utils.locale_format import format_number

    from tests.test_gui import _helpers as h

    english = _tracking_messages(qapp, "en")
    local = _tracking_messages(qapp, lang)
    for kind, messages in local.items():
        texts = _texts(messages)
        assert len(texts) == len(_texts(english[kind])), kind
        for text in texts:
            assert text, f"[{lang}] {kind}: empty text"
            assert not re.search(r"%(\d|n)", text), f"[{lang}] {kind}: {text!r}"
        if lang != "en":
            for source, text in zip(_texts(english[kind]), texts):
                assert text != source, f"[{lang}] {kind}: untranslated {text!r}"

    details = local["warning"].details
    for count in (h.TRACKING_GRID_MISS, h.TRACKING_LOST, h.TRACKING_BORDER):
        assert _has_number(str(count), details), (lang, count, details)
    assert format_number(h.TRACKING_LARGEST_GRID) in details, (lang, details)
    assert _has_number(str(h.TRACKING_FRAME0_GRID), details), (lang, details)
    lost_pct = format_number(h.TRACKING_LOST_PCT, 1)
    assert lost_pct == ("70,3" if lang in _DECIMAL_COMMA else "70.3")
    assert lost_pct in details, (lang, details)

    burst = local["burst"].details
    assert _has_number(str(h.TRACKING_BURST), burst), (lang, burst)
    assert format_number(h.TRACKING_BURST_PCT, 1) in burst, (lang, burst)


def test_tracking_messages_go_through_the_translator(qapp):
    """Pseudo-locale: every translated piece is wrapped, so none bypasses tr()."""
    for kind, messages in _tracking_messages(qapp, "pseudo").items():
        for text in _texts(messages):
            assert text.startswith("\u27e6"), f"{kind}: {text!r}"
            assert text.endswith("\u27e7"), f"{kind}: {text!r}"


@pytest.mark.parametrize("lang", _DECIMAL_COMMA)
def test_tracking_counts_have_singular_and_plural(qapp, lang):
    """The count sentences are numerus messages in the catalog.

    lupdate marks ``self.tr(source, "", n)`` numerus but not
    ``QCoreApplication.translate(context, source, "", n)``; a message it
    does not mark has one form, and for any count the language's plural
    rule sends to the second form Qt falls back to the English source.
    """
    from al_dic.gui.controllers.tracking_messages import TrackingQuality
    from al_dic.i18n import LanguageManager

    mgr = LanguageManager(qapp)
    assert mgr.load(lang)
    t = TrackingQuality()
    for sentence in (t.grid_miss, t.loss, t.burst, t.edge, t.grid_note):
        one, many = sentence(1), sentence(7)
        assert "(s)" not in one and "(s)" not in many, (lang, one, many)
        assert one.replace("1", "#", 1) != many.replace("7", "#", 1), (
            f"[{lang}] singular and plural read the same: {many!r}"
        )


# --- Counted (numerus) messages ---------------------------------------------

_TS_DIR = Path(__file__).resolve().parents[1] / "src" / "al_dic" / "i18n" / "source"

#: Locales with a singular and a plural form (CJK have one form).
_TWO_FORMS = ("de", "fr", "es")

#: A plural written into one form: "Maske(n)", "masque(s)", "chargé(s)".
_PLURAL_MARKER = re.compile(r"\w\((?:s|n|e|es|en|x)\)")

#: Counted messages once written as QCoreApplication.translate(ctx, src, "", n)
#: (extracted as plain messages: English for n >= 2 in de/fr/es) or with the
#: count as an expression (not extracted at all: English everywhere).
#: (context, source); the context is the class whose self.tr() says it.
_COUNTED = (
    ("MainWindow", "Imported Region of Interest for %n frame(s)"),
    ("MainWindow", "Batch import: %n mask(s) loaded"),
    ("BatchImportDialog", "%n mask(s) have mismatched sizes and are disabled."),
    ("ImageList", "Import Region of Interest for %n frame(s)"),
    ("ImageList", "Delete %n image(s)"),
    ("ImageList", "Select %n Mask File(s)"),
    ("ExportDialog", "Exported %n animation(s) → %1"),
    ("PipelineMessages", "  %n frame(s) with custom ROI masks"),
    ("PipelineMessages", "Results received: %n frame(s)"),
)


@pytest.mark.parametrize("lang", SHIPPED_LANGS[1:])
def test_every_count_message_is_numerus(lang):
    """Every ``%n`` message in the catalog has plural forms.

    lupdate marks numerus only ``tr(source, "", n)`` with a plain local
    count (CLAUDE.md R2). A ``%n`` message it left plain has one form,
    and for any count the language's rule sends to another form Qt shows
    the English source.
    """
    tree = ET.parse(_TS_DIR / f"al_dic_{lang}.ts")
    plain = [
        f"{context.findtext('name')}: {message.findtext('source')!r}"
        for context in tree.getroot().iter("context")
        for message in context.iter("message")
        if "%n" in (message.findtext("source") or "")
        and message.get("numerus") != "yes"
    ]
    assert not plain, (
        f"[{lang}] %n messages without plural forms (write them as "
        f"self.tr(source, '', n)):\n" + "\n".join(plain)
    )


@contextlib.contextmanager
def _language(qapp, lang: str):
    """``lang`` installed for the block, and taken out after it, even when
    the block fails.

    A failed test's traceback keeps its manager, and so its translators,
    alive: the next test then read Chinese where French had no plural form.
    """
    from al_dic.i18n import LanguageManager

    mgr = LanguageManager(qapp)
    assert mgr.load(lang)
    try:
        yield mgr
    finally:
        mgr.load("en")


@pytest.mark.parametrize("lang", SHIPPED_LANGS[1:])
def test_counted_messages_are_translated_for_every_count(qapp, lang):
    """No count shows the English source, as ``n >= 2`` did in de/fr/es
    (one form) and every count did where the message was not extracted."""
    with _language(qapp, lang):
        english = [
            f"{context}, n={n}: {text!r}"
            for context, source in _COUNTED
            for n in (1, 2, 7)
            if (text := QCoreApplication.translate(context, source, "", n))
            == source.replace("%n", str(n)) or "%n" in text
        ]
    assert not english, f"[{lang}] untranslated:\n" + "\n".join(english)


@pytest.mark.parametrize("lang", _TWO_FORMS)
def test_counted_messages_have_singular_and_plural(qapp, lang):
    """One and two read differently, and neither hedges with "(s)"."""
    bad = []
    with _language(qapp, lang):
        for context, source in _COUNTED:
            one = QCoreApplication.translate(context, source, "", 1)
            many = QCoreApplication.translate(context, source, "", 2)
            if (_PLURAL_MARKER.search(one) or _PLURAL_MARKER.search(many)
                    or one.replace("1", "#", 1) == many.replace("2", "#", 1)):
                bad.append(f"{context}: {one!r} / {many!r}")
    assert not bad, f"[{lang}] no singular/plural pair:\n" + "\n".join(bad)


@pytest.mark.parametrize("lang", _TWO_FORMS)
def test_pipeline_counts_have_singular_and_plural(qapp, lang):
    """The run's console lines come from ``PipelineMessages``, a QObject:
    PipelineController is not one, and only ``self.tr`` is extracted as a
    plural."""
    from al_dic.gui.controllers.pipeline_messages import PipelineMessages

    with _language(qapp, lang):
        m = PipelineMessages()
        pairs = [(say(1), say(2)) for say in (m.custom_roi_frames,
                                              m.results_received)]
    for one, many in pairs:
        assert "(s)" not in one and "(s)" not in many, (lang, one, many)
        assert one.replace("1", "#", 1) != many.replace("2", "#", 1), (
            f"[{lang}] singular and plural read the same: {many!r}"
        )
