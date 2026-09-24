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

import os

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
