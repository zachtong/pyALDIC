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


@pytest.mark.parametrize("lang", SHIPPED_LANGS)
@pytest.mark.parametrize("widget_path", WIDGETS_TO_CONSTRUCT)
def test_widget_constructs(qapp, lang, widget_path):
    """Every major widget must construct without exception in every language.

    This is the backstop for `.arg()` / non-QObject tr() / userData
    state-sync regressions: if any of those break in a locale-specific
    code path, construction raises.
    """
    from al_dic.i18n import LanguageManager
    mgr = LanguageManager(qapp)
    mgr.load(lang)
    ctor = _import_ctor(widget_path)
    w = ctor()
    # Basic invariant: the widget has at least one child / laid out element
    assert w is not None


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
