"""Light and dark themes: palettes and stylesheet, the Settings menu, live switching, the saved choice.

The switch test is the one that matters. A widget that set its own style sheet
from the palette once, at construction, keeps the old theme's colours after a
live switch -- a dark card on a white window. After switching, no widget of the
main window, the strain window or the export dialog may carry a colour only the
other theme uses. Ported from pyALDVC's theme tests; the palette is the same.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QGraphicsView, QWidget

from al_dic.gui.theme import (
    ARROWS_DIR,
    COLORS,
    DARK,
    LIGHT,
    THEMES,
    build_stylesheet,
    current_colors,
    current_theme,
    set_current_theme,
    style_text,
)
from al_dic.gui.theme_manager import (
    ENV_THEME,
    SETTINGS_KEY,
    qcolor,
    startup_theme,
    theme_manager,
    themed,
)

app = QApplication.instance() or QApplication([])


def _hexes(colors) -> set[str]:
    return {v.lower() for v in asdict(colors).values()
            if isinstance(v, str) and v.startswith("#")}


# Colours only one theme uses. The legacy literals were dark-only greens and
# ambers that sat unreadable on white until they joined the palette, and the
# analysis canvas's hint banner, a dark card that kept dark text in the light
# theme. White is the text on accent and semantic buttons in both themes, so
# it never marks a widget as left behind.
DARK_ONLY = (_hexes(DARK) - _hexes(LIGHT)) | {
    DARK.OVERLAY_BG.lower(), "#4ade80", "#fbbf24", "rgba(20, 25, 41, 215)"}
LIGHT_ONLY = (_hexes(LIGHT) - _hexes(DARK) - {"#ffffff"}) | {LIGHT.OVERLAY_BG.lower()}


@pytest.fixture
def dark_again(tmp_path):
    """Every test starts dark with nothing saved, and leaves it that way:
    other modules expect the default, and the user's own settings are never
    touched (QSettings is redirected to a temporary folder)."""
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    theme_manager(app).set_theme("dark", persist=False)
    yield
    theme_manager(app).set_theme("dark", persist=False)
    app.setStyleSheet("")
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)


def _luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# ---------------------------------------------------------------- palettes, stylesheet
def test_the_dark_theme_is_the_default_and_its_stylesheet_is_the_original_one():
    assert set(THEMES) == {"dark", "light"} and THEMES["dark"] is DARK
    set_current_theme("dark")
    assert current_theme() == "dark" and current_colors() is DARK
    css = build_stylesheet(DARK)
    assert css == build_stylesheet()  # the theme in use
    # the original dark theme: no transparent containers, the style's own selection
    assert "QComboBoxPrivateContainer" not in css and "\nQAbstractItemView {" not in css
    assert "#0b0f1a" in css and "spin_up.svg" in css
    with pytest.raises(ValueError):
        set_current_theme("sepia")


def test_the_light_stylesheet_is_white_with_grey_sidebars_and_dark_arrows():
    css = build_stylesheet(LIGHT)
    assert "QMainWindow,\nQWidget {\n    background: #ffffff;" in css
    assert "background: #f6f7f9;" in css  # the sidebars
    assert "selection-background-color: #e0e7ff;" in css
    assert not [h for h in DARK_ONLY if h in css.lower()]  # nothing of the dark theme
    for c in (DARK, LIGHT):  # every arrow a stylesheet names is shipped
        for name in (c.ARROW_UP, c.ARROW_DOWN, c.ARROW_UP_HOVER, c.ARROW_DOWN_HOVER,
                     c.COMBO_ARROW_HOVER):
            assert (ARROWS_DIR / name).is_file(), name
            assert (ARROWS_DIR / name).as_posix() in build_stylesheet(c)


def test_the_light_palette_is_readable():
    c = LIGHT
    for bg in (c.BG_DARKEST, c.BG_SIDEBAR, c.BG_PANEL, c.BG_INPUT, c.BG_HOVER, c.SELECTION):
        assert _contrast(c.TEXT_PRIMARY, bg) >= 7.0, bg
        assert _contrast(c.TEXT_SECONDARY, bg) >= 4.5, bg
        assert _contrast(c.TEXT_MUTED, bg) >= 3.0, bg  # hints: small, but never faint
    for accent in (c.ACCENT, c.ACCENT_PRESSED):
        assert _contrast("#ffffff", accent) >= 4.5  # white text on the accent buttons
    for semantic in (c.DANGER, c.WARNING):
        assert _contrast(semantic, c.BG_DARKEST) >= 4.5
    assert _contrast(c.SUCCESS, c.BG_DARKEST) >= 3.0


def test_colors_reads_the_theme_in_use():
    set_current_theme("light")
    try:
        assert COLORS.ACCENT == LIGHT.ACCENT and COLORS.IS_DARK is False
    finally:
        set_current_theme("dark")
    assert COLORS.ACCENT == DARK.ACCENT
    with pytest.raises(AttributeError):
        COLORS.NOT_A_COLOUR  # noqa: B018 -- a missing token still fails loudly


def test_style_templates_take_the_colours_of_the_theme():
    template = "QLabel {{ color: {TEXT_PRIMARY}; border: 1px solid {BORDER}; }}"
    assert style_text(template, DARK) == "QLabel { color: #e2e8f0; border: 1px solid #1e293b; }"
    assert style_text(template, LIGHT) == "QLabel { color: #111827; border: 1px solid #d1d5db; }"


def test_qcolor_reads_both_forms_the_palette_holds():
    assert qcolor("#4f46e5").name() == "#4f46e5"
    overlay = qcolor(LIGHT.OVERLAY_BG)
    assert (overlay.red(), overlay.green(), overlay.blue(), overlay.alpha()) == (255, 255, 255, 230)


# ---------------------------------------------------------------- the Settings menu
def test_the_settings_menu_offers_both_themes(dark_again):
    from al_dic.gui.app import MainWindow

    window = MainWindow()
    try:
        actions = window._theme_actions
        assert [a.text() for a in actions.values()] == ["Dark", "Light"]
        assert all(a.isCheckable() for a in actions.values())
        assert actions["dark"].actionGroup().isExclusive()
        assert actions["dark"].isChecked() and not actions["light"].isChecked()
        actions["light"].trigger()
        assert current_theme() == "light" and actions["light"].isChecked()
        assert QSettings().value(SETTINGS_KEY) == "light"  # remembered
    finally:
        window.close()


# ---------------------------------------------------------------- live switching
def _result(img=96):
    from al_dic.core.config import dicpara_default
    from al_dic.core.data_structures import (
        DICMesh, FrameResult, FrameSchedule, PipelineResult, StrainResult,
    )

    xs, ys = np.meshgrid(np.linspace(8, img - 8, 5), np.linspace(8, img - 8, 5))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    n = coords.shape[0]
    mesh = DICMesh(coordinates_fem=coords, elements_fem=np.zeros((0, 8), np.int64))
    u = coords[:, 0] / img
    fr = FrameResult(U=np.repeat(u, 2), U_accum=np.repeat(u, 2))
    zeros, small = np.zeros(n), np.full(n, 0.01)
    sr = StrainResult(disp_u=u, disp_v=zeros, strain_exx=small, strain_eyy=zeros,
                      strain_exy=zeros, strain_principal_max=small,
                      strain_principal_min=zeros, strain_maxshear=small,
                      strain_von_mises=small, strain_rotation=zeros)
    return PipelineResult(
        dic_para=dicpara_default(img_size=(img, img)), dic_mesh=mesh,
        result_disp=[fr, fr], result_def_grad=[fr, fr], result_strain=[sr, sr],
        result_fe_mesh_each_frame=[mesh, mesh],
        frame_schedule=FrameSchedule.from_mode("accumulative", 3))


def _windows():
    from al_dic.gui.app import MainWindow
    from al_dic.gui.app_state import AppState
    from al_dic.gui.dialogs.export_dialog import ExportDialog, VizExportHint
    from al_dic.gui.strain_window import StrainWindow

    return [MainWindow(), StrainWindow(AppState.instance()),
            ExportDialog(_result(), None, VizExportHint(), image_files=[])]


def _offenders(windows, forbidden) -> list[str]:
    bad = []
    for win in windows:
        for widget in (win, *win.findChildren(QWidget)):
            css = widget.styleSheet().lower()
            hits = sorted(h for h in forbidden if h in css)
            if isinstance(widget, QGraphicsView):
                colour = widget.backgroundBrush().color().name().lower()
                if colour in forbidden:
                    hits.append(f"canvas {colour}")
            if hits:
                name = widget.objectName() or "-"
                bad.append(f"{type(win).__name__} > {type(widget).__name__}#{name}: {hits}")
    return bad


def test_a_live_switch_leaves_no_colour_of_the_other_theme(dark_again):
    windows = _windows()
    try:
        theme_manager(app).set_theme("light", persist=False)
        left = _offenders(windows, DARK_ONLY)
        assert not left, f"{len(left)} widget(s) kept dark colours:\n  " + "\n  ".join(left)
        theme_manager(app).set_theme("dark", persist=False)
        left = _offenders(windows, LIGHT_ONLY)
        assert not left, f"{len(left)} widget(s) kept light colours:\n  " + "\n  ".join(left)
    finally:
        for win in windows:
            win.close()


def test_a_window_built_in_the_light_theme_is_light_throughout(dark_again):
    theme_manager(app).set_theme("light", persist=False)
    windows = _windows()
    try:
        left = _offenders(windows, DARK_ONLY)
        assert not left, f"{len(left)} widget(s) built dark:\n  " + "\n  ".join(left)
    finally:
        for win in windows:
            win.close()


def _icon_brightness(toolbar) -> float:
    """Mean brightness, 0 (black) to 1 (white), of a toolbar icon's opaque pixels."""
    action = next(a for a in toolbar.actions() if not a.icon().isNull())
    image = action.icon().pixmap(24, 24).toImage()
    values = [image.pixelColor(x, y).valueF()
              for x in range(image.width()) for y in range(image.height())
              if image.pixelColor(x, y).alpha() > 200]
    assert values, "the icon has no opaque pixels"
    return sum(values) / len(values)


def test_the_chart_toolbar_icons_follow_the_theme(dark_again):
    """matplotlib draws white toolbar icons on a background it finds dark, and
    it found the light theme's transparent containers black: white on white."""
    from al_dic.gui.widgets.mpl_chart import MplChart

    charts = [MplChart()]
    try:
        assert _icon_brightness(charts[0]._toolbar) > 0.8  # light icons, dark theme
        theme_manager(app).set_theme("light", persist=False)
        charts.append(MplChart())  # built light, besides the one switched live
        for chart in charts:
            assert _icon_brightness(chart._toolbar) < 0.2  # dark icons, light theme
        theme_manager(app).set_theme("dark", persist=False)
        for chart in charts:
            assert _icon_brightness(chart._toolbar) > 0.8
    finally:
        for chart in charts:
            chart.close()


def test_a_themed_widget_is_restyled_on_a_switch(dark_again):
    from PySide6.QtWidgets import QLabel

    label = themed(QLabel("x"), "QLabel {{ color: {TEXT_PRIMARY}; }}")
    assert DARK.TEXT_PRIMARY in label.styleSheet()
    theme_manager(app).set_theme("light", persist=False)
    from al_dic.gui.theme_manager import restyle

    restyle(label)  # a window on its own; the manager restyles top-level windows
    assert LIGHT.TEXT_PRIMARY in label.styleSheet()


# ---------------------------------------------------------------- the start-up theme
def test_the_saved_theme_is_applied_at_start_up(dark_again, monkeypatch):
    monkeypatch.delenv(ENV_THEME, raising=False)
    assert startup_theme() == "dark"  # nothing saved
    QSettings().setValue(SETTINGS_KEY, "light")
    assert startup_theme() == "light"
    QSettings().setValue(SETTINGS_KEY, "sepia")  # an unknown value falls back
    assert startup_theme() == "dark"


def test_the_environment_pins_the_start_theme(dark_again, monkeypatch):
    QSettings().setValue(SETTINGS_KEY, "dark")
    monkeypatch.setenv(ENV_THEME, "light")
    assert startup_theme() == "light"
    monkeypatch.setenv(ENV_THEME, "sepia")  # ignored, not an error
    assert startup_theme() == "dark"
