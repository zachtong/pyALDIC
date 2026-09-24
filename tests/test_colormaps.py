"""The colormap registry, and the black rainbow map ported from MATLAB.

The reference values come from ``Black_rainbow.m``, not from this
implementation's own output, so a transcription error in the table would fail
here rather than quietly ship a slightly different colormap.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import colormaps

from al_dic.core.colormaps import (
    BLACK_RAINBOW,
    COLORMAP_NAMES,
    BLACK_RAINBOW_NAME,
    resolve,
)


# --- the ported table ----------------------------------------------------

def test_table_has_the_matlab_row_count():
    assert BLACK_RAINBOW.shape == (64, 3)


def test_endpoints_match_the_matlab_table():
    """First and last rows of black_rainbow in Black_rainbow.m."""
    np.testing.assert_allclose(BLACK_RAINBOW[0], [0.0, 0.0, 0.5625])
    np.testing.assert_allclose(BLACK_RAINBOW[-1], [0.5, 0.0, 0.0])


def test_the_middle_is_black():
    """The whole point of the map: jet's green band replaced by black.

    Row 31 of 64, so black sits marginally below the exact centre. That is
    what the MATLAB table does and it is reproduced rather than corrected --
    a colormap that does not match the one a user's earlier figures were made
    with is worse than one that is 0.8% off centre.
    """
    black_rows = [i for i, row in enumerate(BLACK_RAINBOW) if not row.any()]
    assert black_rows == [31]


def test_values_stay_in_range():
    assert BLACK_RAINBOW.min() >= 0.0
    assert BLACK_RAINBOW.max() <= 1.0


def test_it_goes_blue_then_cyan_then_dark_then_yellow_then_red():
    """The rainbow order, sampled at the anchors the MATLAB table sets."""
    cmap = resolve(BLACK_RAINBOW_NAME)
    tol = 1.0 / 255.0                       # a 256-entry lookup quantises
    assert cmap(0.0)[2] > 0.5 and cmap(0.0)[0] == 0.0        # dark blue
    np.testing.assert_allclose(cmap(24 / 63)[:3], (0.0, 1.0, 1.0), atol=tol)
    np.testing.assert_allclose(cmap(38 / 63)[:3], (1.0, 1.0, 0.0), atol=tol)
    red = cmap(1.0)[:3]
    assert red[0] > 0.4 and red[1] == 0.0 and red[2] == 0.0


def test_the_dark_band_sits_near_the_middle():
    """Interpolated to 256, the black row is approached but never landed on.

    Row 31 of 64 sits at 31/63, and a 256-entry lookup samples at j/255 --
    31/63 is not one of those, so the darkest sampled colour is near-black
    rather than black. MATLAB's own `black_rainbow_plus` has the same gap: its
    linspace(1, 64, 256) misses row 32 by the same fraction. Asserting exact
    black here would demand more of the port than the original delivers.
    """
    cmap = resolve(BLACK_RAINBOW_NAME)
    x = np.linspace(0.0, 1.0, 256)
    luminance = cmap(x)[:, :3].sum(axis=1)
    darkest = int(np.argmin(luminance))

    assert luminance[darkest] < 0.05, "the dark band should be near-black"
    assert 0.47 < x[darkest] < 0.52, "and should sit near the middle"


def test_256_sampling_reproduces_matlab_interp1():
    """`black_rainbow_plus` in the .m file is interp1 to 256 rows.

    matplotlib's from_list does the same linear interpolation, so the 'plus'
    variant needs no separate table -- this asserts the two agree.
    """
    cmap = resolve(BLACK_RAINBOW_NAME)
    got = cmap(np.linspace(0.0, 1.0, 256))[:, :3]

    src = np.arange(64, dtype=float)
    query = np.linspace(0.0, 63.0, 256)
    expected = np.column_stack([
        np.interp(query, src, BLACK_RAINBOW[:, c]) for c in range(3)
    ])
    np.testing.assert_allclose(got, expected, atol=2e-3)


# --- registry ------------------------------------------------------------

def test_it_is_registered_with_matplotlib():
    """Every render path resolves names through matplotlib's registry.

    export_png, colorbar, viz_controller and colorbar_overlay all look the
    name up there, so registration is what makes the map work everywhere at
    once -- and each of them silently falls back to jet on a miss.
    """
    assert BLACK_RAINBOW_NAME in colormaps


def test_registering_twice_is_harmless():
    """Import order must not matter; matplotlib raises on a duplicate name."""
    from al_dic.core.colormaps import register

    register()
    register()
    assert BLACK_RAINBOW_NAME in colormaps


def test_it_is_last_in_the_list():
    assert COLORMAP_NAMES[-1] == BLACK_RAINBOW_NAME


def test_the_previous_options_are_all_still_offered():
    assert set(COLORMAP_NAMES) >= {
        "jet", "viridis", "turbo", "coolwarm",
        "plasma", "inferno", "RdBu_r", "seismic",
    }


def test_resolve_falls_back_to_jet_for_an_unknown_name():
    """A stale session or hand-edited config must not crash the render."""
    assert resolve("no_such_colormap").name == "jet"


@pytest.mark.parametrize("name", COLORMAP_NAMES)
def test_every_offered_name_resolves(name):
    assert resolve(name).name == name


# --- no drift between the registry and the widgets -----------------------

def test_every_colormap_chooser_uses_the_registry():
    """Four widgets used to carry their own hardcoded copy of the list.

    Adding a colormap then meant editing four literals, and forgetting one
    left a chooser silently short an option.
    """
    import ast
    from pathlib import Path

    gui = Path(__file__).resolve().parents[1] / "src" / "al_dic" / "gui"
    offenders: list[str] = []
    for py in sorted(gui.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), str(py))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            literals = [
                e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            if {"jet", "viridis", "turbo"} <= set(literals):
                offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, (
        "hardcoded colormap list(s) at "
        + ", ".join(offenders)
        + " -- use al_dic.core.colormaps.COLORMAP_NAMES"
    )


def test_no_render_path_looks_the_name_up_itself():
    """Each render path used to carry its own `except KeyError: jet`.

    That fallback is silent, so a colormap that failed to register produced a
    jet figure and no error at all. Resolution now happens in one place, which
    is also what guarantees registration has run -- an API-only script imports
    no GUI module, so nothing else would have triggered it.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "al_dic"
    registry = src / "core" / "colormaps.py"
    offenders: list[str] = []
    for py in sorted(src.rglob("*.py")):
        if py == registry:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), str(py))
        for node in ast.walk(tree):
            # colormaps[...]
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "colormaps"):
                offenders.append(f"{py.name}:{node.lineno}")
            # plt.get_cmap(...) / matplotlib.cm.get_cmap(...)
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get_cmap"):
                offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, (
        "direct colormap lookup at "
        + ", ".join(offenders)
        + " -- use al_dic.core.colormaps.resolve()"
    )


def test_the_png_export_lut_renders_it():
    """export_png quantises to a 256-entry BGR LUT rather than calling
    matplotlib per pixel; the new map has to survive that path too."""
    from al_dic.export.export_png import _colormap_bgr_lut

    lut = _colormap_bgr_lut(BLACK_RAINBOW_NAME)
    jet = _colormap_bgr_lut("jet")

    assert lut.shape == (256, 3)
    assert not np.array_equal(lut, jet), "silently fell back to jet"

    # The dark band survives quantisation to 8-bit: entry 125 sits at
    # 125/255 = 0.490, within a LUT step of the black row's 31/63 = 0.492.
    brightness = lut.astype(int).sum(axis=1)
    assert brightness.min() <= 8, "the dark band is missing from the LUT"
    assert 120 <= int(brightness.argmin()) <= 135, "and it drifted off centre"


def test_the_colorbar_renders_it():
    """The exported colorbar resolves the name separately from the field."""
    from al_dic.export.colorbar import render_colorbar_strip

    strip = render_colorbar_strip(
        height=200, cmap_name=BLACK_RAINBOW_NAME,
        vmin=0.0, vmax=1.0, label="test",
    )
    jet = render_colorbar_strip(
        height=200, cmap_name="jet", vmin=0.0, vmax=1.0, label="test",
    )
    assert strip is not None and strip.size > 0
    assert not np.array_equal(strip, jet), "silently fell back to jet"
