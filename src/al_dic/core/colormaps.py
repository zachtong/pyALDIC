"""Colormaps offered by pyALDIC, in one place.

Four widgets each carried their own hardcoded copy of the list, so adding a
colormap meant editing four literals and forgetting one left a chooser
silently short an option. They all read ``COLORMAP_NAMES`` now.

Every render path -- ``export_png``, ``export.colorbar``, the viewport
controller and the colorbar overlay -- resolves a name through matplotlib's
registry and falls back to jet when the lookup misses. That fallback is silent
by design (a stale session naming a colormap that no longer exists should still
draw), which is exactly why a custom map has to be *registered* rather than
special-cased: miss the registration and the user gets jet with no complaint.
``resolve`` is the one place that fallback now lives.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap, LinearSegmentedColormap
from numpy.typing import NDArray

#: Registered name of the map ported from ``Black_rainbow.m``.
BLACK_RAINBOW_NAME = "black_rainbow"

#: The rainbow/jet progression with green replaced by black, transcribed from
#: the ``black_rainbow`` table in ``Black_rainbow.m``.
#:
#: Black lands on row 31 of 64, marginally below the exact centre. That is what
#: the MATLAB table does; reproducing it faithfully matters more than centring
#: it, because figures made with the original must still match.
#:
#: The file's ``black_rainbow_plus`` is this table run through
#: ``interp1(..., linspace(1, P, 256), 'linear')``. ``from_list`` performs the
#: same linear interpolation, so that variant needs no separate table.
BLACK_RAINBOW: NDArray[np.float64] = np.array([
    (       0,        0,   0.5625),
    (       0,        0,  0.60625),
    (       0,        0,     0.65),
    (       0,        0,  0.69375),
    (       0,        0,   0.7375),
    (       0,        0,  0.78125),
    (       0,        0,    0.825),
    (       0,        0,  0.86875),
    (       0,        0,   0.9125),
    (       0,        0,  0.95625),
    (       0,        0,        1),
    (       0, 0.071429,        1),
    (       0,  0.14286,        1),
    (       0,  0.21429,        1),
    (       0,  0.28571,        1),
    (       0,  0.35714,        1),
    (       0,  0.42857,        1),
    (       0,      0.5,        1),
    (       0,  0.57143,        1),
    (       0,  0.64286,        1),
    (       0,  0.71429,        1),
    (       0,  0.78571,        1),
    (       0,  0.85714,        1),
    (       0,  0.92857,        1),
    (       0,        1,        1),
    (       0,  0.85714,  0.85714),
    (       0,  0.71429,  0.71429),
    (       0,  0.57143,  0.57143),
    (       0,  0.42857,  0.42857),
    (       0,  0.28571,  0.28571),
    (       0,  0.14286,  0.14286),
    (       0,        0,        0),
    ( 0.14286,  0.14286,        0),
    ( 0.28571,  0.28571,        0),
    ( 0.42857,  0.42857,        0),
    ( 0.57143,  0.57143,        0),
    ( 0.71429,  0.71429,        0),
    ( 0.85714,  0.85714,        0),
    (       1,        1,        0),
    (       1,  0.92857,        0),
    (       1,  0.85714,        0),
    (       1,  0.78571,        0),
    (       1,  0.71429,        0),
    (       1,  0.64286,        0),
    (       1,  0.57143,        0),
    (       1,      0.5,        0),
    (       1,  0.42857,        0),
    (       1,  0.35714,        0),
    (       1,  0.28571,        0),
    (       1,  0.21429,        0),
    (       1,  0.14286,        0),
    (       1, 0.071429,        0),
    (       1,        0,        0),
    ( 0.95455,        0,        0),
    ( 0.90909,        0,        0),
    ( 0.86364,        0,        0),
    ( 0.81818,        0,        0),
    ( 0.77273,        0,        0),
    ( 0.72727,        0,        0),
    ( 0.68182,        0,        0),
    ( 0.63636,        0,        0),
    ( 0.59091,        0,        0),
    ( 0.54545,        0,        0),
    (     0.5,        0,        0)
], dtype=np.float64)

#: Offered in the interface and accepted in a batch config, in this order.
COLORMAP_NAMES: tuple[str, ...] = (
    "jet",
    "viridis",
    "turbo",
    "coolwarm",
    "plasma",
    "inferno",
    "RdBu_r",
    "seismic",
    BLACK_RAINBOW_NAME,
)

_FALLBACK = "jet"


def register() -> None:
    """Add pyALDIC's own colormaps to matplotlib's registry.

    Idempotent: matplotlib raises on a duplicate name, and this runs on import
    of a module several others import, so it must tolerate being called again.
    """
    if BLACK_RAINBOW_NAME in colormaps:
        return
    colormaps.register(
        LinearSegmentedColormap.from_list(
            BLACK_RAINBOW_NAME, BLACK_RAINBOW, N=256
        ),
        name=BLACK_RAINBOW_NAME,
    )


def resolve(name: str) -> Colormap:
    """The colormap called *name*, or jet if there is no such thing.

    Falling back rather than raising keeps a stale session or a hand-edited
    config from breaking a render, which is the behaviour every call site
    already had -- written out four times.
    """
    register()
    try:
        return colormaps[name]
    except KeyError:
        return colormaps[_FALLBACK]


register()

__all__ = [
    "BLACK_RAINBOW", "BLACK_RAINBOW_NAME", "COLORMAP_NAMES", "register",
    "resolve",
]
