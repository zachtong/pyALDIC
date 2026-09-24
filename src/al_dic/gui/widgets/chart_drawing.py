"""Drawing the Analysis charts on any matplotlib Figure, in a theme.

The chart widget draws on screen in the dark theme; an export draws the same
call again on a fresh Figure in the light one. Keeping the drawing apart from
the widget is what makes an export a redraw, not a recolouring of the screen's
figure -- which would carry the frame cursor and the dark legend along.

What a curve shows about its own gaps
-------------------------------------
Each curve marks its own trouble, in its own colour; nothing is shaded across
the whole chart. (The first version shaded every frame any probe missed,
behind every curve, with no probe named -- one edge-trimmed point greyed out
four good curves.) A gap in the data is a gap in the line. A frame that
carries a value but is marked -- a crack runs through the probe -- gets a
hollow marker, so the reading is shown and flagged at once.

Along a line
------------
A profile draws the current frame in the probe's colour over the other frames
in grey; a kymograph draws every frame at once, distance against frame. In
both, material a crack consumed has its own shade: it is neither a zero nor
a missing measurement, and looking like either would misread the crack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from al_dic.analysis.series import FrameStatus
from al_dic.gui.theme import COLORS

# Above this many points a marker on every frame is noise; markers are kept
# only where a frame is flagged.
_MARKERS_UP_TO = 60


@dataclass(frozen=True)
class ChartTheme:
    """Colours and type sizes of one look of the chart."""

    figure: str
    axes: str
    spine: str
    tick: str
    label: str
    grid: str
    grid_alpha: float
    legend: str
    muted: str          # a profile's other frames; a placeholder's text
    consumed: str       # material a crack consumed
    tick_size: float
    label_size: float
    legend_size: float


# The application's dark look.
DARK = ChartTheme(
    figure=COLORS.BG_PANEL, axes=COLORS.BG_DARKEST, spine=COLORS.BORDER,
    tick=COLORS.TEXT_SECONDARY, label=COLORS.TEXT_PRIMARY, grid=COLORS.BORDER,
    grid_alpha=0.6, legend=COLORS.TEXT_SECONDARY, muted=COLORS.TEXT_MUTED,
    consumed=COLORS.TEXT_SECONDARY, tick_size=9, label_size=10, legend_size=9,
)

# A figure for a paper or a report: white, black type, sizes for a printed
# column rather than a screen.
LIGHT = ChartTheme(
    figure="#ffffff", axes="#ffffff", spine="#333333", tick="#222222",
    label="#000000", grid="#d4d4d4", grid_alpha=1.0, legend="#222222",
    muted="#a3a3a3", consumed="#8c8c8c", tick_size=8, label_size=9,
    legend_size=8,
)


@dataclass(frozen=True)
class Curve:
    """One line on the chart, already in display units."""

    label: str          # the legend's text, a note included
    colour: str
    x: np.ndarray
    y: np.ndarray
    status: Sequence[FrameStatus]
    emphasised: bool = False
    name: str = ""      # the probe alone, for a table; the label if empty


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges where *mask* is True."""
    runs, start = [], None
    for i, on in enumerate(np.asarray(mask, dtype=bool)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _edges(centres: np.ndarray) -> tuple[float, float]:
    """Outer edges of evenly spaced cells around *centres*."""
    c = np.asarray(centres, dtype=np.float64)
    half = (c[1] - c[0]) / 2.0 if len(c) > 1 else 0.5
    return float(c[0] - half), float(c[-1] + half)


def new_axes(fig: Figure, theme: ChartTheme) -> Axes:
    """Clear *fig* and give it one styled axes."""
    fig.clear()
    fig.set_facecolor(theme.figure)
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.axes)
    for spine in ax.spines.values():
        spine.set_color(theme.spine)
    ax.tick_params(colors=theme.tick, labelsize=theme.tick_size)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(theme.label)
        label.set_fontsize(theme.label_size)
    ax.grid(True, color=theme.grid, linewidth=0.5, alpha=theme.grid_alpha)
    return ax


def _legend(ax: Axes, theme: ChartTheme, **kwargs) -> None:
    legend = ax.legend(fontsize=theme.legend_size, framealpha=0.0, **kwargs)
    for text in legend.get_texts():
        text.set_color(theme.legend)


def draw_message(fig: Figure, theme: ChartTheme, message: str | None) -> Axes:
    """Empty axes that say why they are empty."""
    ax = new_axes(fig, theme)
    ax.set_xticks([])
    ax.set_yticks([])
    if message:
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
                transform=ax.transAxes, color=theme.muted, fontsize=theme.label_size)
    return ax


def draw_curves(fig: Figure, theme: ChartTheme, curves: Sequence[Curve], *,
                x_label: str, y_label: str, integer_x: bool = True) -> Axes:
    """*curves* on shared axes. Callers keep them to one quantity."""
    ax = new_axes(fig, theme)
    for c in curves:
        finite = np.isfinite(c.y)
        show_all_markers = len(c.x) <= _MARKERS_UP_TO
        width = 2.4 if c.emphasised else 1.6
        drew = False
        # One polyline per unbroken stretch: a gap stays a gap.
        start = None
        for i in range(len(c.y) + 1):
            ok = i < len(c.y) and finite[i]
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                ax.plot(c.x[start:i], c.y[start:i], color=c.colour, linewidth=width,
                        marker="o" if show_all_markers else None, markersize=3.5,
                        label=c.label if not drew else None,
                        zorder=3 if c.emphasised else 2)
                drew = True
                start = None
        if not drew:
            # Nothing measurable: keep the legend entry, whose label carries
            # the reason, so the probe does not simply vanish.
            ax.plot([], [], color=c.colour, linewidth=width, label=c.label)
        flagged = np.array([st is FrameStatus.CRACK for st in c.status]) & finite
        if flagged.any():
            ax.plot(c.x[flagged], c.y[flagged], linestyle="none", marker="o",
                    markersize=6, markerfacecolor="none", markeredgecolor=c.colour,
                    markeredgewidth=1.4, zorder=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if curves:
        _legend(ax, theme)
    return ax


def draw_profile(fig: Figure, theme: ChartTheme, x: np.ndarray, y: np.ndarray, *,
                 colour: str, label: str, x_label: str, y_label: str,
                 others: Sequence[np.ndarray] = (),
                 consumed: np.ndarray | None = None,
                 consumed_label: str = "") -> Axes:
    """A field along a line on one frame, over the *others* in grey.

    NaN breaks the line, so a gap stays a gap; where *consumed* is True, the
    span is shaded and named in the legend.
    """
    ax = new_axes(fig, theme)
    for other in others:
        ax.plot(x, other, color=theme.muted, linewidth=0.9, alpha=0.5, zorder=1,
                gid="other")
    ax.plot(x, y, color=colour, linewidth=2.0, label=label, zorder=3, gid="current")
    if consumed is not None and len(x):
        half = (x[1] - x[0]) / 2.0 if len(x) > 1 else 0.5
        for i, (a, b) in enumerate(_true_runs(consumed)):
            ax.axvspan(x[a] - half, x[b - 1] + half, color=theme.consumed,
                       alpha=0.18, linewidth=0, zorder=0,
                       label=consumed_label if i == 0 else None)
    if len(x) > 1:
        ax.set_xlim(float(x[0]), float(x[-1]))   # the whole line, gaps too
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _legend(ax, theme)
    return ax


def draw_kymograph(fig: Figure, theme: ChartTheme, values: np.ndarray, *,
                   x: np.ndarray, distance: np.ndarray, x_label: str,
                   y_label: str, value_label: str, colormap: str = "jet",
                   vmin: float | None = None, vmax: float | None = None,
                   consumed: np.ndarray | None = None, consumed_label: str = "",
                   integer_x: bool = True) -> Axes:
    """Distance along a line against frame, the value as colour.

    *values* and *consumed* are ``[sample, frame]``; *x* places each frame
    (frame number or time) and *distance* each sample. Cells without a value
    show the axes behind them; consumed cells are a flat band.
    """
    from al_dic.core.colormaps import resolve

    ax = new_axes(fig, theme)
    ax.grid(False)
    extent = (*_edges(x), *_edges(distance))
    cmap = resolve(colormap).with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", origin="lower",
                      cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                      interpolation="nearest")
    image.set_gid("values")
    if consumed is not None and np.any(consumed):
        band = np.ma.masked_where(~np.asarray(consumed, dtype=bool),
                                  np.ones(np.shape(consumed)))
        layer = ax.imshow(band, aspect="auto", origin="lower",
                          cmap=ListedColormap([theme.consumed]), vmin=0.0, vmax=1.0,
                          extent=extent, interpolation="nearest", alpha=0.85)
        layer.set_gid("consumed")
        _legend(ax, theme, handles=[Patch(facecolor=theme.consumed, label=consumed_label)],
                loc="upper left")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    bar = fig.colorbar(image, ax=ax)
    bar.set_label(value_label, color=theme.label, fontsize=theme.legend_size)
    bar.ax.tick_params(colors=theme.tick, labelsize=theme.tick_size - 1)
    bar.outline.set_edgecolor(theme.spine)
    return ax


# -- the numbers behind a drawing ------------------------------------------------
#
# Each table function takes the arguments of its draw function and gives back
# rows of text, header first, in display units: what the chart shows, for a
# spreadsheet. A missing value is an empty cell, never 0.

def _cell(value: float) -> str:
    return f"{value:.9g}" if np.isfinite(value) else ""


def table_curves(curves: Sequence[Curve], *, x_label: str, y_label: str,
                 **_drawing) -> list[list[str]]:
    header = [x_label] + [f"{c.name or c.label} — {y_label}" for c in curves]
    xs = sorted({float(x) for c in curves for x in c.x})
    by_x = [dict(zip((float(x) for x in c.x), c.y)) for c in curves]
    rows = [[_cell(x)] + [_cell(values.get(x, np.nan)) for values in by_x]
            for x in xs]
    return [header] + rows


def table_profile(x: np.ndarray, y: np.ndarray, *, label: str, x_label: str,
                  y_label: str, **_drawing) -> list[list[str]]:
    header = [x_label, f"{label} — {y_label}"]
    return [header] + [[_cell(a), _cell(b)] for a, b in zip(x, y)]


def table_kymograph(values: np.ndarray, *, x: np.ndarray, distance: np.ndarray,
                    x_label: str, y_label: str, **_drawing) -> list[list[str]]:
    header = [f"{y_label} \\ {x_label}"] + [_cell(v) for v in x]
    return [header] + [[_cell(d)] + [_cell(v) for v in values[s]]
                       for s, d in enumerate(distance)]


__all__ = [
    "DARK", "LIGHT", "ChartTheme", "Curve", "draw_curves", "draw_kymograph",
    "draw_message", "draw_profile", "new_axes", "table_curves",
    "table_kymograph", "table_profile",
]
