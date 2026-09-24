"""An embedded matplotlib canvas for probe curves.

The only place in the application that touches matplotlib's Qt backend, which
is why ``packaging/pyaldic.spec`` must not exclude
``matplotlib.backends.backend_qtagg`` and ``gui/self_test.py`` checks that it
imports inside a frozen bundle.

What a curve shows about its own gaps
-------------------------------------
Each curve marks its own trouble, in its own colour; nothing is shaded across
the whole chart. (The first version shaded every frame any probe missed,
behind every curve, with no probe named -- one edge-trimmed point greyed out
four good curves.) A gap in the data is a gap in the line. A frame that
carries a value but is marked -- a crack runs through the probe -- gets a
hollow marker, so the reading is shown and flagged at once.

The current frame is a separate artist, moved without redrawing the curves,
and a click on the axes reports the x position so the caller can jump there.

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

import matplotlib

matplotlib.use("QtAgg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from PySide6.QtCore import QCoreApplication, Signal  # noqa: E402
from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402

from al_dic.analysis.series import FrameStatus  # noqa: E402
from al_dic.gui.theme import COLORS  # noqa: E402

# Above this many points a marker on every frame is noise; markers are kept
# only where a frame is flagged.
_MARKERS_UP_TO = 60

# The shade of consumed material, in a profile's spans and a kymograph's band.
_CONSUMED = COLORS.TEXT_SECONDARY


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


def status_label(status: FrameStatus) -> str:
    """Short, translated explanation for a marked frame.

    Contexts are literals: lupdate cannot follow a context held in a variable,
    and a string it cannot extract can never be translated.
    """
    if status is FrameStatus.CRACK:
        return QCoreApplication.translate("AnalysisChart", "crack")
    if status is FrameStatus.BELOW_THRESHOLD:
        return QCoreApplication.translate("AnalysisChart", "too few valid points")
    if status is FrameStatus.UNRELIABLE:
        return QCoreApplication.translate(
            "AnalysisChart", "unreliable (strain edge trim)")
    if status is FrameStatus.ENDPOINT_LOST:
        return QCoreApplication.translate("AnalysisChart", "gauge endpoint lost")
    if status is FrameStatus.NOT_COMPUTED:
        return QCoreApplication.translate("AnalysisChart", "not computed")
    return QCoreApplication.translate("AnalysisChart", "no data")


@dataclass(frozen=True)
class Curve:
    """One line on the chart, already in display units."""

    label: str
    colour: str
    x: np.ndarray
    y: np.ndarray
    status: Sequence[FrameStatus]
    emphasised: bool = False


class MplChart(QWidget):
    """A figure with a navigation toolbar, styled for the dark theme."""

    # Data-space x of a left click inside the axes.
    x_clicked = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5.0, 3.5), dpi=100, layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._ax = None
        self._cursor = None
        self._has_data = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._apply_theme()
        self.clear()

    # -- appearance -------------------------------------------------------

    def _apply_theme(self) -> None:
        self._figure.set_facecolor(COLORS.BG_PANEL)
        self._canvas.setStyleSheet(f"background-color: {COLORS.BG_PANEL};")

    def _style_axes(self, ax) -> None:
        ax.set_facecolor(COLORS.BG_DARKEST)
        for spine in ax.spines.values():
            spine.set_color(COLORS.BORDER)
        ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=9)
        ax.xaxis.label.set_color(COLORS.TEXT_PRIMARY)
        ax.yaxis.label.set_color(COLORS.TEXT_PRIMARY)
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.grid(True, color=COLORS.BORDER, linewidth=0.5, alpha=0.6)

    # -- content ----------------------------------------------------------

    @property
    def has_data(self) -> bool:
        """True while real curves are drawn, False for a placeholder."""
        return self._has_data

    def clear(self, message: str | None = None) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        if message:
            ax.text(
                0.5, 0.5, message, ha="center", va="center", wrap=True,
                transform=ax.transAxes, color=COLORS.TEXT_MUTED, fontsize=10,
            )
        self._ax, self._cursor, self._has_data = ax, None, False
        self._canvas.draw_idle()

    def plot_curves(
        self,
        curves: Sequence[Curve],
        *,
        x_label: str,
        y_label: str,
        integer_x: bool = True,
        cursor_x: float | None = None,
    ) -> None:
        """Draw *curves* on shared axes. Callers keep them to one quantity."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)

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
                    ax.plot(
                        c.x[start:i], c.y[start:i], color=c.colour,
                        linewidth=width,
                        marker="o" if show_all_markers else None,
                        markersize=3.5,
                        label=c.label if not drew else None,
                        zorder=3 if c.emphasised else 2,
                    )
                    drew = True
                    start = None
            if not drew:
                # Nothing measurable: keep the legend entry, whose label
                # carries the reason, so the probe does not simply vanish.
                ax.plot([], [], color=c.colour, linewidth=width, label=c.label)
            flagged = np.array([
                st is FrameStatus.CRACK for st in c.status
            ]) & finite
            if flagged.any():
                ax.plot(
                    c.x[flagged], c.y[flagged], linestyle="none", marker="o",
                    markersize=6, markerfacecolor="none",
                    markeredgecolor=c.colour, markeredgewidth=1.4, zorder=4,
                )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if integer_x:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if curves:
            legend = ax.legend(fontsize=9, framealpha=0.0)
            for text in legend.get_texts():
                text.set_color(COLORS.TEXT_SECONDARY)
        self._ax, self._cursor, self._has_data = ax, None, bool(curves)
        self.set_cursor(cursor_x)

    def set_cursor(self, x: float | None) -> None:
        """Move the current-frame marker without redrawing the curves."""
        if self._ax is None or not self._has_data:
            return
        if self._cursor is not None:
            self._cursor.remove()
            self._cursor = None
        if x is not None and np.isfinite(x):
            self._cursor = self._ax.axvline(
                x, color=COLORS.ACCENT, linestyle="--", linewidth=1.0,
                alpha=0.9, zorder=1,
            )
        self._canvas.draw_idle()

    def _on_press(self, event) -> None:
        # Ignore clicks that belong to the toolbar's pan/zoom modes.
        if self._toolbar.mode or event.inaxes is not self._ax:
            return
        if event.button == 1 and event.xdata is not None and self._has_data:
            self.x_clicked.emit(float(event.xdata))

    def plot_profile(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        colour: str,
        label: str,
        x_label: str,
        y_label: str,
        others: Sequence[np.ndarray] = (),
        consumed: np.ndarray | None = None,
        consumed_label: str = "",
    ) -> None:
        """A field along a line on one frame, over the *others* in grey.

        NaN breaks the line, so a gap stays a gap; where *consumed* is True,
        the span is shaded and named in the legend.
        """
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)
        for other in others:
            ax.plot(x, other, color=COLORS.TEXT_MUTED, linewidth=0.9, alpha=0.5,
                    zorder=1, gid="other")
        ax.plot(x, y, color=colour, linewidth=2.0, label=label, zorder=3,
                gid="current")
        if consumed is not None and len(x):
            half = (x[1] - x[0]) / 2.0 if len(x) > 1 else 0.5
            for i, (a, b) in enumerate(_true_runs(consumed)):
                ax.axvspan(x[a] - half, x[b - 1] + half, color=_CONSUMED,
                           alpha=0.18, linewidth=0, zorder=0,
                           label=consumed_label if i == 0 else None)
        if len(x) > 1:
            ax.set_xlim(float(x[0]), float(x[-1]))   # the whole line, gaps too
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        legend = ax.legend(fontsize=9, framealpha=0.0)
        for text in legend.get_texts():
            text.set_color(COLORS.TEXT_SECONDARY)
        self._ax, self._cursor, self._has_data = ax, None, True
        self._canvas.draw_idle()

    def plot_kymograph(
        self,
        values: np.ndarray,
        *,
        x: np.ndarray,
        distance: np.ndarray,
        x_label: str,
        y_label: str,
        value_label: str,
        colormap: str = "jet",
        vmin: float | None = None,
        vmax: float | None = None,
        consumed: np.ndarray | None = None,
        consumed_label: str = "",
        integer_x: bool = True,
        cursor_x: float | None = None,
    ) -> None:
        """Distance along a line against frame, the value as colour.

        *values* and *consumed* are ``[sample, frame]``; *x* places each frame
        (frame number or time) and *distance* each sample. Cells without a
        value show the axes behind them; consumed cells are a flat band.
        """
        from al_dic.core.colormaps import resolve

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)
        ax.grid(False)
        extent = (*_edges(x), *_edges(distance))
        cmap = resolve(colormap).with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
        image = ax.imshow(
            np.ma.masked_invalid(values), aspect="auto", origin="lower",
            cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
            interpolation="nearest",
        )
        image.set_gid("values")
        if consumed is not None and np.any(consumed):
            band = np.ma.masked_where(~np.asarray(consumed, dtype=bool),
                                      np.ones(np.shape(consumed)))
            layer = ax.imshow(
                band, aspect="auto", origin="lower",
                cmap=ListedColormap([_CONSUMED]), vmin=0.0, vmax=1.0,
                extent=extent, interpolation="nearest", alpha=0.85,
            )
            layer.set_gid("consumed")
            legend = ax.legend(
                handles=[Patch(facecolor=_CONSUMED, label=consumed_label)],
                fontsize=9, framealpha=0.0, loc="upper left")
            for text in legend.get_texts():
                text.set_color(COLORS.TEXT_SECONDARY)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if integer_x:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        bar = self._figure.colorbar(image, ax=ax)
        bar.set_label(value_label, color=COLORS.TEXT_PRIMARY, fontsize=9)
        bar.ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=8)
        bar.outline.set_edgecolor(COLORS.BORDER)
        self._ax, self._cursor, self._has_data = ax, None, True
        self.set_cursor(cursor_x)

    # -- export -----------------------------------------------------------

    def save_figure(self, path: str, dpi: int = 200) -> None:
        """Write the figure exactly as drawn on screen."""
        self._figure.savefig(
            path, dpi=dpi, facecolor=self._figure.get_facecolor()
        )

    @property
    def figure(self) -> Figure:
        return self._figure


__all__ = ["Curve", "MplChart", "status_label"]
