"""An embedded matplotlib canvas for probe curves.

The only place in the application that touches matplotlib's Qt backend, which
is why ``packaging/pyaldic.spec`` must not exclude
``matplotlib.backends.backend_qtagg`` and ``gui/self_test.py`` checks that it
imports inside a frozen bundle.

The widget remembers its last drawing call. On screen it is drawn in the dark
theme with a frame cursor, a separate artist moved without redrawing, and a
click on the axes reports the x position so the caller can jump there. For a
paper the same call is drawn again on a fresh figure in the light theme --
16 x 10 cm, 300 dpi, text left editable in SVG and PDF, no cursor.
"""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("QtAgg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QCoreApplication, Signal  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402

from al_dic.analysis.series import FrameStatus  # noqa: E402
from al_dic.gui.theme import COLORS  # noqa: E402
from al_dic.gui.widgets.chart_drawing import (  # noqa: E402
    DARK,
    LIGHT,
    Curve,
    draw_curves,
    draw_kymograph,
    draw_message,
    draw_profile,
    table_curves,
    table_kymograph,
    table_profile,
)

# A figure for a paper: a two-column page's width, a readable height.
PUBLICATION_SIZE_CM = (16.0, 10.0)
PUBLICATION_DPI = 300

# Text stays text: editable in a vector editor, searchable in a PDF.
_EDITABLE_TEXT = {"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}

# Which table goes with which drawing.
_TABLES = {draw_curves: table_curves, draw_profile: table_profile,
           draw_kymograph: table_kymograph}


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
        self._last: tuple[Callable, tuple, dict] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.setStyleSheet(f"background-color: {DARK.figure};")
        self.clear()

    # -- content ----------------------------------------------------------

    @property
    def has_data(self) -> bool:
        """True while real curves are drawn, False for a placeholder."""
        return self._has_data

    @property
    def figure(self) -> Figure:
        return self._figure

    def clear(self, message: str | None = None) -> None:
        self._ax = draw_message(self._figure, DARK, message)
        self._cursor, self._has_data, self._last = None, False, None
        self._canvas.draw_idle()

    def _draw(self, draw: Callable, *args, has_data: bool = True, **kwargs) -> None:
        """Draw on screen, and remember the call for an export to repeat."""
        self._ax = draw(self._figure, DARK, *args, **kwargs)
        self._cursor, self._has_data = None, has_data
        self._last = (draw, args, kwargs) if has_data else None
        self._canvas.draw_idle()

    def plot_curves(self, curves: Sequence[Curve], *, x_label: str, y_label: str,
                    integer_x: bool = True, cursor_x: float | None = None) -> None:
        """Draw *curves* on shared axes. Callers keep them to one quantity."""
        self._draw(draw_curves, curves, has_data=bool(curves), x_label=x_label,
                   y_label=y_label, integer_x=integer_x)
        self.set_cursor(cursor_x)

    def plot_profile(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """A field along a line; see ``chart_drawing.draw_profile``."""
        self._draw(draw_profile, x, y, **kwargs)

    def plot_kymograph(self, values: np.ndarray, *, cursor_x: float | None = None,
                       **kwargs) -> None:
        """Distance against frame; see ``chart_drawing.draw_kymograph``."""
        self._draw(draw_kymograph, values, **kwargs)
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

    # -- publication output ------------------------------------------------

    def publication_figure(self) -> Figure:
        """What is drawn now, redrawn light for a page. No cursor."""
        if self._last is None:
            raise ValueError("Nothing is plotted to export.")
        draw, args, kwargs = self._last
        if draw is draw_curves:
            # Emphasis marks the on-screen selection; a page has none.
            args = ([replace(c, emphasised=False) for c in args[0]], *args[1:])
        width, height = PUBLICATION_SIZE_CM
        fig = Figure(figsize=(width / 2.54, height / 2.54), dpi=PUBLICATION_DPI,
                     layout="constrained")
        draw(fig, LIGHT, *args, **kwargs)
        return fig

    def export_figure(self, path: str | Path, *, dpi: int = PUBLICATION_DPI) -> Path:
        """Write the publication figure; the format follows the extension."""
        fig = self.publication_figure()
        out = Path(path)
        with matplotlib.rc_context(_EDITABLE_TEXT):
            fig.savefig(str(out), dpi=dpi, facecolor=fig.get_facecolor())
        return out

    def publication_image(self, *, dpi: int = PUBLICATION_DPI) -> QImage:
        """The publication figure as an image, for the clipboard."""
        buffer = BytesIO()
        fig = self.publication_figure()
        fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        return QImage.fromData(buffer.getvalue(), "PNG")

    def plotted_table(self) -> list[list[str]]:
        """What is drawn now as rows of text, header first, as displayed."""
        if self._last is None:
            raise ValueError("Nothing is plotted to copy.")
        draw, args, kwargs = self._last
        return _TABLES[draw](*args, **kwargs)


__all__ = ["Curve", "MplChart", "PUBLICATION_DPI", "PUBLICATION_SIZE_CM", "status_label"]
