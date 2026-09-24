"""An embedded matplotlib canvas for probe curves and kymographs.

The only place in the application that touches matplotlib's Qt backend. Screen
and exported figure go through the same code, so what a user sees is what they
get in a file -- which is also why ``packaging/pyaldic.spec`` must not exclude
``matplotlib.backends.backend_qtagg``, and why ``gui/self_test.py`` checks that
it imports inside a frozen bundle.

Gaps in a series stay gaps. ``TimeSeries.contiguous_runs`` gives one polyline
per unbroken stretch and the missing frames are shaded with the reason, because
a line drawn straight through absent data is a claim nobody made.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402

from al_dic.analysis.series import FrameStatus, TimeSeries  # noqa: E402
from al_dic.gui.theme import COLORS  # noqa: E402

#: Shading for frames a probe could not measure.
_GAP_COLOUR = "#94a3b8"
_GAP_ALPHA = 0.18


def _status_label(status: FrameStatus) -> str:
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5.0, 3.5), dpi=100, layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

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
        ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=8)
        ax.xaxis.label.set_color(COLORS.TEXT_PRIMARY)
        ax.yaxis.label.set_color(COLORS.TEXT_PRIMARY)
        ax.title.set_color(COLORS.TEXT_PRIMARY)
        ax.grid(True, color=COLORS.BORDER, linewidth=0.5, alpha=0.6)

    # -- content ----------------------------------------------------------

    def clear(self, message: str | None = None) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        if message:
            ax.text(
                0.5, 0.5, message, ha="center", va="center",
                transform=ax.transAxes, color=COLORS.TEXT_MUTED, fontsize=10,
            )
        self._canvas.draw_idle()

    def plot_series(
        self,
        entries: list[tuple[str, str, TimeSeries]],
        *,
        y_label: str,
        current_frame: int | None = None,
    ) -> None:
        """Draw ``(label, colour, series)`` curves on shared axes.

        Every series must share a unit; the caller enforces that by only
        offering probes of one kind and one field at a time.
        """
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)

        gap_reasons: set[FrameStatus] = set()
        for label, colour, ts in entries:
            frames = ts.frames + 1          # 1-based on screen, as in the file
            drew = False
            for start, stop in ts.contiguous_runs():
                ax.plot(
                    frames[start:stop], ts.values[start:stop],
                    color=colour, linewidth=1.6,
                    marker="o", markersize=3.0,
                    label=label if not drew else None,
                )
                drew = True
            if not drew:
                # Nothing measurable at all: still claim the legend entry so the
                # probe does not simply vanish from the chart.
                ax.plot([], [], color=colour, linewidth=1.6, label=label)
            for i, status in enumerate(ts.status):
                if status is not FrameStatus.OK:
                    gap_reasons.add(status)
                    ax.axvspan(
                        frames[i] - 0.5, frames[i] + 0.5,
                        color=_GAP_COLOUR, alpha=_GAP_ALPHA, linewidth=0,
                    )

        ax.set_xlabel(QCoreApplication.translate("AnalysisChart", "Frame"))
        ax.set_ylabel(y_label)
        if current_frame is not None:
            ax.axvline(
                current_frame + 1, color=COLORS.ACCENT,
                linestyle="--", linewidth=1.0, alpha=0.8,
            )
        if entries:
            legend = ax.legend(fontsize=8, framealpha=0.0)
            for text in legend.get_texts():
                text.set_color(COLORS.TEXT_SECONDARY)
        if gap_reasons:
            ax.set_title(
                QCoreApplication.translate(
                    "AnalysisChart", "Shaded frames: %1"
                ).replace(
                    "%1", ", ".join(sorted(_status_label(s) for s in gap_reasons))
                ),
                fontsize=8, color=COLORS.TEXT_MUTED, loc="left",
            )
        self._canvas.draw_idle()

    def plot_kymograph(
        self,
        data,
        *,
        distance_unit: str,
        length: float,
        value_label: str,
        colormap: str = "jet",
    ) -> None:
        """Position along a line against frame, with the value as colour.

        A growing crack shows up as an invalid boundary sweeping across the
        image -- the crack tip's trajectory, read directly.
        """
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)
        ax.grid(False)

        n_points, n_frames = data.shape
        image = ax.imshow(
            data, aspect="auto", origin="lower", cmap=colormap,
            extent=(0.5, n_frames + 0.5, 0.0, length),
            interpolation="nearest",
        )
        ax.set_xlabel(QCoreApplication.translate("AnalysisChart", "Frame"))
        ax.set_ylabel(
            QCoreApplication.translate(
                "AnalysisChart", "Distance along line (%1)"
            ).replace("%1", distance_unit)
        )
        bar = self._figure.colorbar(image, ax=ax)
        bar.set_label(value_label, color=COLORS.TEXT_PRIMARY, fontsize=9)
        bar.ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=8)
        bar.outline.set_edgecolor(COLORS.BORDER)
        self._canvas.draw_idle()

    # -- export -----------------------------------------------------------

    def save_figure(self, path: str, dpi: int = 200) -> None:
        """Write the figure exactly as drawn on screen."""
        self._figure.savefig(
            path, dpi=dpi, facecolor=self._figure.get_facecolor()
        )

    @property
    def figure(self) -> Figure:
        return self._figure


__all__ = ["MplChart"]
