"""The Analysis tab's words: titles, notes, instructions and messages.

Every string is translated in the one catalog context "AnalysisTab", whichever
class shows it, so splitting the tab into parts moved no translation. Calls
are spelled ``QCoreApplication.translate("AnalysisTab", ...)`` in full each
time: lupdate reads the literal context and cannot follow an alias -- nor a
``tr()`` inside an f-string.
"""

from __future__ import annotations

import html

import numpy as np
from PySide6.QtCore import QCoreApplication

from al_dic.analysis.engine import SampleStatus
from al_dic.analysis.probes import Probe
from al_dic.analysis.series import FrameStatus, TimeSeries
from al_dic.gui.panels.analysis.quantities import Quantity
from al_dic.gui.theme import COLORS
from al_dic.gui.widgets.mpl_chart import status_label
from al_dic.i18n import tr_args


def field_title(name: str) -> str:
    titles = {
        "disp_u": QCoreApplication.translate("AnalysisTab", "Displacement U"),
        "disp_v": QCoreApplication.translate("AnalysisTab", "Displacement V"),
        "disp_magnitude": QCoreApplication.translate(
            "AnalysisTab", "Displacement magnitude"),
        "strain_exx": "εxx",
        "strain_eyy": "εyy",
        "strain_exy": "εxy",
        "strain_principal_max": "ε₁",
        "strain_principal_min": "ε₂",
        "strain_maxshear": "γ max",
        "strain_von_mises": "von Mises",
        "strain_rotation": "ω rot",
    }
    return titles.get(name, name)


def gauge_title(name: str) -> str:
    titles = {
        "strain": QCoreApplication.translate("AnalysisTab", "Extensometer strain"),
        "true_strain": QCoreApplication.translate(
            "AnalysisTab", "Extensometer true strain"),
        "elongation": QCoreApplication.translate("AnalysisTab", "Elongation ΔL"),
        "cod": QCoreApplication.translate("AnalysisTab", "Crack opening"),
        "cod_sliding": QCoreApplication.translate("AnalysisTab", "Crack sliding"),
        "cod_magnitude": QCoreApplication.translate(
            "AnalysisTab", "Crack opening magnitude"),
    }
    return titles.get(name, name)


def statistic_title(name: str) -> str:
    titles = {
        "mean": QCoreApplication.translate(
            "AnalysisTab", "Mean", "Statistic: arithmetic mean"),
        "median": QCoreApplication.translate("AnalysisTab", "Median", "Statistic"),
        "max": QCoreApplication.translate("AnalysisTab", "Maximum", "Statistic"),
        "min": QCoreApplication.translate("AnalysisTab", "Minimum", "Statistic"),
        "std": QCoreApplication.translate("AnalysisTab", "Standard deviation"),
        "valid_fraction": QCoreApplication.translate("AnalysisTab", "Valid fraction"),
    }
    return titles.get(name, name)


def y_title(quantity: Quantity, statistic: str, unit: str, points_only: bool) -> str:
    if quantity.is_gauge:
        text = gauge_title(quantity.name)
    elif points_only and statistic in ("mean", "median", "max", "min"):
        # A point has one value; calling it a mean would be noise.
        text = field_title(quantity.name)
    else:
        text = f"{field_title(quantity.name)} — {statistic_title(statistic)}"
    return f"{text} ({unit})" if unit else text


# -- placement ------------------------------------------------------------------

def tool_name(tool: str) -> str:
    names = {
        "point": QCoreApplication.translate(
            "AnalysisTab", "Point", "Placement tool: a single location"),
        "line": QCoreApplication.translate(
            "AnalysisTab", "Line", "Placement tool: a two-point gauge"),
        "area_rect": QCoreApplication.translate(
            "AnalysisTab", "Rectangle", "Placement tool"),
        "area_circle": QCoreApplication.translate(
            "AnalysisTab", "Circle", "Placement tool"),
        "area_polygon": QCoreApplication.translate(
            "AnalysisTab", "Polygon", "Placement tool"),
        "extensometer": QCoreApplication.translate(
            "AnalysisTab", "Virtual extensometer", "Placement tool"),
        "crack_gauge": QCoreApplication.translate(
            "AnalysisTab", "Crack gauge", "Placement tool: a line across a crack"),
    }
    return names.get(tool, tool)


def tool_instruction(tool: str) -> str:
    instructions = {
        "point": QCoreApplication.translate(
            "AnalysisTab", "Click once to place a point probe."),
        "line": QCoreApplication.translate(
            "AnalysisTab",
            "Click twice: start and end. A line is also a virtual "
            "extensometer and a crack-opening gauge."),
        "area_rect": QCoreApplication.translate(
            "AnalysisTab", "Click twice: opposite corners."),
        "area_circle": QCoreApplication.translate(
            "AnalysisTab", "Click twice: centre, then the edge."),
        "area_polygon": QCoreApplication.translate(
            "AnalysisTab", "Click each vertex, then double-click to close."),
        "extensometer": QCoreApplication.translate(
            "AnalysisTab",
            "Click the two gauge points. The chart then shows the strain "
            "between them."),
        "crack_gauge": QCoreApplication.translate(
            "AnalysisTab",
            "Click one point on each side of the crack. The chart then "
            "shows how far it opens."),
    }
    return instructions.get(tool, "")


def banner_html(tool: str, has_selection: bool) -> str:
    """What the canvas is waiting for: the armed tool's clicks, or how to
    edit the selected probe. Empty when there is nothing to say."""
    if tool != "none":
        cancel = QCoreApplication.translate("AnalysisTab", "Esc cancels placement")
        # Three sentences, a line each; nothing is spliced into another.
        return "<br>".join((
            f"<b>{html.escape(tool_name(tool))}</b>",
            html.escape(tool_instruction(tool)),
            f"<span style='color:{COLORS.TEXT_MUTED}'>{html.escape(cancel)}</span>",
        ))
    if has_selection:
        return html.escape(QCoreApplication.translate(
            "AnalysisTab",
            "Drag to move the probe, or drag a handle to reshape it. "
            "Delete removes it; F2 renames it."))
    return ""


# -- notes ---------------------------------------------------------------------

def series_note(ts: TimeSeries) -> str:
    """Why a curve has gaps or marks, for the legend and the probe list."""
    # Nothing to measure even on the reference frame: the probe was put off
    # the specimen. Say that, rather than blame the run.
    first = ts.status[0] if len(ts.status) else None
    if first is FrameStatus.NO_DATA:
        return QCoreApplication.translate(
            "AnalysisTab", "not plotted: off the measured area")
    if first is FrameStatus.ENDPOINT_LOST:
        return QCoreApplication.translate(
            "AnalysisTab", "not plotted: a gauge end is off the measured area")
    # Frame 0 is the reference: a displacement is 0 there by definition, so
    # it says nothing about whether the probe ever measures.
    values = ts.values[1:] if len(ts.values) > 1 else ts.values
    statuses = list(ts.status[1:]) if len(ts.status) > 1 else list(ts.status)
    if not np.isfinite(values).any():
        reasons = [st for st in statuses if st is not FrameStatus.OK]
        if reasons:
            common = max(set(reasons), key=reasons.count)
            return tr_args(QCoreApplication.translate(
                "AnalysisTab", "no valid data: %1"), status_label(common))
    crack = ts.first_frame(FrameStatus.CRACK)
    if crack is not None:
        return tr_args(QCoreApplication.translate(
            "AnalysisTab", "crack from frame %1"), crack + 1)
    lost = ts.first_frame(FrameStatus.ENDPOINT_LOST)
    if lost is not None:
        return tr_args(QCoreApplication.translate(
            "AnalysisTab", "endpoint lost from frame %1"), lost + 1)
    if any(st is FrameStatus.BELOW_THRESHOLD for st in ts.status):
        return QCoreApplication.translate("AnalysisTab", "gaps: too few valid points")
    if any(st is FrameStatus.UNRELIABLE for st in ts.status):
        return QCoreApplication.translate("AnalysisTab", "gaps: unreliable strain")
    return ""


def not_applicable_note(probe: Probe, quantity: Quantity) -> str:
    if quantity.is_gauge:
        return QCoreApplication.translate(
            "AnalysisTab", "not plotted: gauges need a line")
    if probe.kind == "point":
        return QCoreApplication.translate(
            "AnalysisTab", "not plotted: one point has no spread or coverage")
    return QCoreApplication.translate("AnalysisTab", "not plotted")


def empty_line_message(label: str, status: np.ndarray) -> str:
    """Why a line has nothing to show on any deformed frame.

    The reason named is the one the user can act on, not the commonest: a
    line over a hole and a trimmed ligament is mostly off the material, but
    only the trim can be changed.
    """
    if (status == SampleStatus.UNRELIABLE).any():
        text = QCoreApplication.translate(
            "AnalysisTab",
            "Nothing valid along %1: its strain is trimmed as low-confidence "
            "near an edge or a hole. Plot a displacement, or trim less on "
            "the Strain Field tab.")
    elif (status == SampleStatus.CONSUMED).any():
        text = QCoreApplication.translate(
            "AnalysisTab",
            "Nothing valid along %1: a crack has consumed the material "
            "under it.")
    else:
        text = QCoreApplication.translate(
            "AnalysisTab", "Nothing valid along %1: it lies off the measured area.")
    return tr_args(text, label)


__all__ = [
    "banner_html", "empty_line_message", "field_title", "gauge_title",
    "not_applicable_note", "series_note", "statistic_title", "tool_instruction",
    "tool_name", "y_title",
]
