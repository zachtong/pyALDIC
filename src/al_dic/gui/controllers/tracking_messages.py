"""Translated text of a run's tracking-quality findings (plan D18/D19).

The core assesses ``TrackingDiagnostics`` into findings with English keys
(``al_dic.core.tracking_diagnostics.assess_tracking``); this module turns
them into what the user reads: console lines for every finding, and the
parts of the warning dialog shown for WARNING findings.

Pure: no widget, no signal, inputs never modified. The sentences live on
:class:`TrackingQuality` (translation context ``TrackingQuality``); other
values fill ``%1``... through ``tr_args``, numbers go through
``format_number``, and frames are shown 1-based like everywhere else in
the GUI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PySide6.QtCore import QObject

from al_dic.core._grid_extent import grid_node_count
from al_dic.core.data_structures import DICPara, TrackingDiagnostics
from al_dic.core.tracking_diagnostics import (
    ADVICE_REFERENCE_ROI,
    FINDING_GRID_MISS,
    FINDING_LEFT_MEASURABLE_AREA,
    FINDING_MATERIAL_LOSS,
    LOSS_RULE_BURST,
    SEVERITY_WARNING,
    TrackingFinding,
    assess_tracking,
    tracking_advice,
)
from al_dic.i18n import tr_args
from al_dic.utils.locale_format import format_number

logger = logging.getLogger(__name__)

# Console levels (``AppState.log_message``).
LEVEL_WARN = "warn"
LEVEL_INFO = "info"

# Decimals of the percentages shown.
_PCT_DECIMALS = 1


class TrackingQuality(QObject):
    """The sentences, in the ``TrackingQuality`` translation context.

    A ``QObject`` for ``self.tr(source, "", n)``: lupdate marks that form
    numerus, so the catalogs carry every plural form. It does not mark
    ``QCoreApplication.translate(context, source, "", n)``, and Qt then
    finds no translation for the plural forms of de/fr/es (English shows
    for n >= 2). Keep this class unsubclassed: ``self.tr`` looks up the
    runtime class name.
    """

    def title(self) -> str:
        return self.tr("Tracking Quality Warning")

    def summary(self) -> str:
        return self.tr(
            "Some nodes could not be tracked reliably through the image "
            "sequence.")

    def grid_miss(self, n: int) -> str:
        return self.tr(
            "In frame %1, %n node(s) were more than %2 px from the nearest "
            "node measured on reference frame %3. Their displacement was "
            "copied from that node and may be wrong.", "", n)

    def loss(self, n: int) -> str:
        return self.tr(
            "%n node(s) (%1% of the tracked material) stopped being tracked "
            "by frame %2 because they landed on pixels outside the Region of "
            "Interest. Their displacement is not measured from that frame on "
            "(NaN in exported data).", "", n)

    def burst(self, n: int) -> str:
        return self.tr(
            "In frame %1 alone, %n node(s) (%2% of the tracked material) "
            "stopped being tracked because they landed on pixels outside the "
            "Region of Interest. Their displacement is not measured from that "
            "frame on (NaN in exported data).", "", n)

    def edge(self, n: int) -> str:
        return self.tr(
            "%n node(s) moved too close to the image edge to be measured. "
            "Their displacement there was copied from the nearest measured "
            "node.", "", n)

    def grid_note(self, n: int) -> str:
        return self.tr(
            "Measurement grid enlarged for %n reference frame(s) to cover "
            "their Regions of Interest (largest grid: %1 nodes on frame %2; "
            "frame 1 grid: %3 nodes).", "", n)

    def advice_reference_roi(self) -> str:
        return self.tr(
            "Suggested actions:\n"
            "- Give every reference frame its own Region of Interest that "
            "covers the specimen in that frame (Batch Import can assign "
            "Regions of Interest to many frames at once).\n"
            "- Keep the specimen inside the image for the whole sequence.\n"
            "- If the specimen cracks or has holes, nodes lost along them "
            "are expected.")

    def advice_frame_roi(self) -> str:
        return self.tr(
            "Suggested actions:\n"
            "- Check that each frame's Region of Interest covers the specimen "
            "in that frame.\n"
            "- Keep the specimen inside the image for the whole sequence.\n"
            "- If the specimen cracks or has holes, nodes lost along them are "
            "expected.")


@dataclass(frozen=True)
class TrackingMessages:
    """What the GUI shows about one run's tracking quality.

    Attributes:
        title: Dialog title.
        text: Dialog main text (what happened, in one sentence).
        informative: Dialog text below it: the WARNING findings, then the
            suggested actions.
        details: Every finding, and the grid note, one paragraph each
            (the dialog's "Show Details").
        log_lines: ``(message, level)`` console lines, in order.
    """

    title: str
    text: str
    informative: str
    details: str
    log_lines: tuple[tuple[str, str], ...]

    @property
    def has_warning(self) -> bool:
        """Some finding is a WARNING (the dialog is worth showing)."""
        return any(level == LEVEL_WARN for _text, level in self.log_lines)

    @property
    def dialog(self) -> tuple[str, str, str, str]:
        """``(title, text, informative, details)``, as ``AppState.run_warning``
        carries them."""
        return (self.title, self.text, self.informative, self.details)


def build_tracking_messages(
    diag: TrackingDiagnostics | None, para: DICPara,
) -> TrackingMessages | None:
    """Messages for ``diag`` of a run with parameters ``para`` (its
    ``PipelineResult.dic_para``); ``None`` when there is nothing to report.

    ``None`` diagnostics (unknown) report nothing; so does a run with no
    finding and no reference frame on a grid of its own.
    """
    if diag is None:
        return None
    t = TrackingQuality()
    found = _found(t, diag)
    grid_note = _grid_note(t, diag, para)
    if not found and grid_note is None:
        return None
    notes = [(grid_note, LEVEL_INFO)] if grid_note is not None else []
    advice = _advice(t, diag)
    warnings = [text for text, level in found if level == LEVEL_WARN]
    log_lines = notes + found + ([(advice, LEVEL_INFO)] if warnings else [])
    return TrackingMessages(
        title=t.title(),
        text=t.summary(),
        informative="\n\n".join([*warnings, advice]),
        details="\n\n".join(text for text, _level in notes + found),
        log_lines=tuple(log_lines),
    )


def _found(
    t: TrackingQuality, diag: TrackingDiagnostics,
) -> list[tuple[str, str]]:
    """``(text, level)`` of every finding, in assessment order.

    A finding key with no sentence here (a newer core) is logged and left
    out rather than shown as some other finding.
    """
    found = []
    for f in assess_tracking(diag):
        text = _finding_text(t, diag, f)
        if text is None:
            logger.warning(
                "Tracking finding %r has no GUI text; not shown.", f.key)
            continue
        found.append((text, _level(f)))
    return found


def _level(finding: TrackingFinding) -> str:
    if finding.severity == SEVERITY_WARNING:
        return LEVEL_WARN
    return LEVEL_INFO


def _pct(fraction: float) -> str:
    return format_number(100.0 * fraction, _PCT_DECIMALS)


def _finding_text(
    t: TrackingQuality, diag: TrackingDiagnostics, f: TrackingFinding,
) -> str | None:
    if f.key == FINDING_GRID_MISS:
        far_px = diag.far_extrapolation_px or 0.0
        return tr_args(
            t.grid_miss(f.count), f.frame + 1, format_number(far_px),
            f.ref_frame + 1,
        )
    if f.key == FINDING_MATERIAL_LOSS:
        if f.rule == LOSS_RULE_BURST:
            n_material = diag.n_material
            burst = f.burst_count / n_material if n_material > 0 else 0.0
            return tr_args(t.burst(f.burst_count), f.frame + 1, _pct(burst))
        return tr_args(t.loss(f.count), _pct(f.fraction), f.frame + 1)
    if f.key == FINDING_LEFT_MEASURABLE_AREA:
        return t.edge(f.count)
    return None


def _grid_note(
    t: TrackingQuality, diag: TrackingDiagnostics, para: DICPara,
) -> str | None:
    """The reference frames solved on a grid of their own, if any.

    Compares lattices with lattices: the reference grids are counted as
    regular lattices (``RefGridExtent.n_nodes``), so frame 1's grid is its
    lattice too, not the frame-0 mesh (``diag.n_nodes`` also counts its
    refinement nodes). The core's grid-plan log counts the same way.
    """
    extents = diag.grid_extents
    if not extents:
        return None
    largest = max(extents, key=lambda e: e.n_nodes)
    n_frame0 = grid_node_count(
        para.gridxy_roi_range, tuple(para.img_size), para.winsize,
        para.winstepsize,
    )
    return tr_args(
        t.grid_note(len(extents)), format_number(largest.n_nodes),
        largest.ref_frame + 1, format_number(n_frame0),
    )


def _advice(t: TrackingQuality, diag: TrackingDiagnostics) -> str:
    if tracking_advice(diag) == ADVICE_REFERENCE_ROI:
        return t.advice_reference_roi()
    return t.advice_frame_roi()
