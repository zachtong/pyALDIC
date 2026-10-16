"""Translated dialogs for a run that Starting Points could not carry.

``PipelineWorker`` catches every ``SeedPropagationError`` and shows one
dialog: a title, what happened, the suggested actions, then the core's own
message as the details (English, as raised; the console has it too).
:data:`SEED_ERROR_TEXT` picks the sentences by the exception's exact type;
an unknown subclass gets the generic ones. The table holds methods, not
text: the sentences are looked up when the dialog is shown, in the
language then active.

Through ``run_aldic`` every seed failure of a frame arrives as a
``PartialResultError`` (the run stops and keeps the frames before it), so
that entry is the one users see; the core's message names the cause.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject

from al_dic.gui.controllers.pipeline_messages import (
    LABEL_BATCH_IMPORT,
    LABEL_FFT,
    LABEL_INITIAL_GUESS,
    LABEL_PARAMETERS,
    LABEL_PLACE_POINTS,
    LABEL_SUBSET_SIZE,
    gui_label,
)
from al_dic.i18n import tr_args
from al_dic.solver.seed_propagation import (
    MissingSeedForRegion,
    PartialResultError,
    SeedICGNDiverged,
    SeedNCCBelowThreshold,
    SeedPropagationError,
    SeedQualityError,
    SeedWarpFailure,
)

# Blank line between the dialog's explanation and its details.
_PARAGRAPH = "\n\n"


def _fft_in_initial_guess() -> tuple[str, str]:
    return gui_label(LABEL_FFT), gui_label(LABEL_INITIAL_GUESS)


class SeedErrorMessages(QObject):
    """The sentences, in the ``SeedErrorMessages`` translation context.

    Each body takes the exception (only ``PartialResultError`` reads it).
    Keep this class unsubclassed: ``self.tr`` looks up the runtime class
    name.
    """

    def details(self, detail: str) -> str:
        """The core's message, shown as it is."""
        return tr_args(self.tr("Details: %1"), detail)

    def ncc_title(self) -> str:
        return self.tr("Starting Point match quality too low")

    def ncc(self, _exc: SeedPropagationError) -> str:
        return tr_args(self.tr(
            "A Starting Point could not be matched reliably between the "
            "reference frame and the deformed frame: the peak of its "
            "normalized cross-correlation (NCC) stayed below the acceptance "
            "threshold. The texture around it is probably too uniform.\n\n"
            "Suggested actions:\n"
            "- Move the Starting Point to a more textured area (visible "
            "speckle).\n"
            "- Or choose “%1” in the %2 section."), *_fft_in_initial_guess())

    def icgn_title(self) -> str:
        return self.tr("Starting Point local solver did not converge")

    def icgn(self, _exc: SeedPropagationError) -> str:
        return tr_args(self.tr(
            "The IC-GN solve at a Starting Point reached its iteration limit "
            "without converging. Common causes: the point is too close to "
            "the edge of the Region of Interest, or the texture around it is "
            "too weak.\n\n"
            "Suggested actions:\n"
            "- Move the Starting Point inward, away from the edges of the "
            "Region of Interest.\n"
            "- Try a smaller %1 (%2 section).\n"
            "- Or choose “%3” in the %4 section."),
            gui_label(LABEL_SUBSET_SIZE), gui_label(LABEL_PARAMETERS),
            *_fft_in_initial_guess())

    def quality_title(self) -> str:
        return self.tr("Starting Point flagged as unreliable")

    def quality(self, _exc: SeedPropagationError) -> str:
        return tr_args(self.tr(
            "After the frame was solved, the displacement at a Starting "
            "Point was rejected as an outlier. The frame's field was "
            "propagated from the Starting Points, so the whole frame is "
            "unreliable.\n\n"
            "Suggested actions:\n"
            "- Move the Starting Point to a cleaner, well-textured area, away "
            "from the edges of the Region of Interest and from cracks.\n"
            "- Or choose “%1” in the %2 section."), *_fft_in_initial_guess())

    def warp_title(self) -> str:
        return self.tr(
            "Could not carry Starting Points to the new reference frame")

    def warp(self, _exc: SeedPropagationError) -> str:
        return tr_args(self.tr(
            "When the reference frame changed, the Starting Points could not "
            "be carried over to the new reference frame, and none could be "
            "placed there automatically. The material at the points has "
            "probably moved out of the Region of Interest of the new "
            "reference frame.\n\n"
            "Suggested actions:\n"
            "- Place Starting Points well inside the Region of Interest, away "
            "from its edges.\n"
            "- Give each reference frame a Region of Interest that covers the "
            "specimen in that frame (%1 can assign Regions of Interest to "
            "many frames at once)."), gui_label(LABEL_BATCH_IMPORT))

    def missing_title(self) -> str:
        return self.tr("Region without a Starting Point")

    def missing(self, _exc: SeedPropagationError) -> str:
        return tr_args(self.tr(
            "At least one region (a separate part of the Region of Interest) "
            "has no working Starting Point, even after placing one "
            "automatically was tried. The propagation cannot cross from one "
            "region to another, so every region needs its own Starting "
            "Point.\n\n"
            "Suggested actions:\n"
            "- Make sure every region shown in red has a Starting Point "
            "(“%1” in the %2 section).\n"
            "- Or remove small separate pieces from the Region of Interest, "
            "so that fewer regions need a Starting Point."),
            gui_label(LABEL_PLACE_POINTS), gui_label(LABEL_INITIAL_GUESS))

    def partial_title(self) -> str:
        return self.tr("Analysis stopped early")

    def partial(self, exc: SeedPropagationError) -> str:
        frame = getattr(exc, "stopped_at_frame", None)
        if frame is None:
            return self.other(exc)
        return tr_args(self.tr(
            "Frame %1 could not be computed from the Starting Points, so the "
            "run stopped there; the frames completed before it, if any, are "
            "kept and shown. The details below name the cause.\n\n"
            "Suggested actions:\n"
            "- Place Starting Points on well-textured speckle, away from the "
            "edges of the Region of Interest and from cracks.\n"
            "- Remove small separate pieces from the Regions of Interest: "
            "every piece needs a Starting Point, and none can be placed "
            "automatically on a piece without usable speckle.\n"
            "- Or choose “%2” in the %3 section."),
            frame, *_fft_in_initial_guess())

    def other_title(self) -> str:
        return self.tr("Starting Points error")

    def other(self, _exc: SeedPropagationError) -> str:
        return self.tr(
            "Propagation from the Starting Points failed.\n\n"
            "Suggested actions:\n"
            "- Check the positions of the Starting Points.\n"
            "- See the console log for details.")


_Title = Callable[[SeedErrorMessages], str]
_Body = Callable[[SeedErrorMessages, SeedPropagationError], str]

#: Exception type -> (title, body), matched exactly; others get _OTHER.
SEED_ERROR_TEXT: dict[type[SeedPropagationError], tuple[_Title, _Body]] = {
    SeedNCCBelowThreshold: (SeedErrorMessages.ncc_title, SeedErrorMessages.ncc),
    SeedICGNDiverged: (SeedErrorMessages.icgn_title, SeedErrorMessages.icgn),
    SeedQualityError: (SeedErrorMessages.quality_title,
                       SeedErrorMessages.quality),
    SeedWarpFailure: (SeedErrorMessages.warp_title, SeedErrorMessages.warp),
    MissingSeedForRegion: (SeedErrorMessages.missing_title,
                           SeedErrorMessages.missing),
    PartialResultError: (SeedErrorMessages.partial_title,
                         SeedErrorMessages.partial),
}
_OTHER: tuple[_Title, _Body] = (SeedErrorMessages.other_title,
                                SeedErrorMessages.other)


def seed_error_dialog(exc: SeedPropagationError) -> tuple[str, str]:
    """``(title, text)`` of the dialog for ``exc``, in the active language."""
    m = SeedErrorMessages()
    title, body = SEED_ERROR_TEXT.get(type(exc), _OTHER)
    text = _PARAGRAPH.join((body(m, exc), m.details(str(exc))))
    return title(m), text
