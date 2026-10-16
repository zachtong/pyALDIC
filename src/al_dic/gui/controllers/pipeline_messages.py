"""Translated text the pipeline controller shows about a run.

``PipelineController`` is not a ``QObject``, and lupdate extracts a
counted message as a plural only from ``self.tr(source, "", n)`` on a
``QObject``: ``QCoreApplication.translate(context, source, "", n)`` comes
out as a plain message with one form, and German, French and Spanish then
show the English source for n >= 2. So the run's sentences live here, on
:class:`PipelineMessages` (translation context ``PipelineMessages``): the
console lines with a count, and the confirmation asked at *Run* when some
reference frames have no Region of Interest of their own.

Pure: no widget, no signal; each function returns text and changes
nothing. Frame numbers are shown 1-based, like everywhere else in the GUI.
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QObject

from al_dic.i18n import tr_args

# Frames listed by number in the confirmation; the rest are counted.
MAX_LISTED_FRAMES = 10
# Rich-text paragraph break between the confirmation's sentences.
_PARAGRAPH = "<br><br>"
# Separator of the listed frame numbers (data, not a sentence).
_LIST_SEPARATOR = ", "


class PipelineMessages(QObject):
    """The sentences, in the ``PipelineMessages`` translation context.

    Pass every count positionally as a plain name, ``self.tr(source, "",
    n)``; lupdate does not extract other spellings as plurals
    (``tests/test_i18n_plural_calls.py``). Keep this class unsubclassed:
    ``self.tr`` looks up the runtime class name, and a subclass would look
    in a context that holds no translations.
    """

    def custom_roi_frames(self, n: int) -> str:
        """Console line: ``n`` frames besides frame 1 have their own mask."""
        return self.tr("  %n frame(s) with custom ROI masks", "", n)

    def results_received(self, n: int) -> str:
        """Console line: the run returned results for ``n`` frames."""
        return self.tr("Results received: %n frame(s)", "", n)

    # -- Run-time confirmation: reference frames without their own region --

    def missing_roi_title(self) -> str:
        return self.tr("Incomplete Region of Interest Coverage")

    def missing_roi_frames(self, n: int) -> str:
        """``n`` reference frames have no region; ``%1`` lists them."""
        return self.tr(
            "This incremental run has %n reference frame(s) with no Region "
            "of Interest of their own: frame(s) %1.", "", n)

    def frames_not_listed(self, n: int) -> str:
        """A frame list cut short: ``%1`` the frames listed, ``n`` the rest."""
        return self.tr(
            "%1 and %n more",
            "List of frame numbers cut short: %1 = the frame numbers shown, "
            "%n = how many more frames there are", n)

    def missing_roi_effect(self) -> str:
        return self.tr(
            "A reference frame without a Region of Interest of its own "
            "reuses frame 1's region unchanged: nothing moves the region "
            "with the specimen. Where the specimen has moved out of that "
            "region, the part outside it is not measured from such a frame, "
            "and the nodes there stop being tracked or have their "
            "displacement copied from the nearest measured node. The run "
            "reports these nodes when it finishes.")

    def missing_roi_advice(self) -> str:
        return self.tr(
            "To measure the whole specimen, give each reference frame its "
            "own Region of Interest that covers the specimen in that frame "
            "(Batch Import can assign Regions of Interest to many frames at "
            "once). A reference frame with a region of its own is measured "
            "on a grid that covers that region.")

    def missing_roi_question(self) -> str:
        return self.tr("Continue with frame 1's region?")


def missing_roi_dialog(missing: Iterable[int]) -> tuple[str, str]:
    """``(title, text)`` of the confirmation asked at *Run* in incremental
    mode when the reference frames ``missing`` (0-based) have no Region of
    Interest of their own. ``text`` is rich text: one paragraph per
    sentence group, the closing question in bold."""
    m = PipelineMessages()
    frames = [f + 1 for f in sorted(missing)]
    n_missing = len(frames)
    text = _PARAGRAPH.join((
        tr_args(m.missing_roi_frames(n_missing), _frame_list(m, frames)),
        m.missing_roi_effect(),
        m.missing_roi_advice(),
        f"<b>{m.missing_roi_question()}</b>",
    ))
    return m.missing_roi_title(), text


def _frame_list(m: PipelineMessages, frames: list[int]) -> str:
    """The first ``MAX_LISTED_FRAMES`` frame numbers, then how many more."""
    listed = _LIST_SEPARATOR.join(str(f) for f in frames[:MAX_LISTED_FRAMES])
    n_more = len(frames) - MAX_LISTED_FRAMES
    if n_more <= 0:
        return listed
    return tr_args(m.frames_not_listed(n_more), listed)
