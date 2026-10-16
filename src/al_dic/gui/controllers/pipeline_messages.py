"""Translated text the pipeline controller shows about a run.

``PipelineController`` is not a ``QObject``, and lupdate extracts a
counted message as a plural only from ``self.tr(source, "", n)`` on a
``QObject``: ``QCoreApplication.translate(context, source, "", n)`` comes
out as a plain message with one form, and German, French and Spanish then
show the English source for n >= 2. So the run's sentences live here, on
:class:`PipelineMessages` (translation context ``PipelineMessages``): the
console lines with a count now, and the run-time dialogs as they move in.

Pure: no widget, no signal; each method returns text and changes nothing.
"""

from __future__ import annotations

from PySide6.QtCore import QObject


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
