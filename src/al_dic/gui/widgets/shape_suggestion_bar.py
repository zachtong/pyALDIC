"""The post-run suggestion for subset size and shape function, with Apply.

A Local DIC run may end with a suggestion (``PipelineResult.shape_advice``).
This bar shows it under the run controls and offers one button that writes
the two suggested values into the parameters. It does not start a run.

The bar is derived from the state, with nothing stored of its own: it is
visible while the results carry a suggestion and the parameters are still
the ones that run used. Applying the suggestion, or changing the subset
size, subset step, shape function or solver by hand, therefore hides it;
setting the parameters back shows it again.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from al_dic.gui.app_state import AppState, RunState
from al_dic.gui.theme_manager import themed
from al_dic.i18n import tr_args

_BAR_STYLE = (
    "QFrame#shapeSuggestionBar {{ "
    "background: {BG_INPUT}; "
    "border: 1px solid {BORDER}; "
    "border-left: 3px solid {ACCENT}; "
    "border-radius: 3px; "
    "}}"
)
_TEXT_STYLE = "color: {TEXT_SECONDARY}; font-size: 11px; border: none; background: transparent;"


class ShapeSuggestionBar(QFrame):
    """Suggested subset size and shape function, with an Apply button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        self.setObjectName("shapeSuggestionBar")
        themed(self, _BAR_STYLE)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        self._label = QLabel()
        self._label.setWordWrap(True)
        themed(self._label, _TEXT_STYLE)
        layout.addWidget(self._label)

        row = QHBoxLayout()
        row.addStretch()
        self._apply_btn = QPushButton()
        self._apply_btn.clicked.connect(self._on_apply)
        row.addWidget(self._apply_btn)
        layout.addLayout(row)

        self._state.results_changed.connect(self._refresh)
        self._state.params_changed.connect(self._refresh)
        self._state.run_state_changed.connect(self._refresh)
        self._refresh()

    # ------------------------------------------------------------------

    def pending_advice(self):
        """The suggestion that applies to the current parameters, or None."""
        s = self._state
        results = getattr(s, "results", None)
        advice = getattr(results, "shape_advice", None)
        if advice is None or advice.status != "suggest":
            return None
        if s.use_admm or s.run_state == RunState.RUNNING:
            return None
        # Only while the parameters are those of the run it was made from.
        run_step = getattr(getattr(results, "dic_para", None), "winstepsize", s.subset_step)
        if (s.subset_size != advice.current_winsize
                or s.shape_function_order != advice.current_order
                or s.subset_step != run_step):
            return None
        return advice

    def _order_name(self, order: str) -> str:
        return self.tr("2nd order") if order == "second" else self.tr("1st order")

    def _refresh(self, *_args) -> None:
        advice = self.pending_advice()
        self._apply_btn.setText(self.tr("Apply"))
        self._apply_btn.setToolTip(self.tr(
            "Set the subset size and the shape function to the suggested "
            "values. Nothing is recomputed until you run again."
        ))
        if advice is None:
            self.setVisible(False)
            return
        # The GUI shows the subset as winsize + 1; the gain is a prediction,
        # so it is rounded to the nearest 5 %.
        percent = int(5 * round(100.0 * advice.predicted_gain / 5.0))
        self._label.setText(tr_args(
            self.tr("Suggested: subset size %1 px, %2 shape function "
                    "(predicted to lower the displacement error by about %3 %)."),
            advice.suggested_winsize + 1, self._order_name(advice.suggested_order), percent,
        ))
        self.setVisible(True)

    def _on_apply(self) -> None:
        advice = self.pending_advice()
        if advice is None:
            return
        size, order = advice.suggested_winsize, advice.suggested_order
        self._state.set_param("subset_size", size)
        self._state.set_param("shape_function_order", order)
        self._state.log_message.emit(tr_args(
            self.tr("Suggested setting applied: subset size %1 px, %2 shape "
                    "function. Run again to use it."),
            size + 1, self._order_name(order),
        ), "info")

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.LanguageChange:
            self._refresh()
        super().changeEvent(event)
