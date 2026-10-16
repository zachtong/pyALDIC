"""Tests for P5.6 — tailored fatal_error dialogs per SeedPropagationError.

Exercises the table ``seed_error_messages.SEED_ERROR_TEXT`` and
``seed_error_dialog``. The table holds methods, not text, so the sentences
are looked up in the language active when the dialog is shown; the real
dispatch path (``PipelineWorker.run`` catching each type) and the
translations are covered by ``tests/test_gui/test_pipeline_messages_i18n.py``.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from al_dic.gui.controllers.seed_error_messages import (
    SEED_ERROR_TEXT,
    SeedErrorMessages,
    seed_error_dialog,
)
from al_dic.solver.seed_propagation import (
    MissingSeedForRegion,
    PartialResultError,
    SeedICGNDiverged,
    SeedNCCBelowThreshold,
    SeedPropagationError,
    SeedQualityError,
    SeedWarpFailure,
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _instance(exc_type: type) -> SeedPropagationError:
    if exc_type is PartialResultError:
        return PartialResultError("core text", stopped_at_frame=4)
    return exc_type("core text")


class TestSeedErrorMessages:
    def test_all_concrete_subclasses_covered(self):
        """Every concrete SeedPropagationError subclass has a dialog entry."""
        concrete = {
            SeedNCCBelowThreshold,
            SeedICGNDiverged,
            SeedQualityError,
            SeedWarpFailure,
            MissingSeedForRegion,
            PartialResultError,
        }
        missing = concrete - set(SEED_ERROR_TEXT.keys())
        assert not missing, f"Missing dialog entries for: {missing}"

    def test_messages_have_title_and_suggested_actions(self):
        m = SeedErrorMessages()
        for exc_type, (title, body) in SEED_ERROR_TEXT.items():
            text = body(m, _instance(exc_type))
            assert title(m), f"{exc_type}: empty title"
            assert "Suggested actions" in text, (
                f"{exc_type}: message missing 'Suggested actions' section"
            )
            assert len(text) > 100, (
                f"{exc_type}: message too short to be useful "
                f"({len(text)} chars)"
            )

    def test_subclass_lookup_uses_exact_type(self):
        """A subclass instance resolves to its own entry, not base class."""
        title, _ = seed_error_dialog(SeedNCCBelowThreshold("test"))
        assert "match quality" in title.lower()

    def test_base_class_not_keyed(self):
        """The base SeedPropagationError is handled by a fallback, not as a
        keyed entry; the dialog still has a title and suggested actions."""
        assert SeedPropagationError not in SEED_ERROR_TEXT
        title, text = seed_error_dialog(SeedPropagationError("core text"))
        assert title and "Suggested actions" in text

    def test_dialog_ends_with_the_core_message(self):
        """The core's own message is the dialog's details, as raised."""
        _, text = seed_error_dialog(MissingSeedForRegion("2 region(s) left"))
        assert text.endswith("Details: 2 region(s) left")

    def test_partial_result_names_the_frame(self):
        _, text = seed_error_dialog(
            PartialResultError("core text", stopped_at_frame=12))
        assert "Frame 12 " in text
