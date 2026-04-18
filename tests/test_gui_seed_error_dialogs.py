"""Tests for P5.6 — tailored fatal_error dialogs per SeedPropagationError.

Exercises the static mapping in pipeline_controller._SEED_ERROR_MESSAGES.
The actual dispatch path (try/except routing) is verified by the
integration tests; here we assert that every concrete subclass has
a human-readable message and that messages follow a consistent shape.
"""
from __future__ import annotations

from al_dic.gui.controllers.pipeline_controller import _SEED_ERROR_MESSAGES
from al_dic.solver.seed_propagation import (
    MissingSeedForRegion,
    SeedICGNDiverged,
    SeedNCCBelowThreshold,
    SeedPropagationError,
    SeedQualityError,
    SeedWarpFailure,
)


class TestSeedErrorMessages:
    def test_all_concrete_subclasses_covered(self):
        """Every concrete SeedPropagationError subclass has a dialog entry."""
        concrete = {
            SeedNCCBelowThreshold,
            SeedICGNDiverged,
            SeedQualityError,
            SeedWarpFailure,
            MissingSeedForRegion,
        }
        missing = concrete - set(_SEED_ERROR_MESSAGES.keys())
        assert not missing, f"Missing dialog entries for: {missing}"

    def test_messages_have_title_and_suggested_actions(self):
        for exc_type, (title, suggestions) in _SEED_ERROR_MESSAGES.items():
            assert isinstance(title, str) and title, f"{exc_type}: empty title"
            assert "Suggested actions" in suggestions, (
                f"{exc_type}: message missing 'Suggested actions' section"
            )
            assert len(suggestions) > 100, (
                f"{exc_type}: message too short to be useful "
                f"({len(suggestions)} chars)"
            )

    def test_subclass_lookup_uses_exact_type(self):
        """A subclass instance resolves to its own entry, not base class."""
        instance = SeedNCCBelowThreshold("test")
        title, _ = _SEED_ERROR_MESSAGES[type(instance)]
        assert "match quality" in title.lower()

    def test_base_class_not_keyed(self):
        """The base SeedPropagationError is handled by a fallback in the
        except block (via dict.get default), not as a keyed entry.
        """
        assert SeedPropagationError not in _SEED_ERROR_MESSAGES
