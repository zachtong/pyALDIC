"""Tests for AppState refinement-level math.

Covers:
  * compute_max_refinement_level() — combined integer + physical constraint
  * compute_refinement_min_size() — floor at 2 (qrefine_r mathematical lower
    bound)
  * set_refinement_level() — automatic clamp to current max
"""

from __future__ import annotations

import warnings

import pytest
from PySide6.QtWidgets import QApplication

from al_dic.gui.app_state import AppState


def _safe_disconnect(signal) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            signal.disconnect()
        except (RuntimeError, TypeError):
            pass


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _reset_singleton():
    state = AppState.instance()
    _safe_disconnect(state.params_changed)
    state.reset()
    yield


class TestMinSize:
    """Floor must be 2 px (true qrefine_r lower bound)."""

    @pytest.mark.parametrize(
        "step, level, expected",
        [
            # subset_step=4 — anything beyond L1 is clamped to floor=2
            (4, 1, 2),
            (4, 2, 2),
            # subset_step=8
            (8, 1, 4),
            (8, 2, 2),
            (8, 3, 2),
            # subset_step=16
            (16, 1, 8),
            (16, 2, 4),
            (16, 3, 2),
            (16, 4, 2),
            # subset_step=32
            (32, 1, 16),
            (32, 2, 8),
            (32, 3, 4),
            (32, 4, 2),
            # subset_step=64
            (64, 1, 32),
            (64, 2, 16),
            (64, 3, 8),
            (64, 4, 4),
            (64, 5, 2),
        ],
    )
    def test_compute_refinement_min_size(self, qapp, step, level, expected):
        state = AppState.instance()
        state.subset_step = step
        state.refinement_level = level
        assert state.compute_refinement_min_size() == expected


class TestMaxLevel:
    """max_level = min(integer constraint, physical constraint).

    integer  : floor(log2(subset_step / 2))
    physical : floor(log2(subset_size / 4))
    where subset_size is the *internal* even value (display = internal + 1).
    """

    @pytest.mark.parametrize(
        "subset_size_internal, subset_step, expected_max",
        [
            # ---- subset_size = 20 (display 21) ----
            # physical = floor(log2(5)) = 2
            (20, 4, 1),     # int=1 wins
            (20, 8, 2),     # int=2 wins
            (20, 16, 2),    # phys=2 wins
            (20, 32, 2),    # phys=2 wins
            (20, 64, 2),    # phys=2 wins
            # ---- subset_size = 40 (display 41) ----
            # physical = floor(log2(10)) = 3
            (40, 4, 1),
            (40, 8, 2),
            (40, 16, 3),
            (40, 32, 3),
            (40, 64, 3),
            # ---- subset_size = 80 (display 81) ----
            # physical = floor(log2(20)) = 4
            (80, 16, 3),    # int wins
            (80, 32, 4),    # equal
            (80, 64, 4),    # phys wins
            # ---- subset_size = 160 (display 161) ----
            # physical = floor(log2(40)) = 5
            (160, 32, 4),   # int wins
            (160, 64, 5),   # equal
            # ---- tiny subset_size (display 11) ----
            # physical = floor(log2(2.5)) = 1
            (10, 16, 1),
            (10, 64, 1),
        ],
    )
    def test_compute_max_refinement_level(
        self, qapp, subset_size_internal, subset_step, expected_max
    ):
        state = AppState.instance()
        state.subset_size = subset_size_internal
        state.subset_step = subset_step
        assert state.compute_max_refinement_level() == expected_max

    def test_max_level_always_at_least_one(self, qapp):
        state = AppState.instance()
        # Pathological: tiny subset_size, smallest step
        state.subset_size = 4
        state.subset_step = 4
        assert state.compute_max_refinement_level() >= 1


class TestLevelClamping:
    """set_refinement_level should clamp to current max_level."""

    def test_clamp_above_max(self, qapp):
        state = AppState.instance()
        state.subset_size = 20  # max_level = 2 with step=16
        state.subset_step = 16
        state.set_refinement_level(99)
        assert state.refinement_level == 2

    def test_clamp_below_one(self, qapp):
        state = AppState.instance()
        state.subset_size = 80
        state.subset_step = 32
        state.set_refinement_level(0)
        assert state.refinement_level == 1
        state.set_refinement_level(-5)
        assert state.refinement_level == 1

    def test_valid_value_passes_through(self, qapp):
        state = AppState.instance()
        state.subset_size = 80   # max=4 at step=32
        state.subset_step = 32
        state.set_refinement_level(3)
        assert state.refinement_level == 3

    def test_clamp_emits_params_changed(self, qapp):
        state = AppState.instance()
        state.subset_size = 20
        state.subset_step = 16
        received = []
        state.params_changed.connect(lambda: received.append(True))
        state.set_refinement_level(99)
        assert len(received) == 1
        assert state.refinement_level == 2


class TestSubsetSizeShrinkClampScenario:
    """When subset_size shrinks, an existing high refinement_level becomes
    invalid. Subsequent set_refinement_level calls must clamp."""

    def test_shrinking_subset_size_invalidates_level(self, qapp):
        state = AppState.instance()
        # Start with large subset → L4 valid
        state.subset_size = 80
        state.subset_step = 32
        state.set_refinement_level(4)
        assert state.refinement_level == 4

        # Now user shrinks subset_size — old level becomes invalid
        state.subset_size = 20  # max_level drops to 2
        # Re-applying same level via setter must clamp
        state.set_refinement_level(state.refinement_level)
        assert state.refinement_level == 2
