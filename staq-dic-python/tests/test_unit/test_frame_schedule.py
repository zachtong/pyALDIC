"""Unit tests for FrameSchedule data structure.

Tests cover:
    - Valid creation and DAG constraint enforcement
    - from_mode factory for accumulative and incremental
    - parent(), path_to_root(), children() tree traversal
    - Edge cases: single pair, long chains, branching trees
"""

from __future__ import annotations

import numpy as np
import pytest

from staq_dic.core.data_structures import FrameSchedule


# ---------------------------------------------------------------------------
# Creation and validation
# ---------------------------------------------------------------------------


class TestFrameScheduleCreation:
    """Test FrameSchedule creation and DAG constraint validation."""

    def test_valid_accumulative(self):
        """All frames reference frame 0."""
        s = FrameSchedule(ref_indices=(0, 0, 0, 0))
        assert len(s) == 4
        assert s.ref_indices == (0, 0, 0, 0)

    def test_valid_incremental(self):
        """Each frame references the previous frame."""
        s = FrameSchedule(ref_indices=(0, 1, 2, 3))
        assert len(s) == 4

    def test_valid_mixed(self):
        """Mix of direct and chained references."""
        s = FrameSchedule(ref_indices=(0, 1, 0))
        assert s.ref_indices == (0, 1, 0)

    def test_valid_skip_frame(self):
        """Key-frame every 3: frames 1-3 ref 0, frames 4-6 ref 3."""
        s = FrameSchedule(ref_indices=(0, 0, 0, 3, 3, 3))
        assert len(s) == 6

    def test_valid_single_pair(self):
        """Minimal case: one deformed frame."""
        s = FrameSchedule(ref_indices=(0,))
        assert len(s) == 1

    def test_invalid_negative_ref(self):
        """Negative reference index should fail."""
        with pytest.raises(ValueError, match="negative"):
            FrameSchedule(ref_indices=(0, -1))

    def test_invalid_future_ref(self):
        """Referencing a future frame should fail."""
        with pytest.raises(ValueError, match="future frame"):
            FrameSchedule(ref_indices=(0, 2))

    def test_invalid_self_ref(self):
        """Self-referencing (ref_indices[0] = 1) should fail.

        ref_indices[0] is for deformed frame 1; its ref must be in [0, 0],
        i.e., only frame 0 is allowed.
        """
        with pytest.raises(ValueError, match="future frame"):
            FrameSchedule(ref_indices=(1,))

    def test_invalid_type(self):
        """Non-integer reference index should fail."""
        with pytest.raises(TypeError, match="must be int"):
            FrameSchedule(ref_indices=(0, 1.5))  # type: ignore[arg-type]

    def test_numpy_int_accepted(self):
        """numpy integer types should be accepted."""
        s = FrameSchedule(ref_indices=(np.int64(0), np.int32(1)))
        assert s.parent(1) == 0

    def test_empty_schedule(self):
        """Empty schedule (no deformed frames) should be valid."""
        s = FrameSchedule(ref_indices=())
        assert len(s) == 0


# ---------------------------------------------------------------------------
# Factory: from_mode
# ---------------------------------------------------------------------------


class TestFrameScheduleFromMode:
    """Test from_mode factory method."""

    def test_accumulative_3_frames(self):
        s = FrameSchedule.from_mode("accumulative", 3)
        assert s.ref_indices == (0, 0)

    def test_accumulative_5_frames(self):
        s = FrameSchedule.from_mode("accumulative", 5)
        assert s.ref_indices == (0, 0, 0, 0)

    def test_incremental_3_frames(self):
        s = FrameSchedule.from_mode("incremental", 3)
        assert s.ref_indices == (0, 1)

    def test_incremental_5_frames(self):
        s = FrameSchedule.from_mode("incremental", 5)
        assert s.ref_indices == (0, 1, 2, 3)

    def test_incremental_2_frames(self):
        """Minimum: 2 frames -> 1 pair."""
        s = FrameSchedule.from_mode("incremental", 2)
        assert s.ref_indices == (0,)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown reference mode"):
            FrameSchedule.from_mode("invalid", 3)

    def test_too_few_frames(self):
        with pytest.raises(ValueError, match="n_frames must be >= 2"):
            FrameSchedule.from_mode("accumulative", 1)


# ---------------------------------------------------------------------------
# Tree traversal methods
# ---------------------------------------------------------------------------


class TestFrameScheduleTraversal:
    """Test parent(), path_to_root(), children() methods."""

    def test_parent_accumulative(self):
        s = FrameSchedule.from_mode("accumulative", 5)
        for frame in range(1, 5):
            assert s.parent(frame) == 0

    def test_parent_incremental(self):
        s = FrameSchedule.from_mode("incremental", 5)
        assert s.parent(1) == 0
        assert s.parent(2) == 1
        assert s.parent(3) == 2
        assert s.parent(4) == 3

    def test_parent_out_of_range(self):
        s = FrameSchedule.from_mode("accumulative", 3)
        with pytest.raises(IndexError):
            s.parent(0)
        with pytest.raises(IndexError):
            s.parent(3)

    def test_path_to_root_accumulative(self):
        s = FrameSchedule.from_mode("accumulative", 4)
        assert s.path_to_root(1) == [1, 0]
        assert s.path_to_root(3) == [3, 0]

    def test_path_to_root_incremental(self):
        s = FrameSchedule.from_mode("incremental", 5)
        assert s.path_to_root(1) == [1, 0]
        assert s.path_to_root(4) == [4, 3, 2, 1, 0]

    def test_path_to_root_skip(self):
        """Skip-3 key-frame: (0,0,0,3,3,3)."""
        s = FrameSchedule(ref_indices=(0, 0, 0, 3, 3, 3))
        assert s.path_to_root(1) == [1, 0]
        assert s.path_to_root(4) == [4, 3, 0]
        assert s.path_to_root(6) == [6, 3, 0]

    def test_path_to_root_mixed(self):
        """Mixed: (0, 1, 0) -> frame 1 refs 0, frame 2 refs 1, frame 3 refs 0."""
        s = FrameSchedule(ref_indices=(0, 1, 0))
        assert s.path_to_root(2) == [2, 1, 0]
        assert s.path_to_root(3) == [3, 0]

    def test_children_accumulative(self):
        s = FrameSchedule.from_mode("accumulative", 4)
        assert s.children(0) == [1, 2, 3]
        assert s.children(1) == []
        assert s.children(2) == []

    def test_children_incremental(self):
        s = FrameSchedule.from_mode("incremental", 4)
        assert s.children(0) == [1]
        assert s.children(1) == [2]
        assert s.children(2) == [3]
        assert s.children(3) == []

    def test_children_skip(self):
        s = FrameSchedule(ref_indices=(0, 0, 0, 3, 3, 3))
        assert s.children(0) == [1, 2, 3]
        assert s.children(3) == [4, 5, 6]
        assert s.children(1) == []


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------


class TestFrameScheduleImmutability:
    """FrameSchedule should be immutable (frozen dataclass)."""

    def test_frozen(self):
        s = FrameSchedule(ref_indices=(0, 0))
        with pytest.raises(AttributeError):
            s.ref_indices = (0, 1)  # type: ignore[misc]
