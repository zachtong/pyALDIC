"""Tests for BatchImportDialog — mask assignment logic.

Tests the assignment logic without requiring actual file I/O.
"""

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from al_dic.gui.dialogs.batch_import_dialog import BatchImportDialog


@pytest.fixture
def dialog():
    image_files = [f"/images/frame_{i:02d}.tif" for i in range(5)]
    dlg = BatchImportDialog(image_files)
    return dlg


def test_initial_empty_assignments(dialog):
    assert dialog.get_assignments() == {}


def test_auto_match_by_filename_number(dialog):
    """Auto-match extracts last number from mask filename."""
    dialog._mask_files = [
        "/masks/mask_02.png",
        "/masks/mask_00.png",
        "/masks/mask_04.png",
    ]
    dialog._auto_match()
    assignments = dialog.get_assignments()
    assert assignments.get(0) == "/masks/mask_00.png"
    assert assignments.get(2) == "/masks/mask_02.png"
    assert assignments.get(4) == "/masks/mask_04.png"


def test_auto_match_out_of_range_ignored(dialog):
    """Mask files with numbers beyond frame count are ignored."""
    dialog._mask_files = ["/masks/mask_99.png"]
    dialog._auto_match()
    assert dialog.get_assignments() == {}


def test_sequential_assignment(dialog):
    """Sequential assigns masks in order starting from frame 0."""
    dialog._mask_files = ["/masks/a.png", "/masks/b.png", "/masks/c.png"]
    dialog._assign_sequential()
    assignments = dialog.get_assignments()
    assert assignments[0] == "/masks/a.png"
    assert assignments[1] == "/masks/b.png"
    assert assignments[2] == "/masks/c.png"
    assert 3 not in assignments


def test_sequential_more_masks_than_frames(dialog):
    """Extra masks beyond frame count are ignored."""
    dialog._mask_files = [f"/masks/{i}.png" for i in range(10)]
    dialog._assign_sequential()
    assignments = dialog.get_assignments()
    assert len(assignments) == 5  # only 5 frames


def test_clear_assignments(dialog):
    dialog._mask_files = ["/masks/a.png"]
    dialog._assign_sequential()
    assert len(dialog.get_assignments()) > 0
    dialog._clear_assignments()
    assert dialog.get_assignments() == {}


def test_populate_frames_count(dialog):
    """Frame tree should have one row per image file."""
    assert dialog._assign_tree.topLevelItemCount() == 5


def test_get_assignments_returns_dict(dialog):
    """get_assignments should return a dict copy."""
    result = dialog.get_assignments()
    assert isinstance(result, dict)
    # Modifying the returned dict should not affect internal state
    result[99] = "fake"
    assert 99 not in dialog.get_assignments()
