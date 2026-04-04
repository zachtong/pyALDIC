"""Tests for image loading controller."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from staq_dic.gui.app_state import AppState
from staq_dic.gui.controllers.image_controller import ImageController


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def image_folder(tmp_path):
    for name in ["img_1.tif", "img_10.tif", "img_2.tif", "img_20.tif"]:
        img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        img.save(tmp_path / name)
    return tmp_path


class TestImageController:
    def test_lexicographic_sort_default(self, qapp, image_folder):
        """Default sort is lexicographic (for zero-padded names)."""
        state = AppState.instance()
        state.reset()
        ctrl = ImageController(state)
        ctrl.load_folder(str(image_folder))
        names = [Path(f).name for f in state.image_files]
        # Lexicographic: "1" < "10" < "2" < "20"
        assert names == ["img_1.tif", "img_10.tif", "img_2.tif", "img_20.tif"]

    def test_natural_sort(self, qapp, image_folder):
        """Natural sort treats embedded numbers as integers."""
        state = AppState.instance()
        state.reset()
        ctrl = ImageController(state)
        ctrl.set_natural_sort(True)
        ctrl.load_folder(str(image_folder))
        names = [Path(f).name for f in state.image_files]
        assert names == ["img_1.tif", "img_2.tif", "img_10.tif", "img_20.tif"]

    def test_load_populates_state(self, qapp, image_folder):
        state = AppState.instance()
        state.reset()
        ctrl = ImageController(state)
        ctrl.load_folder(str(image_folder))
        assert len(state.image_files) == 4
        assert state.image_folder == Path(image_folder)

    def test_read_image_returns_float64(self, qapp, image_folder):
        state = AppState.instance()
        state.reset()
        ctrl = ImageController(state)
        ctrl.load_folder(str(image_folder))
        img = ctrl.read_image(0)
        assert img.dtype == np.float64
        assert img.ndim == 2

    def test_empty_folder(self, qapp, tmp_path):
        state = AppState.instance()
        state.reset()
        ctrl = ImageController(state)
        ctrl.load_folder(str(tmp_path))
        assert state.image_files == []

    def test_supported_extensions(self, qapp, tmp_path):
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(tmp_path / "a.tif")
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(tmp_path / "b.png")
        (tmp_path / "c.txt").write_text("not an image")
        state = AppState.instance()
        state.reset()
        ctrl = ImageController(state)
        ctrl.load_folder(str(tmp_path))
        assert len(state.image_files) == 2
