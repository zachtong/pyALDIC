"""Tests for ROI mask controller — add/cut boolean operations."""

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from staq_dic.gui.controllers.roi_controller import ROIController


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class TestROIController:
    def test_add_rectangle(self, qapp):
        ctrl = ROIController(img_shape=(100, 100))
        ctrl.add_rectangle(10, 10, 50, 50, mode="add")
        assert ctrl.mask[30, 30] == True  # (row=30, col=30) inside
        assert ctrl.mask[0, 0] == False

    def test_cut_rectangle(self, qapp):
        ctrl = ROIController(img_shape=(100, 100))
        ctrl.add_rectangle(0, 0, 100, 100, mode="add")
        ctrl.add_rectangle(20, 20, 60, 60, mode="cut")
        assert ctrl.mask[10, 10] == True
        assert ctrl.mask[40, 40] == False

    def test_add_polygon(self, qapp):
        ctrl = ROIController(img_shape=(100, 100))
        pts = [(10, 10), (90, 10), (90, 90), (10, 90)]
        ctrl.add_polygon(pts, mode="add")
        assert ctrl.mask[50, 50] == True
        assert ctrl.mask[0, 0] == False

    def test_add_circle(self, qapp):
        ctrl = ROIController(img_shape=(100, 100))
        ctrl.add_circle(cx=50, cy=50, radius=30, mode="add")
        assert ctrl.mask[50, 50] == True
        assert ctrl.mask[0, 0] == False

    def test_clear(self, qapp):
        ctrl = ROIController(img_shape=(100, 100))
        ctrl.add_rectangle(0, 0, 100, 100, mode="add")
        ctrl.clear()
        assert not np.any(ctrl.mask)

    def test_import_mask(self, qapp, tmp_path):
        mask_img = np.zeros((100, 100), dtype=np.uint8)
        mask_img[20:80, 20:80] = 255
        cv2.imwrite(str(tmp_path / "mask.png"), mask_img)
        ctrl = ROIController(img_shape=(100, 100))
        ctrl.import_mask(str(tmp_path / "mask.png"))
        assert ctrl.mask[50, 50] == True
        assert ctrl.mask[0, 0] == False

    def test_stroke_segment_paints_thick_line(self, qapp):
        ctrl = ROIController(img_shape=(64, 64))
        ctrl.stroke_segment(10, 32, 50, 32, radius=3, mode="add")
        # Line y=32, x in [10..50], +/- 3 px should be set
        assert ctrl.mask[32, 30]
        assert ctrl.mask[29, 30]  # within radius (~3 px above line)
        assert not ctrl.mask[20, 30]  # well outside radius
        assert ctrl.mask[32, 50]
        assert not ctrl.mask[32, 60]  # past end of segment

    def test_stroke_segment_zero_length_acts_as_dot(self, qapp):
        ctrl = ROIController(img_shape=(32, 32))
        ctrl.stroke_segment(16, 16, 16, 16, radius=2, mode="add")
        assert ctrl.mask[16, 16]
        assert ctrl.mask[14, 16]  # within radius
        assert not ctrl.mask[10, 10]  # outside

    def test_stroke_segment_erase_mode(self, qapp):
        ctrl = ROIController(img_shape=(32, 32))
        ctrl.mask[:] = True
        ctrl.stroke_segment(16, 16, 16, 16, radius=2, mode="cut")
        assert not ctrl.mask[16, 16]
        assert ctrl.mask[0, 0]  # untouched corner
