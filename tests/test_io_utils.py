"""Tests for al_dic.io.io_utils — image I/O, normalization, and conversion."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from al_dic.io.io_utils import (
    _read_unchanged,
    _to_grayscale,
    _normalize_to_float64,
    _to_uint8,
    load_images,
    read_mask_as_bool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_png(path: Path, img: np.ndarray) -> None:
    """Write an image to PNG using cv2.imencode (Unicode-safe)."""
    success, buf = cv2.imencode(".png", img)
    assert success
    buf.tofile(str(path))


# ---------------------------------------------------------------------------
# _read_unchanged
# ---------------------------------------------------------------------------

class TestReadUnchanged:
    def test_read_png(self, tmp_path):
        img = np.full((32, 32), 128, dtype=np.uint8)
        p = tmp_path / "test.png"
        _write_png(p, img)
        result = _read_unchanged(p)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _read_unchanged(tmp_path / "nonexistent.png")

    def test_corrupt_file(self, tmp_path):
        p = tmp_path / "corrupt.png"
        p.write_bytes(b"not an image")
        with pytest.raises(IOError):
            _read_unchanged(p)


# ---------------------------------------------------------------------------
# _to_grayscale
# ---------------------------------------------------------------------------

class TestToGrayscale:
    def test_bgr(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # red channel
        result = _to_grayscale(img)
        assert result.ndim == 2

    def test_bgra(self):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[:, :, 1] = 200
        result = _to_grayscale(img)
        assert result.ndim == 2

    def test_already_gray(self):
        img = np.ones((10, 10), dtype=np.uint8) * 128
        result = _to_grayscale(img)
        np.testing.assert_array_equal(result, img)


# ---------------------------------------------------------------------------
# _normalize_to_float64
# ---------------------------------------------------------------------------

class TestNormalizeToFloat64:
    def test_uint8(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        result = _normalize_to_float64(img)
        assert result.dtype == np.float64
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 2] == pytest.approx(1.0)

    def test_uint16(self):
        img = np.array([[0, 65535]], dtype=np.uint16)
        result = _normalize_to_float64(img)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(1.0)

    def test_uint32(self):
        img = np.array([[0, 4294967295]], dtype=np.uint32)
        result = _normalize_to_float64(img)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(1.0)

    def test_float_clips(self):
        img = np.array([[-0.5, 0.5, 1.5]], dtype=np.float32)
        result = _normalize_to_float64(img)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.5)
        assert result[0, 2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _to_uint8
# ---------------------------------------------------------------------------

class TestToUint8:
    def test_from_uint16(self):
        img = np.array([[0, 256, 65535]], dtype=np.uint16)
        result = _to_uint8(img)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_from_float(self):
        img = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
        result = _to_uint8(img)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_identity(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        result = _to_uint8(img)
        np.testing.assert_array_equal(result, img)


# ---------------------------------------------------------------------------
# load_images
# ---------------------------------------------------------------------------

class TestLoadImages:
    def test_sorted_loading(self, tmp_path):
        """Images should be loaded in sorted filename order."""
        for name in ["c.png", "a.png", "b.png"]:
            img = np.full((16, 16), 128, dtype=np.uint8)
            _write_png(tmp_path / name, img)

        result = load_images(tmp_path, pattern="*.png")
        assert len(result) == 3
        assert result[0].dtype == np.float64
        assert result[0].shape == (16, 16)

    def test_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No images"):
            load_images(tmp_path, pattern="*.tif")


# ---------------------------------------------------------------------------
# read_mask_as_bool
# ---------------------------------------------------------------------------

class TestReadMaskAsBool:
    def test_with_resize(self, tmp_path):
        """Mask should be resized to target_shape if different."""
        img = np.full((64, 64), 255, dtype=np.uint8)
        p = tmp_path / "mask.png"
        _write_png(p, img)

        result = read_mask_as_bool(p, target_shape=(32, 32))
        assert result.shape == (32, 32)
        assert result.dtype == np.bool_
        assert result.all()

    def test_threshold(self, tmp_path):
        """Pixels <= 127 should be False, > 127 should be True."""
        img = np.zeros((16, 16), dtype=np.uint8)
        img[:8, :] = 255
        img[8:, :] = 64
        p = tmp_path / "mask.png"
        _write_png(p, img)

        result = read_mask_as_bool(p)
        assert result[:8, :].all()
        assert not result[8:, :].any()
