"""Picking a folder must not decode every image in it.

A user reported that selecting a folder of 3000 frames froze pyALDIC for
15-20 minutes, and asked whether the program reads every image first. It did:
the dimension check that warns about mismatched frames called the full
decoder on every file and used only ``shape[:2]``, so a folder scan decoded
every pixel of every frame. On cloud-synced or network drives, where a full
read pulls the whole file down, that is far worse than the local cost.

The scan now reads image headers instead. These tests pin that: no full
decode during a scan, the sizes it reports still match a real decode, and a
deleted frame stays deleted when the sort mode changes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import al_dic.gui.controllers.image_controller as ic
from al_dic.gui.app_state import AppState
from al_dic.gui.controllers.image_controller import (
    SUPPORTED_EXTENSIONS,
    ImageController,
    _read_image_raw,
    _read_image_size,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _reset_state():
    AppState.instance().reset()
    yield
    AppState.instance().reset()


def _write(path: Path, arr: np.ndarray) -> bool:
    """Write via imencode + tofile so non-ASCII paths work on Windows."""
    ok, buf = cv2.imencode(path.suffix, arr)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


@pytest.fixture
def frames(tmp_path):
    arr = (np.random.default_rng(0).random((48, 64)) * 255).astype(np.uint8)
    for i in range(12):
        _write(tmp_path / f"f{i:03d}.tif", arr)
    return tmp_path


class TestScanDoesNotDecode:
    def test_folder_scan_decodes_nothing(self, qapp, frames, monkeypatch):
        """The cost of picking a folder must not scale with pixels."""
        calls: list[str] = []
        real = ic._read_image_raw
        monkeypatch.setattr(
            ic, "_read_image_raw",
            lambda p: (calls.append(p), real(p))[1],
        )

        ctrl = ImageController(AppState.instance())
        ctrl.load_folder(str(frames))

        assert len(AppState.instance().image_files) == 12
        assert calls == [], (
            f"folder scan fully decoded {len(calls)} image(s); it must read "
            f"headers only"
        )

    def test_mismatched_sizes_are_still_reported(self, qapp, tmp_path):
        """Reading headers must not cost the warning it exists to produce."""
        a = (np.random.default_rng(0).random((48, 64)) * 255).astype(np.uint8)
        b = (np.random.default_rng(1).random((50, 64)) * 255).astype(np.uint8)
        _write(tmp_path / "f000.tif", a)
        _write(tmp_path / "f001.tif", a)
        _write(tmp_path / "f002.tif", b)        # the odd one out

        state = AppState.instance()
        messages: list[tuple[str, str]] = []
        state.log_message.connect(lambda m, lvl="info": messages.append((lvl, m)))

        ImageController(state).load_folder(str(tmp_path))

        errors = [m for lvl, m in messages if lvl == "error"]
        assert errors, "a size mismatch must still be reported"
        assert "f002.tif" in errors[0]
        assert "64x50" in errors[0], errors[0]


class TestHeaderSizeMatchesDecode:
    @pytest.mark.parametrize("ext", sorted(SUPPORTED_EXTENSIONS))
    @pytest.mark.parametrize("dtype,scale", [("uint8", 255), ("uint16", 65535)])
    def test_same_shape_as_a_full_decode(self, tmp_path, ext, dtype, scale):
        arr = (np.random.default_rng(0).random((37, 53)) * scale).astype(dtype)
        p = tmp_path / f"sample{ext}"
        if not _write(p, arr):
            pytest.skip(f"cv2 cannot encode {dtype} {ext}")
        assert _read_image_size(str(p)) == _read_image_raw(str(p)).shape[:2]

    def test_non_ascii_path(self, tmp_path):
        """Windows CJK paths are why the reader is not cv2.imread."""
        d = tmp_path / "试样_测试"
        d.mkdir()
        arr = (np.random.default_rng(0).random((29, 41)) * 255).astype(np.uint8)
        p = d / "帧_0001.tif"
        _write(p, arr)
        assert _read_image_size(str(p)) == (29, 41)

    def test_unreadable_file_falls_back(self, tmp_path):
        """Anything Pillow cannot parse must still go through the decoder."""
        p = tmp_path / "broken.tif"
        p.write_bytes(b"not an image")
        with pytest.raises(Exception):
            _read_image_size(str(p))      # falls through to the full decoder


class TestDeletedFramesStayDeleted:
    def test_sort_toggle_does_not_resurrect(self, qapp, frames):
        """Deleting a frame then changing sort mode must not bring it back.

        The controller kept its own copy of the file list; a delete that
        touched only state.image_files left the two out of step, so the next
        sort toggle restored the deleted frame -- and rescanned every file.
        """
        state = AppState.instance()
        ctrl = ImageController(state)
        ctrl.load_folder(str(frames))
        assert len(state.image_files) == 12

        kept = [f for f in state.image_files if "f005" not in f]
        ctrl.set_files(kept)
        state.image_files = kept

        ctrl.set_natural_sort(not ctrl.natural_sort)

        assert len(state.image_files) == 11
        assert not any("f005" in f for f in state.image_files)
