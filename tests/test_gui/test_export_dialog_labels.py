"""Export dialog: frames numbered like the frame slider, labels shown in full.

Three regressions found while writing the ASE 375 lab handout:

* the FRAME RANGE boxes counted the deformed images from 1, while the frame
  slider counts the reference image as frame 1 -- the 1 kN image was FRAME 11
  on the slider but 10 in the export;
* the tab "Preview & Colorbar" showed as "Preview _Colorbar": a tab label
  turns "&" into a keyboard mnemonic;
* "Browse..." and "Open Folder" had a fixed 90 px width, which cut the label
  at 150 % display scaling (and would cut longer translations).
"""

import re

import numpy as np
from PySide6.QtWidgets import QApplication

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    DICMesh, FrameResult, FrameSchedule, PipelineResult, StrainResult,
)
from al_dic.gui.dialogs.export_dialog import ExportDialog, VizExportHint

app = QApplication.instance() or QApplication([])

QWIDGETSIZE_MAX = (1 << 24) - 1
LONE_AMPERSAND = re.compile(r"(?<!&)&(?!&)")


def _result(n_results: int = 4, img: int = 96) -> PipelineResult:
    xs, ys = np.meshgrid(np.linspace(8, img - 8, 5), np.linspace(8, img - 8, 5))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    n = coords.shape[0]
    mesh = DICMesh(coordinates_fem=coords, elements_fem=np.zeros((0, 8), np.int64))
    u = coords[:, 0] / img
    fr = FrameResult(U=np.repeat(u, 2), U_accum=np.repeat(u, 2))
    zeros, small = np.zeros(n), np.full(n, 0.01)
    sr = StrainResult(disp_u=u, disp_v=zeros, strain_exx=small, strain_eyy=zeros,
                      strain_exy=zeros, strain_principal_max=small,
                      strain_principal_min=zeros, strain_maxshear=small,
                      strain_von_mises=small, strain_rotation=zeros)
    return PipelineResult(
        dic_para=dicpara_default(img_size=(img, img)), dic_mesh=mesh,
        result_disp=[fr] * n_results, result_def_grad=[fr] * n_results,
        result_strain=[sr] * n_results,
        result_fe_mesh_each_frame=[mesh] * n_results,
        frame_schedule=FrameSchedule.from_mode("accumulative", n_results + 1))


def _dialog(n_results: int = 4) -> ExportDialog:
    return ExportDialog(_result(n_results), None, VizExportHint(), image_files=[])


def test_the_frame_range_counts_frames_like_the_slider():
    dlg = _dialog(n_results=4)                 # five images: frames 1..5
    try:
        for prefix in ("img", "anim"):
            start = getattr(dlg, f"_{prefix}_frame_start_spin")
            end = getattr(dlg, f"_{prefix}_frame_end_spin")
            # frame 1 is the reference image, which has no field to export
            assert (start.minimum(), start.maximum()) == (2, 5)
            assert (end.minimum(), end.maximum()) == (2, 5)
            assert (start.value(), end.value()) == (2, 5)
            assert start.toolTip() and "1" in start.toolTip()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_a_frame_picked_in_the_range_is_that_slider_frame():
    dlg = _dialog(n_results=4)
    try:
        dlg._img_all_frames_check.setChecked(False)
        dlg._img_frame_start_spin.setValue(3)
        dlg._img_frame_end_spin.setValue(3)
        dlg._anim_all_frames_check.setChecked(False)
        dlg._anim_frame_start_spin.setValue(2)
        dlg._anim_frame_end_spin.setValue(5)
        cfg = dlg.get_config()
        # slider frame 3 is image 2, i.e. result_disp[1]
        assert (cfg.frame_start, cfg.frame_end) == (1, 1)
        assert (cfg.anim_frame_start, cfg.anim_frame_end) == (0, 3)
    finally:
        dlg.close()
        dlg.deleteLater()


def test_no_tab_label_turns_an_ampersand_into_a_mnemonic():
    dlg = _dialog()
    try:
        labels = [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())]
        assert "Preview && Colorbar" in labels
        assert not [t for t in labels if LONE_AMPERSAND.search(t)], labels
    finally:
        dlg.close()
        dlg.deleteLater()


def test_folder_buttons_are_never_narrower_than_their_label():
    from PySide6.QtWidgets import QPushButton

    dlg = _dialog()
    try:
        buttons = [b for b in dlg.findChildren(QPushButton)
                   if b.text() in ("Browse…", "Open Folder")]
        assert len(buttons) == 2
        for b in buttons:
            assert b.maximumWidth() == QWIDGETSIZE_MAX, b.text()   # no fixed width
            assert b.minimumWidth() >= 90
            assert b.sizeHint().width() >= b.fontMetrics().horizontalAdvance(b.text())
    finally:
        dlg.close()
        dlg.deleteLater()
