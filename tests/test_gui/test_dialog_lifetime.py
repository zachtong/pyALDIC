"""A dialog given a parent must not outlive the call that opened it.

ExportDialog, BatchImportDialog and LoadDataDialog are all created with
``parent=self`` and shown with ``exec()``. Qt hands ownership to the parent,
so when the method returns and drops its reference the dialog stays alive --
for the rest of the session. Every export opened in a session kept its own
reference to the whole PipelineResult, and every one of them was restyled
again on each theme switch.

These tests pin the call sites: the dialog is deleted once it is done with.
"""

from __future__ import annotations

import ast
import gc
from pathlib import Path

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMainWindow

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    DICMesh, FrameResult, FrameSchedule, PipelineResult, StrainResult,
)
from al_dic.gui.dialogs.export_dialog import ExportDialog, VizExportHint

SRC = Path(__file__).resolve().parents[2] / "src" / "al_dic" / "gui"

# Call sites that open a parented dialog, and the method each one lives in.
CALL_SITES = [
    ("app.py", "_on_batch_import", "BatchImportDialog"),
    ("panels/right_sidebar.py", "_on_export", "ExportDialog"),
    ("strain_window.py", "_on_export_strain", "ExportDialog"),
    ("panels/analysis/tab.py", "_on_load_data", "LoadDataDialog"),
]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _result(n=400, frames=3):
    coords = np.random.default_rng(0).random((n, 2)) * 100
    mesh = DICMesh(coordinates_fem=coords, elements_fem=np.zeros((0, 8), np.int64))
    u = np.random.default_rng(1).random(n)
    fr = FrameResult(U=np.repeat(u, 2), U_accum=np.repeat(u, 2))
    sr = StrainResult(
        disp_u=u, disp_v=u, strain_exx=u, strain_eyy=u, strain_exy=u,
        strain_principal_max=u, strain_principal_min=u, strain_maxshear=u,
        strain_von_mises=u, strain_rotation=u,
    )
    return PipelineResult(
        dic_para=dicpara_default(img_size=(128, 128)), dic_mesh=mesh,
        result_disp=[fr] * frames, result_def_grad=[fr] * frames,
        result_strain=[sr] * frames, result_fe_mesh_each_frame=[mesh] * frames,
        frame_schedule=FrameSchedule.from_mode("accumulative", frames + 1),
    )


def _live_export_dialogs() -> int:
    return sum(
        1 for o in gc.get_objects()
        if type(o).__name__ == "ExportDialog" and shiboken6.isValid(o)
    )


def _settle(app) -> None:
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()


class TestDialogsDoNotPileUp:
    def test_deleted_dialog_does_not_survive(self, qapp):
        """The teardown the call sites use must actually free the dialog."""
        parent = QMainWindow()
        result = _result()
        _settle(qapp)
        before = _live_export_dialogs()

        for _ in range(5):
            dlg = ExportDialog(result, None, VizExportHint(),
                               image_files=[], parent=parent)
            dlg.reject()            # what clicking Close does
            dlg.deleteLater()       # what the call sites now do
            del dlg
            _settle(qapp)

        assert _live_export_dialogs() == before, (
            "dialogs are still piling up after being closed and deleted"
        )
        parent.deleteLater()
        _settle(qapp)

    def test_without_deletion_they_would_pile_up(self, qapp):
        """Guard the guard: the check above would catch a regression."""
        parent = QMainWindow()
        result = _result()
        _settle(qapp)
        before = _live_export_dialogs()

        kept = []
        for _ in range(3):
            dlg = ExportDialog(result, None, VizExportHint(),
                               image_files=[], parent=parent)
            dlg.reject()
            kept.append(dlg)        # no deleteLater -- the old behaviour
            del dlg
        _settle(qapp)

        assert _live_export_dialogs() == before + 3

        for dlg in kept:
            dlg.deleteLater()
        kept.clear()
        parent.deleteLater()
        _settle(qapp)


@pytest.mark.parametrize("rel,method,dialog", CALL_SITES)
def test_call_site_deletes_its_dialog(rel: str, method: str, dialog: str):
    """Every parented dialog opened with exec() is deleted in the same method.

    Read as source rather than driven through the GUI: these methods open a
    modal dialog, which a test cannot dismiss without a running event loop.
    """
    tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
    func = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == method),
        None,
    )
    assert func is not None, f"{rel}: no method {method}"

    body = ast.dump(func)
    assert dialog in body, f"{rel}:{method} no longer opens {dialog}"
    assert "'exec'" in body, f"{rel}:{method} no longer calls exec()"
    assert "'deleteLater'" in body, (
        f"{rel}:{method} opens {dialog} with a parent and exec() but never "
        f"deletes it -- it would stay alive for the rest of the session, "
        f"holding whatever it was given"
    )
