"""A saved session must come back exactly as it was left.

Save/load is how users park a project, so anything that silently fails to
round-trip is a data-loss bug: parameters that quietly revert change the next
run, and a lost refinement brush changes the mesh -- and therefore the results.

These tests set every persisted field to a distinctive non-default value and
assert it survives. ``test_every_state_field_is_accounted_for`` is the guard
that matters long-term: it fails when a new piece of AppState appears that is
neither persisted nor explicitly listed as transient, so the next field cannot
be forgotten the way the refinement brush was.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from al_dic.gui.app_state import AppState
from al_dic.gui.session import (
    _PARAM_KEYS, _PHYSICAL_KEYS, _VIEW_SCALAR_KEYS,
    load_session, save_session,
)


def _safe_disconnect(signal) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            signal.disconnect()
        except (RuntimeError, TypeError):
            pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _reset_singleton():
    state = AppState.instance()
    for sig in (
        state.images_changed, state.current_frame_changed, state.roi_changed,
        state.params_changed, state.run_state_changed, state.progress_updated,
        state.results_changed, state.display_changed, state.log_message,
    ):
        _safe_disconnect(sig)
    state.reset()
    yield
    state.reset()


H, W, N_FRAMES = 48, 64, 3

PARAMS = dict(
    subset_size=24, subset_step=8, search_range=14,
    tracking_mode="incremental", inc_ref_mode="interval",
    inc_ref_interval=3, inc_custom_refs=[1, 2],
    use_admm=False, admm_max_iter=7,
    init_guess_mode="fft_every", fft_reset_interval=2, fft_auto_expand=False,
    seed_ncc_threshold=0.42,
    refine_inner=True, refine_outer=True, refinement_level=2,
)
PHYSICAL = dict(use_physical_units=True, pixel_size=2.5, pixel_unit="um",
                frame_rate=30.0)
VIEW = dict(display_field="strain_eyy", show_deformed=False,
            overlay_alpha=0.35, show_mesh=False, mesh_line_color="#ff0000",
            mesh_line_width=3, show_subset_window=True)

# AppState fields that are deliberately not persisted, with the reason.
TRANSIENT = {
    "roi_editing": "UI mode, meaningless across sessions",
    "run_state": "recomputed from whether results were restored",
    "progress": "live run telemetry",
    "progress_message": "live run telemetry",
    "elapsed_seconds": "live run telemetry",
    "deformed_masks": "derived from ROIs + displacement during a run",
    "seeds": "mesh node indices, invalid once ROI/winsize/step change",
    "current_frame": "persisted, but applied by the caller after images load",
    "results": "stored in the bundle's npz, not the config",
    "image_files": "restored by reloading the image folder",
    "image_folder": "top-level key",
    "per_frame_rois": "top-level key",
    "refine_brush_mask": "top-level key",
    "probes": "top-level key",
    "load_data": "top-level key",
}


def _make_project(tmp_path, qapp):
    from al_dic.gui.app import MainWindow

    folder = tmp_path / "img"
    folder.mkdir(parents=True, exist_ok=True)
    from scipy.ndimage import gaussian_filter, shift as ndshift
    rng = np.random.default_rng(3)
    speckle = gaussian_filter(rng.standard_normal((H, W)), 2.0)
    speckle = (speckle - speckle.min()) / (speckle.max() - speckle.min())
    for k in range(N_FRAMES):
        frame = speckle if k == 0 else ndshift(speckle, (0.0, 0.6 * k),
                                               order=3, mode="reflect")
        cv2.imwrite(str(folder / f"f{k}.png"),
                    (np.clip(frame, 0, 1) * 255).astype(np.uint8))

    win = MainWindow()
    win._image_ctrl.load_folder(str(folder))
    state = AppState.instance()

    for k, v in {**PARAMS, **PHYSICAL, **VIEW}.items():
        setattr(state, k, v)

    fs = state.get_field_state("strain_eyy")
    fs.auto, fs.vmin, fs.vmax, fs.colormap = False, -0.03, 0.11, "turbo"

    roi0 = np.zeros((H, W), bool); roi0[10:35, 12:50] = True
    roi2 = np.zeros((H, W), bool); roi2[12:38, 14:52] = True
    state.set_frame_roi(0, roi0)
    state.set_frame_roi(2, roi2)

    brush = np.zeros((H, W), bool); brush[20:28, 25:35] = True
    state.set_refine_brush_mask(brush)
    state.set_current_frame(2)

    return win, state, folder, roi0, roi2, brush


def _reload(tmp_path, qapp, path):
    from al_dic.gui.app import MainWindow

    AppState.instance().reset()
    win = MainWindow()
    win._apply_loaded_session(load_session(path), str(path))
    return win, AppState.instance()


@pytest.fixture
def round_tripped(tmp_path, qapp):
    win, state, folder, roi0, roi2, brush = _make_project(tmp_path, qapp)
    path = tmp_path / "s.aldic"
    save_session(path, state, include_results=False)
    win2, state2 = _reload(tmp_path, qapp, path)
    return state2, win2, folder, roi0, roi2, brush


@pytest.mark.parametrize("key,value", sorted(PARAMS.items()))
def test_solver_parameters_survive(round_tripped, key, value):
    state, *_ = round_tripped
    assert getattr(state, key) == value


@pytest.mark.parametrize("key,value", sorted(PHYSICAL.items()))
def test_physical_units_survive(round_tripped, key, value):
    state, *_ = round_tripped
    assert getattr(state, key) == value


@pytest.mark.parametrize("key,value", sorted(VIEW.items()))
def test_view_state_survives(round_tripped, key, value):
    state, *_ = round_tripped
    assert getattr(state, key) == value


def test_per_field_colour_settings_survive(round_tripped):
    state, *_ = round_tripped
    fs = state.get_field_state("strain_eyy")
    assert (fs.auto, fs.vmin, fs.vmax, fs.colormap) == (False, -0.03, 0.11, "turbo")


def test_images_survive(round_tripped):
    state, _, folder, *_ = round_tripped
    assert len(state.image_files) == N_FRAMES
    assert str(state.image_folder) == str(folder)


def test_every_roi_survives(round_tripped):
    state, _, _, roi0, roi2, _ = round_tripped
    assert sorted(state.per_frame_rois) == [0, 2]
    assert np.array_equal(state.per_frame_rois[0], roi0)
    assert np.array_equal(state.per_frame_rois[2], roi2)


def test_refinement_brush_survives(round_tripped):
    """The painted refinement zones feed BrushRegionCriterion — losing them
    changes the mesh, so a re-run of a restored session would not reproduce
    the results it was saved with."""
    state, _, _, _, _, brush = round_tripped
    assert state.refine_brush_mask is not None, "refinement brush was lost"
    assert np.array_equal(state.refine_brush_mask, brush)


def test_results_survive_with_strain(tmp_path, qapp):
    from al_dic.core.config import dicpara_default
    from al_dic.core.data_structures import GridxyROIRange
    from al_dic.core.pipeline import run_aldic
    from al_dic.io.io_utils import load_images

    win, state, folder, roi0, _, _ = _make_project(tmp_path, qapp)
    images = load_images(folder, pattern="*.png")
    masks = [roi0.astype(np.float64)] * len(images)
    ys, xs = np.where(roi0)
    para = dicpara_default(
        img_size=(H, W), winsize=16, winstepsize=8, use_masks=True,
        img_ref_mask=masks[0],
        gridxy_roi_range=GridxyROIRange((int(xs.min()), int(xs.max())),
                                        (int(ys.min()), int(ys.max()))),
        strain_plane_fit_rad=10.0,
    )
    state.results = run_aldic(para, images, masks, compute_strain=True)
    expect_u = [None if r is None else r.U.copy() for r in state.results.result_disp]
    expect_e = [None if r is None else r.strain_eyy.copy()
                for r in state.results.result_strain]

    path = tmp_path / "with_results.aldic"
    save_session(path, state, include_results=True)
    _, state2 = _reload(tmp_path, qapp, path)

    assert state2.results is not None
    got_u = [None if r is None else r.U for r in state2.results.result_disp]
    got_e = [None if r is None else r.strain_eyy for r in state2.results.result_strain]
    assert len(got_u) == len(expect_u) and len(got_e) == len(expect_e)
    for a, b in zip(expect_u, got_u):
        assert (a is None and b is None) or np.array_equal(a, b, equal_nan=True)
    for a, b in zip(expect_e, got_e):
        assert (a is None and b is None) or np.array_equal(a, b, equal_nan=True)
    # the parameters the results were computed with come back too
    assert state2.results.dic_para.winsize == 16
    assert state2.results.dic_para.strain_plane_fit_rad == 10.0


def test_probes_survive(tmp_path, qapp):
    """A placed, renamed, recoloured probe comes back intact.

    "top-level key" in TRANSIENT records an intention; only a round trip shows
    it was honoured. The refinement brush was registered as saved for a whole
    release while not being saved at all.
    """
    from al_dic.analysis.probes import AreaGeom, LineGeom, PointGeom

    win, state, folder, roi0, roi2, brush = _make_project(tmp_path, qapp)
    state.probes.add("point", PointGeom(12.5, 30.25), label="crack tip",
                     color="#ef4444")
    state.probes.add("line", LineGeom(4.0, 20.0, 36.0, 20.0), label="gauge")
    state.probes.add("area", AreaGeom.polygon([(0, 0), (9, 1), (4, 7)]),
                     label="区域")
    before = state.probes.to_list()

    path = tmp_path / "probes.aldic"
    save_session(path, state, include_results=False)
    _, state2 = _reload(tmp_path, qapp, path)

    assert state2.probes.to_list() == before
    # Ids must not restart, or a restored session collides with itself.
    assert state2.probes.add("point", PointGeom(1.0, 1.0)).id == 4


def test_the_machine_load_record_survives(tmp_path, qapp):
    """Mapping, A0 and the columns in use come back; the images reloading on
    the way (which clears a load record) must not take it with them."""
    import numpy as np

    from al_dic.analysis.load_data import LoadData, LoadSync, LoadTable

    win, state, folder, roi0, roi2, brush = _make_project(tmp_path, qapp)
    table = LoadTable(("t", "F"), (np.array([0.0, 1.0, 2.0]),
                                   np.array([0.0, 1.5, np.nan])), "run7.csv")
    data = LoadData(table, LoadSync(mode="time", load_column="F", time_column="t",
                                    offset_s=0.25, load_unit="kN"), area_mm2=12.5)
    state.set_load_data(data)

    path = tmp_path / "load.aldic"
    save_session(path, state, include_results=False)
    _, state2 = _reload(tmp_path, qapp, path)

    back = state2.load_data
    assert back is not None
    assert back.sync == data.sync and back.area_mm2 == 12.5
    assert back.source == "run7.csv"
    np.testing.assert_array_equal(back.table.column("F"), table.column("F"))


def test_a_session_without_probes_still_loads(tmp_path, qapp):
    """Schema 1 and 2 sessions predate probes and must open with none."""
    win, state, folder, roi0, roi2, brush = _make_project(tmp_path, qapp)
    path = tmp_path / "old.aldic"
    save_session(path, state, include_results=False)

    import json
    import zipfile

    doc = json.loads(zipfile.ZipFile(path).read("session.json").decode("utf-8"))
    doc["schema_version"] = 2
    doc.pop("probes", None)
    rewritten = tmp_path / "v2.aldic"
    with zipfile.ZipFile(rewritten, "w") as zf:
        zf.writestr("session.json", json.dumps(doc))

    _, state2 = _reload(tmp_path, qapp, rewritten)
    assert len(state2.probes) == 0


def test_every_state_field_is_accounted_for(qapp):
    """No AppState field may be silently non-persisted.

    Add a field to AppState and it must either be saved (a param, physical
    unit, view-state or top-level key) or be listed in TRANSIENT with a
    reason. This is the guard that would have caught the refinement brush.
    """
    from PySide6.QtCore import SignalInstance

    state = AppState.instance()
    persisted = set(_PARAM_KEYS) | set(_PHYSICAL_KEYS) | set(_VIEW_SCALAR_KEYS)
    unaccounted = {
        name for name, value in vars(state).items()
        if not name.startswith("_")
        and not isinstance(value, SignalInstance)
        and name not in persisted
        and name not in TRANSIENT
    }
    assert not unaccounted, (
        f"AppState field(s) {sorted(unaccounted)} are neither persisted in a "
        f"session nor declared transient. Persist them, or add them to "
        f"TRANSIENT with the reason."
    )


def test_view_keys_are_restored_not_just_saved():
    """Every saved view key must have a restore path.

    The save list and the restore loop used to be written out separately, so a
    key could be saved and never read back (show_subset_window was).
    """
    import inspect

    from al_dic.gui import session as sess

    src = inspect.getsource(sess._apply_view_state)
    assert "_VIEW_SCALAR_KEYS" in src, (
        "_apply_view_state must iterate _VIEW_SCALAR_KEYS instead of a "
        "hand-maintained copy that can drift from the save side"
    )
