"""Round-trip tests for faithful PipelineResult (de)serialization."""

import io
from dataclasses import dataclass, fields, is_dataclass, replace

import numpy as np
import pytest

from al_dic.core.data_structures import (
    DICMesh, DICPara, FrameResult, FrameSchedule, FrameTrackingStats,
    GridxyROIRange, PipelineResult, RefGridExtent, StrainResult,
    TrackingDiagnostics,
)
from al_dic.io.session_serialize import (
    serialize_result, deserialize_result, save_result_npz, load_result_npz,
)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def _mesh(nx=5, ny=4, offset=0.0):
    xs, ys = np.meshgrid(np.linspace(0, 60, nx), np.linspace(0, 40, ny))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64) + offset
    return DICMesh(
        coordinates_fem=coords,
        elements_fem=np.zeros((0, 8), np.int64),
        mark_coord_hole_edge=np.array([0, 1], np.int64),
        x0=np.linspace(0, 60, nx) + offset,
        y0=np.linspace(0, 40, ny) + offset,
        element_min_size=8,
    )


def _frame(n, seed, accum=True, with_f=True):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(2 * n)
    return FrameResult(
        U=U,
        U_accum=U * 2.0 if accum else None,
        F=rng.standard_normal(4 * n) if with_f else None,
        bad_pt_num=np.array([1, 2, 3], np.int64) if with_f else None,
        ref_frame=0,
    )


def _strain(n, seed, full=True):
    rng = np.random.default_rng(seed)
    if not full:
        return StrainResult(disp_u=rng.standard_normal(n), disp_v=rng.standard_normal(n))
    return StrainResult(
        disp_u=rng.standard_normal(n), disp_v=rng.standard_normal(n),
        dudx=rng.standard_normal(n), dvdx=rng.standard_normal(n),
        dudy=rng.standard_normal(n), dvdy=rng.standard_normal(n),
        strain_exx=rng.standard_normal(n), strain_exy=rng.standard_normal(n),
        strain_eyy=rng.standard_normal(n),
        strain_principal_max=rng.standard_normal(n),
        strain_principal_min=rng.standard_normal(n),
        strain_maxshear=rng.standard_normal(n),
        strain_von_mises=rng.standard_normal(n),
        strain_rotation=rng.standard_normal(n),
        strain_valid=rng.random(n) > 0.3,
    )


def _make_result(n_frames=3, accumulative=True, with_strain=True,
                 distinct_meshes=False, para_arrays=False):
    mesh = _mesh()
    n = mesh.coordinates_fem.shape[0]
    n_pairs = n_frames - 1
    disp = [_frame(n, t) for t in range(n_pairs)]
    defgrad = [_frame(n, 100 + t, with_f=True) for t in range(n_pairs)]
    strain = [_strain(n, 200 + t) for t in range(n_pairs)] if with_strain else []
    if distinct_meshes:
        femesh = [_mesh(offset=float(t + 1)) for t in range(n_pairs)]
    else:
        femesh = [mesh for _ in range(n_pairs)]  # identical -> deduped
    para_kwargs = dict(img_size=(256, 256), winsize=40, winstepsize=16,
                       reference_mode="accumulative" if accumulative else "incremental",
                       gridxy_roi_range=GridxyROIRange(gridx=(0, 200), gridy=(0, 180)))
    if para_arrays:
        para_kwargs["img_ref_mask"] = (np.random.default_rng(9).random((32, 32)) > 0.5).astype(np.float64)
        para_kwargs["winsize_list"] = np.array([40.0, 32.0, 24.0])
    para = DICPara(**para_kwargs)
    return PipelineResult(
        dic_para=para, dic_mesh=mesh, result_disp=disp,
        result_def_grad=defgrad, result_strain=strain,
        result_fe_mesh_each_frame=femesh,
        frame_schedule=FrameSchedule.from_mode(
            "accumulative" if accumulative else "incremental", n_frames),
        ref_switch_frames=(1, 2),
        reseed_events=(),
    )


# --------------------------------------------------------------------------
# recursive equality
# --------------------------------------------------------------------------

def _assert_eq(a, b, path="result"):
    if a is None or b is None:
        assert a is None and b is None, f"{path}: {a!r} != {b!r}"
        return
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), f"{path}: type mismatch"
        assert a.shape == b.shape, f"{path}: shape {a.shape} != {b.shape}"
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} != {b.dtype}"
        np.testing.assert_array_equal(a, b, err_msg=path)
        return
    if is_dataclass(a):
        assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
        for f in fields(a):
            _assert_eq(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
        return
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b), f"{path}: seq type {type(a)} != {type(b)}"
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_eq(x, y, f"{path}[{i}]")
        return
    assert a == b, f"{path}: {a!r} != {b!r}"


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize("accumulative", [True, False])
@pytest.mark.parametrize("with_strain", [True, False])
@pytest.mark.parametrize("distinct_meshes", [True, False])
def test_serialize_roundtrip(accumulative, with_strain, distinct_meshes):
    res = _make_result(n_frames=4, accumulative=accumulative,
                        with_strain=with_strain, distinct_meshes=distinct_meshes)
    arrays, manifest = serialize_result(res)
    back = deserialize_result(arrays, manifest)
    _assert_eq(res, back)


def test_para_arrays_roundtrip():
    res = _make_result(para_arrays=True)
    back = deserialize_result(*serialize_result(res))
    _assert_eq(res, back)


def test_save_load_npz_roundtrip(tmp_path):
    res = _make_result(n_frames=5, with_strain=True)
    p = tmp_path / "r.npz"
    save_result_npz(str(p), res)
    _assert_eq(res, load_result_npz(str(p)))


def test_save_load_filelike_roundtrip():
    res = _make_result(n_frames=3)
    buf = io.BytesIO()
    save_result_npz(buf, res)
    buf.seek(0)
    _assert_eq(res, load_result_npz(buf))


def test_identical_meshes_are_deduped():
    """Accumulative run (identical per-frame meshes) stores mesh arrays once."""
    same = _make_result(n_frames=8, distinct_meshes=False)
    diff = _make_result(n_frames=8, distinct_meshes=True)
    n_same = len(serialize_result(same)[0])
    n_diff = len(serialize_result(diff)[0])
    # distinct meshes add ~4 arrays/frame (coords, x0, y0, ...) => strictly more
    assert n_diff > n_same


def test_save_progress_is_monotonic_to_one():
    res = _make_result(n_frames=6)
    fracs = []
    buf = io.BytesIO()
    save_result_npz(buf, res, progress=fracs.append)
    assert fracs
    assert fracs[-1] == pytest.approx(1.0)
    assert all(0.0 < f <= 1.0 for f in fracs)
    assert fracs == sorted(fracs)          # monotonic non-decreasing
    buf.seek(0)
    _assert_eq(res, load_result_npz(buf))  # still round-trips


# --------------------------------------------------------------------------
# tracking diagnostics (plan D13 / D20)
# --------------------------------------------------------------------------

def _diagnostics(frames=True):
    stats = tuple(
        FrameTrackingStats(
            frame=k, ref_frame=k - 1, n_queried=20, n_extrapolated=3,
            n_extrapolated_far=2, n_extrapolated_border=1,
            max_extrapolation_px=17.25 + k, n_lost_parent_mask=0,
            n_lost_child_mask=k, n_lost_no_data=0, n_tracked=20 - k,
            n_tracked_extrapolated=1,
        )
        for k in (1, 2, 3)
    ) if frames else ()
    extents = (
        RefGridExtent(
            ref_frame=2, roi=GridxyROIRange(gridx=(16, 239), gridy=(8, 199)),
            n_nodes=196,
        ),
    ) if frames else ()
    return TrackingDiagnostics(
        n_nodes=20, n_never_material=0, far_extrapolation_px=16.0,
        measurable_box=(20, 235, 20, 235), has_later_references=True,
        frames=stats, grid_extents=extents,
    )


def _with_diag(diag):
    return replace(_make_result(n_frames=4, accumulative=False), tracking_diagnostics=diag)


@pytest.mark.parametrize("diag", [
    _diagnostics(),
    _diagnostics(frames=False),
    replace(_diagnostics(), far_extrapolation_px=None, measurable_box=None),
    None,
], ids=["populated", "empty_frames", "unclassified", "none"])
def test_tracking_diagnostics_roundtrip(diag, tmp_path):
    res = _with_diag(diag)
    back = deserialize_result(*serialize_result(res))
    _assert_eq(res, back)
    p = tmp_path / "r.npz"
    save_result_npz(str(p), res)
    back = load_result_npz(str(p))
    _assert_eq(res, back)
    if diag is not None:
        assert back.tracking_diagnostics == diag  # tuples, ints, floats only


def test_diagnostics_are_stored_beside_the_root():
    """Builds <= 0.10.0 decode only ``manifest["root"]``, and pass every
    root field to the constructor (a TypeError that ``load_session`` does
    not turn into a ``SessionError`` on an unknown one). So the record sits
    beside the root, and the root never names the field."""
    for diag in (None, _diagnostics()):
        _, manifest = serialize_result(_with_diag(diag))
        assert "tracking_diagnostics" not in manifest["root"]["fields"]
        assert ("tracking_diagnostics" in manifest) == (diag is not None)
    # Required fields are always written, even when None.
    _, manifest = serialize_result(
        _with_diag(replace(_diagnostics(), far_extrapolation_px=None)),
    )
    diag_fields = manifest["tracking_diagnostics"]["fields"]
    assert diag_fields["far_extrapolation_px"] is None


def test_manifest_without_the_field_decodes_to_none():
    arrays, manifest = serialize_result(_make_result())
    manifest["root"]["fields"].pop("tracking_diagnostics", None)
    assert deserialize_result(arrays, manifest).tracking_diagnostics is None
    # A manifest that spells the None out decodes the same way.
    manifest["root"]["fields"]["tracking_diagnostics"] = None
    assert deserialize_result(arrays, manifest).tracking_diagnostics is None


@dataclass(frozen=True)
class _PipelineResultBefore:
    """PipelineResult as builds <= 0.10.0 define it (no diagnostics)."""

    dic_para: DICPara
    dic_mesh: DICMesh
    result_disp: list
    result_def_grad: list
    result_strain: list
    result_fe_mesh_each_frame: list
    frame_schedule: FrameSchedule | None = None
    ref_switch_frames: tuple = ()
    reseed_events: tuple = ()
    stopped_early: bool = False
    stopped_at_frame: int | None = None
    stop_reason: str = ""


def test_old_build_opens_new_sessions_without_the_record(monkeypatch):
    """Decode new manifests the way a build <= 0.10.0 does (its registry,
    ``manifest["root"]`` only): sessions with and without diagnostics both
    open; the record is simply not read."""
    import al_dic.io.session_serialize as ss

    sessions = [serialize_result(_with_diag(d)) for d in (None, _diagnostics())]
    old = {
        k: v for k, v in ss._REGISTRY.items()
        if k not in ("TrackingDiagnostics", "FrameTrackingStats", "RefGridExtent")
    }
    old["PipelineResult"] = _PipelineResultBefore
    monkeypatch.setattr(ss, "_REGISTRY", old)

    for arrays, manifest in sessions:
        assert manifest["serialize_version"] == 1
        loaded = ss._decode(manifest["root"], arrays)
        assert isinstance(loaded, _PipelineResultBefore)
        assert len(loaded.result_disp) == 3


def test_record_from_a_newer_build_is_a_value_error():
    """A field this build does not know (a newer build added it) is the
    ValueError that ``load_session`` reports as a session error."""
    arrays, manifest = serialize_result(_with_diag(_diagnostics()))
    stats = manifest["tracking_diagnostics"]["fields"]["frames"]["__tuple__"][0]
    stats["fields"]["max_in_region_px"] = 3.0
    with pytest.raises(ValueError, match="max_in_region_px"):
        deserialize_result(arrays, manifest)


def test_diagnostics_of_another_type_are_rejected():
    arrays, manifest = serialize_result(_with_diag(_diagnostics()))
    manifest["tracking_diagnostics"] = {"__tuple__": [1, 2]}
    with pytest.raises(ValueError, match="TrackingDiagnostics"):
        deserialize_result(arrays, manifest)


def test_load_rejects_pickle(tmp_path):
    """Loader must refuse pickled object arrays (allow_pickle=False)."""
    p = tmp_path / "evil.npz"
    np.savez(str(p), __manifest__=np.frombuffer(b'{"serialize_version":1,"root":null}', np.uint8),
             obj=np.array([{"x": 1}], dtype=object))
    with pytest.raises(ValueError):
        load_result_npz(str(p))
