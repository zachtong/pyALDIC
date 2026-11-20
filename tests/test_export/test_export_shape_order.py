"""Export of the per-node shape-function order (NPZ and MAT)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.io

from al_dic.export.export_mat import export_mat
from al_dic.export.export_npz import export_npz
from al_dic.export.export_utils import shape_order_matrix

FIELDS = ["disp_u", "disp_v"]


def _with_order(result, orders):
    return replace(result, result_disp=[
        replace(fr, shape_order=o) for fr, o in zip(result.result_disp, orders)
    ])


def _orders(result):
    n = result.dic_mesh.coordinates_fem.shape[0]
    return [(np.arange(n) % 3).astype(np.int8), np.full(n, 2, dtype=np.int8)]


def test_matrix_is_none_when_the_stage_never_ran(minimal_result):
    assert shape_order_matrix(minimal_result) is None


def test_matrix_has_one_column_per_frame(minimal_result):
    orders = _orders(minimal_result)
    mat = shape_order_matrix(_with_order(minimal_result, orders))
    assert mat.dtype == np.int8
    assert mat.shape == (orders[0].size, 2)
    np.testing.assert_array_equal(mat[:, 0], orders[0])
    np.testing.assert_array_equal(mat[:, 1], orders[1])


def test_matrix_is_none_when_a_frame_lives_on_another_mesh(minimal_result):
    # Incremental runs can give a frame pair its own mesh; its per-node array
    # then has no meaning on the exported (frame-0) node list.
    orders = _orders(minimal_result)
    orders[1] = np.ones(orders[1].size + 3, dtype=np.int8)
    assert shape_order_matrix(_with_order(minimal_result, orders)) is None


def test_npz_has_no_shape_order_by_default(tmp_path, minimal_result):
    out = export_npz(tmp_path, "t", "00000000000000", minimal_result, FIELDS)
    assert "shape_order" not in np.load(out).files


def test_npz_single_file_and_per_frame(tmp_path, minimal_result):
    orders = _orders(minimal_result)
    result = _with_order(minimal_result, orders)

    single = np.load(export_npz(tmp_path, "t", "00000000000000", result, FIELDS))
    np.testing.assert_array_equal(single["shape_order"][:, 0], orders[0])

    paths = export_npz(tmp_path, "t", "00000000000001", result, FIELDS, per_frame=True)
    for path, order in zip(paths, orders):
        np.testing.assert_array_equal(np.load(path)["shape_order"], order)


def test_mat_carries_shape_order_only_when_present(tmp_path, minimal_result):
    plain = scipy.io.loadmat(
        str(export_mat(tmp_path, "a", "00000000000000", minimal_result, FIELDS)))
    assert "shape_order" not in plain

    orders = _orders(minimal_result)
    staged = scipy.io.loadmat(str(export_mat(
        tmp_path, "b", "00000000000000", _with_order(minimal_result, orders), FIELDS)))
    np.testing.assert_array_equal(staged["shape_order"][:, 1], orders[1])
