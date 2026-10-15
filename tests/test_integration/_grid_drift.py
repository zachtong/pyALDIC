"""Synthetic drift sequence shared by the incremental grid-coverage tests.

A speckled specimen on a flat background rotates by 1 degree about its
center and translates by (28, 2) px per frame. Every frame has a mask that
follows the specimen (Lagrangian warp of the frame-0 indicator, > 0.5).
With the grid built once from the frame-0 range, the later reference
frames lose the part of the specimen that left that range.

Used by ``test_incremental_grid_coverage.py`` (core) and
``tests/test_gui/test_pipeline_grid_coverage.py`` (GUI path), so both
check the same data against the same thresholds (plan step 2.1).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from al_dic.core.data_structures import GridxyROIRange, PipelineResult

from tests.conftest import apply_displacement_lagrangian, generate_speckle

H, W = 256, 384
X0, X1, Y0, Y1 = 40, 152, 72, 184  # frame-0 specimen: [X0, X1) x [Y0, Y1)
N_DEF = 5
DX, DY, DTH = 28.0, 2.0, math.radians(1.0)
WIN, STEP, SEARCH, BG = 32, 16, 32, 128.0
HW = WIN // 2
CX, CY = (X0 + X1 - 1) / 2.0, (Y0 + Y1 - 1) / 2.0

# Accuracy thresholds, checked at every frame.
INTERIOR_MAX_PX = 0.25
INTERIOR_MEDIAN_PX = 0.05
MATERIAL_DEAD_MAX = 0.10
MATERIAL_MEDIAN_PX = 0.15

_Field = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


def motion(k: int) -> tuple[_Field, _Field]:
    """Displacement (u, v) of frame ``k`` at frame-0 coordinates."""
    th = k * DTH
    c, s = math.cos(th), math.sin(th)
    tx, ty = k * DX, k * DY

    def u(x, y):
        return CX + tx + c * (x - CX) - s * (y - CY) - x

    def v(x, y):
        return CY + ty + s * (x - CX) + c * (y - CY) - y

    return u, v


def drift_data() -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Images (gray levels 20-235, background 128) and per-frame masks."""
    tex = generate_speckle(H, W, sigma=3.0, seed=11)
    ind0 = np.zeros((H, W))
    ind0[Y0:Y1, X0:X1] = 1.0
    ref = np.where(ind0 > 0.5, tex, BG)
    imgs, masks = [ref], [ind0]
    for k in range(1, N_DEF + 1):
        u, v = motion(k)
        imgs.append(apply_displacement_lagrangian(ref, u, v))
        warped = apply_displacement_lagrangian(ind0, u, v)
        masks.append((warped > 0.5).astype(np.float64))
    return imgs, masks


def bbox(mask: NDArray) -> tuple[int, int, int, int]:
    """Inclusive material bbox ``(x_lo, x_hi, y_lo, y_hi)`` of ``mask > 0.5``."""
    rows, cols = np.nonzero(np.asarray(mask) > 0.5)
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())


def range_of(*masks: NDArray) -> GridxyROIRange:
    """Grid range = material bbox of ``masks`` (what the GUI builds)."""
    boxes = [bbox(m) for m in masks]
    return GridxyROIRange(
        gridx=(min(b[0] for b in boxes), max(b[1] for b in boxes)),
        gridy=(min(b[2] for b in boxes), max(b[3] for b in boxes)),
    )


def node_sets(result: PipelineResult) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """``(material, interior)`` nodes of the canonical (frame-0) mesh.

    Interior nodes are at least half a subset (16 px) from the specimen edge.
    """
    x, y = result.dic_mesh.coordinates_fem.T
    material = (x >= X0) & (x <= X1 - 1) & (y >= Y0) & (y <= Y1 - 1)
    edge = np.minimum.reduce([x - X0, X1 - 1 - x, y - Y0, Y1 - 1 - y])
    return material, material & (edge >= HW)


def frame_error(result: PipelineResult, k: int) -> NDArray[np.float64]:
    """``|U_accum - truth|`` (px) of frame ``k`` at every canonical node."""
    x, y = result.dic_mesh.coordinates_fem.T
    ua = result.result_disp[k - 1].U_accum
    u, v = motion(k)
    return np.hypot(ua[0::2] - u(x, y), ua[1::2] - v(x, y))


def interior_median(result: PipelineResult, k: int) -> float:
    """Median error of frame ``k`` over the interior nodes (NaN if any is NaN)."""
    _, interior = node_sets(result)
    return float(np.median(frame_error(result, k)[interior]))


def assert_frame_accuracy(result: PipelineResult, k: int) -> None:
    """Interior: no NaN, max and median bounds; material: dead and median."""
    material, interior = node_sets(result)
    assert interior.sum() >= 9, "fixture too small to judge accuracy"
    err = frame_error(result, k)
    e_in = err[interior]
    assert np.isfinite(e_in).all(), (
        f"frame {k}: {int((~np.isfinite(e_in)).sum())} interior node(s) NaN"
    )
    assert e_in.max() < INTERIOR_MAX_PX, f"frame {k}: interior max {e_in.max():.3f} px"
    assert np.median(e_in) < INTERIOR_MEDIAN_PX, (
        f"frame {k}: interior median {np.median(e_in):.4f} px"
    )
    e_mat = err[material]
    alive = np.isfinite(e_mat)
    dead = 1.0 - alive.mean()
    assert dead <= MATERIAL_DEAD_MAX, f"frame {k}: {dead:.1%} of material dead"
    assert np.median(e_mat[alive]) < MATERIAL_MEDIAN_PX, (
        f"frame {k}: material median {np.median(e_mat[alive]):.4f} px"
    )
