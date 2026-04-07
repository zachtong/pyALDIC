"""Shared GUI test helpers.

Builders for synthetic PipelineResult / AppState wiring used by Task 1, 6,
and 8 of the strain post-processing window plan.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import (
    DICMesh,
    DICPara,
    FrameResult,
    GridxyROIRange,
    PipelineResult,
)


def make_uniform_q4_mesh(
    h: int, w: int, step: int, margin: int | None = None,
) -> DICMesh:
    """Build a uniform Q4 mesh covering the image interior."""
    if margin is None:
        margin = step
    xs = np.arange(margin, w - margin + 1, step, dtype=np.float64)
    ys = np.arange(margin, h - margin + 1, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)

    ny, nx = len(ys), len(xs)
    elements: list[list[int]] = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            n0 = iy * nx + ix
            n1 = n0 + 1
            n2 = n0 + nx + 1
            n3 = n0 + nx
            elements.append([n0, n1, n2, n3, -1, -1, -1, -1])
    elems = (
        np.array(elements, dtype=np.int64)
        if elements else np.empty((0, 8), dtype=np.int64)
    )
    return DICMesh(
        coordinates_fem=coords,
        elements_fem=elems,
        x0=xs,
        y0=ys,
    )


def make_synthetic_pipeline_result(
    n_frames: int = 3,
    shear: float = 0.01,
    img_shape: tuple[int, int] = (256, 256),
    step: int = 16,
) -> tuple[PipelineResult, NDArray[np.float64]]:
    """Build a synthetic PipelineResult with uniform shear deformation.

    Generates a pipeline result whose ``result_disp`` corresponds to a
    pure-shear field ``u = shear * y, v = 0`` applied per-frame so that
    cumulative displacement at frame ``i`` (``i >= 1``) equals
    ``i * shear * y``.

    Args:
        n_frames: Total number of frames including the reference (frame 0).
        shear: Per-frame shear coefficient. Frame ``i`` has cumulative
            ``u_accum = i * shear * y``.
        img_shape: ``(H, W)`` of the synthetic images.
        step: Mesh node spacing in pixels.

    Returns:
        ``(PipelineResult, mask)`` where ``mask`` is the float64 ROI mask
        used to build the strain region map.
    """
    h, w = img_shape
    mesh = make_uniform_q4_mesh(h, w, step)
    n_nodes = mesh.coordinates_fem.shape[0]

    # ROI mask: full-image rectangle
    mask = np.ones((h, w), dtype=np.float64)

    # Per-frame cumulative displacement: u = i * shear * y, v = 0
    result_disp: list[FrameResult] = []
    y = mesh.coordinates_fem[:, 1]
    for i in range(n_frames - 1):
        scale = (i + 1) * shear
        u_inc = scale * y
        v_inc = np.zeros(n_nodes)
        U_inc = np.empty(2 * n_nodes, dtype=np.float64)
        U_inc[0::2] = u_inc
        U_inc[1::2] = v_inc
        result_disp.append(
            FrameResult(U=U_inc.copy(), U_accum=U_inc.copy(), F=None)
        )

    # Build a default DICPara compatible with the helpers
    para: DICPara = dicpara_default(
        winsize=32,
        winstepsize=step,
        winsize_min=8,
        img_size=img_shape,
        gridxy_roi_range=GridxyROIRange(gridx=(0, w - 1), gridy=(0, h - 1)),
        reference_mode="accumulative",
        show_plots=False,
    )
    para = replace(
        para,
        method_to_compute_strain=2,
        strain_plane_fit_rad=20.0,
        strain_smoothness=1e-5,
        strain_type=0,
        img_ref_mask=mask,
    )

    result = PipelineResult(
        dic_para=para,
        dic_mesh=mesh,
        result_disp=result_disp,
        result_def_grad=[],
        result_strain=[],
        result_fe_mesh_each_frame=[mesh] * (n_frames - 1),
    )
    return result, mask
