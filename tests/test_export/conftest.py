"""Shared fixtures for export tests."""

import numpy as np
import pytest

from al_dic.core.data_structures import (
    DICMesh,
    FrameResult,
    FrameSchedule,
    PipelineResult,
    StrainResult,
)
from al_dic.core.config import dicpara_default


@pytest.fixture
def minimal_result():
    """Minimal PipelineResult with 2 frames and 5 nodes for testing."""
    # Use a 3x4 grid of nodes so the geometry is 2D (avoids collinear Delaunay failures)
    xs, ys = np.meshgrid(np.linspace(4, 60, 4), np.linspace(4, 60, 3))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    N = coords.shape[0]  # 12 nodes

    mesh = DICMesh(
        coordinates_fem=coords,
        elements_fem=np.zeros((0, 8), dtype=np.int64),
    )

    U = np.ones(2 * N, dtype=np.float64)
    U_accum = np.ones(2 * N, dtype=np.float64) * 2.0
    fr = FrameResult(U=U, U_accum=U_accum)

    sr = StrainResult(
        disp_u=np.ones(N, dtype=np.float64),
        disp_v=np.zeros(N, dtype=np.float64),
        strain_exx=np.full(N, 0.01, dtype=np.float64),
        strain_eyy=np.full(N, -0.005, dtype=np.float64),
        strain_exy=np.zeros(N, dtype=np.float64),
        strain_principal_max=np.full(N, 0.01, dtype=np.float64),
        strain_principal_min=np.full(N, -0.005, dtype=np.float64),
        strain_maxshear=np.full(N, 0.0075, dtype=np.float64),
        strain_von_mises=np.full(N, 0.013, dtype=np.float64),
        strain_rotation=np.zeros(N, dtype=np.float64),
    )

    para = dicpara_default(img_size=(64, 64))
    return PipelineResult(
        dic_para=para,
        dic_mesh=mesh,
        result_disp=[fr, fr],
        result_def_grad=[fr, fr],
        result_strain=[sr, sr],
        result_fe_mesh_each_frame=[mesh, mesh],
        frame_schedule=FrameSchedule.from_mode("accumulative", 3),
    )


@pytest.fixture
def minimal_result_no_strain(minimal_result):
    """PipelineResult with no strain results."""
    from dataclasses import replace
    return replace(minimal_result, result_strain=[])
