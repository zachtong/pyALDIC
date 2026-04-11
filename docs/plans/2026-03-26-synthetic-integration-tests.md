# Phase 3.1: Synthetic Integration Tests

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port all 10 MATLAB synthetic test cases to Python, validating the AL-DIC pipeline against known displacement/strain fields.

**Architecture:** Each test case generates a 256x256 speckle image, applies a known displacement via Fourier/inverse warp, builds a mesh with ground-truth U0, runs `run_aldic()`, and validates displacement RMSE and strain RMSE against tolerances matching the MATLAB test suite.

**Tech Stack:** pytest (parametrize), NumPy, SciPy (ndimage), conftest fixtures (`generate_speckle`, `apply_displacement`)

---

## Critical Context

### Why U0 = ground truth at nodes

Since `integer_search.py` and `init_disp.py` are stubs (FFT initial guess is deferred), all tests **must** provide `mesh` and `U0` directly to `run_aldic()`. We provide ground-truth displacement at mesh nodes as U0. This bypasses FFT but still exercises:
- Local ICGN (6-DOF refinement from near-exact guess)
- ADMM Subproblem 1 (2-DOF ICGN) + Subproblem 2 (FEM global solve)
- Post-solve corrections, cumulative displacement transform, strain computation

### MATLAB-to-Python coordinate mapping

| Concept | MATLAB | Python |
|---------|--------|--------|
| Image layout | `[W x H]` after transpose | `[H x W]` standard |
| Pixel indices | 1-based (1..256) | 0-based (0..255) |
| Image center | 128 | 127 |
| Displacement | `u = 0.02*(x - 128)` | `u = 0.02*(x - 127)` |
| Mask center | (128, 128) | (127, 127) |
| Interleaved U | `[u1,v1,...,uN,vN]` | `[u0,v0,...,uN,vN]` (same layout) |
| Interleaved F | `[F11,F21,F12,F22,...]` | Same layout |

### Tolerances (from MATLAB)

| Category | disp RMSE | strain RMSE |
|----------|-----------|-------------|
| Standard | < 0.5 px | < 0.03 |
| Large deform (case6) | < 1.0 px | < 0.05 |

---

## Test Cases (ported from `test_aldic_synthetic.m`)

| # | Name | u(x,y) | v(x,y) | GT F11 | GT F22 | GT F12 | GT F21 | Mask | Notes |
|---|------|--------|--------|--------|--------|--------|--------|------|-------|
| 1 | zero | 0 | 0 | 0 | 0 | 0 | 0 | solid | Baseline |
| 2 | translation | 2.5 | -1.8 | 0 | 0 | 0 | 0 | solid | Constant shift |
| 3 | affine | 0.02*(x-127) | 0.02*(y-127) | 0.02 | 0.02 | 0 | 0 | solid | Uniform expansion |
| 4 | annular | 0.02*(x-127) | 0.02*(y-127) | 0.02 | 0.02 | 0 | 0 | annular | Ring mask |
| 5 | shear | 0.015*(y-127) | 0 | 0 | 0 | 0.015 | 0 | solid | Pure shear |
| 6 | large_deform | 0.10*(x-127)+0.05*(y-127) | 0.05*(x-127)+0.10*(y-127) | 0.10 | 0.10 | 0.05 | 0.05 | solid | Wide tol |
| 7 | multiframe_incr | u=1.0/frame | v=0 | 0 | 0 | 0 | 0 | solid | Incremental mode |
| 8 | multiframe_accum | u=1.0/frame | v=0 | 0 | 0 | 0 | 0 | solid | Accumulative mode |
| 9 | local_only | 0.02*(x-127) | 0.02*(y-127) | 0.02 | 0.02 | 0 | 0 | solid | UseGlobalStep=false |
| 10 | rotation | rotation(2deg) | rotation(2deg) | cos(pi/90)-1 | cos(pi/90)-1 | -sin(pi/90) | sin(pi/90) | solid | Rigid rotation |

---

## Task 1: Add shared synthetic test helpers to conftest.py

**Files:**
- Modify: `tests/conftest.py`

### Step 1: Write the helper functions

Add to `tests/conftest.py`:

```python
def make_mesh_for_image(
    h: int = 256,
    w: int = 256,
    step: int = 16,
    margin: int | None = None,
) -> DICMesh:
    """Create a regular Q4 mesh covering the image interior.

    Args:
        h, w: Image height/width.
        step: Node spacing (must be power of 2).
        margin: Margin from image edges. Defaults to step.

    Returns:
        DICMesh with coordinates and Q4 elements.
    """
    if margin is None:
        margin = step
    xs = np.arange(margin, w - margin + 1, step, dtype=np.float64)
    ys = np.arange(margin, h - margin + 1, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)

    ny, nx = len(ys), len(xs)
    elements = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            n0 = iy * nx + ix
            n1 = n0 + 1
            n2 = n0 + nx + 1
            n3 = n0 + nx
            elements.append([n0, n1, n2, n3, -1, -1, -1, -1])

    if not elements:
        elems = np.empty((0, 8), dtype=np.int64)
    else:
        elems = np.array(elements, dtype=np.int64)

    return DICMesh(
        coordinates_fem=coords,
        elements_fem=elems,
        x0=xs,
        y0=ys,
    )


def make_circular_mask(
    h: int = 256,
    w: int = 256,
    cx: float = 127.0,
    cy: float = 127.0,
    radius: float = 90.0,
) -> np.ndarray:
    """Create a circular binary mask.

    Returns:
        (H, W) float64 array with 1.0 inside circle, 0.0 outside.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (dist2 <= radius ** 2).astype(np.float64)


def make_annular_mask(
    h: int = 256,
    w: int = 256,
    cx: float = 127.0,
    cy: float = 127.0,
    r_outer: float = 90.0,
    r_inner: float = 40.0,
) -> np.ndarray:
    """Create an annular (ring) binary mask."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return ((dist2 <= r_outer ** 2) & (dist2 > r_inner ** 2)).astype(np.float64)


def compute_disp_rmse(
    U: np.ndarray,
    coords: np.ndarray,
    gt_u: np.ndarray,
    gt_v: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Compute displacement RMSE on mask-interior nodes.

    Args:
        U: Interleaved displacement (2*n_nodes,).
        coords: Node coordinates (n_nodes, 2), col0=x, col1=y.
        gt_u, gt_v: Ground truth at each node (n_nodes,).
        mask: Binary mask image (H, W).

    Returns:
        (rmse_u, rmse_v).
    """
    u_comp = U[0::2]
    v_comp = U[1::2]

    # Find nodes inside mask
    h, w = mask.shape
    cx = np.clip(np.round(coords[:, 0]).astype(int), 0, w - 1)
    cy = np.clip(np.round(coords[:, 1]).astype(int), 0, h - 1)
    in_mask = mask[cy, cx] > 0.5

    valid = in_mask & np.isfinite(u_comp) & np.isfinite(v_comp)

    if not np.any(valid):
        return np.inf, np.inf

    err_u = u_comp[valid] - gt_u[valid]
    err_v = v_comp[valid] - gt_v[valid]

    return float(np.sqrt(np.mean(err_u ** 2))), float(np.sqrt(np.mean(err_v ** 2)))


def compute_strain_rmse(
    strain_result,
    gt_F11: float,
    gt_F21: float,
    gt_F12: float,
    gt_F22: float,
    coords: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute strain RMSE on mask-interior nodes.

    Args:
        strain_result: StrainResult from pipeline.
        gt_*: Scalar ground truth for each gradient component.
        coords: Node coordinates (n_nodes, 2).
        mask: Binary mask image (H, W).

    Returns:
        Dict with keys 'rmse_F11', 'rmse_F21', 'rmse_F12', 'rmse_F22'.
    """
    h, w = mask.shape
    cx = np.clip(np.round(coords[:, 0]).astype(int), 0, w - 1)
    cy = np.clip(np.round(coords[:, 1]).astype(int), 0, h - 1)
    in_mask = mask[cy, cx] > 0.5
    valid = in_mask & np.isfinite(strain_result.dudx)

    n = len(coords)
    gt = {
        'F11': np.full(n, gt_F11),
        'F21': np.full(n, gt_F21),
        'F12': np.full(n, gt_F12),
        'F22': np.full(n, gt_F22),
    }
    computed = {
        'F11': strain_result.dudx,
        'F21': strain_result.dvdx,
        'F12': strain_result.dudy,
        'F22': strain_result.dvdy,
    }

    result = {}
    for key in ('F11', 'F21', 'F12', 'F22'):
        err = computed[key][valid] - gt[key][valid]
        result[f'rmse_{key}'] = float(np.sqrt(np.mean(err ** 2)))

    return result
```

### Step 2: Run existing tests to verify no regressions

Run: `cd al-dic && python -m pytest tests/ -x -q`
Expected: All existing tests PASS.

### Step 3: Commit

```bash
git add tests/conftest.py
git commit -m "test: add synthetic test helpers (mesh builder, masks, RMSE)"
```

---

## Task 2: Create test_synthetic.py with case definitions

**Files:**
- Create: `tests/test_integration/test_synthetic.py`

### Step 1: Write the test file with case infrastructure

```python
"""Synthetic integration tests for the AL-DIC pipeline.

Ports all 10 test cases from MATLAB test_aldic_synthetic.m.
Validates displacement and strain RMSE against known ground truth.
"""

import numpy as np
import pytest
from dataclasses import replace

from al_dic.core.data_structures import DICPara, DICMesh, GridxyROIRange, merge_uv
from al_dic.core.config import dicpara_default
from al_dic.core.pipeline import run_aldic

from tests.conftest import (
    generate_speckle,
    apply_displacement,
    make_mesh_for_image,
    make_circular_mask,
    make_annular_mask,
    compute_disp_rmse,
    compute_strain_rmse,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H, W = 256, 256
CX, CY = 127.0, 127.0  # Image center (0-based)
STEP = 16
MARGIN = 16


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

def _case_para(**overrides) -> DICPara:
    """Build DICPara for 256x256 synthetic tests."""
    defaults = dict(
        winsize=32,
        winstepsize=STEP,
        winsize_min=8,
        img_size=(H, W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, W - 1), gridy=(0, H - 1)),
        reference_mode="accumulative",
        admm_max_iter=3,
        admm_tol=1e-2,
        method_to_compute_strain=3,
        strain_smoothness=0.0,
        disp_smoothness=0.0,
        smoothness=0.0,
        show_plots=False,
        icgn_max_iter=50,
        tol=1e-2,
        mu=1e-3,
        gauss_pt_order=2,
        alpha=0.0,
    )
    defaults.update(overrides)
    return dicpara_default(**defaults)


CASES = {
    "case1_zero": dict(
        u_func=lambda x, y: np.zeros_like(x),
        v_func=lambda x, y: np.zeros_like(x),
        u3_func=lambda x, y: np.zeros_like(x),
        v3_func=lambda x, y: np.zeros_like(x),
        gt_F11=0, gt_F22=0, gt_F12=0, gt_F21=0,
        mask_type="solid",
        para_overrides={},
        disp_tol=0.5, strain_tol=0.03,
    ),
    "case2_translation": dict(
        u_func=lambda x, y: 2.5 * np.ones_like(x),
        v_func=lambda x, y: -1.8 * np.ones_like(x),
        u3_func=lambda x, y: 5.0 * np.ones_like(x),
        v3_func=lambda x, y: -3.6 * np.ones_like(x),
        gt_F11=0, gt_F22=0, gt_F12=0, gt_F21=0,
        mask_type="solid",
        para_overrides={},
        disp_tol=0.5, strain_tol=0.03,
    ),
    "case3_affine": dict(
        u_func=lambda x, y: 0.02 * (x - CX),
        v_func=lambda x, y: 0.02 * (y - CY),
        u3_func=lambda x, y: 0.04 * (x - CX),
        v3_func=lambda x, y: 0.04 * (y - CY),
        gt_F11=0.02, gt_F22=0.02, gt_F12=0, gt_F21=0,
        mask_type="solid",
        para_overrides={},
        disp_tol=0.5, strain_tol=0.03,
    ),
    "case4_annular": dict(
        u_func=lambda x, y: 0.02 * (x - CX),
        v_func=lambda x, y: 0.02 * (y - CY),
        u3_func=lambda x, y: 0.04 * (x - CX),
        v3_func=lambda x, y: 0.04 * (y - CY),
        gt_F11=0.02, gt_F22=0.02, gt_F12=0, gt_F21=0,
        mask_type="annular",
        para_overrides={"winsize_min": 4},
        disp_tol=0.5, strain_tol=0.03,
    ),
    "case5_shear": dict(
        u_func=lambda x, y: 0.015 * (y - CY),
        v_func=lambda x, y: np.zeros_like(x),
        u3_func=lambda x, y: 0.030 * (y - CY),
        v3_func=lambda x, y: np.zeros_like(x),
        gt_F11=0, gt_F22=0, gt_F12=0.015, gt_F21=0,
        mask_type="solid",
        para_overrides={},
        disp_tol=0.5, strain_tol=0.03,
    ),
    "case6_large_deform": dict(
        u_func=lambda x, y: 0.10 * (x - CX) + 0.05 * (y - CY),
        v_func=lambda x, y: 0.05 * (x - CX) + 0.10 * (y - CY),
        u3_func=lambda x, y: 0.20 * (x - CX) + 0.10 * (y - CY),
        v3_func=lambda x, y: 0.10 * (x - CX) + 0.20 * (y - CY),
        gt_F11=0.10, gt_F22=0.10, gt_F12=0.05, gt_F21=0.05,
        mask_type="solid",
        para_overrides={"winsize": 48},
        disp_tol=1.0, strain_tol=0.05,
    ),
    "case7_multiframe_incr": dict(
        u_func=lambda x, y: 1.0 * np.ones_like(x),
        v_func=lambda x, y: np.zeros_like(x),
        u3_func=lambda x, y: 2.0 * np.ones_like(x),
        v3_func=lambda x, y: np.zeros_like(x),
        gt_F11=0, gt_F22=0, gt_F12=0, gt_F21=0,
        mask_type="solid",
        para_overrides={"reference_mode": "incremental"},
        disp_tol=0.5, strain_tol=0.03,
    ),
    "case8_multiframe_accum": dict(
        u_func=lambda x, y: 1.0 * np.ones_like(x),
        v_func=lambda x, y: np.zeros_like(x),
        u3_func=lambda x, y: 2.0 * np.ones_like(x),
        v3_func=lambda x, y: np.zeros_like(x),
        gt_F11=0, gt_F22=0, gt_F12=0, gt_F21=0,
        mask_type="solid",
        para_overrides={},
        disp_tol=0.5, strain_tol=0.03,
    ),
    "case9_local_only": dict(
        u_func=lambda x, y: 0.02 * (x - CX),
        v_func=lambda x, y: 0.02 * (y - CY),
        u3_func=lambda x, y: 0.04 * (x - CX),
        v3_func=lambda x, y: 0.04 * (y - CY),
        gt_F11=0.02, gt_F22=0.02, gt_F12=0, gt_F21=0,
        mask_type="solid",
        para_overrides={"use_global_step": False},
        disp_tol=0.5, strain_tol=0.03,
    ),
    "case10_rotation": dict(
        u_func=lambda x, y: (x - CX) * (np.cos(np.pi / 90) - 1) - (y - CY) * np.sin(np.pi / 90),
        v_func=lambda x, y: (x - CX) * np.sin(np.pi / 90) + (y - CY) * (np.cos(np.pi / 90) - 1),
        u3_func=lambda x, y: (x - CX) * (np.cos(np.pi / 45) - 1) - (y - CY) * np.sin(np.pi / 45),
        v3_func=lambda x, y: (x - CX) * np.sin(np.pi / 45) + (y - CY) * (np.cos(np.pi / 45) - 1),
        gt_F11=np.cos(np.pi / 90) - 1,
        gt_F22=np.cos(np.pi / 90) - 1,
        gt_F12=-np.sin(np.pi / 90),
        gt_F21=np.sin(np.pi / 90),
        mask_type="solid",
        para_overrides={},
        disp_tol=0.5, strain_tol=0.03,
    ),
}
```

### Step 2: Verify imports work

Run: `cd al-dic && python -c "from tests.test_integration.test_synthetic import CASES; print(f'{len(CASES)} cases defined')"`
Expected: `10 cases defined`

### Step 3: Commit

```bash
git add tests/test_integration/test_synthetic.py
git commit -m "test: add synthetic case definitions (10 cases from MATLAB)"
```

---

## Task 3: Implement displacement-only tests (cases 1-5, 9)

**Files:**
- Modify: `tests/test_integration/test_synthetic.py`

### Step 1: Write the parametrized displacement test

Add to `test_synthetic.py`:

```python
# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ref_speckle():
    """Module-scoped reference speckle image."""
    return generate_speckle(H, W, sigma=3.0, seed=42)


def _build_test_data(case_name: str, ref: np.ndarray):
    """Build images, masks, mesh, U0 for a given test case."""
    case = CASES[case_name]
    para = _case_para(**case["para_overrides"])

    # Build mask
    if case["mask_type"] == "annular":
        mask = make_annular_mask(H, W, CX, CY, 90.0, 40.0)
    else:
        mask = make_circular_mask(H, W, CX, CY, 90.0)

    # Build displacement fields on full image grid
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    u2_field = case["u_func"](xx, yy)
    v2_field = case["v_func"](xx, yy)
    u3_field = case["u3_func"](xx, yy)
    v3_field = case["v3_func"](xx, yy)

    # Generate deformed images
    def2 = apply_displacement(ref, u2_field, v2_field)
    def3 = apply_displacement(ref, u3_field, v3_field)

    images = [ref, def2, def3]
    masks = [mask, mask, mask]

    # Build mesh
    mesh = make_mesh_for_image(H, W, step=STEP, margin=MARGIN)
    n_nodes = mesh.coordinates_fem.shape[0]

    # Ground truth U0 at mesh nodes
    node_x = mesh.coordinates_fem[:, 0]
    node_y = mesh.coordinates_fem[:, 1]
    gt_u2 = case["u_func"](node_x, node_y)
    gt_v2 = case["v_func"](node_x, node_y)
    U0 = merge_uv(gt_u2, gt_v2)

    # Ground truth for frame 3
    gt_u3 = case["u3_func"](node_x, node_y)
    gt_v3 = case["v3_func"](node_x, node_y)

    return {
        "para": para, "images": images, "masks": masks,
        "mesh": mesh, "U0": U0, "mask": mask,
        "gt_u2": gt_u2, "gt_v2": gt_v2,
        "gt_u3": gt_u3, "gt_v3": gt_v3,
        "case": case,
    }


# ---------------------------------------------------------------------------
# Displacement tests
# ---------------------------------------------------------------------------

SINGLE_FRAME_CASES = [
    "case1_zero", "case2_translation", "case3_affine",
    "case4_annular", "case5_shear", "case9_local_only",
]

@pytest.mark.parametrize("case_name", SINGLE_FRAME_CASES)
class TestSyntheticDisplacement:
    """Displacement RMSE validation for standard cases."""

    def test_frame2_displacement(self, ref_speckle, case_name):
        """Frame 2 displacement RMSE should be within tolerance."""
        data = _build_test_data(case_name, ref_speckle)
        case = data["case"]

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        assert len(result.result_disp) >= 1
        U_accum = result.result_disp[0].U_accum
        if U_accum is None:
            U_accum = result.result_disp[0].U

        rmse_u, rmse_v = compute_disp_rmse(
            U_accum, data["mesh"].coordinates_fem,
            data["gt_u2"], data["gt_v2"], data["mask"],
        )

        assert rmse_u < case["disp_tol"], (
            f"{case_name} frame2 RMSE_u={rmse_u:.4f} > tol={case['disp_tol']}"
        )
        assert rmse_v < case["disp_tol"], (
            f"{case_name} frame2 RMSE_v={rmse_v:.4f} > tol={case['disp_tol']}"
        )

    def test_frame3_displacement(self, ref_speckle, case_name):
        """Frame 3 cumulative displacement RMSE should be within tolerance."""
        data = _build_test_data(case_name, ref_speckle)
        case = data["case"]

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        assert len(result.result_disp) >= 2
        U_accum = result.result_disp[1].U_accum
        if U_accum is None:
            U_accum = result.result_disp[1].U

        rmse_u, rmse_v = compute_disp_rmse(
            U_accum, data["mesh"].coordinates_fem,
            data["gt_u3"], data["gt_v3"], data["mask"],
        )

        assert rmse_u < case["disp_tol"], (
            f"{case_name} frame3 RMSE_u={rmse_u:.4f} > tol={case['disp_tol']}"
        )
        assert rmse_v < case["disp_tol"], (
            f"{case_name} frame3 RMSE_v={rmse_v:.4f} > tol={case['disp_tol']}"
        )
```

### Step 2: Run the displacement tests

Run: `cd al-dic && python -m pytest tests/test_integration/test_synthetic.py::TestSyntheticDisplacement -v --timeout=300`
Expected: All 12 tests PASS (6 cases x 2 frames). May be slow (~30s per case).

### Step 3: Commit

```bash
git add tests/test_integration/test_synthetic.py
git commit -m "test: add displacement RMSE tests for cases 1-5, 9"
```

---

## Task 4: Implement strain tests (cases 3, 5, 6, 9, 10)

**Files:**
- Modify: `tests/test_integration/test_synthetic.py`

### Step 1: Write the strain validation tests

Add to `test_synthetic.py`:

```python
STRAIN_CASES = [
    "case3_affine", "case5_shear", "case6_large_deform",
    "case9_local_only", "case10_rotation",
]

@pytest.mark.parametrize("case_name", STRAIN_CASES)
class TestSyntheticStrain:
    """Strain RMSE validation for cases with nonzero ground truth."""

    def test_strain_rmse(self, ref_speckle, case_name):
        """Frame 2 strain RMSE should be within tolerance."""
        data = _build_test_data(case_name, ref_speckle)
        case = data["case"]

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=True,
        )

        assert len(result.result_strain) >= 1
        sr = result.result_strain[0]
        assert sr.dudx is not None

        coords = data["mesh"].coordinates_fem
        rmses = compute_strain_rmse(
            sr,
            case["gt_F11"], case["gt_F21"],
            case["gt_F12"], case["gt_F22"],
            coords, data["mask"],
        )

        tol = case["strain_tol"]
        for key, val in rmses.items():
            assert val < tol, (
                f"{case_name} {key}={val:.5f} > tol={tol}"
            )
```

### Step 2: Run the strain tests

Run: `cd al-dic && python -m pytest tests/test_integration/test_synthetic.py::TestSyntheticStrain -v --timeout=300`
Expected: All 5 tests PASS.

### Step 3: Commit

```bash
git add tests/test_integration/test_synthetic.py
git commit -m "test: add strain RMSE tests for cases 3, 5, 6, 9, 10"
```

---

## Task 5: Implement large deformation test (case 6)

**Files:**
- Modify: `tests/test_integration/test_synthetic.py`

### Step 1: Write case6 displacement test

Case 6 has wider tolerances and uses `winsize=48`. It's already included in strain tests but needs its own displacement test since it's not in SINGLE_FRAME_CASES:

```python
class TestSyntheticLargeDeform:
    """Large deformation case with relaxed tolerances."""

    def test_case6_frame2_displacement(self, ref_speckle):
        """10% stretch + 5% shear: displacement RMSE < 1.0."""
        data = _build_test_data("case6_large_deform", ref_speckle)
        case = data["case"]

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        U_accum = result.result_disp[0].U_accum
        if U_accum is None:
            U_accum = result.result_disp[0].U

        rmse_u, rmse_v = compute_disp_rmse(
            U_accum, data["mesh"].coordinates_fem,
            data["gt_u2"], data["gt_v2"], data["mask"],
        )

        assert rmse_u < 1.0, f"case6 RMSE_u={rmse_u:.4f} > 1.0"
        assert rmse_v < 1.0, f"case6 RMSE_v={rmse_v:.4f} > 1.0"
```

### Step 2: Run the test

Run: `cd al-dic && python -m pytest tests/test_integration/test_synthetic.py::TestSyntheticLargeDeform -v --timeout=300`
Expected: PASS

### Step 3: Commit

```bash
git add tests/test_integration/test_synthetic.py
git commit -m "test: add large deformation displacement test (case 6)"
```

---

## Task 6: Implement multi-frame tests (cases 7 and 8)

**Files:**
- Modify: `tests/test_integration/test_synthetic.py`

### Step 1: Write multi-frame incremental and accumulative tests

```python
class TestSyntheticMultiFrame:
    """Multi-frame tests: incremental vs accumulative reference mode."""

    def test_case7_incremental_frame2(self, ref_speckle):
        """Incremental mode frame 2: cumulative u=1.0."""
        data = _build_test_data("case7_multiframe_incr", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        U_accum = result.result_disp[0].U_accum
        assert U_accum is not None, "Incremental mode should set U_accum"

        rmse_u, rmse_v = compute_disp_rmse(
            U_accum, data["mesh"].coordinates_fem,
            data["gt_u2"], data["gt_v2"], data["mask"],
        )
        assert rmse_u < 0.5, f"case7 frame2 RMSE_u={rmse_u:.4f}"
        assert rmse_v < 0.5, f"case7 frame2 RMSE_v={rmse_v:.4f}"

    def test_case7_incremental_frame3(self, ref_speckle):
        """Incremental mode frame 3: cumulative u=2.0."""
        data = _build_test_data("case7_multiframe_incr", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        assert len(result.result_disp) >= 2
        U_accum = result.result_disp[1].U_accum
        assert U_accum is not None

        rmse_u, rmse_v = compute_disp_rmse(
            U_accum, data["mesh"].coordinates_fem,
            data["gt_u3"], data["gt_v3"], data["mask"],
        )
        assert rmse_u < 0.5, f"case7 frame3 RMSE_u={rmse_u:.4f}"
        assert rmse_v < 0.5, f"case7 frame3 RMSE_v={rmse_v:.4f}"

    def test_case8_accumulative_frame2(self, ref_speckle):
        """Accumulative mode frame 2: u=1.0 vs reference."""
        data = _build_test_data("case8_multiframe_accum", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        U_accum = result.result_disp[0].U_accum
        if U_accum is None:
            U_accum = result.result_disp[0].U

        rmse_u, rmse_v = compute_disp_rmse(
            U_accum, data["mesh"].coordinates_fem,
            data["gt_u2"], data["gt_v2"], data["mask"],
        )
        assert rmse_u < 0.5, f"case8 frame2 RMSE_u={rmse_u:.4f}"
        assert rmse_v < 0.5, f"case8 frame2 RMSE_v={rmse_v:.4f}"

    def test_case8_accumulative_frame3(self, ref_speckle):
        """Accumulative mode frame 3: u=2.0 vs reference."""
        data = _build_test_data("case8_multiframe_accum", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        assert len(result.result_disp) >= 2
        U_accum = result.result_disp[1].U_accum
        if U_accum is None:
            U_accum = result.result_disp[1].U

        rmse_u, rmse_v = compute_disp_rmse(
            U_accum, data["mesh"].coordinates_fem,
            data["gt_u3"], data["gt_v3"], data["mask"],
        )
        assert rmse_u < 0.5, f"case8 frame3 RMSE_u={rmse_u:.4f}"
        assert rmse_v < 0.5, f"case8 frame3 RMSE_v={rmse_v:.4f}"
```

### Step 2: Run multi-frame tests

Run: `cd al-dic && python -m pytest tests/test_integration/test_synthetic.py::TestSyntheticMultiFrame -v --timeout=300`
Expected: All 4 tests PASS.

### Step 3: Commit

```bash
git add tests/test_integration/test_synthetic.py
git commit -m "test: add multi-frame incremental/accumulative tests (cases 7-8)"
```

---

## Task 7: Implement rotation test (case 10)

**Files:**
- Modify: `tests/test_integration/test_synthetic.py`

### Step 1: Write rotation displacement test

```python
class TestSyntheticRotation:
    """Pure rotation (2 degrees) — tests non-trivial deformation gradient."""

    def test_case10_frame2_displacement(self, ref_speckle):
        """Rotation displacement RMSE < 0.5."""
        data = _build_test_data("case10_rotation", ref_speckle)

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=False,
        )

        U_accum = result.result_disp[0].U_accum
        if U_accum is None:
            U_accum = result.result_disp[0].U

        rmse_u, rmse_v = compute_disp_rmse(
            U_accum, data["mesh"].coordinates_fem,
            data["gt_u2"], data["gt_v2"], data["mask"],
        )
        assert rmse_u < 0.5, f"case10 RMSE_u={rmse_u:.4f}"
        assert rmse_v < 0.5, f"case10 RMSE_v={rmse_v:.4f}"

    def test_case10_strain(self, ref_speckle):
        """Rotation strain should match analytical gradients."""
        data = _build_test_data("case10_rotation", ref_speckle)
        case = data["case"]

        result = run_aldic(
            data["para"], data["images"], data["masks"],
            mesh=data["mesh"], U0=data["U0"],
            compute_strain=True,
        )

        sr = result.result_strain[0]
        rmses = compute_strain_rmse(
            sr,
            case["gt_F11"], case["gt_F21"],
            case["gt_F12"], case["gt_F22"],
            data["mesh"].coordinates_fem, data["mask"],
        )

        for key, val in rmses.items():
            assert val < 0.03, f"case10 {key}={val:.5f} > 0.03"
```

### Step 2: Run rotation test

Run: `cd al-dic && python -m pytest tests/test_integration/test_synthetic.py::TestSyntheticRotation -v --timeout=300`
Expected: PASS

### Step 3: Commit

```bash
git add tests/test_integration/test_synthetic.py
git commit -m "test: add rotation displacement/strain test (case 10)"
```

---

## Task 8: Run full test suite and fix failures

### Step 1: Run all synthetic tests

Run: `cd al-dic && python -m pytest tests/test_integration/test_synthetic.py -v --timeout=600`
Expected: All tests PASS.

### Step 2: Run the entire test suite to check for regressions

Run: `cd al-dic && python -m pytest tests/ -v --timeout=600`
Expected: All tests PASS (existing 195 + new synthetic tests).

### Step 3: If any test fails, debug and fix

Common issues to check:
- **NaN in U**: Check mask coverage — nodes outside mask get NaN
- **High RMSE**: Check center offset (127 vs 128), check `apply_displacement` sign convention
- **Timeout**: Reduce `admm_max_iter` to 2, or reduce image size
- **Memory**: 256x256 is small, should be fine

### Step 4: Final commit

```bash
git add -A
git commit -m "test: Phase 3.1 synthetic integration tests complete (10 cases)"
```

---

## Verification Checklist

- [ ] All 10 MATLAB test cases ported
- [ ] Displacement RMSE within tolerance for all single-frame cases
- [ ] Displacement RMSE within tolerance for multi-frame (cases 7, 8)
- [ ] Strain RMSE within tolerance for cases with nonzero GT (3, 5, 6, 9, 10)
- [ ] Large deformation (case6) uses relaxed tolerances
- [ ] Incremental mode cumulative displacement validated (case7)
- [ ] Accumulative mode validated (case8)
- [ ] Local-only mode validated (case9, use_global_step=False)
- [ ] Rotation validated with analytical gradient (case10)
- [ ] No regressions in existing test suite
- [ ] All helpers reusable for future cross-validation tests
