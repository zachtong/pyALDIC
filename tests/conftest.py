"""Shared pytest fixtures for AL-DIC tests.

Provides:
    - Synthetic speckle image generation
    - Known displacement field generation
    - MATLAB checkpoint loading
    - Standard DICPara configurations
    - Simple mesh fixtures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from al_dic.core.data_structures import DICMesh, DICPara, GridxyROIRange
from al_dic.core.config import dicpara_default


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MATLAB_CHECKPOINTS_DIR = FIXTURES_DIR / "matlab_checkpoints"


# ---------------------------------------------------------------------------
# Speckle image generation
# ---------------------------------------------------------------------------

def generate_speckle(
    height: int = 256,
    width: int = 256,
    sigma: float = 3.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate a synthetic speckle pattern using Gaussian-filtered random noise.

    Matches MATLAB test_aldic_synthetic.m generate_speckle().

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        sigma: Gaussian filter std (feature size ~ 2*sigma).
        seed: Random seed for reproducibility.

    Returns:
        (H, W) float64 array with values in [20, 235].
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((height, width))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")

    # Normalize to [20, 235]
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def apply_displacement(
    ref_image: NDArray[np.float64],
    u_field: NDArray[np.float64],
    v_field: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply Eulerian inverse warp to generate a deformed image.

    warped(y, x) = ref(y - v(y,x), x - u(y,x))

    .. deprecated::
        Use :func:`apply_displacement_lagrangian` for spatially-varying fields.
        This Eulerian version has O(du/dx * u) error for non-uniform fields.
        Kept for backward compatibility with existing unit tests.

    Uses order=5 (quintic B-spline) intentionally: the IC-GN solver samples
    with order=3 (cubic). Using a higher order here avoids the "double
    interpolation" artifact where cubic(cubic(ref, +d), -d) != ref, which
    would create an irreducible residual floor and prevent IC-GN convergence.

    Args:
        ref_image: (H, W) reference image.
        u_field: (H, W) x-displacement field.
        v_field: (H, W) y-displacement field.

    Returns:
        (H, W) warped image.
    """
    from scipy.ndimage import map_coordinates

    h, w = ref_image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    src_y = yy - v_field
    src_x = xx - u_field
    coords = np.array([src_y.ravel(), src_x.ravel()])
    warped = map_coordinates(ref_image, coords, order=5, mode="constant", cval=0.0)
    return warped.reshape(h, w)


def apply_displacement_lagrangian(
    ref_image: NDArray[np.float64],
    u_func,
    v_func,
    n_iter: int = 20,
) -> NDArray[np.float64]:
    """Generate deformed image using Lagrangian displacement convention.

    DIC solves for u(X) where X is the reference coordinate:
        x = X + u(X)

    To build g(x) = f(X), we need to invert x = X + u(X) for each
    deformed pixel x.  Fixed-point iteration:
        X_{k+1} = x - u(X_k)

    converges for |du/dX| < 1 (satisfied for typical DIC deformations).

    Uses order=5 (quintic B-spline) to avoid double-interpolation artifacts
    with the IC-GN solver's order=3 cubic sampling.

    Args:
        ref_image: (H, W) reference image.
        u_func: Callable(x, y) -> x-displacement at reference coords.
        v_func: Callable(x, y) -> y-displacement at reference coords.
        n_iter: Number of fixed-point iterations (20 is sufficient for
                |du/dX| < 0.2, converging to < 1e-14 px).

    Returns:
        (H, W) deformed image with exact Lagrangian ground truth.
    """
    from scipy.ndimage import map_coordinates

    h, w = ref_image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # Fixed-point iteration: find reference coords X for each deformed pixel x
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        X = xx - u_func(X, Y)
        Y = yy - v_func(X, Y)

    coords = np.array([Y.ravel(), X.ravel()])
    warped = map_coordinates(ref_image, coords, order=5, mode="nearest")
    return warped.reshape(h, w)


# ---------------------------------------------------------------------------
# Standard fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_dicpara() -> DICPara:
    """Default DICPara for testing."""
    return dicpara_default(
        winsize=32,
        winstepsize=16,
        winsize_min=8,
        img_size=(256, 256),
        gridxy_roi_range=GridxyROIRange(gridx=(0, 255), gridy=(0, 255)),
        reference_mode="accumulative",
        show_plots=False,
    )


@pytest.fixture
def speckle_256() -> NDArray[np.float64]:
    """256x256 synthetic speckle pattern."""
    return generate_speckle(256, 256, sigma=3.0, seed=42)


@pytest.fixture
def simple_quad_mesh() -> DICMesh:
    """Simple 3x3 node regular quadrilateral mesh for unit tests.

    Nodes:
        6 -- 7 -- 8
        |    |    |
        3 -- 4 -- 5
        |    |    |
        0 -- 1 -- 2

    Elements (Q8, but only 4 corner nodes used for simplicity):
        Element 0: [0, 1, 4, 3, -, -, -, -]  (mid-edge nodes = -1 placeholder)
        Element 1: [1, 2, 5, 4, -, -, -, -]
        Element 2: [3, 4, 7, 6, -, -, -, -]
        Element 3: [4, 5, 8, 7, -, -, -, -]
    """
    coords = np.array([
        [0, 0], [16, 0], [32, 0],
        [0, 16], [16, 16], [32, 16],
        [0, 32], [16, 32], [32, 32],
    ], dtype=np.float64)

    # Q8 elements: 4 corners + 4 midside nodes (use -1 as placeholder for no midside)
    elems = np.array([
        [0, 1, 4, 3, -1, -1, -1, -1],
        [1, 2, 5, 4, -1, -1, -1, -1],
        [3, 4, 7, 6, -1, -1, -1, -1],
        [4, 5, 8, 7, -1, -1, -1, -1],
    ], dtype=np.int64)

    return DICMesh(
        coordinates_fem=coords,
        elements_fem=elems,
        x0=np.array([0, 16, 32], dtype=np.float64),
        y0=np.array([0, 16, 32], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Synthetic test helpers
# ---------------------------------------------------------------------------

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
) -> NDArray[np.float64]:
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
) -> NDArray[np.float64]:
    """Create an annular (ring) binary mask."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return ((dist2 <= r_outer ** 2) & (dist2 > r_inner ** 2)).astype(np.float64)


def compute_disp_rmse(
    U: NDArray[np.float64],
    coords: NDArray[np.float64],
    gt_u: NDArray[np.float64],
    gt_v: NDArray[np.float64],
    mask: NDArray[np.float64],
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


def compute_disp_rmse_interior(
    U: NDArray[np.float64],
    coords: NDArray[np.float64],
    gt_u: NDArray[np.float64],
    gt_v: NDArray[np.float64],
    img_size: tuple[int, int],
    edge_margin: int = 32,
) -> tuple[float, float]:
    """Compute displacement RMSE on interior nodes (edge-margin exclusion).

    Excludes nodes near image boundaries where interpolation artifacts
    degrade accuracy.  Preferred over mask-based filtering for solid
    (non-annular) specimens to avoid circular-mask contamination.

    Args:
        U: Interleaved displacement (2*n_nodes,).
        coords: Node coordinates (n_nodes, 2), col0=x, col1=y.
        gt_u, gt_v: Ground truth at each node (n_nodes,).
        img_size: (height, width) of the image.
        edge_margin: Pixels to exclude from each edge.

    Returns:
        (rmse_u, rmse_v).
    """
    u_comp = U[0::2]
    v_comp = U[1::2]

    h, w = img_size
    x, y = coords[:, 0], coords[:, 1]
    interior = (
        (x >= edge_margin) & (x <= w - 1 - edge_margin)
        & (y >= edge_margin) & (y <= h - 1 - edge_margin)
    )
    valid = interior & np.isfinite(u_comp) & np.isfinite(v_comp)

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
    coords: NDArray[np.float64],
    mask: NDArray[np.float64],
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


# ---------------------------------------------------------------------------
# MATLAB checkpoint loading
# ---------------------------------------------------------------------------

def load_matlab_checkpoint(name: str) -> dict:
    """Load a MATLAB checkpoint .mat file from the fixtures directory.

    Args:
        name: Checkpoint filename (e.g., 'checkpoint_S3_frame2.mat').

    Returns:
        Dictionary of MATLAB variables.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    import scipy.io as sio

    path = MATLAB_CHECKPOINTS_DIR / name
    if not path.exists():
        pytest.skip(f"MATLAB checkpoint not found: {path}")
    return sio.loadmat(str(path), squeeze_me=True)
