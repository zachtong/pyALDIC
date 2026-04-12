"""Regression baselines for accuracy and performance.

Locked from benchmark run 2026-04-12. These thresholds guard against
future code changes degrading solver accuracy or throughput.

Config: 512x512, winsize=32, step=8, speckle sigma=3.0, noise=0.005,
        ADMM iter=3, Lagrangian ground truth, Numba JIT post-warmup.

Run all:           pytest tests/test_regression_benchmark.py -m slow
Skip performance:  pytest -m "not perf"  (used in CI — runner CPU varies)
Skip all slow:     pytest -m "not slow"
"""

from __future__ import annotations

import time
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import (
    GridxyROIRange,
    FrameSchedule,
)
from al_dic.core.pipeline import run_aldic
from al_dic.io.image_ops import compute_image_gradient, normalize_images
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.mesh.refinement import RefinementContext, refine_mesh
from al_dic.mesh.criteria.mask_boundary import MaskBoundaryCriterion
from al_dic.solver.init_disp import init_disp
from al_dic.solver.integer_search import integer_search
from al_dic.solver.local_icgn import local_icgn
from al_dic.solver.subpb1_solver import precompute_subpb1, subpb1_solver
from al_dic.solver.subpb2_solver import precompute_subpb2, subpb2_solver
from al_dic.strain.nodal_strain_fem import global_nodal_strain_fem
from al_dic.utils.region_analysis import precompute_node_regions
from al_dic.strain.smooth_field import smooth_field_sparse

# ── Constants (must match benchmark script) ────────────────────────────

IMG_SIZE = 512
WINSIZE = 32
STEP = 8
SIGMA = 3.0
NOISE_STD = 0.005
ADMM_ITER = 3


# ── Image helpers ──────────────────────────────────────────────────────

def _make_speckle(h, w, sigma=SIGMA, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return filtered


def _warp_lagrangian(ref, u_func, v_func, n_iter=20):
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        X = xx - u_func(X, Y)
        Y = yy - v_func(X, Y)
    return map_coordinates(ref, [Y, X], order=5, mode="nearest")


def _add_noise(img, std=NOISE_STD):
    rng = np.random.default_rng(99)
    return np.clip(img + rng.normal(0, std, img.shape), 0, 1)


# ── Field definitions ──────────────────────────────────────────────────

_CX = IMG_SIZE / 2.0
_CY = IMG_SIZE / 2.0


def _field_translation(dx=2.5, dy=1.5):
    return (lambda x, y: np.full_like(x, dx, dtype=np.float64),
            lambda x, y: np.full_like(x, dy, dtype=np.float64))


def _field_rotation(deg=2.0):
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    return (lambda x, y: (x - _CX) * (c - 1) - (y - _CY) * s,
            lambda x, y: (x - _CX) * s + (y - _CY) * (c - 1))


def _field_affine(eps=0.02):
    return (lambda x, y: eps * (x - _CX),
            lambda x, y: eps * (y - _CY))


def _field_quadratic(c=1.5e-4):
    return (lambda x, y: c * (x - _CX) ** 2,
            lambda x, y: c * (y - _CY) ** 2)


def _gt_translation(coords, dx=2.5, dy=1.5):
    n = coords.shape[0]
    return np.full(n, dx), np.full(n, dy)


def _gt_rotation(coords, deg=2.0):
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    x, y = coords[:, 0], coords[:, 1]
    return ((c - 1) * (x - _CX) - s * (y - _CY),
            s * (x - _CX) + (c - 1) * (y - _CY))


def _gt_affine(coords, eps=0.02):
    return eps * (coords[:, 0] - _CX), eps * (coords[:, 1] - _CY)


def _gt_quadratic(coords, c_val=1.5e-4):
    return c_val * (coords[:, 0] - _CX) ** 2, c_val * (coords[:, 1] - _CY) ** 2


# ── Solver helpers ─────────────────────────────────────────────────────

def _estimate_max_disp(u_func, v_func):
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float64)
    return float(max(np.abs(u_func(xx, yy)).max(), np.abs(v_func(xx, yy)).max()))


def _setup_frame(ref_img, def_img, mask, step=STEP, winsize=WINSIZE):
    h, w = ref_img.shape
    wmin = min(8, step)
    para = dicpara_default(
        winsize=winsize, winstepsize=step, winsize_min=wmin,
        img_size=(h, w), admm_max_iter=ADMM_ITER,
        gridxy_roi_range=GridxyROIRange(gridx=(0, w - 1), gridy=(0, h - 1)),
        tol=1e-2, disp_smoothness=5e-4, strain_smoothness=1e-5,
    )
    imgs = [ref_img, def_img]
    img_norm, roi = normalize_images(imgs, para.gridxy_roi_range)
    p = replace(para, gridxy_roi_range=roi, img_size=(h, w))
    Df = compute_image_gradient(img_norm[0], mask)
    x0, y0, ug, vg, fft_i = integer_search(img_norm[0], img_norm[1], p)
    U0 = init_disp(ug, vg, fft_i["cc_max"], x0, y0)
    mesh = mesh_setup(x0, y0, p)
    return p, img_norm, Df, mesh, U0


def _run_local(p, img_norm, Df, mesh, U0):
    U, F, _, _, _, _ = local_icgn(
        U0, mesh.coordinates_fem, Df,
        img_norm[0], img_norm[1], p, p.tol,
    )
    return U, F


def _run_aldic(p, img_norm, Df, mesh, U0, mask):
    h, w = img_norm[0].shape
    n = mesh.coordinates_fem.shape[0]
    region_map = precompute_node_regions(mesh.coordinates_fem, mask, (h, w))
    mu = p.mu
    beta = np.median(p.beta_range) * p.winstepsize ** 2 * mu
    cache2 = precompute_subpb2(mesh, p.gauss_pt_order, beta, mu, 0.0)
    wl = np.full((n, 2), p.winsize, dtype=np.float64)
    p_a = replace(p, winsize_list=wl)
    pre1 = precompute_subpb1(mesh.coordinates_fem, Df, img_norm[0], p_a)

    U_s1, F_s1, _, _, _, _ = local_icgn(
        U0, mesh.coordinates_fem, Df,
        img_norm[0], img_norm[1], p, p.tol,
    )
    sd = p.winstepsize * max(0.3, 500.0 * p.disp_smoothness)
    U_s1 = smooth_field_sparse(U_s1, mesh.coordinates_fem, sd, region_map, 2)
    ss = p.winstepsize * max(0.3, 500.0 * p.strain_smoothness)
    F_s1 = smooth_field_sparse(F_s1, mesh.coordinates_fem, ss, region_map, 4)

    gd = np.zeros(4 * n)
    dd = np.zeros(2 * n)
    U_s2 = subpb2_solver(mesh, p.gauss_pt_order, beta, mu,
                         U_s1, F_s1, gd, dd, 0.0, p.winstepsize,
                         precomputed=cache2)
    F_s2 = global_nodal_strain_fem(mesh, p, U_s2)
    gd = F_s2 - F_s1
    dd = U_s2 - U_s1

    for _ in range(2, p.admm_max_iter + 1):
        U_s1, _, _, _ = subpb1_solver(
            U_s2, F_s2, dd, gd,
            mesh.coordinates_fem, Df, img_norm[0], img_norm[1],
            mu, beta, p_a, p.tol, precomputed=pre1,
        )
        F_s1 = F_s2.copy()
        U_s2 = subpb2_solver(mesh, p.gauss_pt_order, beta, mu,
                             U_s1, F_s1, gd, dd, 0.0, p.winstepsize,
                             precomputed=cache2)
        F_s2 = global_nodal_strain_fem(mesh, p, U_s2)
        gd = F_s2 - F_s1
        dd = U_s2 - U_s1

    return U_s2, F_s2


def _compute_rmse(U, coords, gt_func, mask, max_disp=0.0,
                  hole_center=None, hole_radius=0, hole_margin=0,
                  **gt_kw):
    """RMSE excluding image-edge margin, mask=0 nodes, and optionally hole boundary."""
    h, w = mask.shape
    u_m, v_m = U[0::2], U[1::2]
    gt_u, gt_v = gt_func(coords, **gt_kw)
    margin = max(3 * STEP, int(np.ceil(max_disp + WINSIZE / 2)))
    x, y = coords[:, 0], coords[:, 1]
    sel = ((x > margin) & (x < w - 1 - margin) &
           (y > margin) & (y < h - 1 - margin))
    xi = np.clip(np.round(x).astype(int), 0, w - 1)
    yi = np.clip(np.round(y).astype(int), 0, h - 1)
    sel &= mask[yi, xi] > 0.5
    if hole_margin > 0 and hole_center is not None:
        cx, cy = hole_center
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        sel &= dist >= (hole_radius + hole_margin)
    eu = u_m[sel] - gt_u[sel]
    ev = v_m[sel] - gt_v[sel]
    ok = np.isfinite(eu) & np.isfinite(ev)
    if ok.sum() == 0:
        return float("nan"), 0
    return float(np.sqrt(np.mean(eu[ok] ** 2 + ev[ok] ** 2))), int(ok.sum())


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ref_img():
    return _make_speckle(IMG_SIZE, IMG_SIZE)


@pytest.fixture(scope="module")
def warmup():
    """JIT warmup — runs once per module."""
    r = _make_speckle(128, 128, seed=0)
    u_fn = lambda x, y: np.full_like(x, 1.0)
    v_fn = lambda x, y: np.full_like(x, 0.5)
    d = _warp_lagrangian(r, u_fn, v_fn, n_iter=1)
    m = np.ones((128, 128), dtype=np.float64)
    p, im, Df, mesh, U0 = _setup_frame(r, d, m, step=16, winsize=32)
    _run_local(p, im, Df, mesh, U0)
    # warmup subpb1/subpb2
    n = mesh.coordinates_fem.shape[0]
    wl = np.full((n, 2), 32.0)
    p2 = replace(p, winsize_list=wl)
    pre1 = precompute_subpb1(mesh.coordinates_fem, Df, im[0], p2)
    subpb1_solver(
        np.zeros(2 * n), np.zeros(4 * n),
        np.zeros(2 * n), np.zeros(4 * n),
        mesh.coordinates_fem, Df, im[0], im[1],
        1.0, 1.0, p2, 1e-2, precomputed=pre1,
    )
    return True


# ══════════════════════════════════════════════════════════════════════
#  Category 1 — Deformation type accuracy & performance
# ══════════════════════════════════════════════════════════════════════
#
# Baseline RMSE (2026-04-12):
#   Translation  Local=0.0135  ALDIC=0.0128
#   Rotation     Local=0.0277  ALDIC=0.0282
#   Affine       Local=0.0200  ALDIC=0.0198
#   Quadratic    Local=0.0380  ALDIC=0.0415
#
# Thresholds set to ~2x measured to absorb noise variance.
# Performance lower bounds set to ~0.5x measured.

_CAT1_CASES = [
    # (id, field_factory, field_kw, gt_func, gt_kw,
    #  local_rmse_max, aldic_rmse_max, local_pois_min, aldic_pois_min)
    ("translation",
     _field_translation, {"dx": 2.5, "dy": 1.5},
     _gt_translation, {"dx": 2.5, "dy": 1.5},
     0.030, 0.030, 40_000, 10_000),
    ("rotation",
     _field_rotation, {"deg": 2.0},
     _gt_rotation, {"deg": 2.0},
     0.060, 0.060, 40_000, 10_000),
    ("affine",
     _field_affine, {"eps": 0.02},
     _gt_affine, {"eps": 0.02},
     0.045, 0.045, 40_000, 10_000),
    ("quadratic",
     _field_quadratic, {"c": 1.5e-4},
     _gt_quadratic, {"c_val": 1.5e-4},
     0.080, 0.085, 30_000, 10_000),
]


@pytest.mark.slow
class TestCategory1Accuracy:
    """Deformation-type RMSE must stay below locked thresholds."""

    @pytest.mark.parametrize(
        "field_fn,field_kw,gt_fn,gt_kw,local_max,aldic_max",
        [(c[1], c[2], c[3], c[4], c[5], c[6]) for c in _CAT1_CASES],
        ids=[c[0] for c in _CAT1_CASES],
    )
    def test_local_dic_rmse(self, ref_img, warmup,
                            field_fn, field_kw, gt_fn, gt_kw,
                            local_max, aldic_max):
        mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float64)
        u_fn, v_fn = field_fn(**field_kw)
        def_img = _add_noise(_warp_lagrangian(ref_img, u_fn, v_fn))
        p, im, Df, mesh, U0 = _setup_frame(ref_img, def_img, mask)
        max_d = _estimate_max_disp(u_fn, v_fn)
        U, _ = _run_local(p, im, Df, mesh, U0)
        rmse, n = _compute_rmse(U, mesh.coordinates_fem, gt_fn, mask,
                                max_disp=max_d, **gt_kw)
        assert n > 100, f"Too few eval nodes: {n}"
        assert rmse < local_max, (
            f"Local DIC RMSE={rmse:.4f} exceeds threshold {local_max}")

    @pytest.mark.parametrize(
        "field_fn,field_kw,gt_fn,gt_kw,local_max,aldic_max",
        [(c[1], c[2], c[3], c[4], c[5], c[6]) for c in _CAT1_CASES],
        ids=[c[0] for c in _CAT1_CASES],
    )
    def test_aldic_rmse(self, ref_img, warmup,
                        field_fn, field_kw, gt_fn, gt_kw,
                        local_max, aldic_max):
        mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float64)
        u_fn, v_fn = field_fn(**field_kw)
        def_img = _add_noise(_warp_lagrangian(ref_img, u_fn, v_fn))
        p, im, Df, mesh, U0 = _setup_frame(ref_img, def_img, mask)
        max_d = _estimate_max_disp(u_fn, v_fn)
        U, _ = _run_aldic(p, im, Df, mesh, U0, mask)
        rmse, n = _compute_rmse(U, mesh.coordinates_fem, gt_fn, mask,
                                max_disp=max_d, **gt_kw)
        assert n > 100, f"Too few eval nodes: {n}"
        assert rmse < aldic_max, (
            f"AL-DIC RMSE={rmse:.4f} exceeds threshold {aldic_max}")


@pytest.mark.slow
@pytest.mark.perf
class TestCategory1Performance:
    """Solver throughput must stay above locked lower bounds.

    Marked ``perf`` — excluded from CI (runner CPU varies).
    Run locally with: pytest -m slow
    """

    @pytest.mark.parametrize(
        "field_fn,field_kw,local_pois_min,aldic_pois_min",
        [(c[1], c[2], c[7], c[8]) for c in _CAT1_CASES],
        ids=[c[0] for c in _CAT1_CASES],
    )
    def test_local_dic_throughput(self, ref_img, warmup,
                                  field_fn, field_kw,
                                  local_pois_min, aldic_pois_min):
        mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float64)
        u_fn, v_fn = field_fn(**field_kw)
        def_img = _add_noise(_warp_lagrangian(ref_img, u_fn, v_fn))
        p, im, Df, mesh, U0 = _setup_frame(ref_img, def_img, mask)
        n_nodes = mesh.coordinates_fem.shape[0]
        t0 = time.perf_counter()
        _run_local(p, im, Df, mesh, U0)
        elapsed = time.perf_counter() - t0
        pois = n_nodes / elapsed
        assert pois > local_pois_min, (
            f"Local DIC throughput {pois:,.0f} < {local_pois_min:,} POIs/s")

    @pytest.mark.parametrize(
        "field_fn,field_kw,local_pois_min,aldic_pois_min",
        [(c[1], c[2], c[7], c[8]) for c in _CAT1_CASES],
        ids=[c[0] for c in _CAT1_CASES],
    )
    def test_aldic_throughput(self, ref_img, warmup,
                              field_fn, field_kw,
                              local_pois_min, aldic_pois_min):
        mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float64)
        u_fn, v_fn = field_fn(**field_kw)
        def_img = _add_noise(_warp_lagrangian(ref_img, u_fn, v_fn))
        p, im, Df, mesh, U0 = _setup_frame(ref_img, def_img, mask)
        n_nodes = mesh.coordinates_fem.shape[0]
        t0 = time.perf_counter()
        _run_aldic(p, im, Df, mesh, U0, mask)
        elapsed = time.perf_counter() - t0
        pois = n_nodes / elapsed
        assert pois > aldic_pois_min, (
            f"AL-DIC throughput {pois:,.0f} < {aldic_pois_min:,} POIs/s")

    def test_aldic_ratio(self, ref_img, warmup):
        """AL-DIC should be 2-5x slower than Local DIC (not 10x+)."""
        mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float64)
        u_fn, v_fn = _field_translation(dx=2.5, dy=1.5)
        def_img = _add_noise(_warp_lagrangian(ref_img, u_fn, v_fn))
        p, im, Df, mesh, U0 = _setup_frame(ref_img, def_img, mask)

        t0 = time.perf_counter()
        _run_local(p, im, Df, mesh, U0)
        t_local = time.perf_counter() - t0

        t0 = time.perf_counter()
        _run_aldic(p, im, Df, mesh, U0, mask)
        t_aldic = time.perf_counter() - t0

        ratio = t_aldic / t_local
        assert ratio < 8.0, f"AL-DIC/Local ratio={ratio:.1f}x (expected < 8x)"


# ══════════════════════════════════════════════════════════════════════
#  Category 2 — Mesh refinement (circular hole)
# ══════════════════════════════════════════════════════════════════════
#
# Baseline (2026-04-12, rotation 2 deg, hole r=80, step=16):
#   Unmasked deformed image (matching pipeline.py convention):
#   Uniform:  interior=0.0150  all_valid=0.0152
#   Refined:  interior=0.0150  all_valid=0.0161
#
# Key invariant: interior RMSE must NOT degrade with mesh refinement.

HOLE_R = 80
HOLE_CX = IMG_SIZE / 2
HOLE_CY = IMG_SIZE / 2
CAT2_STEP = 16


def _make_hole_mask():
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float64)
    mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float64)
    mask[((xx - HOLE_CX) ** 2 + (yy - HOLE_CY) ** 2) < HOLE_R ** 2] = 0.0
    return mask


@pytest.mark.slow
class TestCategory2MeshRefinement:
    """Mesh refinement near hole boundary must not degrade interior accuracy."""

    def _solve_with_hole(self, ref_img, mesh, p, img_norm, Df, mask, U0):
        """Run Local DIC on a hole-bearing mesh, return interior + boundary RMSE."""
        max_d = _estimate_max_disp(*_field_rotation(deg=2.0))
        U, _ = _run_local(p, img_norm, Df, mesh, U0)
        coords = mesh.coordinates_fem

        rmse_int, n_int = _compute_rmse(
            U, coords, _gt_rotation, mask, max_disp=max_d,
            hole_center=(HOLE_CX, HOLE_CY), hole_radius=HOLE_R,
            hole_margin=WINSIZE, deg=2.0,
        )
        rmse_all, n_all = _compute_rmse(
            U, coords, _gt_rotation, mask, max_disp=max_d, deg=2.0,
        )
        return rmse_int, n_int, rmse_all, n_all

    def test_interior_rmse_not_degraded(self, ref_img, warmup):
        """Refined mesh interior RMSE <= uniform mesh interior RMSE (+ tolerance)."""
        mask = _make_hole_mask()
        u_fn, v_fn = _field_rotation(deg=2.0)
        def_img = _add_noise(_warp_lagrangian(ref_img, u_fn, v_fn))

        # Uniform — pass unmasked images (mask only for gradient/subset selection)
        p_u, im_u, Df_u, mesh_u, U0_u = _setup_frame(
            ref_img, def_img, mask, step=CAT2_STEP)
        rmse_int_u, n_u, _, _ = self._solve_with_hole(
            ref_img, mesh_u, p_u, im_u, Df_u, mask, U0_u)

        # Refined
        crit = MaskBoundaryCriterion(min_element_size=8)
        ctx = RefinementContext(mesh=mesh_u, mask=mask)
        mesh_r, U0_r = refine_mesh(
            mesh_u, [crit], ctx, U0_u,
            mask=mask, img_size=(IMG_SIZE, IMG_SIZE))
        p_r = replace(p_u)
        rmse_int_r, n_r, _, _ = self._solve_with_hole(
            ref_img, mesh_r, p_r, im_u, Df_u, mask, U0_r)

        # Interior RMSE of refined must not exceed uniform + 20% tolerance
        assert rmse_int_r <= rmse_int_u * 1.20, (
            f"Refined interior RMSE {rmse_int_r:.4f} > uniform {rmse_int_u:.4f} * 1.2")

    def test_interior_rmse_within_threshold(self, ref_img, warmup):
        """Interior RMSE stays below absolute threshold (locked baseline)."""
        mask = _make_hole_mask()
        u_fn, v_fn = _field_rotation(deg=2.0)
        def_img = _add_noise(_warp_lagrangian(ref_img, u_fn, v_fn))
        # Pass unmasked images (mask only for gradient/subset selection)
        p, im, Df, mesh, U0 = _setup_frame(
            ref_img, def_img, mask, step=CAT2_STEP)
        rmse_int, n, _, _ = self._solve_with_hole(
            ref_img, mesh, p, im, Df, mask, U0)
        # Baseline=0.0150 (unmasked def), threshold=0.03 (~2x)
        assert rmse_int < 0.03, (
            f"Interior RMSE {rmse_int:.4f} exceeds threshold 0.03")

    @pytest.mark.perf
    def test_refinement_overhead_reasonable(self, ref_img, warmup):
        """Refinement + solve should be < 5x the uniform solve time."""
        mask = _make_hole_mask()
        u_fn, v_fn = _field_rotation(deg=2.0)
        def_img = _add_noise(_warp_lagrangian(ref_img, u_fn, v_fn))

        # Pass unmasked images (mask only for gradient/subset selection)
        p_u, im_u, Df_u, mesh_u, U0_u = _setup_frame(
            ref_img, def_img, mask, step=CAT2_STEP)
        t0 = time.perf_counter()
        _run_local(p_u, im_u, Df_u, mesh_u, U0_u)
        t_uniform = time.perf_counter() - t0

        crit = MaskBoundaryCriterion(min_element_size=8)
        ctx = RefinementContext(mesh=mesh_u, mask=mask)
        t0 = time.perf_counter()
        mesh_r, U0_r = refine_mesh(
            mesh_u, [crit], ctx, U0_u,
            mask=mask, img_size=(IMG_SIZE, IMG_SIZE))
        p_r = replace(p_u)
        _run_local(p_r, im_u, Df_u, mesh_r, U0_r)
        t_refined = time.perf_counter() - t0

        ratio = t_refined / max(t_uniform, 1e-9)
        assert ratio < 5.0, (
            f"Refined/uniform time ratio={ratio:.1f}x (expected < 5x)")


# ══════════════════════════════════════════════════════════════════════
#  Category 3 — Tracking modes (accumulative vs incremental)
# ══════════════════════════════════════════════════════════════════════
#
# Baseline (2026-04-12):
#   Moderate (20 x 0.5 deg = 10 deg):
#     Accumulative  final=0.1240  avg=0.0675
#     Incremental   final=0.1581  avg=0.0850
#   Large (20 x 2.0 deg = 40 deg):
#     Accumulative  final=4.3722  avg=1.5275  (expected failure)
#     Incremental   final=0.4784  avg=0.2575
#
# Key invariants:
#   - Both modes work for moderate deformation
#   - Incremental handles large deformation where accumulative fails

def _make_frame_sequence(ref_img, n_frames, deg_per_frame):
    mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float64)
    images = [ref_img.copy()]
    masks = [mask.copy()]
    for i in range(1, n_frames + 1):
        total_deg = i * deg_per_frame
        u_fn, v_fn = _field_rotation(deg=total_deg)
        d = _warp_lagrangian(ref_img, u_fn, v_fn)
        d = _add_noise(d, rng_seed_offset=i)
        images.append(d)
        masks.append(mask.copy())
    return images, masks, mask


def _add_noise_seq(img, std=NOISE_STD, rng_seed_offset=0):
    """Noise with unique seed per frame."""
    rng = np.random.default_rng(100 + rng_seed_offset)
    return np.clip(img + rng.normal(0, std, img.shape), 0, 1)


def _run_tracking(ref_img, n_frames, deg_per_frame, mode):
    """Run pipeline and return per-frame RMSEs."""
    mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float64)
    images = [ref_img.copy()]
    masks = [mask.copy()]
    for i in range(1, n_frames + 1):
        total_deg = i * deg_per_frame
        u_fn, v_fn = _field_rotation(deg=total_deg)
        d = _warp_lagrangian(ref_img, u_fn, v_fn)
        d = _add_noise_seq(d, rng_seed_offset=i)
        images.append(d)
        masks.append(mask.copy())

    para = dicpara_default(
        winsize=WINSIZE, winstepsize=STEP,
        img_size=(IMG_SIZE, IMG_SIZE),
        gridxy_roi_range=GridxyROIRange(
            gridx=(0, IMG_SIZE - 1), gridy=(0, IMG_SIZE - 1)),
        reference_mode=mode, admm_max_iter=ADMM_ITER,
        tol=1e-2, disp_smoothness=5e-4, strain_smoothness=1e-5,
    )

    result = run_aldic(para, images, masks, compute_strain=False)
    coords = result.dic_mesh.coordinates_fem

    rmses = []
    for i in range(n_frames):
        fr = result.result_disp[i]
        U_acc = fr.U_accum if fr.U_accum is not None else fr.U
        total_deg = (i + 1) * deg_per_frame
        max_d = _estimate_max_disp(*_field_rotation(deg=total_deg))
        rmse, _ = _compute_rmse(
            U_acc, coords, _gt_rotation, mask,
            max_disp=max_d, deg=total_deg)
        rmses.append(rmse)
    return rmses


@pytest.mark.slow
class TestCategory3TrackingModerate:
    """Moderate deformation (20 x 0.5 deg = 10 deg): both modes work."""

    N_FRAMES = 20
    DEG_PER_FRAME = 0.5

    def test_accumulative_final_rmse(self, ref_img, warmup):
        rmses = _run_tracking(ref_img, self.N_FRAMES, self.DEG_PER_FRAME,
                              "accumulative")
        # Baseline=0.124, threshold=0.25 (~2x)
        assert rmses[-1] < 0.25, (
            f"Accumulative final RMSE={rmses[-1]:.4f} exceeds 0.25")

    def test_incremental_final_rmse(self, ref_img, warmup):
        rmses = _run_tracking(ref_img, self.N_FRAMES, self.DEG_PER_FRAME,
                              "incremental")
        # Baseline=0.158, threshold=0.32 (~2x)
        assert rmses[-1] < 0.32, (
            f"Incremental final RMSE={rmses[-1]:.4f} exceeds 0.32")


@pytest.mark.slow
class TestCategory3TrackingLarge:
    """Large deformation (20 x 2.0 deg = 40 deg): incremental handles it."""

    N_FRAMES = 20
    DEG_PER_FRAME = 2.0

    def test_accumulative_degrades(self, ref_img, warmup):
        """Accumulative mode should visibly degrade (RMSE > 1 px by final frame)."""
        rmses = _run_tracking(ref_img, self.N_FRAMES, self.DEG_PER_FRAME,
                              "accumulative")
        assert rmses[-1] > 1.0, (
            f"Accumulative final RMSE={rmses[-1]:.4f} unexpectedly low "
            f"(expected > 1.0 for 40 deg total)")

    def test_incremental_handles_large(self, ref_img, warmup):
        """Incremental mode should keep RMSE manageable (< 1.0 px)."""
        rmses = _run_tracking(ref_img, self.N_FRAMES, self.DEG_PER_FRAME,
                              "incremental")
        # Baseline=0.478, threshold=1.0 (~2x)
        assert rmses[-1] < 1.0, (
            f"Incremental final RMSE={rmses[-1]:.4f} exceeds 1.0")

    def test_incremental_beats_accumulative(self, ref_img, warmup):
        """For large deformation, incremental must be significantly better."""
        rmses_acc = _run_tracking(ref_img, self.N_FRAMES, self.DEG_PER_FRAME,
                                  "accumulative")
        rmses_inc = _run_tracking(ref_img, self.N_FRAMES, self.DEG_PER_FRAME,
                                  "incremental")
        # Incremental should be at least 3x better
        ratio = rmses_acc[-1] / max(rmses_inc[-1], 1e-9)
        assert ratio > 3.0, (
            f"Incremental advantage ratio={ratio:.1f}x (expected > 3x)")
