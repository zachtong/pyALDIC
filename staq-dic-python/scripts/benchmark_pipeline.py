"""Benchmark script for STAQ-DIC Python pipeline.

Profiles each section of the AL-DIC pipeline on synthetic data
at multiple image sizes to identify performance bottlenecks.

Usage:
    python scripts/benchmark_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import DICMesh, GridxyROIRange
from staq_dic.io.image_ops import compute_image_gradient, normalize_images
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.solver.init_disp import init_disp
from staq_dic.solver.integer_search import integer_search
from staq_dic.solver.local_icgn import local_icgn
from staq_dic.solver.subpb1_solver import subpb1_solver
from staq_dic.solver.subpb2_solver import subpb2_solver
from staq_dic.strain.compute_strain import compute_strain
from staq_dic.strain.nodal_strain_fem import global_nodal_strain_fem
from staq_dic.strain.smooth_field import smooth_field_sparse
from staq_dic.utils.region_analysis import precompute_node_regions


def generate_speckle(h: int, w: int, sigma: float = 3.0, seed: int = 42) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def apply_displacement(ref, u_field, v_field):
    from scipy.ndimage import map_coordinates
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = np.array([(yy - v_field).ravel(), (xx - u_field).ravel()])
    return map_coordinates(ref, coords, order=5, mode="constant", cval=0.0).reshape(h, w)


class Timer:
    """Context manager for timing code sections."""
    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def benchmark_size(H: int, W: int, winstepsize: int = 16, winsize: int = 32,
                   admm_iter: int = 3, n_runs: int = 1):
    """Run full pipeline benchmark at given image size."""
    print(f"\n{'='*70}")
    print(f"  Image: {H}x{W}, winsize={winsize}, step={winstepsize}, ADMM={admm_iter}")
    print(f"{'='*70}")

    # --- Generate synthetic data ---
    ref = generate_speckle(H, W, sigma=3.0, seed=42)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2, H / 2
    u_field = 0.02 * (xx - cx)
    v_field = 0.02 * (yy - cy)
    deformed = apply_displacement(ref, u_field, v_field)
    mask = np.ones((H, W), dtype=np.float64)

    ref_norm = ref / 255.0
    def_norm = deformed / 255.0
    images = [ref_norm, def_norm]
    masks = [mask, mask]

    para = dicpara_default(
        winsize=winsize,
        winstepsize=winstepsize,
        winsize_min=8,
        img_size=(H, W),
        gridxy_roi_range=GridxyROIRange(gridx=(0, W - 1), gridy=(0, H - 1)),
        reference_mode="accumulative",
        admm_max_iter=admm_iter,
        method_to_compute_strain=2,
        strain_plane_fit_rad=20.0,
        show_plots=False,
        tol=1e-2,
        disp_smoothness=5e-4,
        strain_smoothness=1e-5,
    )

    timings = {}

    for run in range(n_runs):
        # --- S2b: Normalize ---
        with Timer("S2b_normalize") as t:
            img_normalized, clamped_roi = normalize_images(images, para.gridxy_roi_range)
            from dataclasses import replace
            para_run = replace(para, gridxy_roi_range=clamped_roi, img_size=(H, W))
        timings.setdefault(t.name, []).append(t.elapsed)

        f_img = img_normalized[0]
        g_img = img_normalized[1]
        f_mask = mask.copy()

        # --- Image gradient ---
        with Timer("S2b_gradient") as t:
            Df = compute_image_gradient(f_img, f_mask)
        timings.setdefault(t.name, []).append(t.elapsed)

        # --- S3: Integer search ---
        with Timer("S3_integer_search") as t:
            x0, y0, u_grid, v_grid, fft_info = integer_search(f_img, g_img, para_run)
        timings.setdefault(t.name, []).append(t.elapsed)

        # --- S3: Init disp ---
        with Timer("S3_init_disp") as t:
            U0 = init_disp(u_grid, v_grid, fft_info["cc_max"], x0, y0)
        timings.setdefault(t.name, []).append(t.elapsed)

        # --- S3: Mesh setup ---
        with Timer("S3_mesh_setup") as t:
            dic_mesh = mesh_setup(x0, y0, para_run)
        timings.setdefault(t.name, []).append(t.elapsed)

        n_nodes = dic_mesh.coordinates_fem.shape[0]
        n_elems = dic_mesh.elements_fem.shape[0]

        # --- S3: Region map ---
        with Timer("S3_region_map") as t:
            region_map = precompute_node_regions(
                dic_mesh.coordinates_fem, f_mask, (H, W)
            )
        timings.setdefault(t.name, []).append(t.elapsed)

        # --- S4: Local IC-GN (6-DOF) ---
        with Timer("S4_local_icgn") as t:
            U_subpb1, F_subpb1, s4_time, conv_iter, bad_pt, mark_hole = local_icgn(
                U0, dic_mesh.coordinates_fem, Df, f_img, g_img, para_run, para_run.tol
            )
        timings.setdefault(t.name, []).append(t.elapsed)

        # --- S5: Smoothing ---
        with Timer("S5_smooth_disp") as t:
            sigma_d = para_run.winstepsize * max(0.3, 500.0 * para_run.disp_smoothness)
            U_subpb1_s = smooth_field_sparse(
                U_subpb1, dic_mesh.coordinates_fem, sigma_d, region_map, n_components=2
            )
        timings.setdefault(t.name, []).append(t.elapsed)

        with Timer("S5_smooth_strain") as t:
            sigma_s = para_run.winstepsize * max(0.3, 500.0 * para_run.strain_smoothness)
            F_subpb1_s = smooth_field_sparse(
                F_subpb1, dic_mesh.coordinates_fem, sigma_s, region_map, n_components=4
            )
        timings.setdefault(t.name, []).append(t.elapsed)

        # --- S5: Beta tuning (3 subpb2 solves) ---
        mu_val = para_run.mu
        beta_list = np.array(para_run.beta_range) * para_run.winstepsize ** 2 * mu_val

        with Timer("S5_beta_tuning") as t:
            udual_zero = np.zeros(4 * n_nodes, dtype=np.float64)
            vdual_zero = np.zeros(2 * n_nodes, dtype=np.float64)
            for beta_k in beta_list:
                U_trial = subpb2_solver(
                    dic_mesh, para_run.gauss_pt_order, beta_k, mu_val,
                    U_subpb1_s, F_subpb1_s, udual_zero, vdual_zero,
                    0.0, para_run.winstepsize,
                )
                F_trial = global_nodal_strain_fem(dic_mesh, para_run, U_trial)
        timings.setdefault(t.name, []).append(t.elapsed)

        # Pick middle beta
        beta_val = beta_list[len(beta_list) // 2]

        # --- S5: Subpb2 solve ---
        grad_dual = np.zeros(4 * n_nodes, dtype=np.float64)
        disp_dual = np.zeros(2 * n_nodes, dtype=np.float64)

        with Timer("S5_subpb2_solve") as t:
            U_subpb2 = subpb2_solver(
                dic_mesh, para_run.gauss_pt_order, beta_val, mu_val,
                U_subpb1_s, F_subpb1_s, grad_dual, disp_dual,
                0.0, para_run.winstepsize,
            )
        timings.setdefault(t.name, []).append(t.elapsed)

        # --- S5: FEM strain ---
        with Timer("S5_fem_strain") as t:
            F_subpb2 = global_nodal_strain_fem(dic_mesh, para_run, U_subpb2)
        timings.setdefault(t.name, []).append(t.elapsed)

        # --- S6: ADMM iterations ---
        admm_timings = {"subpb1": [], "subpb2": [], "fem_strain": [], "smooth": []}
        U_s1, F_s1 = U_subpb1_s.copy(), F_subpb1_s.copy()
        U_s2, F_s2 = U_subpb2.copy(), F_subpb2.copy()
        grad_dual = F_s2 - F_s1
        disp_dual = U_s2 - U_s1

        winsize_list = np.full((n_nodes, 2), para_run.winsize, dtype=np.float64)
        para_admm = replace(para_run, winsize_list=winsize_list)

        with Timer("S6_admm_total") as t_admm:
            for step in range(2, admm_iter + 1):
                t0 = time.perf_counter()
                U_s1, sub1_time, _, _ = subpb1_solver(
                    U_s2, F_s2, disp_dual, grad_dual,
                    dic_mesh.coordinates_fem, Df, f_img, g_img,
                    mu_val, beta_val, para_admm, para_run.tol,
                )
                admm_timings["subpb1"].append(time.perf_counter() - t0)
                F_s1 = F_s2.copy()

                t0 = time.perf_counter()
                U_s2 = subpb2_solver(
                    dic_mesh, para_run.gauss_pt_order, beta_val, mu_val,
                    U_s1, F_s1, grad_dual, disp_dual,
                    0.0, para_run.winstepsize,
                )
                admm_timings["subpb2"].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                F_s2 = global_nodal_strain_fem(dic_mesh, para_run, U_s2)
                admm_timings["fem_strain"].append(time.perf_counter() - t0)

                grad_dual = F_s2 - F_s1
                disp_dual = U_s2 - U_s1
        timings.setdefault(t_admm.name, []).append(t_admm.elapsed)

        # --- S8: Strain computation ---
        with Timer("S8_compute_strain") as t:
            strain_region_map = precompute_node_regions(
                dic_mesh.coordinates_fem, f_mask, (H, W)
            )
            U_accum = U_s2.copy()
            strain_result = compute_strain(dic_mesh, para_run, U_accum, strain_region_map)
        timings.setdefault(t.name, []).append(t.elapsed)

    # --- Print results ---
    print(f"\n  Mesh: {n_nodes} nodes, {n_elems} elements")
    print(f"\n  {'Section':<30} {'Mean (s)':>10} {'% Total':>10}")
    print(f"  {'-'*50}")

    total = sum(np.mean(v) for v in timings.values())
    for name in [
        "S2b_normalize", "S2b_gradient",
        "S3_integer_search", "S3_init_disp", "S3_mesh_setup", "S3_region_map",
        "S4_local_icgn",
        "S5_smooth_disp", "S5_smooth_strain", "S5_beta_tuning",
        "S5_subpb2_solve", "S5_fem_strain",
        "S6_admm_total",
        "S8_compute_strain",
    ]:
        if name in timings:
            mean_t = np.mean(timings[name])
            pct = mean_t / total * 100
            print(f"  {name:<30} {mean_t:>10.4f} {pct:>9.1f}%")

    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<30} {total:>10.4f} {'100.0%':>10}")

    # ADMM breakdown
    if admm_timings["subpb1"]:
        print(f"\n  ADMM breakdown (per step avg, {admm_iter - 1} steps):")
        for k in ["subpb1", "subpb2", "fem_strain"]:
            if admm_timings[k]:
                avg = np.mean(admm_timings[k])
                print(f"    {k:<25} {avg:.4f}s")

    return timings, total


def main():
    print("=" * 70)
    print("  STAQ-DIC Python Pipeline Performance Benchmark")
    print("=" * 70)

    configs = [
        # (H, W, winstepsize, winsize, admm_iter)
        (128, 128, 16, 32, 3),
        (256, 256, 16, 32, 3),
        (512, 512, 16, 32, 3),
    ]

    all_results = []
    for H, W, step, win, admm in configs:
        timings, total = benchmark_size(H, W, step, win, admm)
        all_results.append((f"{H}x{W}", timings, total))

    # Scaling summary
    print(f"\n{'='*70}")
    print(f"  Scaling Summary")
    print(f"{'='*70}")
    print(f"\n  {'Size':<12} {'Total (s)':>10} {'S4 ICGN':>10} {'S5 Subpb2':>10} {'S6 ADMM':>10} {'S8 Strain':>10}")
    print(f"  {'-'*62}")
    for label, timings, total in all_results:
        s4 = np.mean(timings.get("S4_local_icgn", [0]))
        s5 = np.mean(timings.get("S5_subpb2_solve", [0]))
        s6 = np.mean(timings.get("S6_admm_total", [0]))
        s8 = np.mean(timings.get("S8_compute_strain", [0]))
        print(f"  {label:<12} {total:>10.3f} {s4:>10.3f} {s5:>10.3f} {s6:>10.3f} {s8:>10.3f}")


if __name__ == "__main__":
    main()
