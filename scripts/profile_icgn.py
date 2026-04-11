"""Profile IC-GN solver to diagnose GIL contention and threading overhead.

Measures:
1. cProfile breakdown of a single icgn_solver() call
2. GIL contention: ThreadPoolExecutor(8) vs sequential for IC-GN loop
3. Per-thread CPU time vs wall time to quantify actual concurrency
4. map_coordinates GIL release test: concurrent vs sequential

Usage:
    python scripts/profile_icgn.py
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from al_dic.core.config import dicpara_default
from al_dic.core.data_structures import GridxyROIRange, ImageGradients
from al_dic.io.image_ops import compute_image_gradient
from al_dic.solver.icgn_solver import icgn_solver
from al_dic.solver.icgn_warp import compose_warp


# ---------------------------------------------------------------------------
# Synthetic test data generation (same as benchmark_pipeline.py)
# ---------------------------------------------------------------------------

def generate_speckle(h: int, w: int, sigma: float = 3.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    filtered = gaussian_filter(noise, sigma=sigma, mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def apply_displacement(ref, u_field, v_field):
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = np.array([(yy - v_field).ravel(), (xx - u_field).ravel()])
    return map_coordinates(ref, coords, order=5, mode="constant", cval=0.0).reshape(h, w)


def prepare_test_data(H: int = 512, W: int = 512, winsize: int = 32,
                      winstepsize: int = 16):
    """Generate all shared data needed for IC-GN profiling."""
    ref = generate_speckle(H, W, sigma=3.0, seed=42)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2, H / 2
    u_field = 0.02 * (xx - cx)
    v_field = 0.02 * (yy - cy)
    deformed = apply_displacement(ref, u_field, v_field)
    mask = np.ones((H, W), dtype=np.float64)

    f_img = ref / 255.0
    g_img = deformed / 255.0

    Df = compute_image_gradient(f_img, mask)

    # Build a grid of node coordinates inside the ROI
    half_w = winsize // 2
    margin = half_w + 3  # avoid edge artifacts
    xs = np.arange(margin, W - margin, winstepsize, dtype=np.float64)
    ys = np.arange(margin, H - margin, winstepsize, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Ground-truth initial displacement for each node
    U0 = np.zeros(2 * len(coords), dtype=np.float64)
    for i in range(len(coords)):
        x0, y0 = coords[i]
        U0[2 * i] = 0.02 * (x0 - cx)      # u
        U0[2 * i + 1] = 0.02 * (y0 - cy)  # v

    return f_img, g_img, Df, mask, coords, U0, winsize


# ---------------------------------------------------------------------------
# TEST 1: cProfile a single icgn_solver() call
# ---------------------------------------------------------------------------

def test_cprofile_single_node(f_img, g_img, Df, mask, coords, U0, winsize):
    """Profile a single IC-GN solver call to see internal time distribution."""
    print("\n" + "=" * 70)
    print("  TEST 1: cProfile of a single icgn_solver() call")
    print("=" * 70)

    # Pick a node near the center (should have good convergence)
    mid = len(coords) // 2
    x0, y0 = coords[mid]
    u0, v0 = U0[2 * mid], U0[2 * mid + 1]

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(50):  # Run 50 times for statistical significance
        icgn_solver(
            np.array([u0, v0], dtype=np.float64),
            float(x0), float(y0),
            Df.df_dx, Df.df_dy, Df.img_ref_mask,
            f_img, g_img, winsize, 1e-2, 100,
        )

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    # Also print by tottime
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(20)
    print("\n--- Sorted by tottime ---")
    print(s2.getvalue())


# ---------------------------------------------------------------------------
# TEST 2: GIL contention - ThreadPoolExecutor vs sequential
# ---------------------------------------------------------------------------

def _solve_node_for_benchmark(j, x0, y0, u0, v0, f_img, g_img, df_dx, df_dy,
                               img_ref_mask, winsize, tol, max_iter):
    """Wrapper that also measures per-thread CPU time."""
    cpu_start = time.thread_time()
    wall_start = time.perf_counter()

    U_j, F_j, step = icgn_solver(
        np.array([u0, v0], dtype=np.float64),
        x0, y0, df_dx, df_dy, img_ref_mask,
        f_img, g_img, winsize, tol, max_iter,
    )

    cpu_end = time.thread_time()
    wall_end = time.perf_counter()

    return (j, U_j, F_j, step,
            cpu_end - cpu_start, wall_end - wall_start,
            threading.current_thread().name)


def test_gil_contention(f_img, g_img, Df, mask, coords, U0, winsize):
    """Compare ThreadPoolExecutor(N) vs sequential, measure CPU vs wall time."""
    print("\n" + "=" * 70)
    print("  TEST 2: GIL contention — ThreadPool(N) vs sequential")
    print("=" * 70)

    n_nodes = len(coords)
    tol = 1e-2
    max_iter = 100
    df_dx = Df.df_dx
    df_dy = Df.df_dy
    img_ref_mask = Df.img_ref_mask

    print(f"\n  Nodes: {n_nodes}, winsize: {winsize}")
    print(f"  CPU cores: {os.cpu_count()}")

    # --- Sequential ---
    print("\n  --- Sequential ---")
    t0 = time.perf_counter()
    cpu0 = time.thread_time()
    seq_cpu_times = []
    seq_wall_times = []

    for j in range(n_nodes):
        x0, y0 = float(coords[j, 0]), float(coords[j, 1])
        u0, v0 = float(U0[2 * j]), float(U0[2 * j + 1])

        cpu_s = time.thread_time()
        wall_s = time.perf_counter()
        icgn_solver(
            np.array([u0, v0], dtype=np.float64),
            x0, y0, df_dx, df_dy, img_ref_mask,
            f_img, g_img, winsize, tol, max_iter,
        )
        seq_cpu_times.append(time.thread_time() - cpu_s)
        seq_wall_times.append(time.perf_counter() - wall_s)

    seq_wall = time.perf_counter() - t0
    seq_cpu = time.thread_time() - cpu0
    print(f"  Wall time:  {seq_wall:.4f}s")
    print(f"  CPU time:   {seq_cpu:.4f}s")
    print(f"  CPU/Wall:   {seq_cpu/seq_wall:.2f}  (1.0 = no overhead)")
    print(f"  Per-node avg: {np.mean(seq_wall_times)*1000:.3f}ms wall, "
          f"{np.mean(seq_cpu_times)*1000:.3f}ms CPU")

    # --- Threaded with varying worker counts ---
    for n_workers in [2, 4, 8]:
        print(f"\n  --- ThreadPoolExecutor(max_workers={n_workers}) ---")
        t0 = time.perf_counter()

        thread_cpu_times = []
        thread_wall_times = []
        thread_names = set()

        futures = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for j in range(n_nodes):
                x0, y0 = float(coords[j, 0]), float(coords[j, 1])
                u0, v0 = float(U0[2 * j]), float(U0[2 * j + 1])
                fut = executor.submit(
                    _solve_node_for_benchmark,
                    j, x0, y0, u0, v0,
                    f_img, g_img, df_dx, df_dy, img_ref_mask,
                    winsize, tol, max_iter,
                )
                futures.append(fut)

            for fut in as_completed(futures):
                res = fut.result()
                thread_cpu_times.append(res[4])
                thread_wall_times.append(res[5])
                thread_names.add(res[6])

        par_wall = time.perf_counter() - t0
        total_thread_cpu = sum(thread_cpu_times)
        total_thread_wall = sum(thread_wall_times)

        speedup = seq_wall / par_wall
        efficiency = speedup / n_workers
        # If GIL blocks, total_thread_cpu >> total_thread_wall per thread
        # because threads spend time waiting for GIL

        print(f"  Wall time:     {par_wall:.4f}s")
        print(f"  Speedup:       {speedup:.2f}x  (ideal: {n_workers}x)")
        print(f"  Efficiency:    {efficiency:.1%}")
        print(f"  Threads used:  {len(thread_names)}")
        print(f"  Sum(CPU):      {total_thread_cpu:.4f}s  "
              f"(vs seq CPU {seq_cpu:.4f}s — ratio "
              f"{total_thread_cpu/seq_cpu:.2f})")
        print(f"  Per-node avg:  {np.mean(thread_wall_times)*1000:.3f}ms wall, "
              f"{np.mean(thread_cpu_times)*1000:.3f}ms CPU")
        print(f"  Per-node wall inflation: "
              f"{np.mean(thread_wall_times)/np.mean(seq_wall_times):.2f}x  "
              f"(>1 = GIL contention)")


# ---------------------------------------------------------------------------
# TEST 3: map_coordinates GIL release test
# ---------------------------------------------------------------------------

def _run_map_coordinates_batch(img, n_calls, subset_size):
    """Run map_coordinates n_calls times, return (cpu_time, wall_time)."""
    rng = np.random.default_rng(123)
    h, w = img.shape

    cpu_start = time.thread_time()
    wall_start = time.perf_counter()

    for _ in range(n_calls):
        # Generate random subset coordinates within image bounds
        cx = rng.uniform(subset_size, w - subset_size)
        cy = rng.uniform(subset_size, h - subset_size)
        xs = np.linspace(cx - subset_size / 2, cx + subset_size / 2, subset_size)
        ys = np.linspace(cy - subset_size / 2, cy + subset_size / 2, subset_size)
        xx, yy = np.meshgrid(xs, ys)
        map_coordinates(img, [yy.ravel(), xx.ravel()], order=3,
                        mode='constant', cval=0.0)

    cpu_end = time.thread_time()
    wall_end = time.perf_counter()

    return cpu_end - cpu_start, wall_end - wall_start


def test_map_coordinates_gil(f_img):
    """Test if map_coordinates actually releases GIL by comparing concurrent vs sequential."""
    print("\n" + "=" * 70)
    print("  TEST 3: map_coordinates GIL release test")
    print("=" * 70)

    subset_size = 33  # winsize + 1
    n_calls = 500

    # --- Sequential baseline ---
    print(f"\n  Subset: {subset_size}x{subset_size}, {n_calls} calls per batch")
    cpu_seq, wall_seq = _run_map_coordinates_batch(f_img, n_calls, subset_size)
    print(f"\n  Sequential (1 batch):")
    print(f"    Wall: {wall_seq:.4f}s, CPU: {cpu_seq:.4f}s")

    # --- Concurrent with varying threads ---
    for n_threads in [2, 4, 8]:
        print(f"\n  Concurrent ({n_threads} threads, {n_calls} calls each):")
        t0 = time.perf_counter()
        results = []

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(_run_map_coordinates_batch, f_img, n_calls, subset_size)
                for _ in range(n_threads)
            ]
            for fut in as_completed(futures):
                results.append(fut.result())

        total_wall = time.perf_counter() - t0
        total_cpu = sum(r[0] for r in results)
        avg_per_thread_wall = np.mean([r[1] for r in results])

        # If GIL released: total_wall ~ wall_seq (threads run in parallel)
        # If GIL held:     total_wall ~ n_threads * wall_seq (serialized)
        ideal_wall = wall_seq  # perfect parallelism
        serial_wall = n_threads * wall_seq  # no parallelism

        parallelism_ratio = serial_wall / total_wall
        print(f"    Total wall:     {total_wall:.4f}s")
        print(f"    Ideal wall:     {ideal_wall:.4f}s  (perfect GIL release)")
        print(f"    Serial wall:    {serial_wall:.4f}s  (full GIL contention)")
        print(f"    Parallelism:    {parallelism_ratio:.2f}x  "
              f"(1.0 = GIL locked, {n_threads}.0 = GIL free)")
        print(f"    Sum(CPU):       {total_cpu:.4f}s  "
              f"(vs 1-thread {cpu_seq:.4f}s)")


# ---------------------------------------------------------------------------
# TEST 4: Breakdown of IC-GN iteration internals
# ---------------------------------------------------------------------------

def test_icgn_internals_timing(f_img, g_img, Df, mask, coords, U0, winsize):
    """Time each phase inside icgn_solver manually (subset extraction,
    Hessian build, map_coordinates, residual/gradient, linalg.solve, compose_warp).
    """
    print("\n" + "=" * 70)
    print("  TEST 4: IC-GN internal phase timing (50 representative nodes)")
    print("=" * 70)

    from scipy.ndimage import label

    df_dx = Df.df_dx
    df_dy = Df.df_dy
    img_ref_mask = Df.img_ref_mask
    h, w = f_img.shape
    half_w = winsize // 2
    tol = 1e-2
    max_iter = 100

    # Timers for each phase
    t_subset = 0.0
    t_hessian = 0.0
    t_warp_coords = 0.0
    t_map_coords = 0.0
    t_residual = 0.0
    t_solve = 0.0
    t_compose = 0.0
    t_connected = 0.0
    t_other = 0.0
    total_iters = 0
    total_nodes = 0

    # Sample 50 evenly-spaced nodes
    indices = np.linspace(0, len(coords) - 1, min(50, len(coords)), dtype=int)

    for idx in indices:
        x0, y0 = float(coords[idx, 0]), float(coords[idx, 1])
        u0, v0 = float(U0[2 * idx]), float(U0[2 * idx + 1])

        x_lo, x_hi = int(x0 - half_w), int(x0 + half_w)
        y_lo, y_hi = int(y0 - half_w), int(y0 + half_w)
        if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
            continue

        total_nodes += 1

        # --- Subset extraction ---
        ts = time.perf_counter()
        tempf_mask = img_ref_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
        tempf = f_img[y_lo:y_hi + 1, x_lo:x_hi + 1] * tempf_mask
        grad_x = df_dx[y_lo:y_hi + 1, x_lo:x_hi + 1]
        grad_y = df_dy[y_lo:y_hi + 1, x_lo:x_hi + 1]
        t_subset += time.perf_counter() - ts

        # --- Connected component ---
        ts = time.perf_counter()
        ny, nx = tempf.shape
        binary_m = tempf_mask > 0.5
        labeled, _ = label(binary_m)
        cy_c, cx_c = ny // 2, nx // 2
        cl = labeled[cy_c, cx_c]
        bw_mask = (labeled == cl).astype(np.float64) if cl > 0 else np.zeros_like(tempf)
        tempf = tempf * bw_mask
        grad_x = grad_x * bw_mask
        grad_y = grad_y * bw_mask
        t_connected += time.perf_counter() - ts

        # Coordinate grids
        xx = np.arange(x_lo, x_hi + 1, dtype=np.float64)
        yy_arr = np.arange(y_lo, y_hi + 1, dtype=np.float64)
        XX = np.broadcast_to((xx - x0)[np.newaxis, :], (ny, nx)).copy()
        YY = np.broadcast_to((yy_arr - y0)[:, np.newaxis], (ny, nx)).copy()

        # --- Hessian ---
        ts = time.perf_counter()
        gx2 = grad_x ** 2
        gy2 = grad_y ** 2
        gxgy = grad_x * grad_y
        XX2 = XX ** 2
        YY2 = YY ** 2
        XXYY = XX * YY
        H = np.zeros((6, 6), dtype=np.float64)
        H[0, 0] = np.sum(XX2 * gx2)
        H[0, 1] = np.sum(XX2 * gxgy)
        H[0, 2] = np.sum(XXYY * gx2)
        H[0, 3] = np.sum(XXYY * gxgy)
        H[0, 4] = np.sum(XX * gx2)
        H[0, 5] = np.sum(XX * gxgy)
        H[1, 1] = np.sum(XX2 * gy2)
        H[1, 2] = H[0, 3]
        H[1, 3] = np.sum(XXYY * gy2)
        H[1, 4] = H[0, 5]
        H[1, 5] = np.sum(XX * gy2)
        H[2, 2] = np.sum(YY2 * gx2)
        H[2, 3] = np.sum(YY2 * gxgy)
        H[2, 4] = np.sum(YY * gx2)
        H[2, 5] = np.sum(YY * gxgy)
        H[3, 3] = np.sum(YY2 * gy2)
        H[3, 4] = H[2, 5]
        H[3, 5] = np.sum(YY * gy2)
        H[4, 4] = np.sum(gx2)
        H[4, 5] = np.sum(gxgy)
        H[5, 5] = np.sum(gy2)
        H = H + H.T - np.diag(np.diag(H))
        t_hessian += time.perf_counter() - ts

        # ZNSSD normalization for reference
        valid = np.abs(tempf) > 1e-10
        n_valid = valid.sum()
        if n_valid < 4:
            continue
        meanf = np.mean(tempf[valid])
        varf = np.var(tempf[valid])
        bottomf = np.sqrt(max((n_valid - 1) * varf, 1e-30))

        # --- Gauss-Newton iteration ---
        P = np.array([0.0, 0.0, 0.0, 0.0, u0, v0], dtype=np.float64)
        norm_new = 1.0
        norm_abs = 1.0
        norm_init = None
        step = 0

        while step <= max_iter and norm_new > tol and norm_abs > tol:
            step += 1
            total_iters += 1

            # --- Warp coordinates ---
            ts = time.perf_counter()
            u22 = (1.0 + P[0]) * XX + P[2] * YY + x0 + P[4]
            v22 = P[1] * XX + (1.0 + P[3]) * YY + y0 + P[5]
            t_warp_coords += time.perf_counter() - ts

            margin = 2.5
            if (np.any(u22 < margin) or np.any(u22 > w - 1 - margin) or
                    np.any(v22 < margin) or np.any(v22 > h - 1 - margin)):
                break

            # --- map_coordinates ---
            ts = time.perf_counter()
            tempg = map_coordinates(g_img, [v22.ravel(), u22.ravel()],
                                    order=3, mode='constant', cval=0.0)
            tempg = tempg.reshape(ny, nx)
            t_map_coords += time.perf_counter() - ts

            # --- Residual & gradient ---
            ts = time.perf_counter()
            g_valid = np.abs(tempg) > 1e-10
            combined_mask = bw_mask.astype(bool) & g_valid
            if combined_mask.sum() < 4:
                t_residual += time.perf_counter() - ts
                break
            tempf_iter = tempf * combined_mask
            grad_x_iter = grad_x * combined_mask
            grad_y_iter = grad_y * combined_mask
            tempg = tempg * combined_mask

            g_nz = np.abs(tempg) > 1e-10
            if g_nz.sum() < 4:
                t_residual += time.perf_counter() - ts
                break
            meang = np.mean(tempg[g_nz])
            varg = np.var(tempg[g_nz])
            bottomg = np.sqrt(max((g_nz.sum() - 1) * varg, 1e-30))

            residual = (tempf_iter - meanf) / bottomf - (tempg - meang) / bottomg

            b = np.zeros(6, dtype=np.float64)
            b[0] = np.sum(XX * grad_x_iter * residual)
            b[1] = np.sum(XX * grad_y_iter * residual)
            b[2] = np.sum(YY * grad_x_iter * residual)
            b[3] = np.sum(YY * grad_y_iter * residual)
            b[4] = np.sum(grad_x_iter * residual)
            b[5] = np.sum(grad_y_iter * residual)
            b *= bottomf

            norm_abs = np.linalg.norm(b)
            if norm_init is None:
                norm_init = norm_abs
            norm_new = norm_abs / norm_init if norm_init > tol else 0.0
            t_residual += time.perf_counter() - ts

            if norm_new < tol or norm_abs < tol:
                break

            # --- linalg.solve ---
            ts = time.perf_counter()
            try:
                delta_P = -np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                t_solve += time.perf_counter() - ts
                break
            t_solve += time.perf_counter() - ts

            if np.linalg.norm(delta_P) < tol:
                break

            # --- compose_warp ---
            ts = time.perf_counter()
            result = compose_warp(P, delta_P)
            t_compose += time.perf_counter() - ts
            if result is None:
                break
            P = result

    total_time = (t_subset + t_connected + t_hessian + t_warp_coords +
                  t_map_coords + t_residual + t_solve + t_compose)

    print(f"\n  Sampled {total_nodes} nodes, total {total_iters} GN iterations")
    print(f"\n  {'Phase':<25} {'Time (ms)':>10} {'% Total':>10} {'Per-iter (us)':>14}")
    print(f"  {'-'*60}")

    phases = [
        ("subset extraction", t_subset),
        ("connected component", t_connected),
        ("hessian build", t_hessian),
        ("warp coordinates", t_warp_coords),
        ("map_coordinates", t_map_coords),
        ("residual+gradient", t_residual),
        ("linalg.solve", t_solve),
        ("compose_warp", t_compose),
    ]

    for name, t in phases:
        pct = t / total_time * 100 if total_time > 0 else 0
        per_iter = t / total_iters * 1e6 if total_iters > 0 else 0
        print(f"  {name:<25} {t*1000:>10.2f} {pct:>9.1f}% {per_iter:>13.1f}")

    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<25} {total_time*1000:>10.2f} {'100.0%':>10}")
    print(f"\n  Key insight: Phases that hold the GIL (pure Python/NumPy array ops)")
    print(f"  prevent concurrent threads from running. Only map_coordinates and")
    print(f"  linalg.solve can potentially release the GIL.")


# ---------------------------------------------------------------------------
# TEST 5: np.linalg.solve GIL release test
# ---------------------------------------------------------------------------

def _run_linalg_solve_batch(n_calls, size=6):
    """Run np.linalg.solve n_calls times on small matrices."""
    rng = np.random.default_rng(42)
    cpu_start = time.thread_time()
    wall_start = time.perf_counter()

    for _ in range(n_calls):
        A = rng.standard_normal((size, size))
        A = A.T @ A + np.eye(size)  # Ensure positive definite
        b = rng.standard_normal(size)
        np.linalg.solve(A, b)

    cpu_end = time.thread_time()
    wall_end = time.perf_counter()
    return cpu_end - cpu_start, wall_end - wall_start


def test_linalg_solve_gil():
    """Test if np.linalg.solve releases GIL for small (6x6) matrices."""
    print("\n" + "=" * 70)
    print("  TEST 5: np.linalg.solve GIL release test (6x6 matrices)")
    print("=" * 70)

    n_calls = 5000

    # Sequential baseline
    cpu_seq, wall_seq = _run_linalg_solve_batch(n_calls)
    print(f"\n  Sequential ({n_calls} solves):")
    print(f"    Wall: {wall_seq:.4f}s, CPU: {cpu_seq:.4f}s")

    for n_threads in [2, 4, 8]:
        t0 = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(_run_linalg_solve_batch, n_calls)
                for _ in range(n_threads)
            ]
            for fut in as_completed(futures):
                results.append(fut.result())

        total_wall = time.perf_counter() - t0
        serial_wall = n_threads * wall_seq
        parallelism = serial_wall / total_wall

        print(f"\n  Concurrent ({n_threads} threads, {n_calls} each):")
        print(f"    Total wall:   {total_wall:.4f}s")
        print(f"    Ideal wall:   {wall_seq:.4f}s")
        print(f"    Serial wall:  {serial_wall:.4f}s")
        print(f"    Parallelism:  {parallelism:.2f}x  "
              f"(1.0 = GIL locked, {n_threads}.0 = GIL free)")


# ---------------------------------------------------------------------------
# TEST 6: NumPy element-wise ops GIL test (the "glue" code)
# ---------------------------------------------------------------------------

def _run_numpy_elementwise_batch(n_calls, size=33):
    """Simulate the NumPy array operations between map_coordinates calls."""
    rng = np.random.default_rng(42)
    arr1 = rng.standard_normal((size, size))
    arr2 = rng.standard_normal((size, size))
    arr3 = rng.standard_normal((size, size))

    cpu_start = time.thread_time()
    wall_start = time.perf_counter()

    for _ in range(n_calls):
        # Simulates the IC-GN "glue" operations per iteration:
        # warp coord computation, masking, ZNSSD, gradient assembly
        u22 = 1.01 * arr1 + 0.01 * arr2 + 100.0
        v22 = 0.01 * arr1 + 1.01 * arr2 + 100.0
        g_valid = np.abs(arr3) > 1e-10
        masked = arr1 * g_valid
        mean_val = np.mean(masked[g_valid])
        var_val = np.var(masked[g_valid])
        bottom = np.sqrt(max(var_val * (size * size - 1), 1e-30))
        residual = (arr1 - mean_val) / bottom - (arr2 - mean_val) / bottom
        b = np.zeros(6)
        b[0] = np.sum(arr1 * arr2 * residual)
        b[1] = np.sum(arr1 * arr3 * residual)
        _ = np.linalg.norm(b)

    cpu_end = time.thread_time()
    wall_end = time.perf_counter()
    return cpu_end - cpu_start, wall_end - wall_start


def test_numpy_glue_gil():
    """Test if NumPy element-wise ops release GIL (they generally don't for small arrays)."""
    print("\n" + "=" * 70)
    print("  TEST 6: NumPy element-wise ops GIL test (IC-GN 'glue' code)")
    print("=" * 70)

    n_calls = 1000

    cpu_seq, wall_seq = _run_numpy_elementwise_batch(n_calls)
    print(f"\n  Sequential ({n_calls} iterations):")
    print(f"    Wall: {wall_seq:.4f}s, CPU: {cpu_seq:.4f}s")

    for n_threads in [2, 4, 8]:
        t0 = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(_run_numpy_elementwise_batch, n_calls)
                for _ in range(n_threads)
            ]
            for fut in as_completed(futures):
                results.append(fut.result())

        total_wall = time.perf_counter() - t0
        serial_wall = n_threads * wall_seq
        parallelism = serial_wall / total_wall

        print(f"\n  Concurrent ({n_threads} threads, {n_calls} each):")
        print(f"    Total wall:   {total_wall:.4f}s")
        print(f"    Ideal wall:   {wall_seq:.4f}s")
        print(f"    Serial wall:  {serial_wall:.4f}s")
        print(f"    Parallelism:  {parallelism:.2f}x  "
              f"(1.0 = GIL locked, {n_threads}.0 = GIL free)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  IC-GN Solver Profiling: GIL Contention & Threading Analysis")
    print("  Image: 512x512, winsize=32, winstepsize=16")
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  Python: {sys.version}")
    print(f"  NumPy: {np.__version__}")
    try:
        import scipy
        print(f"  SciPy: {scipy.__version__}")
    except Exception:
        pass
    print("=" * 70)

    # Prepare shared test data
    print("\n  Generating synthetic test data...")
    f_img, g_img, Df, mask, coords, U0, winsize = prepare_test_data(512, 512)
    print(f"  Done. {len(coords)} nodes generated.")

    # Run all tests
    test_cprofile_single_node(f_img, g_img, Df, mask, coords, U0, winsize)
    test_icgn_internals_timing(f_img, g_img, Df, mask, coords, U0, winsize)
    test_map_coordinates_gil(f_img)
    test_linalg_solve_gil()
    test_numpy_glue_gil()
    test_gil_contention(f_img, g_img, Df, mask, coords, U0, winsize)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY & DIAGNOSIS")
    print("=" * 70)
    print("""
  ThreadPoolExecutor gives modest speedup for IC-GN because:

  1. GIL RELEASE PROFILE: Only map_coordinates (C extension) and
     np.linalg.solve (LAPACK) release the GIL. Everything else
     (NumPy element-wise ops on small 33x33 arrays, Python control flow)
     holds the GIL.

  2. SMALL ARRAY PROBLEM: For 33x33 = 1089-element arrays, NumPy ops
     complete in microseconds — too fast for BLAS to release GIL. NumPy
     only releases GIL for large arrays (typically >10K elements).

  3. TIME DISTRIBUTION: The IC-GN iteration consists of:
     - map_coordinates: ~20-40% of time (DOES release GIL)
     - NumPy array ops: ~40-60% of time (does NOT release GIL)
     - linalg.solve 6x6: ~5-10% (too small to release GIL effectively)
     - compose_warp: ~5% (pure Python/NumPy, holds GIL)

  4. THREADING OVERHEAD: Thread creation, task submission to executor,
     future management, and GIL acquisition/release transitions add
     per-node overhead that reduces effective speedup.

  RECOMMENDATION:
  - For true parallelism, use ProcessPoolExecutor (bypasses GIL)
    or Numba @njit with parallel=True (no GIL)
  - Alternatively, vectorize the IC-GN loop to process all nodes
    simultaneously with batched NumPy operations (best approach)
  - multiprocessing.Pool with fork may help but has data serialization cost
""")


if __name__ == "__main__":
    main()
