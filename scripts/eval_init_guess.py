"""Evaluation harness for IC-GN initial-guess strategies.

This script compares five init-guess strategies on a diverse catalog
of motion scenarios and reports accuracy, speed, and convergence.

Strategies
----------
1. ``previous``         — warm start from U_{i-1} (current pyALDIC default)
2. ``linear_extrap``    — U_i = 2 U_{i-1} - U_{i-2}
3. ``narrow_fft``       — small FFT window (±5 px) centered on U_{i-1}
4. ``adaptive_fallback``— ``previous`` + auto-switch to ``fft_every``
                          when IC-GN convergence rate drops below a
                          threshold. Threshold is scanned.
5. ``fft_every``        — full FFT every frame (baseline)

All FFT-based strategies use the pipeline's auto-expand logic
(retry with 2x search radius until NCC peaks are no longer clipped,
capped at image half-size).

Scenarios
---------
Temporal (8):   zero, const velocity, const accel, sinusoidal, step
                impulse, chirp, stop-and-go, random walk
Spatial  (6):   uniform, rotation, expand, hotspot, crack,
                shear band

Scenario matrix (15 cases, not full 48 cross-product):
  All 8 temporal x S_uniform                               (8)
  T2,T4 x S_rotation     (uniform rotation + oscillation) (2)
  T2,T5 x S_hotspot      (localized + impulse)            (2)
  T2,T5 x S_crack        (discontinuity)                  (2)
  T2   x S_shear_band                                     (1)

Output
------
PDF report to reports/init_guess_eval.pdf (gitignored). Invoke with
``--preview`` to just render scenario previews without running the
full evaluation (faster iteration while iterating on scenarios).

Not a pytest test — this is an evaluation script. Runs take a few
minutes on a typical laptop.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "tests"))

# Lazy imports for heavy deps (al_dic, scipy) happen inside function
# bodies so that `--preview` works even without the full pipeline stack.

# ---------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------

IMG_H = IMG_W = 256
N_FRAMES = 20
CX = (IMG_W - 1) / 2.0
CY = (IMG_H - 1) / 2.0

# Peak inter-frame motion. Chosen to be large enough that differences
# between strategies are visible, but small enough that all strategies
# at least have a chance of working (a 30 px jump would trivially kill
# every non-FFT strategy).
PEAK_DISP_PX = 12.0

SPECKLE_SIGMA = 2.5
SPECKLE_SEED = 2025

# ---------------------------------------------------------------------
# Temporal patterns: amplitude(t) in [-1, 1]
# ---------------------------------------------------------------------

def temporal_zero(t: int) -> float:
    return 0.0


def temporal_constant_velocity(t: int) -> float:
    """Linear ramp 0 -> 1."""
    return t / max(1, N_FRAMES - 1)


def temporal_constant_acceleration(t: int) -> float:
    """Quadratic ramp 0 -> 1 (velocity grows linearly)."""
    x = t / max(1, N_FRAMES - 1)
    return x * x


def temporal_sinusoidal(t: int) -> float:
    """1.5 full cycles over the sequence -> 3 direction reversals."""
    return float(np.sin(2.0 * np.pi * 1.5 * t / max(1, N_FRAMES - 1)))


def temporal_step_impulse(t: int) -> float:
    """Zero for first half, then jump to 1. Tests sudden motion onset."""
    return 0.0 if t < N_FRAMES // 2 else 1.0


def temporal_chirp(t: int) -> float:
    """sin(2 pi * (t/T)^2 * k) — frequency grows linearly with time."""
    x = t / max(1, N_FRAMES - 1)
    phase = 2.0 * np.pi * 2.0 * x * x   # integrated linear freq
    return float(np.sin(phase))


def temporal_stop_and_go(t: int) -> float:
    """Alternating rest/motion every 5 frames."""
    chunk = t // 5
    return 0.0 if chunk % 2 == 0 else 1.0


_random_walk_cache: dict[int, float] = {}   # lazy-filled per seed
_random_walk_seed = 7


def temporal_random_walk(t: int) -> float:
    """Brownian walk, pre-generated and cached so repeated calls for
    the same t return the same value."""
    if not _random_walk_cache:
        rng = np.random.default_rng(_random_walk_seed)
        # Step size chosen so cumulative range stays ~[-1, 1] across N frames
        step_std = 1.5 / np.sqrt(N_FRAMES)
        values = np.cumsum(rng.normal(0.0, step_std, N_FRAMES))
        # Normalize to [-1, 1]
        vmax = max(abs(values.min()), abs(values.max()), 1e-9)
        values = values / vmax
        for i, v in enumerate(values):
            _random_walk_cache[i] = float(v)
    return _random_walk_cache.get(t, 0.0)


TEMPORAL_FNS: dict[str, Callable[[int], float]] = {
    "T1_zero": temporal_zero,
    "T2_constant_velocity": temporal_constant_velocity,
    "T3_constant_acceleration": temporal_constant_acceleration,
    "T4_sinusoidal": temporal_sinusoidal,
    "T5_step_impulse": temporal_step_impulse,
    "T6_chirp": temporal_chirp,
    "T7_stop_and_go": temporal_stop_and_go,
    "T8_random_walk": temporal_random_walk,
}


# ---------------------------------------------------------------------
# Spatial patterns: (u, v) = spatial(xx, yy, amplitude)
# ---------------------------------------------------------------------

def spatial_uniform(
    xx: NDArray, yy: NDArray, amp: float,
) -> tuple[NDArray, NDArray]:
    """Pure x-translation. Every node sees the same displacement."""
    u = np.full_like(xx, amp * PEAK_DISP_PX, dtype=np.float64)
    v = np.zeros_like(xx, dtype=np.float64)
    return u, v


def spatial_rotation(
    xx: NDArray, yy: NDArray, amp: float,
) -> tuple[NDArray, NDArray]:
    """Rigid rotation around image center. Peak angle at amp=1 is 5 deg
    (edge displacement ~11 px at R=128)."""
    theta = amp * np.radians(5.0)
    c, s = np.cos(theta), np.sin(theta)
    dx = xx - CX
    dy = yy - CY
    u = dx * (c - 1) - dy * s
    v = dx * s + dy * (c - 1)
    return u, v


def spatial_expand(
    xx: NDArray, yy: NDArray, amp: float,
) -> tuple[NDArray, NDArray]:
    """Radial expansion. Peak magnitude at corners ~10 px."""
    factor = amp * 0.08
    u = factor * (xx - CX)
    v = factor * (yy - CY)
    return u, v


_hotspot_cx = 0.7 * IMG_W
_hotspot_cy = 0.3 * IMG_H
_hotspot_sigma = 35.0


def spatial_hotspot(
    xx: NDArray, yy: NDArray, amp: float,
) -> tuple[NDArray, NDArray]:
    """Gaussian bump of x-translation at an off-center location."""
    dx = xx - _hotspot_cx
    dy = yy - _hotspot_cy
    g = np.exp(-(dx * dx + dy * dy) / (2.0 * _hotspot_sigma ** 2))
    u = amp * PEAK_DISP_PX * g
    v = np.zeros_like(xx, dtype=np.float64)
    return u, v


def spatial_crack(
    xx: NDArray, yy: NDArray, amp: float,
) -> tuple[NDArray, NDArray]:
    """Upper half moves +x, lower half moves -x. Sharp discontinuity
    at y = H/2. Relative displacement across interface = 2 * amp * 6 px."""
    half = 0.5 * PEAK_DISP_PX
    u = np.where(yy < IMG_H / 2.0, amp * half, -amp * half).astype(np.float64)
    v = np.zeros_like(xx, dtype=np.float64)
    return u, v


def spatial_shear_band(
    xx: NDArray, yy: NDArray, amp: float,
) -> tuple[NDArray, NDArray]:
    """Narrow band (width H/4) at y = H/2 moves +x; outside moves -x.
    Two discontinuities at band edges, each with relative ΔU = amp * 8 px."""
    band_half_width = IMG_H / 8.0
    in_band = np.abs(yy - IMG_H / 2.0) < band_half_width
    u = np.where(in_band, amp * 6.0, -amp * 2.0).astype(np.float64)
    v = np.zeros_like(xx, dtype=np.float64)
    return u, v


SPATIAL_FNS: dict[
    str,
    Callable[[NDArray, NDArray, float], tuple[NDArray, NDArray]],
] = {
    "S_uniform": spatial_uniform,
    "S_rotation": spatial_rotation,
    "S_expand": spatial_expand,
    "S_hotspot": spatial_hotspot,
    "S_crack": spatial_crack,
    "S_shear_band": spatial_shear_band,
}


# ---------------------------------------------------------------------
# Scenario dataclass + registry
# ---------------------------------------------------------------------

@dataclass
class Scenario:
    """One motion scenario = (temporal pattern x spatial pattern)."""

    name: str
    temporal: str
    spatial: str
    description: str

    def temporal_fn(self) -> Callable[[int], float]:
        return TEMPORAL_FNS[self.temporal]

    def spatial_fn(self):
        return SPATIAL_FNS[self.spatial]

    def displacement_field(
        self, t: int,
    ) -> tuple[NDArray, NDArray]:
        """Ground-truth per-pixel displacement field at frame t."""
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
        amp = self.temporal_fn()(t)
        return self.spatial_fn()(xx, yy, amp)


def _build_scenarios() -> list[Scenario]:
    """Construct the 15-case evaluation matrix."""
    cases: list[Scenario] = []
    # (1) All 8 temporal patterns x S_uniform — isolates temporal effect
    for tname in TEMPORAL_FNS:
        cases.append(Scenario(
            name=f"{tname}+S_uniform",
            temporal=tname,
            spatial="S_uniform",
            description=(
                f"Temporal pattern {tname!r} driving pure x-translation."
            ),
        ))
    # (2) Rotation with smooth + reversing temporal
    for tname in ("T2_constant_velocity", "T4_sinusoidal"):
        cases.append(Scenario(
            name=f"{tname}+S_rotation",
            temporal=tname,
            spatial="S_rotation",
            description=f"Rotation of up to 5 deg driven by {tname}.",
        ))
    # (3) Hotspot (localized non-uniform) with smooth + impulsive temporal
    for tname in ("T2_constant_velocity", "T5_step_impulse"):
        cases.append(Scenario(
            name=f"{tname}+S_hotspot",
            temporal=tname,
            spatial="S_hotspot",
            description=(
                f"Gaussian-shaped bump of x-motion driven by {tname}."
            ),
        ))
    # (4) Crack (discontinuity) with smooth + impulsive temporal
    for tname in ("T2_constant_velocity", "T5_step_impulse"):
        cases.append(Scenario(
            name=f"{tname}+S_crack",
            temporal=tname,
            spatial="S_crack",
            description=(
                f"Upper and lower halves move opposite directions; "
                f"driven by {tname}."
            ),
        ))
    # (5) Shear band (thin strip opposite to surroundings) with uniform time
    cases.append(Scenario(
        name="T2_constant_velocity+S_shear_band",
        temporal="T2_constant_velocity",
        spatial="S_shear_band",
        description=(
            "Horizontal shear band moves opposite to the surrounding "
            "material (constant velocity)."
        ),
    ))
    return cases


SCENARIOS: list[Scenario] = _build_scenarios()


# ---------------------------------------------------------------------
# Frame generation via Lagrangian warp
# ---------------------------------------------------------------------

def _make_reference_image() -> NDArray:
    """Speckle reference image (deterministic for reproducibility)."""
    from conftest import generate_speckle   # lives in tests/
    return generate_speckle(IMG_H, IMG_W, sigma=SPECKLE_SIGMA, seed=SPECKLE_SEED)


def generate_scenario_frames(
    scenario: Scenario,
    ref: NDArray,
) -> tuple[list[NDArray], list[tuple[NDArray, NDArray]]]:
    """Return (images, ground_truth_fields) for N_FRAMES.

    images[0] is the reference (identical to ``ref``).
    images[t] for t >= 1 is the Lagrangian warp of ``ref`` by the
    scenario's displacement field at frame t.

    ground_truth_fields[t] is the per-pixel (u, v) field used to warp
    frame t (frame 0 gets a zero field so indexing is uniform).
    """
    from conftest import apply_displacement_lagrangian

    images = [ref.copy()]
    gt_fields: list[tuple[NDArray, NDArray]] = [
        (np.zeros_like(ref, dtype=np.float64),
         np.zeros_like(ref, dtype=np.float64))
    ]
    for t in range(1, N_FRAMES):
        # Closure over t: callable(x, y) -> displacement component
        def _u(xs, ys, _scn=scenario, _t=t):
            yy = np.asarray(ys, dtype=np.float64)
            xx = np.asarray(xs, dtype=np.float64)
            u, _ = _scn.spatial_fn()(xx, yy, _scn.temporal_fn()(_t))
            return u

        def _v(xs, ys, _scn=scenario, _t=t):
            yy = np.asarray(ys, dtype=np.float64)
            xx = np.asarray(xs, dtype=np.float64)
            _, v = _scn.spatial_fn()(xx, yy, _scn.temporal_fn()(_t))
            return v

        deformed = apply_displacement_lagrangian(ref, _u, _v)
        images.append(deformed)
        gt_fields.append(scenario.displacement_field(t))
    return images, gt_fields


# ---------------------------------------------------------------------
# IC-GN runner (pyALDIC wrapper) and FFT wrapper with auto-expand
# ---------------------------------------------------------------------

# Mesh + IC-GN parameters used throughout evaluation.
WINSIZE = 32        # IC-GN subset full width (even)
WINSTEP = 16        # node spacing
MARGIN = 16         # ROI border inside image
SEARCH_INIT = 8     # FFT search radius used as the starting point for
                    # both fft_every and narrow_fft; auto-expand will
                    # grow this if peaks hit the boundary.
ICGN_TOL = 1e-2
ICGN_MAX_ITER = 50


def _build_dic_para():
    """A DICPara for this evaluation's image geometry.

    Separated so strategies can ``replace`` its fields (e.g. auto-
    expand logic adjusting ``size_of_fft_search_region``).
    """
    from al_dic.core.config import dicpara_default
    from al_dic.core.data_structures import GridxyROIRange
    return dicpara_default(
        winsize=WINSIZE,
        winstepsize=WINSTEP,
        winsize_min=8,
        img_size=(IMG_H, IMG_W),
        gridxy_roi_range=GridxyROIRange(
            gridx=(MARGIN, IMG_W - 1 - MARGIN),
            gridy=(MARGIN, IMG_H - 1 - MARGIN),
        ),
        size_of_fft_search_region=SEARCH_INIT,
        show_plots=False,
        icgn_max_iter=ICGN_MAX_ITER,
        tol=ICGN_TOL,
    )


def _node_coordinates(para) -> NDArray:
    """Mesh node coordinates derived from the DICPara grid."""
    roi = para.gridxy_roi_range
    half_w = para.winsize // 2
    xs = np.arange(
        max(roi.gridx[0], half_w),
        min(roi.gridx[1], IMG_W - 1 - half_w) + 1,
        para.winstepsize, dtype=np.float64,
    )
    ys = np.arange(
        max(roi.gridy[0], half_w),
        min(roi.gridy[1], IMG_H - 1 - half_w) + 1,
        para.winstepsize, dtype=np.float64,
    )
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _run_icgn(
    f_img: NDArray, g_img: NDArray, coords: NDArray, U0: NDArray, para,
) -> tuple[NDArray, NDArray, float]:
    """Run local IC-GN with given initial guess.

    Returns ``(U_solved, conv_iter, elapsed_seconds)``. ``conv_iter``
    is per-node iteration count; negative values / `>= max_iter` flag
    non-converged nodes (see :func:`detect_bad_points`).
    """
    import time as _time

    from al_dic.io.image_ops import compute_image_gradient
    from al_dic.solver.local_icgn import local_icgn

    Df = compute_image_gradient(f_img, np.ones_like(f_img))
    t0 = _time.perf_counter()
    U, _F, _local_time, conv_iter, _bad_num, _mark = local_icgn(
        U0, coords, Df, f_img, g_img, para, tol=ICGN_TOL,
    )
    elapsed = _time.perf_counter() - t0
    return U, conv_iter, elapsed


def _convergence_rate(
    conv_iter: NDArray, max_iter: int,
) -> float:
    """Fraction of nodes that converged cleanly (0..1)."""
    n = conv_iter.size
    if n == 0:
        return 0.0
    good = (conv_iter > 0) & (conv_iter < max_iter)
    return float(good.sum()) / float(n)


def _full_fft_with_auto_expand(
    f_img: NDArray, g_img: NDArray, para,
) -> tuple[NDArray, dict]:
    """Run integer_search and auto-expand until no peaks clip.

    Mirrors the retry loop in pipeline.py (lines 796-833) so strategy
    comparisons use identical FFT semantics. Returns (U0 as interleaved
    vector, info dict with {search_final, retries, time_s}).
    """
    import time as _time

    from dataclasses import replace
    from al_dic.solver.init_disp import init_disp
    from al_dic.solver.integer_search import integer_search

    img_h, img_w = f_img.shape[:2]
    max_search_cap = max(32, min(img_h, img_w) // 2)

    t0 = _time.perf_counter()
    current_search = para.size_of_fft_search_region
    para_iter = para
    x0, y0, u_grid, v_grid, info = integer_search(f_img, g_img, para_iter)
    retries = 0
    for _ in range(6):
        if not info.get("peak_clipped", False):
            break
        max_disp = info["max_abs_disp"]
        needed = int(np.ceil(max_disp * 2.0)) + 2
        grown = current_search * 2
        new_search = max(needed, grown)
        if new_search >= max_search_cap:
            new_search = max_search_cap
        para_iter = replace(para, size_of_fft_search_region=new_search)
        x0, y0, u_grid, v_grid, info = integer_search(
            f_img, g_img, para_iter,
        )
        current_search = new_search
        retries += 1
        if current_search >= max_search_cap:
            break

    U0 = init_disp(u_grid, v_grid, info["cc_max"], x0, y0)
    elapsed = _time.perf_counter() - t0
    return U0, {
        "search_final": current_search,
        "retries": retries,
        "time_s": elapsed,
        "n_clipped": info.get("n_clipped", 0),
    }


def _narrow_fft_per_node(
    f_img: NDArray,
    g_img: NDArray,
    coords: NDArray,
    U_prev: NDArray,
    half_win: int,
    narrow_radius: int,
    max_expand: int = 3,
) -> tuple[NDArray, dict]:
    """Per-node narrow FFT centered on U_prev, with auto-expand.

    For each node (x, y):
        1. Extract an ``f_img`` subset centered at (x, y), size
           (2*half_win+1)^2.
        2. Extract a ``g_img`` window centered at
           (x + u_prev, y + v_prev), size (2*half_win + 2*R + 1)^2,
           where R starts at ``narrow_radius`` and doubles up to
           ``max_expand`` times if the correlation peak sits on the
           window boundary.
        3. Cross-correlate (FFT-based) and take the peak.
        4. ``U0[node] = U_prev[node] + peak_offset``.

    Returns ``(U0, info)``. ``info`` has {time_s, mean_radius,
    n_expanded}. n_expanded counts nodes that required at least one
    expansion.
    """
    import time as _time

    from scipy.signal import fftconvolve

    t0 = _time.perf_counter()
    n = coords.shape[0]
    U0 = U_prev.copy()
    H, W = f_img.shape
    n_expanded = 0
    radii_used = []

    for i in range(n):
        x, y = coords[i]
        u_prev = U_prev[2 * i]
        v_prev = U_prev[2 * i + 1]

        # Reference subset (integer pixel bounds)
        xi = int(round(x))
        yi = int(round(y))
        x_lo, x_hi = xi - half_win, xi + half_win + 1
        y_lo, y_hi = yi - half_win, yi + half_win + 1
        if x_lo < 0 or x_hi > W or y_lo < 0 or y_hi > H:
            continue   # leave U0 = U_prev at out-of-bounds nodes
        ref_sub = f_img[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
        ref_sub = ref_sub - ref_sub.mean()
        ref_norm = float(np.linalg.norm(ref_sub))
        if ref_norm < 1e-10:
            continue

        # Adaptive window: start narrow, expand if peak hits boundary
        R = narrow_radius
        for attempt in range(max_expand + 1):
            gx_center = xi + int(round(u_prev))
            gy_center = yi + int(round(v_prev))
            gx_lo = gx_center - half_win - R
            gx_hi = gx_center + half_win + R + 1
            gy_lo = gy_center - half_win - R
            gy_hi = gy_center + half_win + R + 1
            if (gx_lo < 0 or gx_hi > W or gy_lo < 0 or gy_hi > H):
                # Can't fit the expanded window in the image; stop.
                break
            def_win = g_img[gy_lo:gy_hi, gx_lo:gx_hi].astype(np.float64)
            def_win = def_win - def_win.mean()

            # FFT-based normalized cross-correlation
            # corr shape = (2R+1, 2R+1) when windows are sized as above
            corr = fftconvolve(def_win, ref_sub[::-1, ::-1], mode="valid")
            peak_y, peak_x = np.unravel_index(int(corr.argmax()), corr.shape)

            # peak_x, peak_y indexed from corner of (2R+1)-wide search region
            du_int = peak_x - R
            dv_int = peak_y - R

            # Check if peak clipped at boundary (may indicate true peak
            # lies outside this window): expand.
            clipped = (
                abs(du_int) >= R or abs(dv_int) >= R
            )
            if not clipped or attempt == max_expand:
                U0[2 * i] = u_prev + du_int
                U0[2 * i + 1] = v_prev + dv_int
                radii_used.append(R)
                break
            R *= 2
            n_expanded += 1

    elapsed = _time.perf_counter() - t0
    return U0, {
        "time_s": elapsed,
        "mean_radius": float(np.mean(radii_used)) if radii_used else 0.0,
        "n_expanded": n_expanded,
    }


# ---------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------

@dataclass
class FrameRecord:
    """Per-frame outcome for one (strategy, scenario) run."""

    frame: int
    init_time_s: float
    icgn_time_s: float
    convergence_rate: float
    mean_iter: float
    # RMSE on the INITIAL guess U0 (pre-IC-GN), averaged over all
    # nodes. Isolates the quality of the init strategy itself.
    rmse_init: float
    # RMSE on the IC-GN output U_sol, averaged over only those nodes
    # that converged cleanly. This is what end-user accuracy depends
    # on; rmse_init - rmse_total is the "IC-GN refinement delta".
    rmse_u: float
    rmse_v: float
    rmse_total: float
    fft_retries: int
    mode_used: str   # e.g. "previous", "fft", "linear_extrap"
    # seed_propagation-specific metadata (defaulted for all other modes)
    n_seeds: int = 0
    max_bfs_depth: int = 0
    seed_ncc_min: float = float("nan")
    n_solve_calls: int = 0


class InitGuessStrategy:
    """Common interface for all init-guess strategies.

    Subclasses must implement :meth:`initial_guess`.  Each strategy
    instance is *per (scenario, strategy)* and maintains whatever
    state it needs (e.g. displacement history).
    """

    name: str

    def __init__(self) -> None:
        self.history_U: list[NDArray] = []

    def initial_guess(
        self,
        t: int,
        f_img: NDArray,
        g_img: NDArray,
        coords: NDArray,
        para,
    ) -> tuple[NDArray, dict]:
        """Return (U0, metadata). Called once per frame."""
        raise NotImplementedError

    def record_solved(self, t: int, U_solved: NDArray,
                      conv_iter: NDArray, max_iter: int) -> None:
        """Called by harness after IC-GN. Default: store U_solved."""
        self.history_U.append(U_solved.copy())


class PreviousStrategy(InitGuessStrategy):
    name = "previous"

    def initial_guess(self, t, f_img, g_img, coords, para):
        if not self.history_U:
            # Frame 1 bootstraps via full FFT.
            U0, info = _full_fft_with_auto_expand(f_img, g_img, para)
            return U0, {"mode": "fft_bootstrap",
                        "init_time_s": info["time_s"],
                        "fft_retries": info["retries"]}
        return self.history_U[-1].copy(), {
            "mode": "previous", "init_time_s": 0.0, "fft_retries": 0,
        }


class LinearExtrapStrategy(InitGuessStrategy):
    name = "linear_extrap"

    def initial_guess(self, t, f_img, g_img, coords, para):
        if len(self.history_U) == 0:
            U0, info = _full_fft_with_auto_expand(f_img, g_img, para)
            return U0, {"mode": "fft_bootstrap",
                        "init_time_s": info["time_s"],
                        "fft_retries": info["retries"]}
        if len(self.history_U) == 1:
            # Only one history sample; fall back to previous.
            return self.history_U[-1].copy(), {
                "mode": "previous", "init_time_s": 0.0, "fft_retries": 0,
            }
        # U_t = 2 U_{t-1} - U_{t-2}
        u_prev = self.history_U[-1]
        u_pprev = self.history_U[-2]
        U0 = 2.0 * u_prev - u_pprev
        return U0, {
            "mode": "linear_extrap", "init_time_s": 0.0, "fft_retries": 0,
        }


class NarrowFFTStrategy(InitGuessStrategy):
    def __init__(self, narrow_radius: int = 5) -> None:
        super().__init__()
        self.narrow_radius = narrow_radius
        self.name = f"narrow_fft_R{narrow_radius}"

    def initial_guess(self, t, f_img, g_img, coords, para):
        if not self.history_U:
            U0, info = _full_fft_with_auto_expand(f_img, g_img, para)
            return U0, {"mode": "fft_bootstrap",
                        "init_time_s": info["time_s"],
                        "fft_retries": info["retries"]}
        U0, info = _narrow_fft_per_node(
            f_img, g_img, coords, self.history_U[-1],
            half_win=para.winsize // 2,
            narrow_radius=self.narrow_radius,
        )
        return U0, {
            "mode": "narrow_fft", "init_time_s": info["time_s"],
            "fft_retries": info["n_expanded"],
        }


class FFTEveryStrategy(InitGuessStrategy):
    name = "fft_every"

    def initial_guess(self, t, f_img, g_img, coords, para):
        U0, info = _full_fft_with_auto_expand(f_img, g_img, para)
        return U0, {"mode": "fft", "init_time_s": info["time_s"],
                    "fft_retries": info["retries"]}


class AdaptiveFallbackStrategy(InitGuessStrategy):
    """Start with ``previous``; retry with full FFT if IC-GN
    convergence rate drops below ``threshold``.

    The retry is invoked AFTER IC-GN returns a poor result — so the
    frame pays two IC-GN solves in the fallback case. That cost is
    recorded in metadata.
    """

    def __init__(self, threshold: float = 0.90) -> None:
        super().__init__()
        self.threshold = threshold
        self.name = f"adaptive_t{int(threshold * 100)}"
        self._pending_retry = False

    def initial_guess(self, t, f_img, g_img, coords, para):
        if not self.history_U:
            U0, info = _full_fft_with_auto_expand(f_img, g_img, para)
            return U0, {"mode": "fft_bootstrap",
                        "init_time_s": info["time_s"],
                        "fft_retries": info["retries"]}
        # Default: previous frame. If record_solved() decided last
        # frame was poor, force FFT this frame.
        if self._pending_retry:
            self._pending_retry = False
            U0, info = _full_fft_with_auto_expand(f_img, g_img, para)
            return U0, {"mode": "fft_fallback",
                        "init_time_s": info["time_s"],
                        "fft_retries": info["retries"]}
        return self.history_U[-1].copy(), {
            "mode": "previous", "init_time_s": 0.0, "fft_retries": 0,
        }

    def record_solved(self, t, U_solved, conv_iter, max_iter):
        """Note: this records the *actual* result. Next frame's
        initial_guess() will decide whether to FFT based on whether
        THIS frame's convergence was poor."""
        super().record_solved(t, U_solved, conv_iter, max_iter)
        rate = _convergence_rate(conv_iter, max_iter)
        if rate < self.threshold:
            self._pending_retry = True


class SeedPropagationStrategy(InitGuessStrategy):
    """Auto-placed seeds + F-aware BFS propagation.

    On the first frame: places one seed per connected region at the
    node with the highest single-point NCC (via the 'highest-NCC per
    region' heuristic from the Phase 5 UI plan). Auto-placement
    samples every Nth node to keep setup cost bounded.

    Each subsequent frame: runs propagate_from_seeds with the same
    seeds (no ref switch in the eval scenarios). The returned U0 is
    already post-IC-GN for reached nodes (propagate_from_seeds runs
    IC-GN per BFS layer); the harness then runs IC-GN once more on
    top — a second convergence pass that is typically a no-op.
    """

    name = "seed_propagation"

    def __init__(
        self,
        ncc_threshold: float = 0.70,
        ncc_sample_stride: int = 3,
    ) -> None:
        super().__init__()
        self.ncc_threshold = ncc_threshold
        self.ncc_sample_stride = ncc_sample_stride
        self.seed_set = None
        self.adjacency = None
        self.region_map = None

    def initial_guess(self, t, f_img, g_img, coords, para):
        import time as _time

        from al_dic.io.image_ops import compute_image_gradient
        from al_dic.solver.local_icgn import local_icgn_precompute
        from al_dic.solver.seed_propagation import (
            Seed, SeedPropagationError, SeedSet,
            build_node_adjacency, propagate_from_seeds, seed_single_point_fft,
        )
        from al_dic.mesh.mesh_setup import mesh_setup
        from al_dic.utils.region_analysis import NodeRegionMap

        t0 = _time.perf_counter()
        n_nodes = coords.shape[0]

        if self.seed_set is None:
            # One-time per-run setup.
            roi = para.gridxy_roi_range
            half_w = para.winsize // 2
            xs = np.arange(
                max(roi.gridx[0], half_w),
                min(roi.gridx[1], IMG_W - 1 - half_w) + 1,
                para.winstepsize, dtype=np.float64,
            )
            ys = np.arange(
                max(roi.gridy[0], half_w),
                min(roi.gridy[1], IMG_H - 1 - half_w) + 1,
                para.winstepsize, dtype=np.float64,
            )
            mesh = mesh_setup(xs, ys, para)
            self.adjacency = build_node_adjacency(
                mesh.elements_fem, n_nodes,
            )
            self.region_map = NodeRegionMap(
                region_node_lists=[np.arange(n_nodes, dtype=np.int64)],
                n_regions=1,
            )

            seeds = []
            for region_idx, nodes in enumerate(
                self.region_map.region_node_lists,
            ):
                best_node, best_ncc = -1, -1.0
                for node_idx in nodes[::self.ncc_sample_stride]:
                    r = seed_single_point_fft(
                        f_img, g_img,
                        (float(coords[node_idx, 0]), float(coords[node_idx, 1])),
                        para.winsize,
                        para.size_of_fft_search_region,
                    )
                    if r.valid and r.ncc_peak > best_ncc:
                        best_ncc = r.ncc_peak
                        best_node = int(node_idx)
                if best_node < 0:
                    raise RuntimeError(
                        f"SeedPropagationStrategy: region {region_idx} has "
                        f"no node with a valid single-point NCC (all windows "
                        f"out of bounds?). Check ROI + winsize."
                    )
                seeds.append(
                    Seed(node_idx=best_node, region_id=region_idx),
                )
            self.seed_set = SeedSet(
                seeds=tuple(seeds),
                ncc_threshold=self.ncc_threshold,
            )

        Df = compute_image_gradient(f_img, np.ones_like(f_img))
        ctx = local_icgn_precompute(coords, Df, f_img, para)

        try:
            result = propagate_from_seeds(
                ctx, self.seed_set, self.adjacency, f_img, g_img,
                search_radius=para.size_of_fft_search_region,
                tol=ICGN_TOL,
                node_region_map=self.region_map,
            )
        except SeedPropagationError as exc:
            # Treat algorithm-level failure as a benchmark-reportable
            # outcome (rather than aborting the whole scenario sweep):
            # return zero init and let the harness record low quality.
            elapsed = _time.perf_counter() - t0
            return np.zeros(2 * n_nodes, dtype=np.float64), {
                "mode": f"seed_prop_error:{type(exc).__name__}",
                "init_time_s": elapsed,
                "fft_retries": 0,
                "n_seeds": len(self.seed_set.seeds),
                "max_bfs_depth": 0,
                "seed_ncc_min": float("nan"),
                "n_solve_calls": 0,
            }

        U0 = np.empty(2 * n_nodes, dtype=np.float64)
        U0[0::2] = result.U_2d[:, 0]
        U0[1::2] = result.U_2d[:, 1]
        U0 = np.nan_to_num(U0, nan=0.0)

        elapsed = _time.perf_counter() - t0
        return U0, {
            "mode": "seed_propagation",
            "init_time_s": elapsed,
            "fft_retries": 0,
            "n_seeds": int(result.n_seeds),
            "max_bfs_depth": int(result.max_bfs_depth_reached),
            "seed_ncc_min": float(result.seed_ncc_min),
            "n_solve_calls": int(result.n_solve_calls),
        }


def build_strategies(
    adaptive_thresholds: tuple[float, ...] = (0.75, 0.90),
    narrow_radii: tuple[int, ...] = (5,),
    include_seed_propagation: bool = True,
) -> list[InitGuessStrategy]:
    """Factory returning one instance of each strategy variant."""
    strategies: list[InitGuessStrategy] = [
        PreviousStrategy(),
        LinearExtrapStrategy(),
        FFTEveryStrategy(),
    ]
    for r in narrow_radii:
        strategies.append(NarrowFFTStrategy(narrow_radius=r))
    for th in adaptive_thresholds:
        strategies.append(AdaptiveFallbackStrategy(threshold=th))
    if include_seed_propagation:
        strategies.append(SeedPropagationStrategy())
    return strategies


# ---------------------------------------------------------------------
# Evaluation loop (stage C)
# ---------------------------------------------------------------------

@dataclass
class ScenarioRun:
    """All per-frame records for one (strategy, scenario) pair."""

    strategy_name: str
    scenario_name: str
    frames: list[FrameRecord]

    @property
    def mean_rmse(self) -> float:
        vals = [
            f.rmse_total for f in self.frames
            if np.isfinite(f.rmse_total)
        ]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_rmse_init(self) -> float:
        """Mean RMSE of the init guess itself (before IC-GN)."""
        vals = [
            f.rmse_init for f in self.frames
            if np.isfinite(f.rmse_init)
        ]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_convergence(self) -> float:
        return float(np.mean([f.convergence_rate for f in self.frames]))

    @property
    def total_time_s(self) -> float:
        return float(sum(f.init_time_s + f.icgn_time_s for f in self.frames))


def _eval_single(
    strategy: InitGuessStrategy,
    scenario: Scenario,
    ref: NDArray,
    images: list[NDArray],
    gt_fields: list[tuple[NDArray, NDArray]],
    para,
    coords: NDArray,
) -> ScenarioRun:
    """Run one strategy on one scenario. Returns all frame outcomes."""
    records: list[FrameRecord] = []

    # Evaluate ground-truth U at node centers once per frame.
    yi = np.clip(coords[:, 1].astype(int), 0, IMG_H - 1)
    xi = np.clip(coords[:, 0].astype(int), 0, IMG_W - 1)

    for t in range(1, N_FRAMES):
        U0, meta = strategy.initial_guess(t, ref, images[t], coords, para)

        # --- Init-guess quality (pre-IC-GN) ---
        # Averaged over ALL nodes because every node received an init.
        gt_u, gt_v = gt_fields[t]
        ref_u = gt_u[yi, xi]
        ref_v = gt_v[yi, xi]
        err_u_init = U0[0::2] - ref_u
        err_v_init = U0[1::2] - ref_v
        finite = np.isfinite(err_u_init) & np.isfinite(err_v_init)
        if finite.any():
            rmse_init = float(np.sqrt(np.mean(
                err_u_init[finite] ** 2 + err_v_init[finite] ** 2
            )))
        else:
            rmse_init = float("nan")

        # --- IC-GN solve ---
        U_sol, conv_iter, icgn_s = _run_icgn(
            ref, images[t], coords, U0, para,
        )
        rate = _convergence_rate(conv_iter, ICGN_MAX_ITER)

        # --- Output quality (post-IC-GN, converged nodes only) ---
        err_u = U_sol[0::2] - ref_u
        err_v = U_sol[1::2] - ref_v
        ok = (conv_iter > 0) & (conv_iter < ICGN_MAX_ITER)
        if ok.any():
            rmse_u = float(np.sqrt(np.mean(err_u[ok] ** 2)))
            rmse_v = float(np.sqrt(np.mean(err_v[ok] ** 2)))
            rmse_total = float(np.sqrt(np.mean(
                err_u[ok] ** 2 + err_v[ok] ** 2
            )))
            mean_iter = float(conv_iter[ok].mean())
        else:
            rmse_u = rmse_v = rmse_total = float("nan")
            mean_iter = float("nan")

        records.append(FrameRecord(
            frame=t,
            init_time_s=meta.get("init_time_s", 0.0),
            icgn_time_s=icgn_s,
            convergence_rate=rate,
            mean_iter=mean_iter,
            rmse_init=rmse_init,
            rmse_u=rmse_u,
            rmse_v=rmse_v,
            rmse_total=rmse_total,
            fft_retries=int(meta.get("fft_retries", 0)),
            mode_used=str(meta.get("mode", "?")),
            n_seeds=int(meta.get("n_seeds", 0)),
            max_bfs_depth=int(meta.get("max_bfs_depth", 0)),
            seed_ncc_min=float(meta.get("seed_ncc_min", float("nan"))),
            n_solve_calls=int(meta.get("n_solve_calls", 0)),
        ))
        strategy.record_solved(t, U_sol, conv_iter, ICGN_MAX_ITER)

    return ScenarioRun(
        strategy_name=strategy.name,
        scenario_name=scenario.name,
        frames=records,
    )


def evaluate_all(
    strategies: list[InitGuessStrategy] | None = None,
    scenarios: list[Scenario] | None = None,
    progress: bool = True,
) -> list[ScenarioRun]:
    """Run every strategy x scenario combination.

    Returns a flat list of :class:`ScenarioRun`. Note: each (strategy,
    scenario) pair gets a *fresh* strategy instance — otherwise the
    strategy state from run N leaks into run N+1.
    """
    import time as _time

    scenarios = scenarios if scenarios is not None else SCENARIOS
    if strategies is None:
        # Defer call so the factory produces one fresh instance per run
        build = build_strategies
    else:
        build = lambda: strategies   # type: ignore[misc]

    ref = _make_reference_image()
    para = _build_dic_para()
    coords = _node_coordinates(para)

    runs: list[ScenarioRun] = []
    n_scenarios = len(scenarios)
    n_strategies = len(build())
    total = n_scenarios * n_strategies
    done = 0
    t0 = _time.perf_counter()

    for scenario in scenarios:
        if progress:
            print(f"\n[scenario] {scenario.name}", flush=True)
        images, gt_fields = generate_scenario_frames(scenario, ref)
        # Fresh strategy instances for this scenario — critical.
        for strategy in build():
            done += 1
            if progress:
                print(f"  ({done}/{total}) {strategy.name}...", end="", flush=True)
            run = _eval_single(
                strategy, scenario, ref, images, gt_fields, para, coords,
            )
            runs.append(run)
            if progress:
                print(f" conv={run.mean_convergence*100:.0f}%, "
                      f"rmse={run.mean_rmse:.3f} px, "
                      f"total={run.total_time_s:.2f}s",
                      flush=True)

    if progress:
        elapsed = _time.perf_counter() - t0
        print(f"\n[done] {total} runs in {elapsed:.1f}s")
    return runs


# ---------------------------------------------------------------------
# PDF report (stage D)
# ---------------------------------------------------------------------

def _render_report(
    runs: list[ScenarioRun],
    output_path: Path,
) -> None:
    """Build the final multi-page evaluation PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Index: {(strategy, scenario) -> ScenarioRun}
    idx: dict[tuple[str, str], ScenarioRun] = {
        (r.strategy_name, r.scenario_name): r for r in runs
    }
    strategies = sorted({r.strategy_name for r in runs})
    scenarios = sorted({r.scenario_name for r in runs})

    # Stable ordering: original insertion order
    strategy_order: list[str] = []
    scenario_order: list[str] = []
    for r in runs:
        if r.strategy_name not in strategy_order:
            strategy_order.append(r.strategy_name)
        if r.scenario_name not in scenario_order:
            scenario_order.append(r.scenario_name)

    with PdfPages(str(output_path)) as pdf:
        _page_cover(pdf, runs, strategy_order, scenario_order)
        _page_summary_matrix(pdf, idx, strategy_order, scenario_order)
        _page_init_vs_solved(pdf, runs)
        _page_pareto(pdf, runs)
        _page_per_scenario_curves(pdf, idx, strategy_order, scenario_order)
        _page_recommendation(pdf, idx, strategy_order, scenario_order)

    print(f"\nReport saved to: {output_path}")


def _page_cover(pdf, runs, strategies, scenarios) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.72, "IC-GN Initial-Guess Strategy Evaluation",
             fontsize=22, ha="center", weight="bold")
    fig.text(0.5, 0.66,
             f"{len(strategies)} strategies \u00d7 {len(scenarios)} scenarios "
             f"\u00d7 {N_FRAMES-1} frames each",
             fontsize=12, ha="center", color="0.3")
    total_time = sum(r.total_time_s for r in runs)
    fig.text(0.5, 0.60,
             f"Total IC-GN + init time: {total_time:.1f}s",
             fontsize=11, ha="center", color="0.4")

    body = (
        "Goal\n"
        "----\n"
        "Quantify how much the choice of initial guess affects IC-GN\n"
        "accuracy, convergence, and runtime across diverse motion\n"
        "scenarios (constant velocity, acceleration, reversal,\n"
        "impulse, chirp, stop-and-go, random, plus spatial\n"
        "discontinuities).\n\n"
        "Strategies\n"
        "----------\n"
        "  previous        - U_t = U_{t-1}  (current default)\n"
        "  linear_extrap   - U_t = 2 U_{t-1} - U_{t-2}\n"
        "  narrow_fft_R5   - per-node FFT, \u00b15 px window around\n"
        "                    U_{t-1}, auto-expands on clipped peaks\n"
        "  adaptive_t{N}   - 'previous' with FFT fallback when the\n"
        "                    previous IC-GN convergence rate fell\n"
        "                    below N%\n"
        "  fft_every       - full FFT every frame (baseline)\n\n"
        "All FFT paths use the pyALDIC auto-expand loop (2\u00d7 retries up\n"
        "to image/2)."
    )
    fig.text(0.12, 0.50, body, fontsize=10, family="monospace",
             va="top", linespacing=1.4)
    pdf.savefig(fig)
    plt.close(fig)


def _page_summary_matrix(pdf, idx, strategy_order, scenario_order) -> None:
    """Three side-by-side heatmaps: RMSE, convergence, total time."""
    import matplotlib.pyplot as plt

    n_s = len(strategy_order)
    n_c = len(scenario_order)
    rmse = np.full((n_c, n_s), np.nan)
    conv = np.full((n_c, n_s), np.nan)
    tsec = np.full((n_c, n_s), np.nan)
    for i, sc in enumerate(scenario_order):
        for j, st in enumerate(strategy_order):
            r = idx.get((st, sc))
            if r is None:
                continue
            rmse[i, j] = r.mean_rmse
            conv[i, j] = r.mean_convergence
            tsec[i, j] = r.total_time_s

    fig, axes = plt.subplots(1, 3, figsize=(16, 0.4 * n_c + 2))
    fig.suptitle("Per-(strategy, scenario) summary",
                 fontsize=13, weight="bold")

    # Short scenario labels (drop the T#_ prefix to save horizontal space)
    sc_labels = [s for s in scenario_order]

    for ax, data, title, cmap, fmt in [
        (axes[0], rmse, "mean RMSE (px, log)", "viridis_r", "{:.2f}"),
        (axes[1], conv * 100, "mean conv rate (%)", "viridis", "{:.0f}"),
        (axes[2], tsec, "total time (s)", "viridis_r", "{:.2f}"),
    ]:
        if title.startswith("mean RMSE"):
            plot_data = np.log10(np.maximum(data, 1e-4))
            im = ax.imshow(plot_data, cmap=cmap, aspect="auto")
        else:
            im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_s))
        ax.set_xticklabels(strategy_order, rotation=45, ha="right",
                           fontsize=7)
        ax.set_yticks(range(n_c))
        ax.set_yticklabels(sc_labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        # Annotate
        for i in range(n_c):
            for j in range(n_s):
                v = data[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=6, color="white")
        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _page_init_vs_solved(pdf, runs) -> None:
    """Init-guess RMSE vs IC-GN output RMSE per (strategy, scenario).

    Exposes three regimes:
      - diagonal  : IC-GN left init untouched (probably already solved)
      - below y=x : IC-GN refined the init (expected healthy case)
      - above y=x : IC-GN diverged from init (pathological — IC-GN
                    walked away from a good starting point)
    """
    import matplotlib.pyplot as plt

    by_strat: dict[str, list[ScenarioRun]] = {}
    for r in runs:
        by_strat.setdefault(r.strategy_name, []).append(r)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(by_strat)))
    for (strat, rs), color in zip(by_strat.items(), colors):
        xs, ys = [], []
        for r in rs:
            xi_val = r.mean_rmse_init
            yi_val = r.mean_rmse
            if np.isfinite(xi_val) and np.isfinite(yi_val):
                xs.append(xi_val)
                ys.append(yi_val)
        if xs:
            ax.scatter(xs, ys, label=strat, s=60, alpha=0.65, color=color)

    # y = x reference
    lim = [1e-4, 20]
    ax.plot(lim, lim, "k--", alpha=0.3, linewidth=1,
            label="y = x (no IC-GN change)")

    ax.set_xlabel("Init guess RMSE (pre-IC-GN, all nodes) [px]")
    ax.set_ylabel("IC-GN output RMSE (post-IC-GN, converged) [px]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(
        "Init quality vs IC-GN output\n"
        "points below y=x: IC-GN refined init.  "
        "points above y=x: IC-GN diverged.",
        fontsize=11, weight="bold",
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_pareto(pdf, runs) -> None:
    """Accuracy vs speed scatter, aggregated across scenarios."""
    import matplotlib.pyplot as plt
    # Per strategy: median RMSE and median total_time
    by_strat: dict[str, list[ScenarioRun]] = {}
    for r in runs:
        by_strat.setdefault(r.strategy_name, []).append(r)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(by_strat)))
    for (strat, rs), color in zip(by_strat.items(), colors):
        # Filter NaN RMSE
        rm = [r.mean_rmse for r in rs if np.isfinite(r.mean_rmse)]
        ts = [r.total_time_s for r in rs
              if np.isfinite(r.mean_rmse)]
        if not rm:
            continue
        ax.scatter(ts, rm, label=strat, s=60, alpha=0.6, color=color)
        med_t = float(np.median(ts))
        med_r = float(np.median(rm))
        ax.scatter([med_t], [med_r], s=200, marker="*",
                   edgecolor="black", linewidth=1.2, color=color, zorder=10)

    ax.set_xlabel("Total per-scenario time (init + IC-GN) [s]")
    ax.set_ylabel("Mean RMSE on converged nodes [px]")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Accuracy vs. speed — dots = scenarios, \u2605 = median "
                 "per strategy",
                 fontsize=11, weight="bold")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_per_scenario_curves(pdf, idx, strategy_order, scenario_order) -> None:
    """One page per scenario: per-frame RMSE + per-frame time for all
    strategies."""
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_order)))

    for sc in scenario_order:
        fig, (ax_init, ax_r, ax_t) = plt.subplots(
            3, 1, figsize=(11, 9), sharex=True,
        )
        fig.suptitle(f"{sc}", fontsize=12, weight="bold")

        for strat, color in zip(strategy_order, colors):
            r = idx.get((strat, sc))
            if r is None:
                continue
            frames = [f.frame for f in r.frames]
            rmse_init = [f.rmse_init for f in r.frames]
            rmse = [f.rmse_total for f in r.frames]
            tms = [(f.init_time_s + f.icgn_time_s) * 1000
                   for f in r.frames]
            ax_init.plot(frames, rmse_init, "o-", label=strat, color=color,
                         markersize=4, linewidth=1.2, alpha=0.8)
            ax_r.plot(frames, rmse, "o-", label=strat, color=color,
                      markersize=4, linewidth=1.2, alpha=0.8)
            ax_t.plot(frames, tms, "o-", label=strat, color=color,
                      markersize=4, linewidth=1.2, alpha=0.8)

        ax_init.set_ylabel("init RMSE [px]\n(pre IC-GN, all nodes)")
        ax_init.set_yscale("log")
        ax_init.grid(True, which="both", alpha=0.3)
        ax_init.set_title(
            "Init guess error  —  IC-GN output error  —  per-frame cost",
            fontsize=9,
        )
        ax_init.legend(fontsize=7, loc="best", ncol=3)

        ax_r.set_ylabel("solved RMSE [px]\n(post IC-GN, converged)")
        ax_r.set_yscale("log")
        ax_r.grid(True, which="both", alpha=0.3)

        ax_t.set_xlabel("frame")
        ax_t.set_ylabel("per-frame time [ms]")
        ax_t.set_yscale("log")
        ax_t.grid(True, which="both", alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)


def _page_recommendation(pdf, idx, strategy_order, scenario_order) -> None:
    """Text page with an auto-derived recommendation table.

    For each scenario, pick the 'best' strategy by composite score =
    rmse / baseline_rmse + log10(time / baseline_time).
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.96, "Scenario \u2192 recommended strategy",
             fontsize=14, ha="center", weight="bold")

    headers = ["Scenario", "Best strategy",
               "RMSE (px)", "Conv %", "Time (s)", "Notes"]
    rows: list[list[str]] = []

    for sc in scenario_order:
        # Pick the row among candidates with best composite: min RMSE
        # subject to conv >= 90%. If nothing meets conv bar, pick min
        # RMSE overall.
        candidates = [
            (st, idx.get((st, sc))) for st in strategy_order
            if idx.get((st, sc)) is not None
        ]
        if not candidates:
            rows.append([sc, "-", "-", "-", "-", "no data"])
            continue

        good = [
            (st, r) for st, r in candidates
            if r.mean_convergence >= 0.90 and np.isfinite(r.mean_rmse)
        ]
        pool = good if good else [
            (st, r) for st, r in candidates if np.isfinite(r.mean_rmse)
        ]
        if not pool:
            rows.append([sc, "-", "-", "-", "-", "all diverged"])
            continue
        pool.sort(key=lambda x: (x[1].mean_rmse, x[1].total_time_s))
        st_best, r_best = pool[0]

        # Note whether the best is materially faster/slower than fft_every
        fft_r = idx.get(("fft_every", sc))
        note = ""
        if fft_r is not None and st_best != "fft_every":
            tspd = fft_r.total_time_s / max(r_best.total_time_s, 1e-6)
            if tspd >= 1.5:
                note = f"{tspd:.1f}\u00d7 faster than fft_every"

        rows.append([
            sc,
            st_best,
            f"{r_best.mean_rmse:.3f}",
            f"{r_best.mean_convergence * 100:.0f}",
            f"{r_best.total_time_s:.2f}",
            note,
        ])

    ax = fig.add_subplot(111)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers,
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#d5e8f0")
        table[0, j].set_text_props(weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------
# Preview mode — render each scenario as a PDF for visual sanity check
# ---------------------------------------------------------------------

def _preview(output_path: Path) -> None:
    """Render a one-page-per-scenario preview PDF.

    Each page shows: reference image, deformed frame at mid + last, and
    the ground-truth displacement magnitude at the last frame.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref = _make_reference_image()

    with PdfPages(str(output_path)) as pdf:
        # Cover
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.68, "Init-Guess Evaluation — Scenario Preview",
                 fontsize=22, ha="center", weight="bold")
        fig.text(0.5, 0.62, f"{len(SCENARIOS)} scenarios, "
                 f"{N_FRAMES} frames, {IMG_H}x{IMG_W} images",
                 fontsize=12, ha="center", color="0.4")
        lines = [f"  {s.name}" for s in SCENARIOS]
        fig.text(0.12, 0.5, "\n".join(lines), fontsize=9,
                 family="monospace", va="top")
        pdf.savefig(fig)
        plt.close(fig)

        for scn in SCENARIOS:
            print(f"  rendering {scn.name}", flush=True)
            images, gt_fields = generate_scenario_frames(scn, ref)

            fig, axes = plt.subplots(2, 3, figsize=(13, 8))
            fig.suptitle(f"{scn.name}\n{scn.description}",
                         fontsize=11, weight="bold")

            mid = N_FRAMES // 2
            last = N_FRAMES - 1

            axes[0, 0].imshow(images[0], cmap="gray")
            axes[0, 0].set_title("Frame 0 (reference)", fontsize=9)
            axes[0, 1].imshow(images[mid], cmap="gray")
            axes[0, 1].set_title(f"Frame {mid}", fontsize=9)
            axes[0, 2].imshow(images[last], cmap="gray")
            axes[0, 2].set_title(f"Frame {last}", fontsize=9)

            # Ground-truth |U| at three frames
            for col_idx, frame in enumerate((1, mid, last)):
                u, v = gt_fields[frame]
                mag = np.sqrt(u * u + v * v)
                im = axes[1, col_idx].imshow(
                    mag, cmap="viridis",
                    vmin=0, vmax=max(PEAK_DISP_PX, mag.max()),
                )
                axes[1, col_idx].set_title(
                    f"|U| at frame {frame}  (max={mag.max():.1f} px)",
                    fontsize=9,
                )
                plt.colorbar(im, ax=axes[1, col_idx], shrink=0.7)

            for ax in axes.ravel():
                ax.set_xticks([])
                ax.set_yticks([])

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nPreview PDF saved to: {output_path}", flush=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def _smoke() -> None:
    """Tiny end-to-end smoke test: one strategy on one scenario, 3 frames.

    Verifies that the FFT bootstrap + IC-GN + strategy update path all
    hang together before investing in the full evaluation loop.
    """
    print("Smoke test: PreviousStrategy x T2_constant_velocity+S_uniform, "
          "3 frames")
    scenario = SCENARIOS[1]  # T2_constant_velocity + S_uniform
    assert scenario.temporal == "T2_constant_velocity"

    ref = _make_reference_image()
    para = _build_dic_para()
    coords = _node_coordinates(para)
    print(f"  {coords.shape[0]} nodes, winsize={para.winsize}, "
          f"step={para.winstepsize}, search={para.size_of_fft_search_region}")

    images, gt_fields = generate_scenario_frames(scenario, ref)
    strat = PreviousStrategy()
    for t in (1, 2, 3):
        U0, meta = strat.initial_guess(t, ref, images[t], coords, para)
        U_sol, conv_iter, icgn_t = _run_icgn(
            ref, images[t], coords, U0, para,
        )
        rate = _convergence_rate(conv_iter, ICGN_MAX_ITER)

        # Ground-truth U at node centers
        yy = coords[:, 1].astype(int)
        xx = coords[:, 0].astype(int)
        gt_u, gt_v = gt_fields[t]
        ref_u = gt_u[yy, xx]
        ref_v = gt_v[yy, xx]
        err_u = U_sol[0::2] - ref_u
        err_v = U_sol[1::2] - ref_v
        # Only evaluate on converged nodes
        ok = (conv_iter > 0) & (conv_iter < ICGN_MAX_ITER)
        if ok.any():
            rmse = float(np.sqrt(np.mean(err_u[ok] ** 2 + err_v[ok] ** 2)))
        else:
            rmse = float("nan")
        print(f"  t={t} [{meta['mode']:15s}] "
              f"init={meta['init_time_s']*1000:.0f} ms "
              f"icgn={icgn_t*1000:.0f} ms "
              f"conv={rate*100:.1f}% "
              f"mean_iter={conv_iter[ok].mean():.1f} "
              f"rmse={rmse:.4f} px")
        strat.record_solved(t, U_sol, conv_iter, ICGN_MAX_ITER)
    print("OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", action="store_true",
        help="Render a scenario preview PDF and exit (no evaluation).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run a minimal end-to-end integration check (one strategy "
             "x one scenario, 3 frames).",
    )
    parser.add_argument(
        "--eval-fast", action="store_true",
        help="Run the full evaluation loop on a subset of scenarios to "
             "verify timing / convergence without committing to the full "
             "15-scenario run.",
    )
    args = parser.parse_args()

    reports_dir = BASE / "reports"

    if args.preview:
        _preview(reports_dir / "init_guess_scenarios_preview.pdf")
        return

    if args.smoke:
        _smoke()
        return

    if args.eval_fast:
        subset = [
            s for s in SCENARIOS
            if s.name in (
                "T2_constant_velocity+S_uniform",
                "T5_step_impulse+S_uniform",
                "T2_constant_velocity+S_crack",
            )
        ]
        runs = evaluate_all(scenarios=subset)
        _render_report(runs, reports_dir / "init_guess_eval_fast.pdf")
        return

    # Default: full evaluation across all 15 scenarios.
    runs = evaluate_all()
    _render_report(runs, reports_dir / "init_guess_eval.pdf")


if __name__ == "__main__":
    main()
