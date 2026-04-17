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
    rmse_u: float   # px, on converged nodes
    rmse_v: float
    rmse_total: float
    fft_retries: int
    mode_used: str   # e.g. "previous", "fft", "linear_extrap"


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


def build_strategies(
    adaptive_thresholds: tuple[float, ...] = (0.75, 0.90),
    narrow_radii: tuple[int, ...] = (5,),
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
    return strategies


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
    args = parser.parse_args()

    reports_dir = BASE / "reports"

    if args.preview:
        _preview(reports_dir / "init_guess_scenarios_preview.pdf")
        return

    if args.smoke:
        _smoke()
        return

    # Stages C, D will slot in here in subsequent commits.
    raise NotImplementedError(
        "Full evaluation (stages C-D) is not yet implemented. "
        "Run with --preview for scenario rendering or --smoke for an "
        "integration check."
    )


if __name__ == "__main__":
    main()
