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

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", action="store_true",
        help="Render a scenario preview PDF and exit (no evaluation).",
    )
    args = parser.parse_args()

    reports_dir = BASE / "reports"

    if args.preview:
        _preview(reports_dir / "init_guess_scenarios_preview.pdf")
        return

    # Stages B, C, D will slot in here in subsequent commits.
    raise NotImplementedError(
        "Full evaluation (stages B-D) is not yet implemented. "
        "Run with --preview to render scenario sanity check."
    )


if __name__ == "__main__":
    main()
