"""Throughput of a complete two-frame analysis with the Local DIC and AL-DIC solvers.

Synthetic speckle with a 5-pixel horizontal translation; 31 x 31-pixel subsets
(``winsize=30``), 8-pixel spacing, FFT initial guess, three ADMM iterations for
AL-DIC. Points of interest per second = nodes / wall time of the whole
``run_aldic`` call, timed after a warm-up run that compiles the Numba kernels.
The median of ``--repeats`` runs is reported.

    python throughput.py                    # installed pyALDIC
    python throughput.py --src ../v041/src  # another checkout
"""

from __future__ import annotations

import argparse
import statistics
import time

from _common import generate_speckle, machine_info, use_source, warp_lagrangian

SIZES = (512, 1024)
MARGIN = 40


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--src", help="src directory of another pyALDIC checkout")
    args = ap.parse_args()
    use_source(args.src)

    import numpy as np
    import al_dic
    from al_dic.core.config import dicpara_default
    from al_dic.core.data_structures import GridxyROIRange
    from al_dic.core.pipeline import run_aldic

    def images(size: int):
        ref = generate_speckle(size, size, seed=1)
        defo = warp_lagrangian(ref, lambda x, y: np.full_like(x, 5.0),
                               lambda x, y: np.zeros_like(x))
        return [ref, defo]

    def run(size: int, local: bool, imgs) -> tuple[int, float]:
        para = dicpara_default(
            winsize=30, winstepsize=8, winsize_min=8, img_size=(size, size),
            gridxy_roi_range=GridxyROIRange(gridx=(MARGIN, size - 1 - MARGIN),
                                            gridy=(MARGIN, size - 1 - MARGIN)),
            reference_mode="accumulative", use_global_step=not local,
            admm_max_iter=1 if local else 3, icgn_max_iter=100, tol=1e-2,
            init_guess_mode="fft", size_of_fft_search_region=20,
            fft_auto_expand_search=True, show_plots=False,
        )
        mask = np.ones((size, size))
        t0 = time.perf_counter()
        res = run_aldic(para, imgs, [mask, mask], compute_strain=False)
        return res.dic_mesh.coordinates_fem.shape[0], time.perf_counter() - t0

    print(f"pyALDIC {al_dic.__version__} | {machine_info()}")
    warm = images(128)
    run(128, True, warm)
    run(128, False, warm)
    print(f"{'image':>10} {'solver':>7} {'nodes':>6} {'median s':>9} {'POI/s':>8}")
    for size in SIZES:
        imgs = images(size)
        for local in (True, False):
            times, nodes = [], 0
            for _ in range(args.repeats):
                nodes, dt = run(size, local, imgs)
                times.append(dt)
            med = statistics.median(times)
            print(f"{size:>5}x{size:<4} {'Local' if local else 'AL-DIC':>7} {nodes:>6} "
                  f"{med:>9.3f} {nodes / med:>8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
