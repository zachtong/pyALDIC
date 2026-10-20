"""Peak memory of a 4096 x 4096-pixel, three-frame analysis.

Synthetic speckle under uniform shear (u = 0.005 * frame * y), 31 x 31-pixel
subsets, 16-pixel spacing, FFT initial guess with a 16-pixel search region,
accumulative reference. Peak memory is the highest traced allocation
(``tracemalloc``, which includes NumPy arrays) during ``run_aldic``; it is the
``xlarge_4096_3frames`` case of tests/test_perf_release.py, written to run on
versions that predate that test as well.

    python peak_memory.py                    # installed pyALDIC
    python peak_memory.py --src ../v041/src  # another checkout
"""

from __future__ import annotations

import argparse
import time
import tracemalloc

from _common import generate_speckle, machine_info, use_source, warp_lagrangian

SIZE, FRAMES, SUBSET, STEP, SEARCH, SHEAR = 4096, 3, 31, 16, 16, 0.005


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", help="src directory of another pyALDIC checkout")
    args = ap.parse_args()
    use_source(args.src)

    import numpy as np
    import al_dic
    from al_dic.core.config import dicpara_default
    from al_dic.core.data_structures import GridxyROIRange
    from al_dic.core.pipeline import run_aldic

    ref = generate_speckle(SIZE, SIZE, sigma=3.0, seed=42)
    images = [ref] + [
        warp_lagrangian(ref, lambda x, y, s=i * SHEAR: s * y, lambda x, y: np.zeros_like(x))
        for i in range(1, FRAMES)
    ]
    masks = [np.ones((SIZE, SIZE), dtype=bool)] * FRAMES
    para = dicpara_default(
        winsize=SUBSET - 1, winstepsize=STEP, winsize_min=min(8, STEP),
        size_of_fft_search_region=SEARCH, img_size=(SIZE, SIZE),
        gridxy_roi_range=GridxyROIRange(gridx=(0, SIZE - 1), gridy=(0, SIZE - 1)),
        reference_mode="accumulative", init_guess_mode="fft", show_plots=False,
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    run_aldic(para, images, masks, compute_strain=False)
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"pyALDIC {al_dic.__version__} | {machine_info()}")
    print(f"{SIZE}x{SIZE} x {FRAMES} frames, step {STEP}: peak {peak / 2**30:.2f} GiB, "
          f"wall {wall:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
