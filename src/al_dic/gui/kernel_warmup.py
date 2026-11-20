"""Compile the Numba kernels in the background, before the user needs them.

The solver kernels are JIT-compiled on first use. Numba caches the result on
disk, so this is a once-per-installation cost -- but it is a large one, and it
lands at the worst possible moment: the user clicks *Run DIC Analysis* and the
interface stops responding for tens of seconds with nothing to explain it.

Measured on a 24-thread workstation, with every cache cleared and both halves
timed in one process so no disk cache can flatter the result:

    no warm-up      first real correlation   32.1 s
    with warm-up    warm-up (background)     41.1 s
                    first real correlation    0.1 s

A four-core laptop is slower on both counts.

To someone who installed by unzipping a folder and double-clicking, that reads
as a crash -- and killing the process is the one response that makes it
permanent, because the cache is written only when compilation finishes.

So compilation starts a couple of seconds after the window appears and overlaps
what the user does first: browsing for images in a file dialog, drawing a
region of interest. The warm-up correlation is small, but not arbitrarily so --
see WARMUP_IMAGE_PX. Numba's compiler runs Python bytecode and holds the GIL, so
the interface is less responsive while this runs. That is a better trade than a
dead window after a button press, and it is not repeated.

Why a daemon ``threading.Thread`` and not a ``QThread``: compilation cannot be
interrupted part-way, so a QThread would leave the application choosing between
aborting on close -- Qt calls ``abort()`` when a running QThread is destroyed,
the same failure mode as the export-dialog crash fixed in 0.7.2 -- and blocking
the exit for however long compilation has left. A daemon thread is reclaimed at
interpreter shutdown with neither problem. The one Qt object here is the signal
carrier, which lives on the main thread and receives a queued emission.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

# How long after the window appears to begin, in milliseconds. Long enough for
# the first paint, and for the user to reach for the mouse.
START_DELAY_MS = 2000

# Below this, the cache was already warm and there is nothing worth saying.
REPORT_THRESHOLD_S = 1.0


class KernelWarmup(QObject):
    """Runs one tiny correlation on a daemon thread to populate the JIT cache."""

    # Emitted on completion with the elapsed seconds. Not emitted at all when
    # the kernels were already compiled, or when warm-up failed.
    compiled = Signal(float)

    def start(self) -> None:
        thread = threading.Thread(
            target=self._run, name="pyALDIC-kernel-warmup", daemon=True
        )
        thread.start()

    def _run(self) -> None:
        started = time.perf_counter()
        try:
            _correlate_tiny_pair()
        except Exception:
            # A warm-up failure must never reach the user: the real run will
            # compile the same kernels itself and report its own errors.
            logger.exception("Kernel warm-up failed; continuing without it")
            return
        elapsed = time.perf_counter() - started
        logger.info("Kernel warm-up finished in %.1fs", elapsed)
        if elapsed >= REPORT_THRESHOLD_S:
            self.compiled.emit(elapsed)


# Image size for the warm-up correlation. Not free to lower: the batch
# precompute kernel -- the largest single compilation in the pipeline -- is
# only reached when a frame has at least 50 nodes (icgn_batch.py). With
# winsize 32 and step 16, a size x size region gives ((size - 32) / 16 + 1)^2
# nodes, so 96 px yields 16 and skips it entirely. 192 px yields 100.
#
# This was measured, not reasoned about: at 96 px the warm-up finished in a
# quick 4.9 s and left the first real correlation exactly as slow as it had
# been without any warm-up. See test_kernel_warmup.py, which fails if the
# geometry ever stops reaching those kernels.
WARMUP_IMAGE_PX = 192


def _correlate_tiny_pair() -> None:
    """Correlate a small synthetic pair through the real pipeline.

    Going through ``run_aldic`` rather than calling the kernels directly is
    deliberate: compilation is per type signature and the backends dispatch on
    problem size, so taking the path the real run takes is the only reliable
    way to compile the signatures the real run will need.
    """
    from dataclasses import replace

    import numpy as np
    from scipy.ndimage import gaussian_filter

    from al_dic.core.config import dicpara_default
    from al_dic.core.data_structures import GridxyROIRange
    from al_dic.core.pipeline import run_aldic

    size = WARMUP_IMAGE_PX
    rng = np.random.default_rng(0)
    ref = gaussian_filter(rng.standard_normal((size, size)), sigma=3.0,
                          mode="nearest")
    ref -= ref.min()
    ref = 20.0 + 215.0 * (ref / ref.max())
    deformed = np.roll(ref, shift=(1, 1), axis=(0, 1))

    para = dicpara_default(
        winsize=32,
        winstepsize=16,
        gridxy_roi_range=GridxyROIRange(gridx=(0, size - 1), gridy=(0, size - 1)),
        # Two ADMM iterations, not one: a single iteration returns before the
        # subproblem-1 solver runs, leaving icgn_2dof_parallel uncompiled and
        # about 2.5 s of the compile still on the user's first click.
        admm_max_iter=2,
        icgn_max_iter=5,
        show_plots=False,
    )
    mask = np.ones((size, size), np.uint8)
    with warnings.catch_warnings():
        # A small image trips the FFT search-region auto-scale notice, which
        # means nothing for a warm-up.
        warnings.simplefilter("ignore")
        run_aldic(para, [ref, deformed], [mask, mask], compute_strain=False)
        # Second pass, Local DIC with the second-order shape function: the
        # AL-DIC pass above never reaches its refinement kernel, and Local
        # DIC is what a new project starts on.
        second_order = replace(
            para, use_global_step=False, shape_function_order="second",
        )
        run_aldic(second_order, [ref, deformed], [mask, mask],
                  compute_strain=False)
