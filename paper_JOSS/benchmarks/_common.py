"""Shared helpers for the paper's benchmarks: synthetic images, machine info,
and an optional switch to benchmark another checkout of pyALDIC."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def use_source(src: str | None) -> None:
    """Import ``al_dic`` from *src* (a checkout's ``src`` directory, e.g. a
    ``git worktree`` of an older tag) instead of the installed package.

    An editable install registers an import hook that would load the working
    tree regardless of ``sys.path``, so that hook is removed first.
    """
    if not src:
        return
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if "editable" not in type(f).__name__.lower()
        and "editable" not in getattr(f, "__module__", "").lower()
    ]
    sys.path.insert(0, str(Path(src).resolve()))


def machine_info() -> str:
    """CPU, logical cores, Python and NumPy versions, for the results header."""
    cpu = platform.processor() or platform.machine()
    return (f"{cpu}; {os.cpu_count()} logical cores; Python {platform.python_version()}; "
            f"NumPy {np.__version__}; {platform.system()} {platform.release()}")


def generate_speckle(height: int, width: int, sigma: float = 3.0,
                     seed: int = 42) -> np.ndarray:
    """Gaussian-filtered random speckle, gray levels 20-235, feature size
    about 2*sigma pixels; deterministic for a fixed seed (the generator of
    tests/conftest.py)."""
    rng = np.random.default_rng(seed)
    filtered = gaussian_filter(rng.standard_normal((height, width)), sigma=sigma,
                               mode="nearest")
    filtered -= filtered.min()
    filtered /= filtered.max()
    return 20.0 + 215.0 * filtered


def warp_lagrangian(ref_image: np.ndarray, u_func, v_func, n_iter: int = 20) -> np.ndarray:
    """Deformed image under x = X + u(X): the reference position of each
    deformed pixel is found by fixed-point iteration, then sampled with a
    quintic spline (avoids sharing the solver's cubic interpolation)."""
    h, w = ref_image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx.copy(), yy.copy()
    for _ in range(n_iter):
        X = xx - u_func(X, Y)
        Y = yy - v_func(X, Y)
    warped = map_coordinates(ref_image, np.array([Y.ravel(), X.ravel()]), order=5,
                             mode="nearest")
    return warped.reshape(h, w)
