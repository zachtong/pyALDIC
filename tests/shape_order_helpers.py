"""Synthetic image pairs and solver setup shared by the shape-order tests.

Ground truth is exact: the deformed image is built by a Lagrangian
fixed-point inversion of ``x = X + u(X)`` with quintic sampling.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates

from al_dic.core.config import dicpara_default
from al_dic.io.image_ops import compute_image_gradient
from al_dic.solver.local_icgn import (
    LocalICGNContext, local_icgn_precompute, local_icgn_solve_subset,
)

Field = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


def make_speckle(height: int, width: int, seed: int, sigma: float = 3.0) -> NDArray[np.float64]:
    """Gaussian-filtered white noise scaled to [0, 1] (about 6 px features)."""
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.standard_normal((height, width)), sigma)
    return (img - img.min()) / (img.max() - img.min())


def warp_lagrangian(ref: NDArray[np.float64], u_fn: Field, v_fn: Field,
                    n_iter: int = 40) -> NDArray[np.float64]:
    """Deformed image ``g(x) = f(X)`` with ``x = X + u(X)``."""
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    x_ref, y_ref = xx.copy(), yy.copy()
    for _ in range(n_iter):
        x_new = xx - u_fn(x_ref, y_ref)
        y_new = yy - v_fn(x_ref, y_ref)
        x_ref, y_ref = x_new, y_new
    return map_coordinates(ref, [y_ref, x_ref], order=5, mode="nearest")


def add_noise(img: NDArray[np.float64], noise_sd: float,
              rng: np.random.Generator) -> NDArray[np.float64]:
    """Additive Gaussian noise; ``noise_sd`` is a fraction of the grey range."""
    return img + rng.standard_normal(img.shape) * noise_sd


def node_grid(height: int, width: int, step: int, margin: int) -> NDArray[np.float64]:
    """(N, 2) integer node coordinates ``[x, y]`` inside ``margin``."""
    xs = np.arange(margin, width - margin, step)
    ys = np.arange(margin, height - margin, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)


def first_order_solution(
    ref: NDArray[np.float64],
    deformed: NDArray[np.float64],
    coords: NDArray[np.float64],
    winsize: int,
    u0: NDArray[np.float64],
    mask: NDArray[np.float64] | None = None,
) -> tuple[LocalICGNContext, NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], float]:
    """Run the production first-order solve; return (ctx, U_2d, F_2d, conv_iter, tol)."""
    para = dicpara_default(winsize=winsize, winstepsize=16, use_global_step=False)
    if mask is None:
        mask = np.ones_like(ref)
    grads = compute_image_gradient(ref * mask, mask, img_raw=ref)
    ctx = local_icgn_precompute(coords, grads, ref, para)
    u_2d, f_2d, conv_iter = local_icgn_solve_subset(ctx, None, u0, deformed, para.tol)
    return ctx, u_2d, f_2d, conv_iter, para.tol
