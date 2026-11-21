"""Predict, node by node, which subset shape-function order is more accurate.

Model (Zhao & Pan, Mech. Syst. Signal Process. 222 (2025) 111792), for a
subset of half-width ``M``:

    random error       sigma_1 = sqrt(2) * sigma_n / sqrt(SSSIG),  sigma_2 = 2 sigma_1
    first-order bias   e_1 = M (M + 1) / 6 * |laplacian(u)|
    second-order bias  e_2 = M^2 (M + 1)^2 / 2520
                             * |9 (u_xxxx + u_yyyy) + 70 u_xxyy|

``sigma_n`` is the image noise, estimated from the residual of the
first-order match; ``SSSIG`` is the sum of squared intensity gradients over
the subset, read from the first-order Hessian. The displacement derivatives
come from a degree-4 least-squares polynomial fitted to the first-order
displacement of the nodes within a radius, so the fit works on refined
meshes and next to holes; it does not look across a masked barrier.

Second order is predicted better where ``e_1^2 + sigma_1^2 > e_2^2 +
4 sigma_1^2``. A node whose fit cannot be trusted is not selected.

One step is added to the published rule: ``e_1`` only counts as far as the
fitted Laplacian stands out from the noise of its own estimate, judged from
the scatter of the displacement about the fit. Real images scatter more than
their image noise explains, and without this step that scatter is read as
curvature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._numba_compat import JIT_CACHE, njit, prange
from ..strain.platefit_kernel import _seg_hits_mask, build_neighbor_cache
from .icgn2_kernels import first_order_znssd_parallel
from .local_icgn import LocalICGNContext, _detect_bad_points

__all__ = [
    "ShapeOrderPrediction", "predict_shape_order", "fit_displacement_derivatives",
    "fit_displacement_model", "fit_radius",
]

#: Radius of the polynomial fit, in subset sizes and in node steps; the larger
#: of the two is used. A degree-4 fit has 15 coefficients per component, so it
#: needs a few dozen nodes, and a fit narrower than the subset cannot see the
#: curvature the subset itself averages over. A wider fit smooths the
#: curvature away: of the factors 1.0, 1.4 and 2.0, 1.0 gave the lowest error
#: across the test scenes (by up to a third with a 65 px subset) at the cost
#: of at most 1.5 % with a 33 px subset in heavy noise.
FIT_RADIUS_SUBSETS = 1.0
FIT_RADIUS_STEPS = 4.0

#: Most nodes a fit uses, on average. Where the mesh is denser than that
#: (small node step, large subset) the fit takes an evenly thinned set of
#: nodes: memory and time then stay bounded, and little is lost because
#: nodes closer than a subset share most of their pixels. The validated
#: configurations (up to about 215 nodes per fit) are below the limit.
MAX_FIT_NEIGHBORS = 250

#: Fewest nodes a fit may use (twice its 15 coefficients).
MIN_FIT_NEIGHBORS = 30

#: Minimum squared pivot of the Jacobi-scaled Cholesky factor of the fit's
#: normal matrix; below it the neighbourhood is too degenerate to fit.
FIT_PIVOT_MIN = 1e-6

#: Largest distance, in fit radii, between a node and the centroid of the
#: nodes its fit uses. Next to the ROI edge or a hole the neighbourhood is
#: one-sided and the degree-4 fit extrapolates; its derivatives are then
#: noise, which read as bias. Such a node is not fitted and stays first
#: order (a half disc has an offset of 0.42).
MAX_FIT_OFFSET = 0.2

_N_TERMS = 15
# Monomial exponents (a, b) of x^a y^b, total degree 0..4.
_EXPONENTS = np.array(
    [(a, d - a) for d in range(5) for a in range(d, -1, -1)], dtype=np.int64)
_IDX = {(int(a), int(b)): k for k, (a, b) in enumerate(_EXPONENTS)}
_I20, _I02 = _IDX[(2, 0)], _IDX[(0, 2)]
_I40, _I04, _I22 = _IDX[(4, 0)], _IDX[(0, 4)], _IDX[(2, 2)]


@dataclass(frozen=True)
class ShapeOrderPrediction:
    """Per-node error prediction for the two shape-function orders.

    Attributes:
        select: (N,) bool; second order is predicted to be more accurate.
        fitted: (N,) bool; the displacement derivatives could be fitted.
        bias_first: (N,) predicted first-order bias (px); NaN where not fitted.
        bias_second: (N,) predicted second-order bias (px); NaN where not fitted.
        sigma_first: (N,) predicted first-order random error (px), the
            vector norm over the two components; NaN where unknown.
        noise_sd: image-noise s.d. estimated from the first-order residual,
            in the units of the image; NaN if it could not be estimated.
        radius: fit radius used (px).
        half: half-width M of the subset the prediction is for (px).
        spacing: spacing of the nodes the fit used (px): the node step, or
            wider where a dense mesh was thinned.
        lap_gain, quart_gain: (N,) variance of the fitted Laplacian and
            quartic term per unit variance of independent nodal noise.
        fit_residual: (N,) r.m.s. scatter of the nodes about the local
            polynomial fit (px).
    """

    select: NDArray[np.bool_]
    fitted: NDArray[np.bool_]
    bias_first: NDArray[np.float64]
    bias_second: NDArray[np.float64]
    sigma_first: NDArray[np.float64]
    noise_sd: float
    radius: float
    half: int
    spacing: float
    lap_gain: NDArray[np.float64]
    quart_gain: NDArray[np.float64]
    fit_residual: NDArray[np.float64]


def fit_radius(winsize: int, winstepsize: int) -> float:
    """Radius (px) of the derivative fit for a subset size and node step."""
    return float(max(FIT_RADIUS_SUBSETS * (winsize + 1), FIT_RADIUS_STEPS * winstepsize))


@njit(cache=JIT_CACHE)
def _solve_two(normal, rhs_u, rhs_v, pivot_min):
    """Solve ``normal x = rhs`` for two right-hand sides; False if unfit.

    Jacobi-scaled Cholesky; the right-hand sides are overwritten with the
    solutions.
    """
    n = normal.shape[0]
    scale = np.empty(n)
    for k in range(n):
        if not (normal[k, k] > 0.0):
            return False
        scale[k] = np.sqrt(normal[k, k])
    chol = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = normal[i, j] / (scale[i] * scale[j])
            for k in range(j):
                s -= chol[i, k] * chol[j, k]
            if i == j:
                if not (s > pivot_min):
                    return False
                chol[i, i] = np.sqrt(s)
            else:
                chol[i, j] = s / chol[j, j]
    for rhs in (rhs_u, rhs_v):
        for i in range(n):
            s = rhs[i] / scale[i]
            for k in range(i):
                s -= chol[i, k] * rhs[k]
            rhs[i] = s / chol[i, i]
        for i in range(n - 1, -1, -1):
            s = rhs[i]
            for k in range(i + 1, n):
                s -= chol[k, i] * rhs[k]
            rhs[i] = s / chol[i, i]
        for i in range(n):
            rhs[i] /= scale[i]
    return True


@njit(parallel=True, cache=JIT_CACHE)
def _fit_kernel(coords, u, v, usable, indptr, indices, radius, mask, use_mask,
                exponents, min_neighbors, pivot_min, max_offset,
                i20, i02, i40, i04, i22):
    n = coords.shape[0]
    n_terms = exponents.shape[0]
    h, w = mask.shape
    lap = np.full((n, 2), np.nan)
    quart = np.full((n, 2), np.nan)
    gain = np.full((n, 2), np.nan)     # noise gain^2 of [laplacian, quartic]
    resid = np.full(n, np.nan)         # r.m.s. scatter of the nodes about the fit
    inv_r2 = 1.0 / (radius * radius)

    for i in prange(n):
        if not usable[i]:
            continue
        x0 = coords[i, 0]
        y0 = coords[i, 1]
        normal = np.zeros((n_terms, n_terms))
        rhs_u = np.zeros(n_terms)
        rhs_v = np.zeros(n_terms)
        phi = np.empty(n_terms)
        count = 0
        sum_x = 0.0
        sum_y = 0.0
        for jj in range(indptr[i], indptr[i + 1]):
            j = indices[jj]
            if not usable[j]:
                continue
            x1 = coords[j, 0]
            y1 = coords[j, 1]
            if use_mask and j != i and _seg_hits_mask(x0, y0, x1, y1, mask, h, w):
                continue
            xs = (x1 - x0) / radius
            ys = (y1 - y0) / radius
            for k in range(n_terms):
                phi[k] = xs ** exponents[k, 0] * ys ** exponents[k, 1]
            for a in range(n_terms):
                rhs_u[a] += phi[a] * u[j]
                rhs_v[a] += phi[a] * v[j]
                for b in range(a + 1):
                    normal[a, b] += phi[a] * phi[b]
            count += 1
            sum_x += xs
            sum_y += ys
        if count < min_neighbors:
            continue
        if np.hypot(sum_x, sum_y) > max_offset * count:
            continue
        for a in range(n_terms):
            for b in range(a):
                normal[b, a] = normal[a, b]
        if not _solve_two(normal, rhs_u, rhs_v, pivot_min):
            continue
        # Coefficients are in units of radius; a derivative of total order p
        # carries a! b! / radius^p.
        lap[i, 0] = 2.0 * (rhs_u[i20] + rhs_u[i02]) * inv_r2
        lap[i, 1] = 2.0 * (rhs_v[i20] + rhs_v[i02]) * inv_r2
        quart[i, 0] = (9.0 * 24.0 * (rhs_u[i40] + rhs_u[i04])
                       + 70.0 * 4.0 * rhs_u[i22]) * inv_r2 * inv_r2
        quart[i, 1] = (9.0 * 24.0 * (rhs_v[i40] + rhs_v[i04])
                       + 70.0 * 4.0 * rhs_v[i22]) * inv_r2 * inv_r2

        # Scatter of the neighbours about the fitted polynomial.
        sq = 0.0
        for jj in range(indptr[i], indptr[i + 1]):
            j = indices[jj]
            if not usable[j]:
                continue
            x1 = coords[j, 0]
            y1 = coords[j, 1]
            if use_mask and j != i and _seg_hits_mask(x0, y0, x1, y1, mask, h, w):
                continue
            xs = (x1 - x0) / radius
            ys = (y1 - y0) / radius
            fit_u = 0.0
            fit_v = 0.0
            for k in range(n_terms):
                term = xs ** exponents[k, 0] * ys ** exponents[k, 1]
                fit_u += rhs_u[k] * term
                fit_v += rhs_v[k] * term
            sq += (u[j] - fit_u) ** 2 + (v[j] - fit_v) ** 2
        if count > n_terms:
            resid[i] = np.sqrt(sq / (2.0 * (count - n_terms)))

        # Noise gain of the two estimators: for independent nodal noise of
        # variance s^2, var(estimate) = s^2 * c^T N^-1 c.
        c_lap = np.zeros(n_terms)
        c_lap[i20] = 2.0
        c_lap[i02] = 2.0
        c_quart = np.zeros(n_terms)
        c_quart[i40] = 9.0 * 24.0
        c_quart[i04] = 9.0 * 24.0
        c_quart[i22] = 70.0 * 4.0
        x_lap = c_lap.copy()
        x_quart = c_quart.copy()
        if _solve_two(normal, x_lap, x_quart, pivot_min):
            g_lap = 0.0
            g_quart = 0.0
            for k in range(n_terms):
                g_lap += c_lap[k] * x_lap[k]
                g_quart += c_quart[k] * x_quart[k]
            gain[i, 0] = g_lap * inv_r2 * inv_r2
            gain[i, 1] = g_quart * inv_r2 * inv_r2 * inv_r2 * inv_r2
    return lap, quart, gain, resid


def fit_displacement_derivatives(
    coords: NDArray[np.float64],
    U_2d: NDArray[np.float64],
    usable: NDArray[np.bool_],
    radius: float,
    mask: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Laplacian and quartic term of the displacement at every usable node.

    A degree-4 polynomial is fitted by least squares to the displacement of
    the usable nodes within ``radius`` of each node. With ``mask`` given, a
    neighbour whose straight path to the node crosses a masked pixel is left
    out, so the fit does not span a hole or a crack.

    Returns:
        (laplacian, quartic), each (N, 2) for the two displacement
        components: ``u_xx + u_yy`` and ``9 (u_xxxx + u_yyyy) + 70 u_xxyy``.
        NaN where the node is not usable or its fit is not determined (too
        few neighbours, a one-sided or degenerate neighbourhood).
    """
    lap, quart, _, _ = fit_displacement_model(coords, U_2d, usable, radius, mask)
    return lap, quart


def _fit_neighbors(
    coords: NDArray[np.float64], usable: NDArray[np.bool_], radius: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], float]:
    """Neighbour lists (CSR) for the fit, and the spacing of the nodes in them.

    Normally every node within *radius*; the returned spacing is then 0
    (the mesh's own). On a mesh with more than ``MAX_FIT_NEIGHBORS`` usable
    nodes per fit on average, one usable node per square cell is kept, the
    cell sized so that a fit sees about that many, and the cell size is
    returned as the spacing.
    """
    from scipy.spatial import cKDTree

    n_usable = int(np.count_nonzero(usable))
    if n_usable == 0:
        return (*build_neighbor_cache(coords, radius), 0.0)
    usable_idx = np.flatnonzero(usable)
    tree = cKDTree(coords[usable_idx])
    per_fit = tree.count_neighbors(tree, radius) / n_usable
    if per_fit <= MAX_FIT_NEIGHBORS:
        return (*build_neighbor_cache(coords, radius), 0.0)

    cell = radius * np.sqrt(np.pi / MAX_FIT_NEIGHBORS)
    keys = np.floor(coords[usable_idx] / cell).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    kept = usable_idx[np.sort(first)]
    lists = cKDTree(coords[kept]).query_ball_point(coords, radius)
    n = coords.shape[0]
    counts = np.fromiter((len(lst) for lst in lists), dtype=np.int64, count=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    if indptr[-1] == 0:
        return indptr, np.zeros(0, dtype=np.int64), float(cell)
    indices = kept[np.concatenate([np.asarray(lst, dtype=np.int64) for lst in lists])]
    return indptr, indices.astype(np.int64), float(cell)


def fit_displacement_model(
    coords: NDArray[np.float64],
    U_2d: NDArray[np.float64],
    usable: NDArray[np.bool_],
    radius: float,
    mask: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64]]:
    """The fit of :func:`fit_displacement_derivatives` with its diagnostics.

    Returns:
        (laplacian, quartic, gain, residual). ``gain`` is (N, 2): the
        variance of the Laplacian and of the quartic estimate per unit
        variance of independent nodal noise. ``residual`` is (N,): the
        r.m.s. scatter of the neighbouring displacements about the fitted
        polynomial (px).
    """
    return _fit_with_spacing(coords, U_2d, usable, radius, mask)[:4]


def _fit_with_spacing(coords, U_2d, usable, radius, mask):
    """(laplacian, quartic, gain, residual, spacing of the nodes used, 0 = the mesh's)."""
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    usable = np.ascontiguousarray(usable, dtype=np.bool_)
    indptr, indices, spacing = _fit_neighbors(coords, usable, radius)
    use_mask = mask is not None
    mask_arr = (np.ascontiguousarray(mask, dtype=np.float64) if use_mask
                else np.ones((1, 1), dtype=np.float64))
    lap, quart, gain, resid = _fit_kernel(
        coords,
        np.ascontiguousarray(U_2d[:, 0], dtype=np.float64),
        np.ascontiguousarray(U_2d[:, 1], dtype=np.float64),
        usable,
        indptr.astype(np.int64), indices.astype(np.int64), float(radius),
        mask_arr, use_mask, _EXPONENTS, MIN_FIT_NEIGHBORS, FIT_PIVOT_MIN,
        MAX_FIT_OFFSET, _I20, _I02, _I40, _I04, _I22,
    )
    return lap, quart, gain, resid, spacing


def _noise_and_sigma(
    ctx: LocalICGNContext,
    U_2d: NDArray[np.float64],
    F_2d: NDArray[np.float64],
    usable: NDArray[np.bool_],
    g_img: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Image-noise estimate and predicted first-order random error per node."""
    pre = ctx.pre
    znssd, n_pix = first_order_znssd_parallel(
        np.round(ctx.coordinates_fem).astype(np.float64),
        np.ascontiguousarray(U_2d, dtype=np.float64),
        np.ascontiguousarray(F_2d, dtype=np.float64),
        np.ascontiguousarray(usable, dtype=np.bool_),
        pre["ref_all"], pre["mask_all"], pre["XX_all"], pre["YY_all"],
        pre["meanf_all"], pre["bottomf_all"],
        np.ascontiguousarray(g_img, dtype=np.float64),
    )
    # The residual carries the noise of both images; 6 warp parameters and
    # the two ZNSSD statistics are fitted to it.
    dof = n_pix - 8
    with np.errstate(invalid="ignore", divide="ignore"):
        per_node = pre["bottomf_all"] * np.sqrt(znssd / (2.0 * dof))
    good = usable & np.isfinite(per_node) & (dof > 0)
    if not np.any(good):
        return float("nan"), np.full(usable.shape[0], np.nan)
    noise_sd = float(np.median(per_node[good]))

    hess = pre["H_all"]
    sssig_x = hess[:, 4, 4]
    sssig_y = hess[:, 5, 5]
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.sqrt(2.0) * noise_sd * np.sqrt(1.0 / sssig_x + 1.0 / sssig_y)
    sigma = np.where(usable & (sssig_x > 0) & (sssig_y > 0), sigma, np.nan)
    return noise_sd, sigma


def predict_shape_order(
    ctx: LocalICGNContext,
    U_2d: NDArray[np.float64],
    F_2d: NDArray[np.float64],
    conv_iter: NDArray[np.int64],
    g_img: NDArray[np.float64],
    winstepsize: int,
) -> ShapeOrderPrediction:
    """Predict the more accurate shape-function order at every node.

    Args:
        ctx: Context from ``local_icgn_precompute``.
        U_2d, F_2d, conv_iter: First-order result of ``local_icgn_solve_subset``.
        g_img: Deformed image (H, W).
        winstepsize: Node step of the (unrefined) mesh, in pixels.

    Returns:
        A :class:`ShapeOrderPrediction`. Nodes the first-order solve did not
        converge at, bad points and nodes whose fit is not determined are
        never selected.
    """
    n = ctx.n_nodes
    u_in = np.asarray(U_2d, dtype=np.float64)
    f_in = np.asarray(F_2d, dtype=np.float64)
    usable = (
        np.asarray(ctx.pre["valid"], dtype=np.bool_)
        & (conv_iter >= 1) & (conv_iter <= ctx.max_iter)
        & np.all(np.isfinite(u_in), axis=1) & np.all(np.isfinite(f_in), axis=1)
    )
    bad_pts, _ = _detect_bad_points(ctx, conv_iter)
    usable[bad_pts] = False

    radius = fit_radius(ctx.winsize, winstepsize)
    half = ctx.winsize // 2
    nothing = np.full(n, np.nan)
    if not np.any(usable):
        return ShapeOrderPrediction(
            select=np.zeros(n, dtype=np.bool_), fitted=np.zeros(n, dtype=np.bool_),
            bias_first=nothing, bias_second=nothing.copy(), sigma_first=nothing.copy(),
            noise_sd=float("nan"), radius=radius, half=half,
            spacing=float(winstepsize),
            lap_gain=nothing.copy(), quart_gain=nothing.copy(),
            fit_residual=nothing.copy())

    noise_sd, sigma = _noise_and_sigma(ctx, u_in, f_in, usable, g_img)
    lap, quart, gain, resid, thinned = _fit_with_spacing(
        ctx.coordinates_fem, u_in, usable, radius, ctx.img_ref_mask)
    # Spacing of the nodes the fit used: the node step, or wider if thinned.
    spacing = max(float(winstepsize), thinned)

    bias_first = half * (half + 1) / 6.0 * np.hypot(lap[:, 0], lap[:, 1])
    bias_second = (half ** 2 * (half + 1) ** 2 / 2520.0
                   * np.hypot(quart[:, 0], quart[:, 1]))
    fitted = usable & np.isfinite(bias_first) & np.isfinite(bias_second)
    known = fitted & np.isfinite(sigma) & np.isfinite(resid) & np.isfinite(gain[:, 0])
    # The fitted Laplacian carries the noise of its own estimate. Only the
    # part that stands out from it counts as bias: the estimate's variance is
    # the scatter of the displacement about the fit (or the predicted noise,
    # if larger) times the gain of the fit. Nodes closer than a subset share
    # pixels, so their errors are correlated and the independent-noise gain
    # understates that variance by about (subset / step)^2. On real images
    # the scatter is mostly systematic (interpolation bias, lens distortion);
    # without this step it reads as curvature and even a rigid rotation is
    # sent to second order.
    size = 2 * half + 1
    correlation = max(1.0, (size / spacing) ** 2)
    with np.errstate(invalid="ignore"):
        scatter_sq = np.maximum(2.0 * resid ** 2, sigma ** 2)
        significant_sq = np.maximum(
            bias_first ** 2
            - (half * (half + 1) / 6.0) ** 2 * scatter_sq * gain[:, 0] * correlation, 0.0)
        better = significant_sq + sigma ** 2 > bias_second ** 2 + 4.0 * sigma ** 2
    return ShapeOrderPrediction(
        select=known & better, fitted=fitted,
        bias_first=np.where(fitted, bias_first, np.nan),
        bias_second=np.where(fitted, bias_second, np.nan),
        sigma_first=sigma, noise_sd=noise_sd, radius=radius, half=half,
        spacing=spacing,
        lap_gain=gain[:, 0], quart_gain=gain[:, 1], fit_residual=resid,
    )
