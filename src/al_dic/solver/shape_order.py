"""Second-order shape-function stage for Local DIC.

Runs after the first-order IC-GN solve and before its post-processing.
Each node ends in exactly one of three states, recorded in ``order``:

    2  refined with the 12-parameter shape function and accepted;
    1  kept at its first-order result (bit for bit), because the stage
       was not asked to refine it or a guard refused the refinement;
    0  not solved by the first-order step; never touched here.

The stage does not raise for numerical reasons and never introduces a
non-finite value: on any doubt a node keeps its first-order result, and
if the kernel itself fails the whole field does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .icgn2_kernels import (
    N_PAR, STATUS_ILL_CONDITIONED, STATUS_NON_FINITE, STATUS_NOT_CONVERGED,
    STATUS_NOT_RUN, STATUS_OK, STATUS_OUT_OF_IMAGE, STATUS_TOO_FEW_PIXELS,
    icgn2_refine_parallel,
)
from .local_icgn import LocalICGNContext

logger = logging.getLogger(__name__)

__all__ = [
    "ShapeOrderResult", "refine_shape_order",
    "STATUS_NOT_RUN", "STATUS_OK", "STATUS_NOT_CONVERGED", "STATUS_OUT_OF_IMAGE",
    "STATUS_ILL_CONDITIONED", "STATUS_TOO_FEW_PIXELS", "STATUS_NON_FINITE",
    "STATUS_SHIFT_TOO_LARGE",
]

#: Stage-level status: the kernel converged but moved the subset centre
#: further than ``MAX_SHIFT_SECOND``.
STATUS_SHIFT_TOO_LARGE = 6

#: The second-order stop tolerance (largest pixel motion of an update) is
#: this fraction of the first-order ``tol``. The first-order stop rule is
#: relative to its starting gradient and cannot be reused: this stage
#: starts from a converged state.
SECOND_TOL_FACTOR = 0.1

#: Minimum connected fraction of the subset window for a second-order
#: solve: the same half window the first-order solve requires. Measured on
#: windows cut by a hole edge, the second-order error relative to the
#: first-order one does not grow as the window is cut down to that limit
#: (it falls, because the first-order bias is largest at the edge), so no
#: stricter limit is justified; a degenerate footprint is caught by
#: ``PIVOT_MIN`` instead.
MIN_FILL_SECOND = 0.5

#: Minimum squared pivot of the Jacobi-scaled Cholesky factor of the
#: 12x12 Hessian (1 for an orthogonal parameter set). Whole windows on a
#: speckle pattern measure 0.2-0.35 and windows cut in half by a hole edge
#: 0.03-0.1; anything more than three times worse conditioned than that
#: keeps its first-order result.
PIVOT_MIN = 1e-2

#: A refinement that moves the subset centre further than this many pixels
#: from the first-order result is rejected: the stage corrects a bias, it
#: does not search. The first-order bias is M(M+1)/6 * laplacian(u); for it
#: to reach half a pixel with a 31 px subset the strain would have to change
#: by about 1 % per pixel, so a larger move is noise or a mismatch (seen at
#: texture-poor nodes of real images), not a correction.
MAX_SHIFT_SECOND = 0.5

# Columns of the kernel parameter vector
# [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy].
_COLS_U = (0, 6)
_COLS_F = (1, 7, 2, 8)                 # al_dic layout [ux, vx, uy, vy]
_COLS_SECOND = (3, 4, 5, 9, 10, 11)    # [uxx, uxy, uyy, vxx, vxy, vyy]


@dataclass(frozen=True)
class ShapeOrderResult:
    """Outcome of the second-order stage for one frame pair.

    Attributes:
        U_2d: (N, 2) displacement at the subset centre.
        F_2d: (N, 4) displacement gradient at the centre, ``[ux, vx, uy, vy]``.
        order: (N,) int8; 0 not solved, 1 first order, 2 second order.
        second: (N, 6) ``[uxx, uxy, uyy, vxx, vxy, vyy]``; zero unless
            ``order == 2``.
        iters: (N,) second-order iterations (0 where the stage did not run).
        status: (N,) int8 status code; ``STATUS_NOT_RUN`` where the stage
            did not run.
        znssd: (N,) ZNSSD of the accepted second-order match, NaN elsewhere.
    """

    U_2d: NDArray[np.float64]
    F_2d: NDArray[np.float64]
    order: NDArray[np.int8]
    second: NDArray[np.float64]
    iters: NDArray[np.int64]
    status: NDArray[np.int8]
    znssd: NDArray[np.float64]


def _first_order_result(
    u_in: NDArray[np.float64],
    f_in: NDArray[np.float64],
    solved: NDArray[np.bool_],
) -> ShapeOrderResult:
    n = u_in.shape[0]
    return ShapeOrderResult(
        U_2d=u_in.copy(),
        F_2d=f_in.copy(),
        order=np.where(solved, 1, 0).astype(np.int8),
        second=np.zeros((n, 6), dtype=np.float64),
        iters=np.zeros(n, dtype=np.int64),
        status=np.full(n, STATUS_NOT_RUN, dtype=np.int8),
        znssd=np.full(n, np.nan),
    )


def refine_shape_order(
    ctx: LocalICGNContext,
    U_2d: NDArray[np.float64],
    F_2d: NDArray[np.float64],
    conv_iter: NDArray[np.int64],
    g_img: NDArray[np.float64],
    tol: float,
    *,
    select: NDArray[np.bool_] | None = None,
) -> ShapeOrderResult:
    """Refine first-order IC-GN results with a second-order shape function.

    Args:
        ctx: Context from ``local_icgn_precompute`` for these nodes.
        U_2d: (N, 2) first-order displacements from ``local_icgn_solve_subset``.
        F_2d: (N, 4) first-order gradients ``[ux, vx, uy, vy]``.
        conv_iter: (N,) first-order iteration counts (``> ctx.max_iter``
            marks a node that did not converge).
        g_img: Deformed image (H, W).
        tol: The first-order tolerance (``DICPara.tol``).
        select: Optional (N,) bool; only these nodes may be refined.
            ``None`` refines every solved node.

    Returns:
        A :class:`ShapeOrderResult`. The inputs are not modified.
    """
    u_in = np.asarray(U_2d, dtype=np.float64)
    f_in = np.asarray(F_2d, dtype=np.float64)
    pre = ctx.pre

    solved = (
        np.asarray(pre["valid"], dtype=np.bool_)
        & (conv_iter >= 1) & (conv_iter <= ctx.max_iter)
        & np.all(np.isfinite(u_in), axis=1)
        & np.all(np.isfinite(f_in), axis=1)
    )
    active = solved if select is None else solved & np.asarray(select, dtype=np.bool_)
    base = _first_order_result(u_in, f_in, solved)
    if not np.any(active):
        return base

    start = np.zeros((u_in.shape[0], N_PAR), dtype=np.float64)
    start[np.ix_(active, _COLS_U)] = u_in[active]
    start[np.ix_(active, _COLS_F)] = f_in[active]

    try:
        refined, iters, status, znssd = icgn2_refine_parallel(
            np.round(ctx.coordinates_fem).astype(np.float64), start, active,
            pre["ref_all"], pre["gx_all"], pre["gy_all"], pre["mask_all"],
            pre["XX_all"], pre["YY_all"], pre["meanf_all"], pre["bottomf_all"],
            np.ascontiguousarray(g_img, dtype=np.float64),
            SECOND_TOL_FACTOR * tol, ctx.max_iter, MIN_FILL_SECOND, PIVOT_MIN,
        )
    except Exception:
        logger.warning(
            "The second-order shape-function stage failed; keeping the "
            "first-order result for every node.",
            exc_info=True,
        )
        return base

    shift = np.hypot(refined[:, 0] - start[:, 0], refined[:, 6] - start[:, 6])
    converged = active & (status == STATUS_OK) & np.all(np.isfinite(refined), axis=1)
    too_far = converged & ~(shift <= MAX_SHIFT_SECOND)
    accepted = converged & ~too_far

    status = status.copy()
    status[too_far] = STATUS_SHIFT_TOO_LARGE

    u_out = u_in.copy()
    f_out = f_in.copy()
    second = np.zeros((u_in.shape[0], 6), dtype=np.float64)
    u_out[accepted] = refined[np.ix_(accepted, _COLS_U)]
    f_out[accepted] = refined[np.ix_(accepted, _COLS_F)]
    second[accepted] = refined[np.ix_(accepted, _COLS_SECOND)]

    order = base.order.copy()
    order[accepted] = 2

    return ShapeOrderResult(
        U_2d=u_out,
        F_2d=f_out,
        order=order,
        second=second,
        iters=np.where(active, iters, 0),
        status=status,
        znssd=np.where(accepted, znssd, np.nan),
    )
