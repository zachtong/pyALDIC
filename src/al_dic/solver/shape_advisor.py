"""Post-run suggestion for the subset size and shape function (Local DIC).

Given the error prediction of one finished frame pair
(:mod:`al_dic.solver.shape_order_criterion`), rank a short ladder of
subset sizes, each with the first-order shape function and with the
automatic choice, by their predicted displacement error, and report the
best one only when it is clearly better than the current setting.

The advisor is built to stay quiet rather than to guess:

* it only ranks candidates near the current subset size (never a closed
  form that could return an extreme value), each one inside hard limits;
* the curvature it extrapolates from is first corrected for the noise of
  its own estimate, so noise is not mistaken for deformation;
* scatter of the measured displacement that image noise does not explain
  (interpolation bias, lens distortion, unresolved detail) is carried as an
  error floor that no candidate is predicted to remove;
* it does not suggest a larger subset where the deformation already
  varies within the current one;
* it needs a clear predicted gain before it suggests a change;
* when its inputs do not check out it says that it has no suggestion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import ShapeAdvice
from .shape_order_criterion import ShapeOrderPrediction

__all__ = ["advise_shape_settings", "candidate_winsizes"]

#: Candidate subset sizes, as factors of the current one.
SUBSET_LADDER = (0.75, 1.0, 1.5, 2.0)

#: Subset-size range of the GUI control (it shows winsize + 1: 11 to 201 px).
WINSIZE_MIN = 10
WINSIZE_MAX = 200

#: A subset may span at most this fraction of the smaller ROI dimension.
MAX_ROI_FRACTION = 0.25

#: Predicted error must drop by at least this fraction for a suggestion.
MIN_GAIN = 0.2

#: Candidates whose predicted error is within this fraction of the best one
#: count as equal; the one closest to the current setting is preferred.
TIE = 0.03

#: If the first-order bias exceeds the noise at more than this share of the
#: nodes, no larger subset is suggested. The curvature is measured through
#: the current subset, which already attenuates what it cannot resolve; a
#: larger subset would hide more of it from the next run, so the predicted
#: gain of growing cannot be trusted there.
MAX_BIAS_SHARE = 0.05

#: Fewest nodes with a prediction, and the smallest share of all nodes.
MIN_NODES = 200
MIN_FITTED_FRACTION = 0.3

#: Share of the largest predicted errors left out of the average.
TRIM = 0.02

#: Scatter of the displacement about its local fit up to this multiple of
#: the predicted noise is treated as noise (it shrinks with a larger
#: subset); the noise formula undercounts by up to about this factor for
#: small subsets. Scatter beyond it is an error floor no setting removes.
NOISE_SLACK = 2.0

#: Accepted range of (scatter of the displacement about its local fit) /
#: (predicted first-order noise). Outside it the noise model does not
#: describe this run and no suggestion is made.
SELF_CHECK = (0.1, 3.0)


def _even(value: float) -> int:
    return int(2 * round(value / 2.0))


def candidate_winsizes(winsize: int, winstepsize: int, roi_min_dim: float | None) -> list[int]:
    """Subset sizes the advisor may suggest, always including the current one.

    Each candidate is an even ``winsize`` inside the GUI range, no smaller
    than the node step, and no larger than a quarter of the smaller ROI
    dimension. A ladder value outside those limits is dropped, not clamped.
    """
    upper = WINSIZE_MAX
    if roi_min_dim is not None and np.isfinite(roi_min_dim):
        upper = min(upper, _even(MAX_ROI_FRACTION * roi_min_dim) - 2)
    lower = max(WINSIZE_MIN, _even(winstepsize))
    sizes = {int(winsize)}
    for factor in SUBSET_LADDER:
        candidate = _even(winsize * factor)
        if lower <= candidate <= upper:
            sizes.add(candidate)
    return sorted(sizes)


def _trimmed_rms(sq_error: NDArray[np.float64]) -> float:
    keep = max(1, int(np.ceil(sq_error.size * (1.0 - TRIM))))
    return float(np.sqrt(np.mean(np.partition(sq_error, keep - 1)[:keep])))


def advise_shape_settings(
    prediction: ShapeOrderPrediction,
    *,
    winsize: int,
    winstepsize: int,
    current_order: str,
    roi_min_dim: float | None = None,
    frame: int = 0,
) -> ShapeAdvice:
    """Suggest a subset size and shape function, or say that none is needed.

    Args:
        prediction: Error prediction of the frame pair, made at ``winsize``.
        winsize: Current subset size parameter (the subset spans ``winsize + 1`` px).
        winstepsize: Node step (px).
        current_order: ``"first"``, ``"second"`` or ``"auto"``.
        roi_min_dim: Smaller dimension of the ROI (px), if known.
        frame: Index of the frame pair, recorded in the advice.

    Returns:
        A :class:`ShapeAdvice` with ``status`` ``"suggest"``, ``"keep"`` or
        ``"unavailable"`` (then ``reason`` says why).
    """
    def result(status: str, reason: str = "", suggested=(winsize, current_order),
               gain: float = 0.0, n_nodes: int = 0) -> ShapeAdvice:
        return ShapeAdvice(
            status=status, reason=reason, frame=int(frame),
            current_winsize=int(winsize), current_order=current_order,
            suggested_winsize=int(suggested[0]), suggested_order=suggested[1],
            predicted_gain=float(gain), n_nodes=int(n_nodes),
        )

    sigma = prediction.sigma_first
    nodes = (prediction.fitted & np.isfinite(sigma) & (sigma > 0)
             & np.isfinite(prediction.lap_gain) & np.isfinite(prediction.quart_gain)
             & np.isfinite(prediction.fit_residual))
    n_nodes = int(nodes.sum())
    if not np.isfinite(prediction.noise_sd):
        return result("unavailable", "noise_unknown", n_nodes=n_nodes)
    if n_nodes < MIN_NODES or n_nodes < MIN_FITTED_FRACTION * nodes.size:
        return result("unavailable", "too_few_nodes", n_nodes=n_nodes)

    sigma = sigma[nodes]
    # sigma is the norm over the two components; the fit residual is per component.
    scatter_sq = 2.0 * prediction.fit_residual[nodes] ** 2
    check = float(np.median(np.sqrt(scatter_sq) / sigma))
    if not (SELF_CHECK[0] <= check <= SELF_CHECK[1]):
        return result("unavailable", "model_mismatch", n_nodes=n_nodes)
    # Random error of each node at the current setting: the predicted noise,
    # or the measured scatter where that is larger (within NOISE_SLACK).
    noise_sq = np.clip(scatter_sq, sigma ** 2, (NOISE_SLACK * sigma) ** 2)
    # Scatter beyond that has an unknown cause, so it is assumed to stay
    # whatever the setting: the advisor predicts little gain where it dominates.
    floor_sq = np.maximum(scatter_sq - noise_sq, 0.0)
    noisy_sq = np.maximum(scatter_sq, sigma ** 2)

    half = prediction.half
    size = 2 * half + 1
    lap = prediction.bias_first[nodes] * 6.0 / (half * (half + 1))
    quart = prediction.bias_second[nodes] * 2520.0 / (half ** 2 * (half + 1) ** 2)
    # Remove the part of the measured derivatives that is their own noise.
    # Nodes closer than a subset share pixels, so their noise is correlated
    # and the independent-noise gain understates the variance by (size/step)^2.
    correlation = max(1.0, (size / max(float(winstepsize), prediction.spacing)) ** 2)
    lap_sq = np.maximum(lap ** 2 - noisy_sq * prediction.lap_gain[nodes] * correlation, 0.0)
    quart_sq = np.maximum(
        quart ** 2 - noisy_sq * prediction.quart_gain[nodes] * correlation, 0.0)

    def predicted(candidate: int, order: str) -> float:
        m = candidate // 2
        scale = (candidate + 1) / float(size)
        var = noise_sq / scale ** 2
        first = (m * (m + 1) / 6.0) ** 2 * lap_sq + var + floor_sq
        second = (m ** 2 * (m + 1) ** 2 / 2520.0) ** 2 * quart_sq + 4.0 * var + floor_sq
        if order == "first":
            return _trimmed_rms(first)
        if order == "second":
            return _trimmed_rms(second)
        return _trimmed_rms(np.minimum(first, second))

    current = predicted(winsize, current_order)
    bias_now_sq = (half * (half + 1) / 6.0) ** 2 * lap_sq
    may_grow = float(np.mean(bias_now_sq > noise_sq)) <= MAX_BIAS_SHARE
    sizes = [w for w in candidate_winsizes(winsize, winstepsize, roi_min_dim)
             if may_grow or w <= winsize]
    scored = [(predicted(w, order), w, order) for w in sizes for order in ("first", "auto")]
    best_error = min(s[0] for s in scored)
    # Among near-equal candidates prefer the smallest change: same subset
    # size, then same shape function, then the smaller subset.
    near = [s for s in scored if s[0] <= best_error * (1.0 + TIE)]
    _, best_w, best_order = min(
        near, key=lambda s: (s[1] != winsize, s[2] != current_order, abs(s[1] - winsize), s[1]))
    best = predicted(best_w, best_order)
    gain = 1.0 - best / current if current > 0 else 0.0

    if gain >= MIN_GAIN and (best_w, best_order) != (winsize, current_order):
        return result("suggest", suggested=(best_w, best_order), gain=gain, n_nodes=n_nodes)
    return result("keep", n_nodes=n_nodes)
