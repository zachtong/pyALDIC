"""The post-run suggestion for subset size and shape function.

The requirement these tests pin: a suggestion is never an absurd value.
Whatever the inputs, it is a ladder value near the current subset size
and inside the hard limits, or there is no suggestion at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from al_dic.solver import shape_advisor
from al_dic.solver.shape_advisor import advise_shape_settings, candidate_winsizes
from al_dic.solver.shape_order_criterion import ShapeOrderPrediction

N = 1200


def _prediction(*, half=16, lap=0.0, quart=0.0, sigma=0.02, lap_gain=0.0, quart_gain=0.0,
                residual_ratio=0.5, fitted=None, noise_sd=0.01, n=N):
    """A synthetic prediction with uniform per-node values (arrays also accepted)."""
    def arr(value):
        return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()

    sigma_arr = arr(sigma)
    lap_arr, quart_arr = arr(lap), arr(quart)
    return ShapeOrderPrediction(
        select=np.zeros(n, dtype=bool),
        fitted=np.ones(n, dtype=bool) if fitted is None else fitted,
        bias_first=half * (half + 1) / 6.0 * lap_arr,
        bias_second=half ** 2 * (half + 1) ** 2 / 2520.0 * quart_arr,
        sigma_first=sigma_arr, noise_sd=noise_sd, radius=46.0, half=half, spacing=8.0,
        lap_gain=arr(lap_gain), quart_gain=arr(quart_gain),
        fit_residual=residual_ratio * sigma_arr / np.sqrt(2.0),
    )


def _advise(pred, winsize=32, step=8, order="first", roi=600.0):
    return advise_shape_settings(pred, winsize=winsize, winstepsize=step,
                                 current_order=order, roi_min_dim=roi, frame=3)


# --- candidates -----------------------------------------------------------

def test_candidates_are_a_short_ladder_around_the_current_size():
    assert candidate_winsizes(32, 8, 600.0) == [24, 32, 48, 64]


def test_candidates_respect_the_gui_range_the_step_and_the_roi():
    assert candidate_winsizes(10, 4, 600.0) == [10, 16, 20]          # 0.75x is below the minimum
    assert max(candidate_winsizes(160, 8, 2000.0)) <= shape_advisor.WINSIZE_MAX
    assert candidate_winsizes(16, 16, 600.0) == [16, 24, 32]         # not below the node step
    assert candidate_winsizes(32, 8, 150.0) == [24, 32]              # a quarter of a 150 px ROI
    assert candidate_winsizes(64, 8, 100.0) == [64]                  # current size is always kept


@pytest.mark.parametrize("seed", range(40))
def test_suggestion_is_never_outside_the_ladder_or_the_limits(seed):
    """Random, including degenerate, inputs: whatever comes out is sane."""
    rng = np.random.default_rng(seed)
    winsize = int(rng.choice([10, 16, 24, 32, 40, 64, 100, 200]))
    step = int(rng.choice([4, 8, 16, 32]))
    roi = float(rng.choice([80.0, 300.0, 1500.0, np.nan]))
    n = int(rng.choice([50, 400, 3000]))
    scale = 10.0 ** rng.uniform(-9, 2)
    pred = _prediction(
        half=winsize // 2, n=n,
        lap=np.abs(rng.standard_normal(n)) * scale,
        quart=np.abs(rng.standard_normal(n)) * scale * 10.0 ** rng.uniform(-6, 0),
        sigma=np.abs(rng.standard_normal(n)) * 10.0 ** rng.uniform(-4, 0) + 1e-9,
        lap_gain=10.0 ** rng.uniform(-12, -2), quart_gain=10.0 ** rng.uniform(-20, -6),
        residual_ratio=10.0 ** rng.uniform(-2, 1),
        fitted=rng.random(n) > rng.uniform(0, 1),
        noise_sd=float(rng.choice([0.01, np.nan])),
    )
    order = str(rng.choice(["first", "second"]))

    advice = _advise(pred, winsize=winsize, step=step, order=order, roi=roi)

    assert advice.status in ("suggest", "keep", "unavailable")
    assert advice.current_winsize == winsize and advice.current_order == order
    allowed = candidate_winsizes(winsize, step, roi)
    assert advice.suggested_winsize in allowed
    assert 0.74 * winsize <= advice.suggested_winsize <= 2.01 * winsize
    assert advice.suggested_winsize % 2 == 0
    if advice.status == "suggest":
        assert advice.suggested_order in ("first", "second")
        assert advice.predicted_gain >= shape_advisor.MIN_GAIN
        assert (advice.suggested_winsize, advice.suggested_order) != (winsize, order)
    else:
        assert (advice.suggested_winsize, advice.suggested_order) == (winsize, order)
        assert advice.predicted_gain == 0.0


# --- behaviour ------------------------------------------------------------

def test_no_curvature_suggests_a_larger_first_order_subset():
    advice = _advise(_prediction(lap=0.0, quart=0.0))
    assert advice.status == "suggest"
    assert (advice.suggested_winsize, advice.suggested_order) == (64, "first")
    assert advice.predicted_gain == pytest.approx(1.0 - 33.0 / 65.0, abs=0.01)
    assert advice.frame == 3


def test_measured_curvature_that_is_only_noise_is_not_treated_as_bias():
    """A Laplacian estimate no larger than its own noise must not stop the
    advisor from seeing that a larger subset is safe."""
    sigma, gain, step, size = 0.02, 1.2e-7, 8, 33
    noise_lap = sigma * np.sqrt(gain) * (size / step)          # one s.d. of pure noise
    advice = _advise(_prediction(lap=noise_lap, lap_gain=gain, sigma=sigma))
    assert advice.status == "suggest"
    assert advice.suggested_order == "first" and advice.suggested_winsize > 32


def test_bias_dominated_field_suggests_second_order_not_a_larger_subset():
    """Where the deformation already varies within the subset, the measured
    curvature is attenuated and a larger subset would hide more of it: the
    advisor may change the shape function or shrink, never grow."""
    half, sigma = 16, 0.02
    lap = 5.0 * sigma / (half * (half + 1) / 6.0)      # bias 5x the noise
    advice = _advise(_prediction(lap=lap, quart=lap * 1e-4, sigma=sigma))
    assert advice.status == "suggest"
    assert advice.suggested_order == "second"
    assert advice.suggested_winsize <= 32


def test_bias_in_a_small_region_does_not_block_a_larger_subset():
    half, sigma = 16, 0.02
    lap = np.zeros(N)
    lap[: int(0.03 * N)] = 1.5 * sigma / (half * (half + 1) / 6.0)   # 3 % of nodes
    advice = _advise(_prediction(lap=lap, sigma=sigma))
    assert advice.status == "suggest"
    assert advice.suggested_winsize > 32


def test_scatter_that_noise_does_not_explain_silences_the_advisor():
    """Real images: the displacement scatters more than image noise predicts
    (interpolation bias, lens distortion). That part is not removed by any
    setting, so an apparent curvature there must not produce a suggestion."""
    half, sigma = 16, 0.002
    lap = 2.0 * sigma / (half * (half + 1) / 6.0)       # looks like bias 2x the noise
    quiet = _advise(_prediction(lap=lap, quart=lap * 1e-4, sigma=sigma, residual_ratio=0.9))
    assert quiet.status == "suggest"                     # noise-limited: believed
    floored = _advise(_prediction(lap=lap, quart=lap * 1e-4, sigma=sigma, residual_ratio=2.9))
    assert floored.status == "keep"


def test_scatter_within_twice_the_predicted_noise_counts_as_noise():
    """Small subsets: the noise formula undercounts. Scatter up to twice the
    prediction still shrinks with a larger subset, so growing is suggested."""
    advice = _advise(_prediction(lap=0.0, residual_ratio=1.6), winsize=16)
    assert advice.status == "suggest"
    assert (advice.suggested_winsize, advice.suggested_order) == (32, "first")


def test_balanced_setting_is_kept():
    # Bias about equal to the noise: smaller is noisier, larger is more biased.
    half, sigma = 16, 0.02
    lap = 1.0 * sigma / (half * (half + 1) / 6.0)
    quart = 2520.0 * 1.5 * sigma / (half ** 2 * (half + 1) ** 2)   # second order no better
    advice = _advise(_prediction(lap=lap, quart=quart, sigma=sigma))
    assert advice.status == "keep"
    assert advice.suggested_winsize == 32 and advice.suggested_order == "first"


def test_already_on_the_best_setting_is_kept():
    advice = _advise(_prediction(lap=0.0), winsize=200, order="first", roi=5000.0)
    assert advice.status == "keep"        # 200 is the top of the range: nothing larger


@pytest.mark.parametrize("kwargs, reason", [
    ({"noise_sd": float("nan")}, "noise_unknown"),
    ({"fitted": np.arange(N) < 50}, "too_few_nodes"),
    ({"fitted": np.arange(N) < 0.2 * N}, "too_few_nodes"),
    ({"residual_ratio": 30.0}, "model_mismatch"),
    ({"residual_ratio": 0.001}, "model_mismatch"),
])
def test_unusable_inputs_give_no_suggestion(kwargs, reason):
    advice = _advise(_prediction(**kwargs))
    assert advice.status == "unavailable"
    assert advice.reason == reason
    assert advice.suggested_winsize == 32


def test_a_few_wild_nodes_do_not_drive_the_suggestion():
    lap = np.zeros(N)
    lap[:10] = 1.0                                   # under 1 % of nodes, absurd curvature
    advice = _advise(_prediction(lap=lap))
    assert (advice.suggested_winsize, advice.suggested_order) == (64, "first")
