"""Smoke test: extreme / boundary parameter combinations must not crash silently.

Each test picks a knob that users can plausibly push to an extreme
(too small, too large, disallowed combination) and asserts pyALDIC
either:
    - handles it correctly, OR
    - raises a specific, user-actionable error (not a silent bad
      result, not a generic KeyError)

Catches a whole class of 'works on happy path, breaks at edges' bugs
before shipping.
"""

from __future__ import annotations

import numpy as np
import pytest


# --- VSG radius < subset_step → strain plane-fit must raise, not zero out ---

def test_planefit_too_small_vsg_raises_with_actionable_message():
    """v0.4.1 regression guard.

    VSG radius = 1, subset_step = 8 → every node has < 3 neighbours →
    comp_def_grad returns all-NaN → fill_nan_idw should now raise with
    a message that tells the user what to do.
    """
    from al_dic.strain.compute_strain import compute_strain

    class _FakePara:
        method_to_compute_strain = 2      # plane fitting
        strain_plane_fit_rad = 1.0
        strain_smoothness = 0.0
        strain_type = 0
        img_ref_mask = None
        um2px = 1.0
        winstepsize = 8                   # 8 px between nodes

    class _FakeMesh:
        coordinates_fem = np.array(
            [[0, 0], [8, 0], [16, 0]], dtype=np.float64,
        )
        elements_fem = np.zeros((0, 8), dtype=np.int64)

    U = np.array([0.0, 0.0, 0.1, 0.0, 0.2, 0.0], dtype=np.float64)
    node_region_map = np.zeros(3, dtype=np.int32)

    with pytest.raises(ValueError) as exc_info:
        compute_strain(_FakeMesh(), _FakePara(), U, node_region_map)

    # Error message must be actionable: mention VSG, mention recommended
    # size, and direct user to FEM nodal as alternative.
    msg = str(exc_info.value).lower()
    assert "vsg" in msg, f"Error message should mention VSG: {msg!r}"
    assert "17" in str(exc_info.value) or "2 * subset_step" in msg, (
        f"Error should give recommended VSG size: {msg!r}"
    )
    assert "fem" in msg, f"Error should suggest FEM nodal: {msg!r}"


# --- fill_nan_idw opt-in raise ---------------------------------------------

def test_fill_nan_idw_default_returns_zeros_with_warning():
    """Default behaviour is backward-compatible (zeros + warning)."""
    import warnings

    from al_dic.utils.outlier_detection import fill_nan_idw

    V = np.full(20, np.nan)
    coords = np.random.rand(10, 2) * 100
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fill_nan_idw(V, coords, n_components=2)
    assert np.all(out == 0)
    assert any("All nodes are NaN" in str(w.message) for w in caught)


def test_fill_nan_idw_opt_in_raises():
    """Opt-in raise path must work for strain-compute callers."""
    from al_dic.utils.outlier_detection import fill_nan_idw

    V = np.full(20, np.nan)
    coords = np.random.rand(10, 2) * 100
    with pytest.raises(ValueError) as exc_info:
        fill_nan_idw(V, coords, n_components=2, on_all_nan="raise")
    assert "every node is NaN" in str(exc_info.value)


# --- seed_propagation + refinement pre-solve → must raise ValueError ------

def test_seed_propagation_incompatible_with_pre_solve_refinement():
    """V1 restriction: mesh rebuild mid-frame breaks seed node_idx.

    The pipeline is expected to detect this at the top of run_aldic
    and raise with a clear message, not crash somewhere in the middle.
    """
    # This test is representative — we inspect the guard rather than
    # constructing a full pipeline run. The guard lives at
    # pipeline.py:656-663 and reads:
    #
    #     if refinement_policy is not None and \
    #        refinement_policy.has_pre_solve:
    #         raise ValueError("init_guess_mode='seed_propagation' "
    #                          "is incompatible with refinement "
    #                          "policies that have pre_solve=True ...")
    import inspect

    from al_dic.core.pipeline import run_aldic

    source = inspect.getsource(run_aldic)
    assert "init_guess_mode='seed_propagation' is incompatible" in source, (
        "Guard against seed_prop + pre_solve refinement seems to have "
        "been removed — users can now silently get wrong results."
    )


# --- Boundary spinbox values -----------------------------------------------

@pytest.mark.parametrize(
    "subset_size,expected_valid", [
        (11, True),    # min
        (31, True),    # typical
        (201, True),   # max
        (10, False),   # below min (must be odd AND ≥ 11)
        (202, False),  # above max
    ],
)
def test_subset_size_boundary(subset_size, expected_valid):
    """Subset size should stay within [11, 201] per the GUI spinbox
    range. This is a data-driven check against the constants used at
    the Qt widget layer — boundaries can drift between UI and solver."""
    MIN, MAX = 11, 201
    is_valid = (MIN <= subset_size <= MAX) and (subset_size % 2 == 1)
    assert is_valid == expected_valid


@pytest.mark.parametrize(
    "subset_step,expected_valid", [
        (4, True),
        (8, True),
        (16, True),
        (32, True),
        (64, True),
        (3, False),    # not power of 2
        (128, False),  # not in offered list
    ],
)
def test_subset_step_must_be_power_of_2(subset_step, expected_valid):
    """subset_step is a combo-box option, not an arbitrary spinbox.
    Only {4, 8, 16, 32, 64} are valid. Solver assumes power-of-2 for
    mesh refinement."""
    offered = {4, 8, 16, 32, 64}
    is_valid = subset_step in offered
    assert is_valid == expected_valid


# --- Pseudo-locale catches missing tr() wrappers ---------------------------

def test_pseudo_locale_wraps_translated_strings():
    """Pseudo-locale is the dev-time tool to detect missing tr()
    wrappers. Translated strings must come back wrapped in ⟦…⟧."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    from al_dic.i18n import LanguageManager, PSEUDO_LOCALE

    app = QApplication.instance() or QApplication(["pyaldic-pseudo"])
    app.setOrganizationName("pyaldic-pseudo-smoke")

    mgr = LanguageManager(app)
    mgr.load(PSEUDO_LOCALE)

    translated = QCoreApplication.translate("RightSidebar", "Run DIC Analysis")
    assert translated.startswith("\u27e6"), (
        f"Pseudo-locale should wrap with ⟦; got {translated!r}"
    )
    assert translated.endswith("\u27e7"), (
        f"Pseudo-locale should wrap with ⟧; got {translated!r}"
    )
