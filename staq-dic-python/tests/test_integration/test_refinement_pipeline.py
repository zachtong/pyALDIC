"""End-to-end tests for pipeline + adaptive refinement."""
import numpy as np
import pytest

from staq_dic.core.pipeline import run_aldic
from staq_dic.core.config import dicpara_default
from staq_dic.core.data_structures import GridxyROIRange
from staq_dic.mesh.refinement import RefinementPolicy
from staq_dic.mesh.criteria.mask_boundary import MaskBoundaryCriterion


def _make_speckle(h, w, seed=42):
    """Generate synthetic speckle image."""
    rng = np.random.default_rng(seed)
    img = rng.random((h, w))
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=1.5)


class TestPreSolveRefinement:
    def test_no_policy_uses_uniform_mesh(self):
        """Without refinement_policy, pipeline uses uniform mesh."""
        h, w = 128, 128
        ref = _make_speckle(h, w, seed=1)
        defm = _make_speckle(h, w, seed=1)  # zero displacement
        mask = np.ones((h, w), dtype=np.float64)
        para = dicpara_default(
            winstepsize=16, winsize=32, winsize_min=8,
            gridxy_roi_range=GridxyROIRange(gridx=(16, 112), gridy=(16, 112)),
        )
        result = run_aldic(para, [ref, defm], [mask, mask],
                          compute_strain=False, refinement_policy=None)
        # Uniform mesh: all elements same size
        mesh = result.result_fe_mesh_each_frame[0]
        corners = mesh.elements_fem[:, :4]
        sizes = (mesh.coordinates_fem[corners[:, 2], 0] -
                 mesh.coordinates_fem[corners[:, 0], 0])
        assert np.all(np.abs(sizes - sizes[0]) < 1e-6), "Should be uniform"

    def test_mask_criterion_refines_near_hole(self):
        """MaskBoundaryCriterion should produce non-uniform mesh."""
        h, w = 128, 128
        ref = _make_speckle(h, w, seed=1)
        defm = _make_speckle(h, w, seed=1)
        mask = np.ones((h, w), dtype=np.float64)
        yy, xx = np.mgrid[0:h, 0:w]
        mask[(xx - 64)**2 + (yy - 64)**2 < 25**2] = 0.0
        para = dicpara_default(
            winstepsize=16, winsize=32, winsize_min=4,
            gridxy_roi_range=GridxyROIRange(gridx=(16, 112), gridy=(16, 112)),
        )
        policy = RefinementPolicy(
            pre_solve=[MaskBoundaryCriterion(min_element_size=4)],
        )
        result = run_aldic(para, [ref, defm], [mask, mask],
                          compute_strain=False, refinement_policy=policy)
        mesh = result.result_fe_mesh_each_frame[0]
        corners = mesh.elements_fem[:, :4]
        sizes = (mesh.coordinates_fem[corners[:, 2], 0] -
                 mesh.coordinates_fem[corners[:, 0], 0])
        # Non-uniform: at least two different element sizes
        assert len(np.unique(np.round(sizes))) > 1, "Should be non-uniform near hole"

    def test_backward_compatible_no_policy(self):
        """Omitting refinement_policy produces same result as before."""
        h, w = 128, 128
        ref = _make_speckle(h, w, seed=1)
        defm = _make_speckle(h, w, seed=1)
        mask = np.ones((h, w), dtype=np.float64)
        para = dicpara_default(
            winstepsize=16, winsize=32, winsize_min=8,
            gridxy_roi_range=GridxyROIRange(gridx=(16, 112), gridy=(16, 112)),
        )
        r1 = run_aldic(para, [ref, defm], [mask, mask], compute_strain=False)
        r2 = run_aldic(para, [ref, defm], [mask, mask], compute_strain=False,
                      refinement_policy=None)
        np.testing.assert_allclose(r1.result_disp[0].U, r2.result_disp[0].U)
