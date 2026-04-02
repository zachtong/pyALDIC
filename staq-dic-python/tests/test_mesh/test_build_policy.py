"""Tests for build_refinement_policy factory function."""
import numpy as np

from staq_dic.mesh.refinement import RefinementPolicy, build_refinement_policy
from staq_dic.mesh.criteria.mask_boundary import MaskBoundaryCriterion
from staq_dic.mesh.criteria.brush_region import BrushRegionCriterion
from staq_dic.mesh.criteria.roi_edge import ROIEdgeCriterion


class TestBuildRefinementPolicy:
    def test_no_options_returns_none(self):
        """No refinement requested -> returns None."""
        policy = build_refinement_policy()
        assert policy is None

    def test_inner_boundary_only(self):
        """refine_inner_boundary=True -> MaskBoundaryCriterion."""
        policy = build_refinement_policy(refine_inner_boundary=True)
        assert policy is not None
        assert len(policy.pre_solve) == 1
        assert isinstance(policy.pre_solve[0], MaskBoundaryCriterion)

    def test_outer_boundary_only(self):
        """refine_outer_boundary=True -> ROIEdgeCriterion."""
        policy = build_refinement_policy(refine_outer_boundary=True)
        assert policy is not None
        assert len(policy.pre_solve) == 1
        assert isinstance(policy.pre_solve[0], ROIEdgeCriterion)

    def test_outer_boundary_half_win(self):
        """half_win is propagated to ROIEdgeCriterion."""
        policy = build_refinement_policy(
            refine_outer_boundary=True, half_win=24,
        )
        assert policy is not None
        crit = policy.pre_solve[0]
        assert isinstance(crit, ROIEdgeCriterion)
        assert crit.half_win == 24

    def test_brush_only(self):
        """refinement_mask provided -> BrushRegionCriterion."""
        rmask = np.ones((64, 64), dtype=np.float64)
        policy = build_refinement_policy(refinement_mask=rmask)
        assert policy is not None
        assert len(policy.pre_solve) == 1
        assert isinstance(policy.pre_solve[0], BrushRegionCriterion)

    def test_all_three_combined(self):
        """All three options -> policy with three criteria."""
        rmask = np.ones((64, 64), dtype=np.float64)
        policy = build_refinement_policy(
            refine_inner_boundary=True,
            refine_outer_boundary=True,
            refinement_mask=rmask,
        )
        assert policy is not None
        assert len(policy.pre_solve) == 3
        types = {type(c) for c in policy.pre_solve}
        assert MaskBoundaryCriterion in types
        assert ROIEdgeCriterion in types
        assert BrushRegionCriterion in types

    def test_inner_plus_brush(self):
        """Inner boundary + brush -> two criteria."""
        rmask = np.ones((64, 64), dtype=np.float64)
        policy = build_refinement_policy(
            refine_inner_boundary=True, refinement_mask=rmask,
        )
        assert policy is not None
        assert len(policy.pre_solve) == 2
        types = {type(c) for c in policy.pre_solve}
        assert MaskBoundaryCriterion in types
        assert BrushRegionCriterion in types

    def test_min_element_size_propagated(self):
        """min_element_size should be set on all criteria."""
        rmask = np.ones((64, 64), dtype=np.float64)
        policy = build_refinement_policy(
            refine_inner_boundary=True,
            refine_outer_boundary=True,
            refinement_mask=rmask,
            min_element_size=6,
        )
        assert policy is not None
        for crit in policy.pre_solve:
            assert crit.min_element_size == 6

    def test_empty_refinement_mask_returns_policy(self):
        """All-zero refinement mask is still a valid mask (returns policy)."""
        rmask = np.zeros((64, 64), dtype=np.float64)
        policy = build_refinement_policy(refinement_mask=rmask)
        assert policy is not None

    def test_returns_refinement_policy_type(self):
        """Return type is RefinementPolicy."""
        policy = build_refinement_policy(refine_inner_boundary=True)
        assert isinstance(policy, RefinementPolicy)
