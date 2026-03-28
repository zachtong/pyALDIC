"""Tests for PosteriorErrorCriterion."""
import numpy as np
import pytest

from staq_dic.mesh.criteria.posterior_error import PosteriorErrorCriterion
from staq_dic.mesh.refinement import RefinementContext, RefinementCriterion
from staq_dic.mesh.mesh_setup import mesh_setup
from staq_dic.core.data_structures import DICPara


@pytest.fixture
def mesh_4x4():
    para = DICPara(winstepsize=16, winsize=32, winsize_min=4)
    x0 = np.arange(16, 64, 16, dtype=np.float64)
    y0 = np.arange(16, 64, 16, dtype=np.float64)
    return mesh_setup(x0, y0, para)


class TestPosteriorErrorCriterion:
    def test_implements_protocol(self):
        criterion = PosteriorErrorCriterion()
        assert isinstance(criterion, RefinementCriterion)

    def test_no_data_no_marks(self, mesh_4x4):
        """Without U/conv_iterations, marks nothing."""
        ctx = RefinementContext(mesh=mesh_4x4)
        criterion = PosteriorErrorCriterion()
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_uniform_convergence_no_marks(self, mesh_4x4):
        """All nodes converge equally -> no outliers -> no marks."""
        n_nodes = mesh_4x4.coordinates_fem.shape[0]
        conv = np.full(n_nodes, 5, dtype=np.int32)
        ctx = RefinementContext(mesh=mesh_4x4, conv_iterations=conv)
        criterion = PosteriorErrorCriterion(sigma_factor=1.0)
        marks = criterion.mark(ctx)
        assert not marks.any()

    def test_high_iteration_nodes_get_marked(self, mesh_4x4):
        """Nodes with high IC-GN iterations -> their elements get marked."""
        n_nodes = mesh_4x4.coordinates_fem.shape[0]
        conv = np.full(n_nodes, 5, dtype=np.int32)
        conv[0] = 50
        conv[1] = 45
        ctx = RefinementContext(mesh=mesh_4x4, conv_iterations=conv)
        criterion = PosteriorErrorCriterion(sigma_factor=1.0)
        marks = criterion.mark(ctx)
        assert marks.any(), "Should mark elements containing slow-converging nodes"

    def test_sigma_factor_controls_sensitivity(self, mesh_4x4):
        """Higher sigma_factor -> fewer marks (less sensitive)."""
        n_nodes = mesh_4x4.coordinates_fem.shape[0]
        conv = np.full(n_nodes, 5, dtype=np.int32)
        conv[0] = 20
        ctx = RefinementContext(mesh=mesh_4x4, conv_iterations=conv)

        strict = PosteriorErrorCriterion(sigma_factor=0.5)
        lenient = PosteriorErrorCriterion(sigma_factor=3.0)
        marks_strict = strict.mark(ctx)
        marks_lenient = lenient.mark(ctx)

        assert marks_strict.sum() >= marks_lenient.sum()

    def test_frozen_dataclass(self):
        """PosteriorErrorCriterion should be immutable (frozen)."""
        criterion = PosteriorErrorCriterion(sigma_factor=2.0)
        with pytest.raises(AttributeError):
            criterion.sigma_factor = 1.0  # type: ignore[misc]

    def test_default_values(self):
        """Default metric, sigma_factor, and min_element_size."""
        criterion = PosteriorErrorCriterion()
        assert criterion.metric == "conv_iterations"
        assert criterion.sigma_factor == 1.0
        assert criterion.min_element_size == 4

    def test_marks_dtype_and_shape(self, mesh_4x4):
        """Output has correct dtype and shape."""
        ctx = RefinementContext(mesh=mesh_4x4)
        criterion = PosteriorErrorCriterion()
        marks = criterion.mark(ctx)
        assert marks.dtype == np.bool_
        assert marks.shape == (mesh_4x4.elements_fem.shape[0],)
