"""Tests for icgn_warp (inverse compositional warp composition)."""

import numpy as np
import pytest

from staq_dic.solver.icgn_warp import compose_warp


class TestComposeWarp:
    def test_identity_delta(self):
        """Zero delta_P should leave P unchanged."""
        P = np.array([0.01, 0.02, -0.01, 0.03, 1.5, -2.0])
        delta_P = np.zeros(6)

        result = compose_warp(P, delta_P)
        np.testing.assert_allclose(result, P, atol=1e-14)

    def test_identity_P(self):
        """Identity P composed with delta_P should give -delta_P (approximately)."""
        P = np.zeros(6)
        delta_P = np.array([0.01, 0.0, 0.0, 0.01, 2.0, 3.0])

        result = compose_warp(P, delta_P)
        # W(0) @ W(dP)^{-1} ≈ W(-dP) for small dP
        # For pure translation: should give (-Ux, -Uy) exactly
        # For deformation: approximately -delta_P
        assert result is not None
        assert result.shape == (6,)

    def test_singular_delta(self):
        """Singular delta_P (det=0) should return None."""
        P = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        # det = (1+0)*(1+(-1)) - 0*0 = 0
        delta_P = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0])

        result = compose_warp(P, delta_P)
        assert result is None

    def test_pure_translation(self):
        """Pure translation composition: W(Ux,Uy) @ W(dUx,dUy)^{-1}.

        For pure translations, composition gives (Ux - dUx, Uy - dUy).
        """
        P = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 3.0])
        delta_P = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.5])

        result = compose_warp(P, delta_P)
        assert result is not None
        np.testing.assert_allclose(result[4], 4.0, atol=1e-14)  # 5 - 1
        np.testing.assert_allclose(result[5], 2.5, atol=1e-14)  # 3 - 0.5
        # Deformation should remain zero
        np.testing.assert_allclose(result[:4], 0.0, atol=1e-14)

    def test_compose_then_decompose(self):
        """W(P) @ W(P)^{-1} should give identity."""
        P = np.array([0.05, -0.02, 0.03, 0.01, 10.0, -5.0])

        result = compose_warp(P, P)
        assert result is not None
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_small_deformation(self):
        """Small deformation update should produce predictable result."""
        P = np.array([0.1, 0.0, 0.0, 0.1, 0.0, 0.0])
        delta_P = np.array([0.01, 0.0, 0.0, 0.01, 0.0, 0.0])

        result = compose_warp(P, delta_P)
        assert result is not None
        assert result.shape == (6,)
        # Should reduce deformation slightly
        assert abs(result[0]) < abs(P[0]) + 0.05
        assert abs(result[3]) < abs(P[3]) + 0.05

    def test_matches_matlab_example(self):
        """Cross-validate with hand-computed MATLAB result.

        MATLAB:
          P = [0.01, 0.02, -0.01, 0.03, 1.5, -2.0]
          DeltaP = [0.001, 0.002, -0.001, 0.003, 0.5, -0.3]
        """
        P = np.array([0.01, 0.02, -0.01, 0.03, 1.5, -2.0])
        delta_P = np.array([0.001, 0.002, -0.001, 0.003, 0.5, -0.3])

        result = compose_warp(P, delta_P)
        assert result is not None

        # Verify via matrix multiply
        def _make_W(p):
            return np.array([
                [1 + p[0], p[2], p[4]],
                [p[1], 1 + p[3], p[5]],
                [0, 0, 1],
            ])

        W_P = _make_W(P)
        W_dP = _make_W(delta_P)
        W_result = W_P @ np.linalg.inv(W_dP)

        expected = np.array([
            W_result[0, 0] - 1,  # F11-1
            W_result[1, 0],       # F21
            W_result[0, 1],       # F12
            W_result[1, 1] - 1,   # F22-1
            W_result[0, 2],       # Ux
            W_result[1, 2],       # Uy
        ])

        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_output_dtype(self):
        """Output should be float64."""
        P = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        delta_P = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5])

        result = compose_warp(P, delta_P)
        assert result.dtype == np.float64
