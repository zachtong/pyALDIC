"""Initialize displacement from FFT cross-correlation results.

Port of MATLAB solver/init_disp.m (Jin Yang, Caltech).

Processes the raw FFT-based displacement guess: removes outliers via
median filtering and neighbor-based smoothing, then assembles into the
interleaved displacement vector format.

MATLAB/Python differences:
    - MATLAB ``inpaint_nans`` (spring model, method=4) →
      ``scipy.interpolate`` or a custom inpainting implementation.
    - MATLAB neighbor-averaging uses explicit 8-neighbor loops;
      Python should use ``scipy.ndimage.generic_filter`` or vectorized
      NumPy operations.
    - MATLAB arranges u, v as 2-D grids; Python maintains the same
      grid layout internally but converts to the 1-D interleaved format
      for output.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def init_disp(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    cc_max: NDArray[np.float64],
    x0: NDArray[np.float64],
    y0: NDArray[np.float64],
    method: int = 1,
) -> NDArray[np.float64]:
    """Initialize displacement field from FFT cross-correlation results.

    Cleans up the raw displacement grids by:
        1. Filling NaN values via inpainting (spring model).
        2. Removing outliers via 8-neighbor median/mean comparison.
        3. Re-inpainting after outlier removal.
        4. Assembling into the interleaved displacement vector.

    Args:
        u: Raw x-displacement grid (N, M), float64.  May contain NaN.
        v: Raw y-displacement grid (N, M), float64.  May contain NaN.
        cc_max: Cross-correlation coefficient grid (N, M), float64.
            Values near 1.0 indicate good matches.  Not used in the
            current implementation but kept for API compatibility.
        x0: 1-D grid x-coordinates (M,).
        y0: 1-D grid y-coordinates (N,).
        method: Smoothing method index.
            1 = median + neighbor averaging (default).
            Other values reserved for future methods.

    Returns:
        Interleaved displacement vector (2 * N * M,), layout
        [u0, v0, u1, v1, ...] where nodes are ordered column-major
        (Fortran-style, matching MATLAB's ``(:)`` linearization).

    Notes:
        - The outlier detection uses a weighted 8-neighbor average:
          ``w_cardinal = 1/8``, ``w_diagonal = 1/16``, then checks if
          the center value deviates by more than ``3 * neighbor_std``.
        - The ``method`` parameter corresponds to MATLAB's ``index``
          argument.  Method 1 is the only one actively used.
    """
    # TODO: Port from MATLAB: init_disp.m
    raise NotImplementedError("Port from MATLAB: init_disp.m")
