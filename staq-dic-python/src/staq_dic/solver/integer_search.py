"""FFT-based integer-pixel displacement search.

Port of MATLAB solver/integer_search.m + integer_search_mg.m +
integer_search_kernel.m (Jin Yang, Caltech).  Redesigned for Python.

Computes the initial integer-pixel displacement guess by FFT-based
normalized cross-correlation between the reference and deformed images.
The MATLAB version uses three separate files; the Python version
consolidates them into a single module.

MATLAB/Python differences:
    - MATLAB ``normxcorr2`` → ``cv2.matchTemplate`` with
      ``cv2.TM_CCOEFF_NORMED`` or ``scipy.signal.fftconvolve``.
    - MATLAB multi-grid (mg) search with coarse-to-fine refinement
      is preserved but uses OpenCV for the correlation kernel.
    - MATLAB ``input()`` prompts for FFT search method are replaced
      by ``DICPara.init_fft_search_method``.
    - The 3-file MATLAB structure (integer_search → integer_search_mg →
      integer_search_kernel) is flattened into this single module with
      internal helper functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import DICPara


def integer_search(
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    para: DICPara,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    dict,
]:
    """Compute initial integer-pixel displacement via FFT cross-correlation.

    Performs normalized cross-correlation between reference and deformed
    images to find the best integer-pixel displacement at each grid point.

    Args:
        f_img: Reference image (H, W), float64, normalized [0, 1].
        g_img: Deformed image (H, W), float64, normalized [0, 1].
        para: DIC parameters.  Uses:
            - ``gridxy_roi_range``: ROI bounds for grid generation.
            - ``winsize``: Correlation window size.
            - ``winstepsize``: Grid spacing.
            - ``size_of_fft_search_region``: Search region in multiples
              of ``winstepsize``.
            - ``init_fft_search_method``: Search algorithm variant (1-3).

    Returns:
        A tuple ``(x0, y0, u, v, info)`` where:
            - ``x0``: 1-D grid x-coordinates (M,).
            - ``y0``: 1-D grid y-coordinates (N,).
            - ``u``: x-displacement grid (N, M), integer pixel values.
            - ``v``: y-displacement grid (N, M), integer pixel values.
            - ``info``: Dict with diagnostic information:
              ``{'cc_max': NDArray, 'search_region_warning': bool}``.

    Notes:
        - The grid is generated from ``gridxy_roi_range`` with spacing
          ``winstepsize``.  Grid points too close to image boundaries
          (within ``winsize/2``) are excluded.
        - ``init_fft_search_method``:
            1 = Full search at each grid point (most accurate, slowest).
            2 = Multi-grid coarse-to-fine (default, good balance).
            3 = Single global FFT + local refinement (fastest, may miss
                large deformations).
        - The ``search_region_warning`` flag in ``info`` is ``True`` if
          the FFT search region was auto-scaled due to being too large
          relative to the image size.
    """
    # TODO: Port from MATLAB: integer_search.m + integer_search_mg.m + integer_search_kernel.m
    raise NotImplementedError(
        "Port from MATLAB: integer_search.m, integer_search_mg.m, integer_search_kernel.m"
    )
