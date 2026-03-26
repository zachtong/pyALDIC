"""DIC parameter configuration and validation.

Port of MATLAB config/dicpara_default.m.
"""

from __future__ import annotations

from dataclasses import replace

from .data_structures import DICPara, GridxyROIRange


def dicpara_default(**overrides) -> DICPara:
    """Return a DICPara with all fields set to defaults, optionally overridden.

    This is the Python equivalent of MATLAB's ``dicpara_default(struct(...))``::

        para = dicpara_default(winsize=32, winstepsize=16)

    Args:
        **overrides: Any DICPara field names with new values.

    Returns:
        Validated DICPara instance.

    Raises:
        ValueError: If any parameter fails validation.
    """
    para = DICPara(**overrides)
    validate_dicpara(para)
    return para


def validate_dicpara(p: DICPara) -> None:
    """Check DICPara fields for common configuration errors.

    Port of MATLAB's validate_dicpara().  Raises ValueError on failure.
    """

    def _is_pow2(v: int) -> bool:
        return v > 0 and (v & (v - 1)) == 0

    # winstepsize: power of 2
    if not _is_pow2(p.winstepsize):
        raise ValueError(
            f"winstepsize={p.winstepsize} must be a positive power of 2."
        )

    # winsizeMin: power of 2, <= winstepsize
    if not _is_pow2(p.winsize_min):
        raise ValueError(
            f"winsize_min={p.winsize_min} must be a positive power of 2."
        )
    if p.winsize_min > p.winstepsize:
        raise ValueError(
            f"winsize_min={p.winsize_min} must be <= winstepsize={p.winstepsize}."
        )

    # winsize: positive even integer
    if p.winsize <= 0 or p.winsize % 2 != 0:
        raise ValueError(
            f"winsize={p.winsize} must be a positive even integer."
        )

    # mu: positive
    if p.mu <= 0:
        raise ValueError(f"mu must be positive (got {p.mu}).")

    # tol: (0, 1)
    if not (0 < p.tol < 1):
        raise ValueError(f"tol must be in (0, 1) (got {p.tol}).")

    # ADMM_maxIter: >= 1
    if p.admm_max_iter < 1:
        raise ValueError(
            f"admm_max_iter must be a positive integer >= 1 (got {p.admm_max_iter})."
        )

    # GaussPtOrder: 2 or 3
    if p.gauss_pt_order not in (2, 3):
        raise ValueError(
            f"gauss_pt_order must be 2 or 3 (got {p.gauss_pt_order})."
        )

    # referenceMode: enum
    if p.reference_mode not in ("incremental", "accumulative"):
        raise ValueError(
            f"reference_mode must be 'incremental' or 'accumulative' "
            f"(got '{p.reference_mode}')."
        )

    # cluster_no: non-negative integer
    if p.cluster_no < 0:
        raise ValueError(
            f"cluster_no must be a non-negative integer (got {p.cluster_no})."
        )

    # icgn_max_iter: positive integer
    if p.icgn_max_iter < 1:
        raise ValueError(
            f"icgn_max_iter must be a positive integer (got {p.icgn_max_iter})."
        )

    # size_of_fft_search_region: positive
    if p.size_of_fft_search_region <= 0:
        raise ValueError(
            f"size_of_fft_search_region must be positive "
            f"(got {p.size_of_fft_search_region})."
        )

    # strain_plane_fit_rad: positive
    if p.strain_plane_fit_rad <= 0:
        raise ValueError(
            f"strain_plane_fit_rad must be positive (got {p.strain_plane_fit_rad})."
        )

    # Smoothness: non-negative
    if p.disp_smoothness < 0:
        raise ValueError(
            f"disp_smoothness must be non-negative (got {p.disp_smoothness})."
        )
    if p.strain_smoothness < 0:
        raise ValueError(
            f"strain_smoothness must be non-negative (got {p.strain_smoothness})."
        )

    # alpha: non-negative
    if p.alpha < 0:
        raise ValueError(f"alpha must be non-negative (got {p.alpha}).")
