"""Core data structures for the AL-DIC pipeline.

All data structures use frozen dataclasses for immutability.
Use dataclasses.replace() for creating modified copies.

IMPORTANT — Index conventions:
    - All node/element indices are 0-based (unlike MATLAB's 1-based).
    - Image coordinates follow NumPy convention: row=y, col=x (H x W).
    - Displacement vectors are interleaved: U = [u0, v0, u1, v1, ..., uN, vN].
    - Deformation gradient vectors: F = [F11_0, F21_0, F12_0, F22_0, ...].
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..solver.seed_propagation import SeedSet


# ---------------------------------------------------------------------------
# Frame schedule — generalized reference-frame pairing for multi-frame DIC
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSchedule:
    """Generalized frame-pairing schedule for multi-frame DIC tracking.

    Each deformed frame ``i+1`` (1-indexed) is tracked against reference
    frame ``ref_indices[i]`` (0-indexed).  The DAG constraint ensures
    ``ref_indices[i] in [0, i]`` — no frame can reference a future frame.

    Special cases:
        - Accumulative: ``ref_indices = (0, 0, ..., 0)``
        - Incremental:  ``ref_indices = (0, 1, 2, ..., n-2)``

    Attributes:
        ref_indices: Tuple of length ``n_frames - 1``.
            ``ref_indices[i]`` is the 0-based reference frame index
            for deformed frame ``i + 1``.
    """

    ref_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate DAG constraint: ref_indices[i] must be in [0, i]."""
        for i, ref in enumerate(self.ref_indices):
            if not isinstance(ref, (int, np.integer)):
                raise TypeError(
                    f"ref_indices[{i}] must be int (got {type(ref).__name__})"
                )
            if ref < 0:
                raise ValueError(
                    f"ref_indices[{i}]={ref} is negative"
                )
            if ref > i:
                raise ValueError(
                    f"ref_indices[{i}]={ref} references future frame "
                    f"(must be <= {i})"
                )

    @classmethod
    def from_mode(cls, mode: str, n_frames: int) -> FrameSchedule:
        """Create a FrameSchedule from a legacy reference mode string.

        Args:
            mode: 'accumulative' or 'incremental'.
            n_frames: Total number of frames (including reference frame 0).

        Returns:
            FrameSchedule instance.

        Raises:
            ValueError: If mode is unknown or n_frames < 2.
        """
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2 (got {n_frames})")
        n_pairs = n_frames - 1
        if mode == "accumulative":
            return cls(ref_indices=tuple(0 for _ in range(n_pairs)))
        if mode == "incremental":
            return cls(ref_indices=tuple(range(n_pairs)))
        raise ValueError(
            f"Unknown reference mode '{mode}'. "
            f"Use 'accumulative' or 'incremental'."
        )

    def parent(self, frame: int) -> int:
        """Return the reference frame index for a given deformed frame.

        Args:
            frame: Deformed frame index (1-based, so frame >= 1).

        Returns:
            0-based reference frame index.
        """
        if frame < 1 or frame > len(self.ref_indices):
            raise IndexError(
                f"frame={frame} out of range [1, {len(self.ref_indices)}]"
            )
        return self.ref_indices[frame - 1]

    def path_to_root(self, frame: int) -> list[int]:
        """Trace the reference chain from *frame* back to frame 0.

        Returns a list of frame indices starting from *frame* and ending
        at 0.  For accumulative mode this is always ``[frame, 0]``.
        For incremental mode it is ``[frame, frame-1, ..., 1, 0]``.

        Args:
            frame: Starting deformed frame (>= 1).

        Returns:
            List of frame indices from *frame* to 0 (inclusive).
        """
        path = [frame]
        current = frame
        while current > 0:
            current = self.parent(current)
            path.append(current)
        return path

    def children(self, frame: int) -> list[int]:
        """Return all deformed frames that directly reference *frame*.

        Args:
            frame: Reference frame index (0-based).

        Returns:
            Sorted list of deformed frame indices (1-based).
        """
        return sorted(
            i + 1
            for i, ref in enumerate(self.ref_indices)
            if ref == frame
        )

    @classmethod
    def from_every_n(cls, n: int, n_frames: int) -> FrameSchedule:
        """Create a schedule where every n-th frame is a new reference.

        Reference frames are placed at 0, n, 2n, ... (as long as < n_frames - 1).
        Each deformed frame references the nearest preceding reference frame.

        Args:
            n: Reference frame interval. n=1 is equivalent to incremental mode.
            n_frames: Total number of frames (including reference frame 0).

        Returns:
            FrameSchedule instance.

        Raises:
            ValueError: If n < 1 or n_frames < 2.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1 (got {n})")
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2 (got {n_frames})")
        refs: list[int] = []
        for deformed in range(1, n_frames):
            # Nearest preceding ref: largest multiple of n that is < deformed
            ref = (deformed - 1) // n * n
            refs.append(ref)
        return cls(ref_indices=tuple(refs))

    @classmethod
    def from_custom(cls, custom_refs: list[int], n_frames: int) -> FrameSchedule:
        """Create a schedule from user-specified reference frame indices.

        Frame 0 is always included as a reference even if not in the list.
        The last frame (n_frames - 1) cannot be a reference frame.
        Each deformed frame references the nearest preceding reference frame.

        Args:
            custom_refs: List of 0-based reference frame indices.
            n_frames: Total number of frames (including reference frame 0).

        Returns:
            FrameSchedule instance.

        Raises:
            ValueError: If n_frames < 2, any ref is negative, out of range,
                or equal to the last frame.
        """
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2 (got {n_frames})")
        for ref in custom_refs:
            if ref < 0:
                raise ValueError(
                    f"Reference frame index {ref} is negative"
                )
            if ref >= n_frames:
                raise ValueError(
                    f"Reference frame index {ref} is out of range "
                    f"[0, {n_frames - 1})"
                )
            if ref == n_frames - 1:
                raise ValueError(
                    f"Reference frame index {ref} is the last frame; "
                    f"the last frame cannot be a reference"
                )
        # Ensure frame 0 is always included, then sort
        ref_set = sorted(set(custom_refs) | {0})
        refs: list[int] = []
        for deformed in range(1, n_frames):
            # Find the largest ref that is < deformed
            best_ref = 0
            for r in ref_set:
                if r < deformed:
                    best_ref = r
                else:
                    break
            refs.append(best_ref)
        return cls(ref_indices=tuple(refs))

    @property
    def ref_frame_set(self) -> set[int]:
        """Return the set of unique reference frame indices.

        Always includes frame 0 (implicit root of the DAG).
        """
        return set(self.ref_indices) | {0}

    def __len__(self) -> int:
        """Number of frame pairs (= n_frames - 1)."""
        return len(self.ref_indices)


@dataclass(frozen=True)
class GridxyROIRange:
    """Region of interest in pixel coordinates.

    Attributes:
        gridx: (xmin, xmax) — column pixel range.
        gridy: (ymin, ymax) — row pixel range.
    """

    gridx: tuple[int, int] = (0, 0)
    gridy: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class DICPara:
    """Immutable DIC parameter container.

    Mirrors every field in MATLAB's dicpara_default.m.
    Use ``dataclasses.replace(para, field=value)`` for updates.

    All pixel-unit parameters use Python's 0-based image coordinates.
    """

    # --- 1. Image loading & ROI ---
    load_img_method: int = 0
    gridxy_roi_range: GridxyROIRange = field(default_factory=GridxyROIRange)
    img_folder: str = ""
    mask_folder: str = ""
    show_plots: bool = True
    img_size: tuple[int, int] = (0, 0)  # (height, width) in Python convention
    dim: int = 2
    img_bit_depth: int = 8
    img_ref_mask: NDArray[np.float64] | None = None
    use_masks: bool = False

    # --- 2. Mesh & subset ---
    winsize: int = 40
    winstepsize: int = 16
    winsize_min: int = 8
    mesh_type: Literal["uniform", "quadtree"] = "uniform"
    winsize_list: NDArray[np.float64] | None = None

    # --- 3. Image sequence ---
    img_seq_inc_unit: int = 1
    img_seq_inc_roi_update: int = 0

    # --- 4. FFT initial guess ---
    new_fft_search: int = 1
    init_fft_search_method: int = 1
    size_of_fft_search_region: int = 20
    init_guess_mode: Literal[
        "auto", "fft", "previous", "seed_propagation",
    ] = "auto"
    # When init_guess_mode == "seed_propagation": user-placed seed set.
    # Seeds are automatically warped to new ref coords on ref switches.
    seed_set: SeedSet | None = None
    # Periodic FFT reset: force a fresh FFT every N frames (0 = disabled).
    # Limits warm-start error propagation in accumulative mode.
    fft_reset_interval: int = 0
    # Whether to auto-enlarge the search region when FFT peaks are clipped.
    fft_auto_expand_search: bool = True
    discontinuity_threshold_cc: float = 0.85
    k_nearest_neighbors: int = 3

    # --- 5. POD-GPR ---
    use_pod_gpr: bool = False
    pod_n_time: int = 5
    pod_n_basis: int = 3
    pod_start_frame: int = 7

    # --- 6. IC-GN solver ---
    tol: float = 1e-2
    cluster_no: int = 0
    icgn_max_iter: int = 100
    min_valid_ratio: float = 0.5  # min fraction of subset pixels in mask

    # --- 7. ADMM ---
    mu: float = 1e-3
    beta_range: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    beta: float | None = None
    admm_max_iter: int = 3
    admm_tol: float = 1e-2
    gauss_pt_order: int = 2
    alpha: float = 0.0
    outlier_sigma_factor: float = 0.25
    outlier_min_threshold: int = 10
    use_global_step: bool = True

    # --- 8. Subproblem 2 ---
    subpb2_fd_or_fem: int = 2

    # --- 9. Smoothing ---
    disp_filter_size: int = 0
    disp_filter_std: float = 0.0
    strain_filter_size: int = 0
    strain_filter_std: float = 0.0
    disp_smoothness: float = 5e-4
    strain_smoothness: float = 1e-5
    skip_extra_smoothing: int = 1
    smoothness: float = 0.0

    # --- 10. Strain ---
    method_to_compute_strain: int = 2
    strain_plane_fit_rad: float = 20.0
    strain_type: int = 0

    # --- 11. Stress ---
    material_model: int = 1
    youngs_modulus: float | None = None
    poissons_ratio: float | None = None

    # --- 12. Visualization ---
    um2px: float = 1.0
    image2plot_results: int = 1
    method_to_save_fig: int = 1
    orig_dic_img_transparency: float = 1.0
    output_file_path: str | None = None
    reference_mode: Literal["incremental", "accumulative"] = "incremental"
    frame_schedule: FrameSchedule | None = None


@dataclass
class DICMesh:
    """Quadtree FE mesh for DIC computation.

    All indices are 0-based.

    Attributes:
        coordinates_fem: Node coordinates (n_nodes, 2) — col 0 = x, col 1 = y.
        elements_fem: Element connectivity (n_elements, 8) — Q8 nodes, 0-based.
        irregular: Hanging-node constraints (n_irregular, 3), 0-based.
        mark_coord_hole_edge: Indices of nodes on hole/edge boundaries.
        dirichlet: Node indices with fixed (Dirichlet) displacement (n_dir,).
        neumann: Boundary edges with normals (n_neu, 4) — [node1, node2, nx, ny].
        coordinates_fem_world: World-space node coordinates (n_nodes, 2).
        x0: Initial grid x-coordinates (1D array).
        y0: Initial grid y-coordinates (1D array).
        element_min_size: Minimum element size for quadtree refinement.
    """

    coordinates_fem: NDArray[np.float64]
    elements_fem: NDArray[np.int64]
    irregular: NDArray[np.int64] = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    mark_coord_hole_edge: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    dirichlet: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    neumann: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 4), dtype=np.float64))
    coordinates_fem_world: NDArray[np.float64] | None = None
    x0: NDArray[np.float64] | None = None
    y0: NDArray[np.float64] | None = None
    element_min_size: int = 8


@dataclass(frozen=True)
class ImageGradients:
    """Pre-computed image gradients for the reference image.

    Attributes:
        df_dx: x-derivative of reference image (H, W).
        df_dy: y-derivative of reference image (H, W).
        img_ref_mask: Binary mask for reference image (H, W).
        img_size: (height, width).
    """

    df_dx: NDArray[np.float64]
    df_dy: NDArray[np.float64]
    img_ref_mask: NDArray[np.float64]
    img_size: tuple[int, int]


@dataclass(frozen=True)
class FrameResult:
    """Displacement result for a single frame pair.

    Attributes:
        U: Incremental displacement vector (2*n_nodes,) interleaved [u0,v0,...].
        U_accum: Cumulative displacement from frame 1, same layout.
        F: Deformation gradient vector (4*n_nodes,) interleaved [F11,F21,F12,F22,...].
        bad_pt_num: Number of bad points detected per ADMM step.
    """

    U: NDArray[np.float64]
    U_accum: NDArray[np.float64] | None = None
    F: NDArray[np.float64] | None = None
    bad_pt_num: NDArray[np.int64] | None = None
    ref_frame: int = 0


@dataclass(frozen=True)
class StrainResult:
    """Strain computation result for a single frame.

    All arrays have shape (n_nodes,).

    Attributes:
        disp_u: World-space x-displacement.
        disp_v: World-space y-displacement.
        dudx, dvdx, dudy, dvdy: Displacement gradient components.
        strain_exx, strain_exy, strain_eyy: Strain tensor components.
        strain_principal_max, strain_principal_min: Principal strains.
        strain_maxshear: Maximum shear strain.
        strain_von_mises: Von Mises equivalent strain.
    """

    disp_u: NDArray[np.float64]
    disp_v: NDArray[np.float64]
    dudx: NDArray[np.float64] | None = None
    dvdx: NDArray[np.float64] | None = None
    dudy: NDArray[np.float64] | None = None
    dvdy: NDArray[np.float64] | None = None
    strain_exx: NDArray[np.float64] | None = None
    strain_exy: NDArray[np.float64] | None = None
    strain_eyy: NDArray[np.float64] | None = None
    strain_principal_max: NDArray[np.float64] | None = None
    strain_principal_min: NDArray[np.float64] | None = None
    strain_maxshear: NDArray[np.float64] | None = None
    strain_von_mises: NDArray[np.float64] | None = None
    strain_rotation: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated output from the full run_aldic pipeline.

    Attributes:
        dic_para: Final DICPara (may differ from input due to auto-scaling).
        dic_mesh: Final mesh used for computation.
        result_disp: Per-frame displacement results.
        result_def_grad: Per-frame deformation gradient results.
        result_strain: Per-frame strain results (empty if strain not computed).
        result_fe_mesh_each_frame: Per-frame FE mesh snapshots.
        ref_switch_frames: Displacement-frame indices where the active
            reference frame changed. Empty on pure accumulative runs or
            single-ref incremental runs. Produced by every init_guess_mode,
            not just seed_propagation; consumers render a marker in the
            frame navigator.
        reseed_events: Auto-reseed log from seed_propagation mode. Each
            entry records a frame where warp_seeds_to_new_ref failed and
            the pipeline fell back to auto-placing fresh seeds on the
            new reference frame. Empty on non-seed-prop runs and on
            seed-prop runs where every warp succeeded.
    """

    dic_para: DICPara
    dic_mesh: DICMesh
    result_disp: list[FrameResult]
    result_def_grad: list[FrameResult]
    result_strain: list[StrainResult]
    result_fe_mesh_each_frame: list[DICMesh]
    frame_schedule: FrameSchedule | None = None
    ref_switch_frames: tuple[int, ...] = ()
    reseed_events: tuple = ()  # tuple[ReseedEvent, ...] — avoid import cycle


# ---------------------------------------------------------------------------
# Utility: structured views into interleaved vectors
# ---------------------------------------------------------------------------

def split_uv(U: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split interleaved displacement vector into u and v components.

    Args:
        U: (2*N,) interleaved as [u0, v0, u1, v1, ...].

    Returns:
        (u, v) each of shape (N,).
    """
    return U[0::2].copy(), U[1::2].copy()


def merge_uv(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Merge u and v arrays into a single interleaved displacement vector.

    Args:
        u: (N,) x-displacements.
        v: (N,) y-displacements.

    Returns:
        (2*N,) interleaved as [u0, v0, u1, v1, ...].
    """
    U = np.empty(2 * len(u), dtype=np.float64)
    U[0::2] = u
    U[1::2] = v
    return U


def split_F(F: NDArray[np.float64]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split interleaved deformation gradient into components.

    Args:
        F: (4*N,) interleaved as [F11_0, F21_0, F12_0, F22_0, ...].

    Returns:
        (F11, F21, F12, F22) each of shape (N,).
    """
    return F[0::4].copy(), F[1::4].copy(), F[2::4].copy(), F[3::4].copy()


def merge_F(
    F11: NDArray[np.float64],
    F21: NDArray[np.float64],
    F12: NDArray[np.float64],
    F22: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Merge deformation gradient components into interleaved vector.

    Returns:
        (4*N,) interleaved as [F11_0, F21_0, F12_0, F22_0, ...].
    """
    n = len(F11)
    F = np.empty(4 * n, dtype=np.float64)
    F[0::4] = F11
    F[1::4] = F21
    F[2::4] = F12
    F[3::4] = F22
    return F
