"""Pipeline glue for seed_propagation init-guess mode.

Encapsulates the per-run mutable state and per-frame logic for
init_guess_mode='seed_propagation', keeping run_aldic's main loop body
slim. Public functions:

- ``SeedPropagationState``: mutable state carried across frames in one
  run (adjacency cache, previous-frame coords/U for ref-switch warp).
- ``build_grid_for_roi``: derive x0/y0 grids directly from DICPara ROI
  without running an FFT search (mesh_setup's input when no FFT path).
- ``compute_seed_prop_init_guess``: run propagate_from_seeds for one
  frame and return the interleaved U0 vector.
- ``capture_for_next_frame``: quality-gate the frame's converged U and
  save it for a future ref-switch warp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..utils.region_analysis import precompute_node_regions
from .integer_search import _centered_arange
from .local_icgn import local_icgn_precompute

logger = logging.getLogger(__name__)
from .seed_propagation import (
    SeedQualityError,
    SeedSet,
    SeedWarpFailure,
    build_node_adjacency,
    propagate_from_seeds,
    warp_seeds_to_new_ref,
)

if TYPE_CHECKING:
    from ..core.data_structures import DICMesh, DICPara, ImageGradients


@dataclass(frozen=True)
class ReseedEvent:
    """Record of an auto-reseed triggered by a ref-switch warp failure.

    Attributes:
        frame_idx: Global frame index where the ref-switch occurred.
        ref_idx: Reference-frame index that the switch was going to.
        reason: Human-readable message describing why the warp failed
            (e.g. "All seeds failed to warp to the new reference").
        n_new_seeds: Number of seeds auto-placed on the new ref frame.
    """

    frame_idx: int
    ref_idx: int
    reason: str
    n_new_seeds: int


@dataclass
class SeedPropagationState:
    """Per-run state for seed_propagation mode.

    Attributes:
        current_seeds: Live seed set for the next propagate call.
            Updated in-place when a ref switch triggers warping.
        prev_coords_fem: Coordinates of the mesh used in the most recent
            successful frame (needed for warping seeds to a new ref).
            None until the first successful frame completes.
        prev_U_2d: (n_nodes, 2) converged displacements from the most
            recent successful frame, in its mesh's node ordering.
        adjacency_cache: Cache of node_adjacency results keyed by
            ``id(elements_fem)`` — the same elements_fem array means
            the same adjacency; no need to rebuild per frame.
        reseed_events: Auto-reseed log for the current run. Populated
            each time a ref-switch warp fails and the pipeline falls
            back to ``auto_place_seeds_on_mesh``.
    """

    current_seeds: SeedSet
    prev_coords_fem: NDArray[np.float64] | None = None
    prev_U_2d: NDArray[np.float64] | None = None
    adjacency_cache: dict[int, list[set[int]]] = field(default_factory=dict)
    reseed_events: list[ReseedEvent] = field(default_factory=list)

    @classmethod
    def from_para(cls, para: DICPara) -> SeedPropagationState:
        """Factory: build state from a DICPara's seed_set."""
        if para.seed_set is None or len(para.seed_set.seeds) == 0:
            raise ValueError(
                "SeedPropagationState.from_para requires para.seed_set "
                "with at least one seed (validate_dicpara should catch "
                "this earlier)."
            )
        return cls(current_seeds=para.seed_set)


def build_grid_for_roi(
    para: DICPara,
    img_h: int,
    img_w: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct x0, y0 grid coordinates from DICPara's ROI.

    Mirrors the logic in ``integer_search`` so that a mesh built with
    these grids is structurally identical to the FFT-path mesh — only
    the FFT computation itself is skipped.
    """
    roi = para.gridxy_roi_range
    winsize = para.winsize
    winstepsize = para.winstepsize
    half_w = winsize // 2

    min_x = max(roi.gridx[0], half_w)
    max_x = min(roi.gridx[1], img_w - 1 - half_w)
    min_y = max(roi.gridy[0], half_w)
    max_y = min(roi.gridy[1], img_h - 1 - half_w)

    if min_x >= max_x or min_y >= max_y:
        raise ValueError(
            f"ROI too small for winsize={winsize} on image ({img_h}x{img_w}); "
            f"no grid points can be generated."
        )

    x0 = _centered_arange(min_x, max_x, winstepsize)
    y0 = _centered_arange(min_y, max_y, winstepsize)

    if len(x0) == 0 or len(y0) == 0:
        raise ValueError(
            f"ROI [{min_x},{max_x}]x[{min_y},{max_y}] with winstepsize="
            f"{winstepsize} produced an empty grid."
        )
    return x0, y0


def compute_seed_prop_init_guess(
    state: SeedPropagationState,
    dic_mesh: DICMesh,
    f_img: NDArray[np.float64],
    g_img: NDArray[np.float64],
    f_mask: NDArray[np.float64],
    Df: ImageGradients,
    para: DICPara,
    tol: float,
    ref_switched: bool,
    max_snap_distance: float | None = None,
    frame_idx: int = -1,
    ref_idx: int = -1,
) -> NDArray[np.float64]:
    """Produce the interleaved U0 initial guess for one frame.

    - Caches adjacency per mesh (keyed by ``id(elements_fem)``).
    - Rebuilds NodeRegionMap per call (mask may change across frames).
    - On ``ref_switched=True``, warps state.current_seeds to the new
      mesh using state.prev_coords_fem / prev_U_2d. If warp raises
      ``SeedWarpFailure`` (e.g. material point moved outside the new
      ROI), falls back to ``auto_place_seeds_on_mesh`` on the new
      reference and records a ``ReseedEvent`` in ``state.reseed_events``
      so the pipeline can surface the frame in its result.
    - Runs ``propagate_from_seeds`` with the NCC + region gates active.
    - Returns an interleaved U0 (length 2*n_nodes) with NaN on
      unsolved nodes, so the pipeline's existing mask-NaN pass and
      downstream fill_nan_idw handle them uniformly.

    ``frame_idx`` and ``ref_idx`` are advisory — they are only used to
    tag ``ReseedEvent`` entries on the fallback path. Defaults (-1) are
    safe for tests that don't care.
    """
    from .seed_auto_place import AutoPlaceConfig, auto_place_seeds_on_mesh

    n_nodes = dic_mesh.coordinates_fem.shape[0]

    mesh_key = id(dic_mesh.elements_fem)
    if mesh_key not in state.adjacency_cache:
        state.adjacency_cache[mesh_key] = build_node_adjacency(
            dic_mesh.elements_fem, n_nodes,
        )
    adjacency = state.adjacency_cache[mesh_key]

    region_map = precompute_node_regions(
        dic_mesh.coordinates_fem, f_mask, f_img.shape,
    )

    if ref_switched:
        if state.prev_coords_fem is None or state.prev_U_2d is None:
            raise RuntimeError(
                "ref_switched=True but no prev-frame state was captured. "
                "Caller must invoke capture_for_next_frame at the end of "
                "each successful frame."
            )
        try:
            state.current_seeds = warp_seeds_to_new_ref(
                state.current_seeds,
                state.prev_coords_fem,
                state.prev_U_2d,
                dic_mesh.coordinates_fem,
                region_map,
                max_snap_distance=max_snap_distance,
            )
        except SeedWarpFailure as exc:
            # Fallback: auto-place fresh seeds on the new ref frame.
            # If auto-place itself cannot find any viable candidate the
            # original warp failure is re-raised so the pipeline still
            # aborts loudly in the un-recoverable case.
            logger.warning(
                "Frame %d ref-switch (-> ref %d): %s. Auto-placing new seeds.",
                frame_idx, ref_idx, exc,
            )
            ap_config = AutoPlaceConfig(
                ncc_threshold=state.current_seeds.ncc_threshold,
            )
            ap_result = auto_place_seeds_on_mesh(
                coordinates_fem=dic_mesh.coordinates_fem,
                elements_fem=dic_mesh.elements_fem,
                node_region_map=region_map,
                f_img=f_img,
                g_img=g_img,
                mask=f_mask,
                winsize=para.winsize,
                search_radius=int(para.size_of_fft_search_region),
                config=ap_config,
                adjacency=adjacency,
            )
            if len(ap_result.seed_set.seeds) == 0:
                raise SeedWarpFailure(
                    f"{exc} Auto-place fallback also found no viable "
                    f"candidates on the new reference frame "
                    f"(n_regions_skipped={ap_result.n_regions_skipped}). "
                    f"Manually re-place seeds on the new reference and "
                    f"re-run."
                ) from exc
            state.current_seeds = ap_result.seed_set
            state.reseed_events.append(
                ReseedEvent(
                    frame_idx=frame_idx,
                    ref_idx=ref_idx,
                    reason=str(exc),
                    n_new_seeds=len(ap_result.seed_set.seeds),
                ),
            )
            for msg in ap_result.warnings:
                logger.warning("Auto-place fallback: %s", msg)

    # No prev-frame hint_uv warm-start: seeds are a handful of points,
    # and asymmetric single-point NCC happily searches up to image
    # half-size on every frame. The small per-frame cost (a few ms per
    # seed) isn't worth the extra state and failure modes that come
    # with tracking a rolling hint. The Seed.user_hint_uv field remains
    # part of the algorithm API for callers who have a prior they trust.

    ctx = local_icgn_precompute(dic_mesh.coordinates_fem, Df, f_img, para)
    result = propagate_from_seeds(
        ctx,
        state.current_seeds,
        adjacency,
        f_img,
        g_img,
        search_radius=para.size_of_fft_search_region,
        tol=tol,
        node_region_map=region_map,
    )

    U0 = np.empty(2 * n_nodes, dtype=np.float64)
    U0[0::2] = result.U_2d[:, 0]
    U0[1::2] = result.U_2d[:, 1]
    if result.unsolved_nodes.size > 0:
        U0[2 * result.unsolved_nodes] = np.nan
        U0[2 * result.unsolved_nodes + 1] = np.nan
    return U0


def capture_for_next_frame(
    state: SeedPropagationState,
    dic_mesh: DICMesh,
    final_U: NDArray[np.float64],
) -> None:
    """Quality-gate the frame's result and save state for future warps.

    Raises SeedQualityError if any current seed node has NaN in
    ``final_U`` after the frame solve. This is the gate described in
    design decision 16 / feedback_seed_quality: the seed's IC-GN result
    drives the BFS tree, so if the seed itself became a bad_point the
    whole frame's propagation is unreliable — fail loud rather than
    trust an IDW-filled seed.

    On success, stores a copy of the mesh coords and reshaped U for the
    next frame's potential ref-switch warp.
    """
    n_nodes = dic_mesh.coordinates_fem.shape[0]
    if final_U.size != 2 * n_nodes:
        raise ValueError(
            f"final_U length {final_U.size} != 2 * n_nodes ({2 * n_nodes})."
        )
    U_2d = final_U.reshape(n_nodes, 2)

    seed_nodes = np.array(
        [s.node_idx for s in state.current_seeds.seeds], dtype=np.int64,
    )
    seed_rows = U_2d[seed_nodes]
    bad_mask = np.any(np.isnan(seed_rows), axis=1)
    if bad_mask.any():
        bad_ids = seed_nodes[bad_mask].tolist()
        raise SeedQualityError(
            f"Seed node(s) {bad_ids} have NaN displacement after frame "
            f"solve — the seed was flagged as a bad_point by "
            f"detect_bad_points, meaning its IC-GN result drove "
            f"unreliable BFS propagation. Move seed(s) to a more "
            f"textured region, or loosen detect_bad_points' sigma_factor."
        )

    state.prev_coords_fem = dic_mesh.coordinates_fem.copy()
    state.prev_U_2d = U_2d.copy()
