"""Post-pipeline strain computation controller.

The displacement pipeline (``pipeline_controller``) intentionally runs with
``compute_strain=False`` for fast iteration. This controller offers a
user-driven, decoupled path: read the existing displacement results, run
the strain pipeline once per frame against an isolated ``DICPara``
override, and write the result back into ``state.results.result_strain``
via :func:`dataclasses.replace`.

Design principles
-----------------
* No new ``AppState`` fields. Strain results live inside the existing
  ``PipelineResult.result_strain`` list.
* The base ``DICPara`` is never mutated. Overrides go through
  :func:`dataclasses.replace` and are restricted to a small whitelist so
  that callers cannot accidentally re-tune displacement parameters.
* The mesh and region map are taken from frame 1, exactly mirroring
  ``pipeline.py`` Section 8.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from al_dic.core.data_structures import (
    DICMesh,
    DICPara,
    PipelineResult,
    StrainResult,
)
from al_dic.gui.app_state import AppState
from al_dic.strain.compute_strain import compute_strain
from al_dic.utils.region_analysis import (
    NodeRegionMap,
    precompute_node_regions,
)

ProgressCallback = Callable[[float, str], None]


# Whitelist of parameters that can be overridden when computing strain.
# Anything outside this set would re-tune the displacement pipeline and is
# rejected to keep the strain post-processing path strictly read-only with
# respect to the existing ``result_disp`` field.
ALLOWED_OVERRIDES: frozenset[str] = frozenset({
    "method_to_compute_strain",
    "strain_plane_fit_rad",
    "strain_smoothness",
    "strain_type",
})


class StrainController:
    """Drive ``compute_strain`` over an existing ``PipelineResult``."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all_frames(
        self,
        override: dict[str, object],
        progress_cb: ProgressCallback | None = None,
    ) -> list[StrainResult]:
        """Compute strain for every frame in ``state.results.result_disp``.

        Args:
            override: Strain-only parameter overrides. Keys must be in
                :data:`ALLOWED_OVERRIDES`; otherwise a ``ValueError`` is
                raised.
            progress_cb: Optional ``(fraction, message)`` callback invoked
                once per processed frame. ``fraction`` is in ``[0, 1]``.

        Returns:
            List of ``StrainResult`` aligned with ``result_disp``.

        Raises:
            RuntimeError: If ``state.results`` is ``None``.
            ValueError: If any override key is outside ``ALLOWED_OVERRIDES``.
        """
        result = self._require_results()
        self._validate_override(override)

        strain_mesh, region_map, para_strain = self._build_strain_context(
            result, override,
        )

        n_frames = len(result.result_disp)
        out: list[StrainResult] = []
        for i, frame in enumerate(result.result_disp):
            U = frame.U_accum if frame.U_accum is not None else frame.U
            sr = compute_strain(strain_mesh, para_strain, U, region_map)
            out.append(sr)
            if progress_cb is not None:
                progress_cb(
                    (i + 1) / max(1, n_frames),
                    f"Strain frame {i + 1}/{n_frames}",
                )
        return out

    def compute_and_store(
        self,
        override: dict[str, object],
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        """Compute strain for all frames and write back into ``state.results``.

        The replacement uses :func:`dataclasses.replace` so that
        ``PipelineResult`` stays a frozen dataclass and listeners receive
        a single ``results_changed`` notification at the end.
        """
        new_strain = self.compute_all_frames(override, progress_cb=progress_cb)
        current = self._require_results()
        self._state.results = replace(current, result_strain=new_strain)
        self._state.results_changed.emit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_results(self) -> PipelineResult:
        result = self._state.results
        if result is None:
            raise RuntimeError(
                "StrainController: no displacement results available. "
                "Run DIC first before computing strain."
            )
        return result

    @staticmethod
    def _validate_override(override: dict[str, object]) -> None:
        unknown = set(override) - ALLOWED_OVERRIDES
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(
                f"Override keys not allowed for strain post-processing: "
                f"{joined}. Allowed keys: {sorted(ALLOWED_OVERRIDES)}."
            )

    def _build_strain_context(
        self,
        result: PipelineResult,
        override: dict[str, object],
    ) -> tuple[DICMesh, NodeRegionMap, DICPara]:
        """Build the (mesh, region_map, para) triple used by ``compute_strain``.

        Mirrors the construction in ``pipeline.py`` Section 8 (frame-1
        mesh + per-frame mask), but pulls inputs from ``state.results``
        and the user override instead of from a fresh pipeline run.
        """
        if not result.result_fe_mesh_each_frame:
            raise RuntimeError(
                "StrainController: PipelineResult has no per-frame meshes."
            )
        ref_mesh = result.result_fe_mesh_each_frame[0]
        strain_mesh = DICMesh(
            coordinates_fem=ref_mesh.coordinates_fem,
            elements_fem=ref_mesh.elements_fem,
            mark_coord_hole_edge=ref_mesh.mark_coord_hole_edge,
        )

        mask = self._resolve_reference_mask(result)
        h, w = mask.shape
        region_map = precompute_node_regions(
            strain_mesh.coordinates_fem, mask, (h, w),
        )

        # Always layer the resolved mask in so that ``compute_strain``
        # sees a consistent ROI even if the original ``dic_para``
        # carried ``img_ref_mask=None``.
        para_strain = replace(
            result.dic_para,
            img_ref_mask=mask,
            **override,
        )
        return strain_mesh, region_map, para_strain

    def _resolve_reference_mask(
        self, result: PipelineResult,
    ) -> NDArray[np.float64]:
        """Find a reference-frame mask for region-map construction.

        Order of preference:
            1. ``dic_para.img_ref_mask`` (set during pipeline run).
            2. ``state.per_frame_rois[0]`` (user-painted ROI).

        Raises:
            RuntimeError: If neither source is available.
        """
        ref_mask = result.dic_para.img_ref_mask
        if ref_mask is not None:
            return np.asarray(ref_mask, dtype=np.float64)

        gui_mask = self._state.per_frame_rois.get(0)
        if gui_mask is not None:
            return gui_mask.astype(np.float64)

        raise RuntimeError(
            "StrainController: no reference-frame mask available "
            "(dic_para.img_ref_mask is None and per_frame_rois[0] is unset)."
        )
