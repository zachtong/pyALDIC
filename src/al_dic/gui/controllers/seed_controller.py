"""Data-layer manager for init_guess_mode='seed_propagation' UI.

Owns the preview mesh and NodeRegionMap derived from the current
ROI + subset_size + subset_step. Canvas click coordinates are snapped
to the nearest mesh node; the node's region determines the seed's
region_id. Existing seeds re-snap automatically when the mesh changes
(Q3-B decision in the Phase 5 UX plan).

This controller is GUI-internal but has no Qt widget dependencies —
pure data manipulation on AppState. Unit-tested directly without a
QApplication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject

from al_dic.core.data_structures import DICMesh, DICPara, GridxyROIRange
from al_dic.gui.app_state import AppState, SeedRecord
from al_dic.mesh.mesh_setup import mesh_setup
from al_dic.solver.seed_prop_pipeline import build_grid_for_roi
from al_dic.utils.region_analysis import NodeRegionMap, precompute_node_regions


@dataclass(frozen=True)
class _MeshCacheKey:
    """Tuple detector for preview-mesh cache invalidation."""

    roi_hash: int
    winsize: int
    winstepsize: int
    img_size: tuple[int, int]


class SeedController(QObject):
    """Mutates ``state.seeds`` in response to canvas clicks and
    mesh-parameter changes.

    Public API
    ----------
    add_seed_at_xy(x, y) : bool
        Left-click handler. Snap to nearest mesh node, verify its
        region, append to state.seeds.
    remove_seed_near(x, y, radius) : bool
        Right-click handler. Drop the seed whose xy_canvas is closest
        to (x, y) and within radius.
    clear_seeds()
        Wipe all seeds (e.g. when mode switches away).

    UI queries
    ----------
    nearest_node_preview(x, y) : (node_idx, node_x, node_y) | None
        For hover snap-preview rendering.
    is_xy_in_mask(x, y) : bool
        Cursor validity — used by canvas to switch shapes.
    regions_status() : list[(region_id, has_seed, node_indices)]
        For yellow/green region overlay.
    all_regions_seeded() : bool
        Run-button gating.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        self._preview_mesh: DICMesh | None = None
        self._region_map: NodeRegionMap | None = None
        self._node_to_region: NDArray[np.int64] | None = None
        self._cache_key: _MeshCacheKey | None = None

        # Mesh-affecting state changes → rebuild + re-snap
        self._state.roi_changed.connect(self._on_mesh_params_changed)
        self._state.params_changed.connect(self._on_mesh_params_changed)

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_seed_at_xy(self, x: float, y: float) -> bool:
        """Snap (x, y) to nearest mesh node, verify region, append seed.

        Returns True on success. Fails (returns False) if:
          - preview mesh can't be built (no ROI / invalid winsize)
          - nearest node isn't inside any tracked region
          - (x, y) isn't inside the ROI mask (pixel-level check, Q8-A)
        """
        if not self.is_xy_in_mask(x, y):
            return False
        self._ensure_preview_mesh()
        if self._preview_mesh is None or self._node_to_region is None:
            return False

        coords = self._preview_mesh.coordinates_fem
        d2 = (coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2
        nearest = int(np.argmin(d2))
        region_id = int(self._node_to_region[nearest])
        if region_id < 0:
            return False

        # xy_canvas stores the snapped node position, not the raw click,
        # so (a) the seed marker renders exactly on the node and (b) the
        # re-snap logic has a clean, ambiguity-free anchor to re-match
        # against after mesh changes.
        node_xy = (float(coords[nearest, 0]), float(coords[nearest, 1]))
        self._state.seeds.append(
            SeedRecord(
                node_idx=nearest,
                region_id=region_id,
                is_warped=False,
                xy_canvas=node_xy,
            ),
        )
        self._state.seeds_changed.emit()
        return True

    def remove_seed_near(
        self, x: float, y: float, radius: float,
    ) -> bool:
        """Drop the seed closest to (x, y) within ``radius`` pixels."""
        seeds = self._state.seeds
        if not seeds:
            return False

        # Distance from click to each seed's xy_canvas (or node coord if
        # xy_canvas is None — e.g., seed added programmatically via REPL).
        positions: list[tuple[float, float]] = []
        for s in seeds:
            if s.xy_canvas is not None:
                positions.append(s.xy_canvas)
            elif self._preview_mesh is not None:
                c = self._preview_mesh.coordinates_fem[s.node_idx]
                positions.append((float(c[0]), float(c[1])))
            else:
                positions.append((float("inf"), float("inf")))

        pos = np.asarray(positions)
        d2 = (pos[:, 0] - x) ** 2 + (pos[:, 1] - y) ** 2
        nearest_i = int(np.argmin(d2))
        if d2[nearest_i] > radius ** 2:
            return False

        del self._state.seeds[nearest_i]
        self._state.seeds_changed.emit()
        return True

    def clear_seeds(self) -> None:
        if not self._state.seeds:
            return
        self._state.seeds.clear()
        self._state.seeds_changed.emit()

    # ------------------------------------------------------------------
    # UI queries
    # ------------------------------------------------------------------

    def nearest_node_preview(
        self, x: float, y: float,
    ) -> tuple[int, float, float] | None:
        """Return (node_idx, node_x, node_y) for hover highlight.

        None when no mesh is available. Does NOT gate on region —
        callers decide whether to render the preview based on
        ``is_xy_in_mask``.
        """
        self._ensure_preview_mesh()
        if self._preview_mesh is None:
            return None
        coords = self._preview_mesh.coordinates_fem
        d2 = (coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2
        nearest = int(np.argmin(d2))
        return nearest, float(coords[nearest, 0]), float(coords[nearest, 1])

    def is_xy_in_mask(self, x: float, y: float) -> bool:
        """Pixel-level check: is ``(x, y)`` inside the current ROI mask.

        Used by the canvas to toggle cursor shapes (Q8-A — the user
        sees the mask directly, so this is the most intuitive probe).
        """
        mask = self._state.per_frame_rois.get(0)
        if mask is None:
            return False
        h, w = mask.shape
        xi = int(round(x))
        yi = int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            return False
        return bool(mask[yi, xi])

    def regions_status(
        self,
    ) -> list[tuple[int, bool, NDArray[np.int64]]]:
        """Per-region seeding status for yellow/green overlay.

        Returns a list of (region_id, has_seed, node_indices).
        Empty list when preview mesh isn't ready.
        """
        self._ensure_preview_mesh()
        if self._region_map is None:
            return []
        seeded = {s.region_id for s in self._state.seeds}
        return [
            (i, i in seeded, nodes)
            for i, nodes in enumerate(self._region_map.region_node_lists)
        ]

    def all_regions_seeded(self) -> bool:
        """Run-button gating: every region has at least one seed."""
        status = self.regions_status()
        if not status:
            return False
        return all(has for _, has, _ in status)

    def auto_place_seeds(
        self,
        ref_img: NDArray[np.float64],
        def_img: NDArray[np.float64],
        winsize: int,
        search_radius: int,
        stride: int = 3,
        only_unseeded_regions: bool = False,
    ) -> int:
        """Place one seed per region at the interior node with highest NCC.

        Uses two biases to avoid the "seed at a mask corner" failure mode
        observed in practice:
          1. **Interior preference** — a node whose mesh adjacency lies
             entirely within the same region (no neighbour on the mask
             boundary) is strictly preferred. Boundary nodes are only
             considered if the region has no interior candidates.
          2. **Highest-NCC** tie-breaker among the preferred set, same
             as before.

        Args:
            ref_img: Reference frame.
            def_img: Deformed frame for the NCC.
            winsize: Template window (= subset_size).
            search_radius: Single-point NCC half-width.
            stride: Evaluate every Nth node per region.
            only_unseeded_regions: If True, leave regions that already
                contain a seed untouched and only fill regions that
                have no seed yet. Used by the ROI-ready auto-placer so
                multi-region edits progressively fill in.

        Returns:
            Number of seeds newly placed this call.
        """
        from al_dic.solver.seed_propagation import seed_single_point_fft

        self._ensure_preview_mesh()
        if self._preview_mesh is None or self._region_map is None:
            return 0

        if not only_unseeded_regions:
            # Manual-button path: the user asked for a full re-auto-place.
            if self._state.seeds:
                self._state.seeds.clear()

        existing = {s.region_id for s in self._state.seeds}

        # Build an 'interior' mask where a full subset window fits inside
        # the ROI. A node that lives on top of an interior pixel has
        # every sample in its winsize x winsize template covered by the
        # mask -- no edge artifacts from mask-cut pixels. Nodes on the
        # mask boundary usually cause IC-GN divergence and auto-place
        # should avoid them.
        from scipy.ndimage import binary_erosion
        mask = self._state.per_frame_rois.get(0)
        if mask is None:
            return 0
        half_w = max(1, winsize // 2)
        struct = np.ones((2 * half_w + 1, 2 * half_w + 1), dtype=bool)
        interior_mask = binary_erosion(mask, structure=struct)
        h, w = interior_mask.shape

        coords = self._preview_mesh.coordinates_fem
        placed = 0
        for region_id, nodes in enumerate(
            self._region_map.region_node_lists,
        ):
            if only_unseeded_regions and region_id in existing:
                continue
            interior: list[int] = []
            for node_idx in nodes:
                n_idx = int(node_idx)
                nx = int(np.clip(round(coords[n_idx, 0]), 0, w - 1))
                ny = int(np.clip(round(coords[n_idx, 1]), 0, h - 1))
                if interior_mask[ny, nx]:
                    interior.append(n_idx)
            candidates = interior if interior else [int(n) for n in nodes]
            candidates = candidates[::stride] if stride > 1 else candidates

            best_node = -1
            best_ncc = -1.0
            best_xy: tuple[float, float] | None = None
            for node_idx in candidates:
                nx = float(coords[node_idx, 0])
                ny = float(coords[node_idx, 1])
                r = seed_single_point_fft(
                    ref_img, def_img, (nx, ny), winsize, search_radius,
                )
                if r.valid and r.ncc_peak > best_ncc:
                    best_ncc = r.ncc_peak
                    best_node = node_idx
                    best_xy = (nx, ny)
            if best_node < 0 or best_xy is None:
                continue
            self._state.seeds.append(
                SeedRecord(
                    node_idx=best_node,
                    region_id=region_id,
                    is_warped=False,
                    ncc_peak=float(best_ncc),
                    xy_canvas=best_xy,
                ),
            )
            placed += 1

        if placed > 0:
            self._state.seeds_changed.emit()
        return placed

    def region_label_image(self) -> NDArray[np.int64] | None:
        """Return (H, W) pixel labels aligned with ``regions_status()``.

        Values:
          -1  outside any tracked region (mask hole, or filtered-out region)
          0..N-1  region_id matching ``regions_status``

        Used by the canvas region-color overlay (P5.2c). Alignment with
        ``regions_status`` is established by mapping each filtered region
        back to its raw scipy-label via the first node's pixel position,
        so the color painted for region i in the overlay is consistent
        with the ``has_seed`` flag returned by ``regions_status``.
        """
        self._ensure_preview_mesh()
        mask = self._state.per_frame_rois.get(0)
        if mask is None or self._region_map is None or self._preview_mesh is None:
            return None

        from scipy.ndimage import label as scipy_label

        struct = np.ones((3, 3), dtype=np.int32)
        labeled, _ = scipy_label(mask, structure=struct)

        h, w = labeled.shape
        result = np.full_like(labeled, -1, dtype=np.int64)
        coords = self._preview_mesh.coordinates_fem
        for region_id, nodes in enumerate(self._region_map.region_node_lists):
            if nodes.size == 0:
                continue
            n0 = int(nodes[0])
            x0 = int(np.clip(round(coords[n0, 0]), 0, w - 1))
            y0 = int(np.clip(round(coords[n0, 1]), 0, h - 1))
            raw_label = int(labeled[y0, x0])
            if raw_label == 0:
                continue
            result[labeled == raw_label] = region_id
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_mesh_params_changed(self) -> None:
        """Rebuild preview mesh + re-snap seeds to the new node indices.

        Re-snap rule (user's simplification at DC-E1):
          - xy_canvas is the seed's source-of-truth coordinate.
          - After rebuild, find the new nearest-node for each seed.
          - If that xy_canvas no longer lies in any tracked region
            (because ROI edit deleted that region), drop the seed
            silently — the region's color reverts to yellow in the
            UI overlay, signalling that user action is required.
        """
        new_key = self._compute_cache_key()
        if new_key == self._cache_key:
            return  # mesh-relevant params unchanged

        # Invalidate and rebuild
        self._cache_key = None
        self._preview_mesh = None
        self._region_map = None
        self._node_to_region = None
        self._ensure_preview_mesh()

        if not self._state.seeds:
            return
        if self._preview_mesh is None or self._node_to_region is None:
            # Mesh can't be built (mask was cleared) — leave seeds as-is.
            # They'll re-snap next time params change and a mesh exists.
            return

        coords = self._preview_mesh.coordinates_fem
        kept: list[SeedRecord] = []
        mask = self._state.per_frame_rois.get(0)
        for s in self._state.seeds:
            if s.xy_canvas is None:
                # Programmatic seed — no canvas coord to re-snap from;
                # keep node_idx as-is (caller takes responsibility).
                kept.append(s)
                continue
            xc, yc = s.xy_canvas
            # If xy_canvas is no longer inside the mask, the region
            # containing this seed was deleted → drop silently.
            if mask is not None:
                h, w = mask.shape
                xi = int(round(xc))
                yi = int(round(yc))
                if not (0 <= xi < w and 0 <= yi < h) or not bool(mask[yi, xi]):
                    continue
            d2 = (coords[:, 0] - xc) ** 2 + (coords[:, 1] - yc) ** 2
            nearest = int(np.argmin(d2))
            region_id = int(self._node_to_region[nearest])
            if region_id < 0:
                continue  # nearest node filtered out of region map
            kept.append(
                SeedRecord(
                    node_idx=nearest,
                    region_id=region_id,
                    is_warped=s.is_warped,
                    ncc_peak=None,  # stale after mesh change
                    xy_canvas=s.xy_canvas,
                ),
            )

        # Always replace — either seeds dropped or node_idx/region_id updated
        self._state.seeds[:] = kept
        self._state.seeds_changed.emit()

    def _compute_cache_key(self) -> _MeshCacheKey | None:
        """Hash the mesh-affecting state into a cache key.

        Returns None if mesh can't be built (no mask yet).
        """
        mask = self._state.per_frame_rois.get(0)
        if mask is None:
            return None
        roi_hash = hash(mask.tobytes())
        return _MeshCacheKey(
            roi_hash=roi_hash,
            winsize=self._state.subset_size,
            winstepsize=self._state.subset_step,
            img_size=mask.shape,
        )

    def _ensure_preview_mesh(self) -> None:
        """Build preview mesh + NodeRegionMap if cache is stale."""
        key = self._compute_cache_key()
        if key is None:
            self._preview_mesh = None
            self._region_map = None
            self._node_to_region = None
            self._cache_key = None
            return
        if self._cache_key == key and self._preview_mesh is not None:
            return

        mask_bool = self._state.per_frame_rois.get(0)
        assert mask_bool is not None  # compute_cache_key would have returned None
        h, w = mask_bool.shape
        mask = mask_bool.astype(np.float64)

        # Derive ROI from mask bounds so mesh_setup has a valid grid box.
        ys, xs = np.where(mask_bool)
        if ys.size == 0:
            self._preview_mesh = None
            self._region_map = None
            self._node_to_region = None
            self._cache_key = key
            return

        roi = GridxyROIRange(
            gridx=(int(xs.min()), int(xs.max())),
            gridy=(int(ys.min()), int(ys.max())),
        )
        para = DICPara(
            winsize=self._state.subset_size,
            winstepsize=self._state.subset_step,
            img_size=(h, w),
            gridxy_roi_range=roi,
            img_ref_mask=mask,
        )

        try:
            x0, y0 = build_grid_for_roi(para, h, w)
            mesh = mesh_setup(x0, y0, para)
        except (ValueError, RuntimeError):
            self._preview_mesh = None
            self._region_map = None
            self._node_to_region = None
            self._cache_key = key
            return

        region_map = precompute_node_regions(
            mesh.coordinates_fem, mask, (h, w),
        )
        n_nodes = mesh.coordinates_fem.shape[0]
        node_to_region = np.full(n_nodes, -1, dtype=np.int64)
        for i, nodes in enumerate(region_map.region_node_lists):
            node_to_region[nodes] = i

        self._preview_mesh = mesh
        self._region_map = region_map
        self._node_to_region = node_to_region
        self._cache_key = key
