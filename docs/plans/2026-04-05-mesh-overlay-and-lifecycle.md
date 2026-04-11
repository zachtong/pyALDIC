# Mesh Overlay & Lifecycle Redesign

**Date:** 2026-04-05
**Status:** Approved (UI/UX decisions confirmed)

## Problem Statement

1. **Mesh lifecycle bug**: `mesh_setup` and `mark_inside` are called only once
   (first frame). In incremental mode with different per-frame ROIs, the mesh
   element trimming is stale — areas that were holes in frame 0 but valid in a
   new reference frame have no elements (Subpb2 FEM gap).
2. **No mesh visualization**: Users cannot see the computational mesh or
   understand the relationship between subset_step, subset_size, and spatial
   resolution before or after running DIC.

## UI/UX Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Visibility | ROI editing + results viewing | Preview before DIC, verify after |
| Deformed toggle | Follows current toggle | Consistent with field overlay |
| Visual style | Grid lines + node dots | Shows both element shape and node positions |
| Subset window | Mouse hover follow | Natural scan of boundary regions |
| Toggle placement | Canvas top toolbar | Display attribute, not DIC parameter |
| Parameter response | Debounced 300ms | Balance responsiveness vs compute |
| Brush warp on ref switch | Displacement field warp | Accurate material tracking |

## Visual Specification

- **Element edges**: white (#ffffff), 1px, opacity 0.4
  - Draw Q4 corners (cols 0-3) as quadrilateral edges
  - Works for both uniform and refined mesh
- **Node dots**: green (#22c55e), 3px diameter, opacity 0.7
  - Skip NaN nodes (outside mask)
- **Subset window (hover)**: yellow (#facc15), 2px dashed rect, opacity 0.8
  - Size = winsize x winsize, centered on nearest node
  - Red dashed for portions extending beyond ROI
- **Toggle buttons**: `[Grid]` and `[Window]` in canvas toolbar
  - `[Window]` enabled only when `[Grid]` is on

## Architecture: Mesh Lifecycle

### Current (broken)

```
Frame 1: mesh_setup → mark_inside(mask_0) → dic_mesh  [CREATED]
Frame 2: reuse dic_mesh                                [STALE if mask changed]
Frame 3: reuse dic_mesh                                [STALE if mask changed]
```

### Correct

```
Pipeline init:
  base_mesh = mesh_setup(x0, y0, para)       # full grid, NO trim
  prev_mask_hash = None

Per-frame loop:
  f_mask = masks[ref_idx]
  mask_hash = hash(f_mask.tobytes())

  if mask_hash != prev_mask_hash:
      dic_mesh = base_mesh                    # start from full grid
      dic_mesh = refine(base_mesh, criteria, mask)  # if refinement_policy
      dic_mesh = trim(dic_mesh, mask)         # mark_inside
      if prev_mask_hash is not None:
          # Uniform: nodes unchanged, U_accum preserved
          # Refined: interpolate U_accum to new nodes
          reinterpolate_U_accum()
      prev_mask_hash = mask_hash
      invalidate subpb2/strain caches
```

### Brush Warp on Ref Switch

When reference frame changes and BrushRegionCriterion is active:
1. Take original brush_mask (frame 0 pixel coordinates)
2. Warp using U_accum: `X_new = X_old + U_accum(X_old)`
3. Rasterize warped brush to new reference pixel coordinates
4. Update BrushRegionCriterion with warped mask

Reuses inverse-warp logic from `viz_controller.py`.

## Architecture: Mesh Overlay

### New Widget: `MeshOverlay`

- QPainter-based widget, parented to `QGraphicsView.viewport()`
- Same pattern as `ColorbarOverlay`: `WA_TransparentForMouseEvents`
- Paints in screen space (not scene space) — transform node coords via
  `QGraphicsView.mapFromScene()`

### Data Flow

**Preview mode (ROI editing, pre-DIC):**
```
param change / ROI edit
  → QTimer.singleShot(300ms)
  → mesh_setup(roi_params) → trim(roi_mask)
  → MeshOverlay.set_mesh(coords, elems, None)
```

**Results mode (post-DIC):**
```
frame change
  → mesh = result.result_fe_mesh_each_frame[frame]
  → if deformed: nodes = mesh.coords + U_accum[frame]
  → MeshOverlay.set_mesh(coords, elems, winsize)
```

### Subset Window Hover

- `ImageCanvas.mouseMoveEvent`: if mesh overlay visible + window toggle on
- Find nearest node via brute-force distance (< 500 nodes typically)
- `MeshOverlay.set_hover_node(idx)` → draws dashed rect on next paint
- `leaveEvent` → clears hover

## Implementation Phases

### Phase 1: Pipeline — mesh lifecycle fix

**Files:** `src/al_dic/core/pipeline.py`

1. After first `mesh_setup`, store as `base_mesh` (untrimmed)
2. Add `prev_mask_hash` tracking in frame loop
3. On mask change: re-trim from `base_mesh` (uniform case)
4. When refinement_policy active: re-refine from `base_mesh` + re-trim
5. Invalidate subpb2 cache on mesh change
6. Ensure `result_fe_mesh_each_frame[frame]` stores per-frame mesh snapshot
7. For uniform mesh: nodes unchanged, `current_U0` preserved as-is
8. For refined mesh: interpolate `current_U0` via `_interpolate_u0`

**Tests:**
- Extend `scripts/report_mesh_behavior.py` scenarios
- Verify: scenario C (inc + different masks) now shows multiple mark_inside calls
- Verify: scenario D (custom schedule + ref switch) re-trims with new mask

### Phase 2: MeshOverlay widget

**Files:** `src/al_dic/gui/widgets/mesh_overlay.py` (NEW)

1. `MeshOverlay(QWidget)` with `WA_TransparentForMouseEvents`
2. `set_mesh(coords, elems, view_transform)` — stores mesh data
3. `set_hover_node(idx, winsize)` — stores hover state
4. `paintEvent`: draw element edges, node dots, hover window
5. Transform node coords: scene → viewport via `mapFromScene`

### Phase 3: Canvas integration

**Files:** `src/al_dic/gui/panels/canvas_area.py`, `src/al_dic/gui/app_state.py`

1. Add `show_mesh: bool` and `show_subset_window: bool` to AppState
2. Add toggle buttons `[Grid]` `[Window]` to canvas toolbar
3. Create `MeshOverlay` instance, parent to canvas viewport
4. Wire toggle signals → overlay visibility
5. In `_refresh_overlay`: if show_mesh, compute mesh data and update overlay
6. Preview mode: debounced mesh generation on param/ROI change
7. Results mode: read from `result_fe_mesh_each_frame[frame]`

### Phase 4: Subset window hover

**Files:** `src/al_dic/gui/panels/canvas_area.py`

1. Override `mouseMoveEvent` in ImageCanvas: if mesh visible + window toggle
2. `mapToScene(pos)` → find nearest node in current mesh coords
3. Call `MeshOverlay.set_hover_node(idx, winsize)`
4. `leaveEvent` → clear hover
5. In deformed mode: window center at deformed node position

### Phase 5: Brush warp (deferred — implement with brush refinement)

**Files:** `src/al_dic/core/pipeline.py`, `src/al_dic/mesh/refinement.py`

1. On ref switch with BrushRegionCriterion active:
   - Warp `brush_mask` from original ref coords to new ref coords
   - Use accumulated displacement field for forward warp
2. Update criterion with warped mask before re-refine

This phase is deferred until brush refinement UI is implemented.

## Dependency Graph

```
Phase 1 (pipeline fix) ──┐
                         ├── Phase 3 (canvas integration)
Phase 2 (overlay widget)─┘        │
                                  ├── Phase 4 (hover)
                                  │
                          Phase 5 (brush warp, deferred)
```

Phases 1 and 2 are independent and can be developed in parallel.
Phase 3 depends on both. Phase 4 depends on Phase 3.
Phase 5 is deferred.
