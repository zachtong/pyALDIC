# Changelog

All notable user-facing changes to pyALDIC are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning follows [Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-04-19

### Added

- **Tolerant multi-seed bootstrap.** A single bad Starting Point
  (low correlation in a later frame, IC-GN divergence, etc.) is now
  dropped rather than killing the run. The pipeline keeps going on
  whichever seeds survive. If every seed in a connected region fails,
  the auto-place algorithm is called on that region to try to fill
  the gap before the run aborts. `PropagationResult` now exposes
  `dropped_seeds` and `rescued_seeds` so callers can see exactly
  what happened.
- **Resilient ref-switching for Starting Points.** On an incremental
  or hybrid frame schedule, when the active reference frame changes
  and a Starting Point would otherwise land outside the new ROI, the
  pipeline now auto-places fresh Starting Points on the new reference
  with the same 3-tier quality / edge-distance / BFS-depth rule the
  GUI **Auto-place** button uses, and continues the run instead of
  aborting. The run only aborts when auto-placement itself cannot
  find any viable node on the new reference.
- **Product demo animations** in `assets/`:
  `seed_propagation_demo.gif` (BFS wave around a cracked specimen
  with a locally refined mesh) and `fft_spectral_overlay_3d.gif`
  (two magnitude spectra merging into a 3D correlation peak that
  collapses back into a whole-field displacement map). Used in the
  README and the user guide.
- **`PipelineResult.ref_switch_frames`** and
  **`PipelineResult.reseed_events`** expose the ref-switch /
  auto-reseed events produced during a run so downstream code and
  reports can surface them too. Both default to empty tuples on
  runs that did not switch references.

### Changed

- **Starting Points NCC threshold default lowered from 0.70 to 0.55.**
  The previous default was tuned for small-deformation scenarios and
  systematically rejected valid seeds in accumulative runs once
  deformation grew past the reference-frame template. 0.55 is still
  well above random-noise correlation while tolerating the typical
  NCC degradation large accumulative displacement produces. The
  Advanced panel in the GUI exposes the threshold for users who
  want to tune it.
- **Starting Points Auto-place algorithm is now shared between the
  GUI and the pipeline fallback.** The 3-tier selection (NCC ≥ 0.85,
  top 30% by mask-edge distance, then lowest BFS max depth) lives in
  `al_dic.solver.seed_auto_place.auto_place_seeds_on_mesh`. The
  GUI's **Auto-place** button is now a thin wrapper around it.
- **User guide** now documents the multi-seed tolerance, ref-switch
  warp, and the auto-reseed fallback (section 07 *Starting Points
  workflow* and an updated entry in section 14 *Troubleshooting*).

### Fixed

- Seed warp near slot-interior hanging midsides no longer drags a
  handful of un-measurable nodes into the active seed set on quadtree
  meshes with a crack inside the refinement region.

## [0.2.0] — 2026-04-17

### Fixed (packaging)

- **`matplotlib` is now a core runtime dependency.** It was previously
  only listed under the `dev` extras, so a clean `pip install al-dic`
  crashed on first launch with `ModuleNotFoundError: No module named
  'matplotlib'`. Several core modules (canvas rendering, colorbar
  overlay, viz controller, PNG and animation exporters) import it.

### Added

- **Session save / load.** `File → Save Session` (Ctrl+S) writes a
  single `.aldic.json` with parameters, physical units, and every
  per-frame Region of Interest mask (base64-encoded PNG inline).
  `File → Open Session` (Ctrl+O) restores them. Missing image folder
  is a non-fatal warning so sessions stay portable across machines.
- **Search Range** parameter promoted from the Advanced collapsible
  section to the main Parameters panel. Users no longer need to
  expand Advanced to set the FFT search radius.
- **Workflow Type section** at the top of the left sidebar, above
  Region of Interest. Holds Tracking Mode, Solver, and Reference
  Update policy. Deciding these first determines which frames need
  Regions of Interest, and avoids drawing regions on wrong frames.
- **Live Region of Interest hint** below the section header. Rewrites
  itself based on workflow type to tell users exactly which frame
  numbers need a region.
- **Modal dialog for fatal pipeline errors** (missing images,
  undefined Region of Interest, ROI too small, runtime exceptions).
  Errors previously only appeared in the console log, below the
  fold, and were easy to miss.
- **Strain window auto-opens** when a Run completes. Removes the
  redundant "Would you like to open Strain?" confirmation dialog.
- **Export: colorbar ON by default** for both images and animations.
  Exported fields now have a scale without toggling the checkbox
  every time.
- **Region of Interest full name** used in every user-facing label
  and message instead of the jargon "ROI". The image-list column
  header shortens to "Region" to fit in 50 px.
- **Ref Update / Ref Frames** labels expanded to **Reference Update**
  / **Reference Frames**.

### Changed

- **Strain field smoothing presets rebalanced.** Old Light (σ = 0.25
  × step) had no measurable effect because the Gaussian kernel
  couldn't reach any neighbour node. New presets:
  - Off (0)
  - Light (σ = 0.5 × step)
  - Medium (σ = 1 × step) — recommended
  - Strong (σ = 2 × step) — marked with a warning glyph.
- **Pause + Stop buttons merged into a single Cancel button.** The
  previous Pause was never a true pause (UI state did not reset on
  resume) and Stop was a hard kill; Cancel is a clean stop with
  partial results kept.
- **Strain window "Export Strain" button renamed to "Export
  Results"** to match the main window. Both open the exact same
  dialog.
- **"Smoothing" UI in Strain window** now a single dropdown
  (Off / Light / Medium / Strong) instead of a checkbox plus
  dropdown. Section renamed to "Strain field smoothing" and a
  tooltip explains that it smooths the strain field after
  computation (not displacement before it).
- **Refine brush button** is now disabled (dashed border, muted
  colours) on any frame other than frame 1, with a tooltip
  explaining why. Previously the frame-1-only restriction was only
  in a tooltip.

### Fixed

- **Refine brush overlay no longer bleeds onto later frames.** The
  brush mask lives in frame-1 coordinates; showing it on a deformed
  frame painted strokes at the wrong material points.
- **`File → Save Session` no longer crashes** with `NameError: name
  'Path' is not defined`. The `pathlib` import was missing from
  `gui/app.py` when the feature first landed.

### Removed

- Unused `matplotlib` listing under `dev` extras (it is now a
  required runtime dep).

## [0.1.1]

Previous release. See git tags / GitHub releases for earlier
history.

[0.3.0]: https://github.com/zachtong/pyALDIC/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/zachtong/pyALDIC/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/zachtong/pyALDIC/releases/tag/v0.1.1
