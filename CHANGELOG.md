# Changelog

All notable user-facing changes to pyALDIC are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning follows [Semantic Versioning](https://semver.org/).

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

[0.2.0]: https://github.com/zachtong/pyALDIC/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/zachtong/pyALDIC/releases/tag/v0.1.1
