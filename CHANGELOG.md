# Changelog

All notable user-facing changes to pyALDIC are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.9.0] — 2026-09-24

pyALDIC now installs like any other program on Windows and macOS, and gains
an Analysis tab for the question a field map raises next: how much did this
point, this line, this area strain -- and against what load.

### Added

- **A Windows installer and a macOS app.** From the release page,
  `pyALDIC-Windows-Setup.exe` installs for the current user -- no
  administrator rights, so it works on lab computers -- with a Start menu
  entry, an optional desktop shortcut, the `.aldic` file association and a
  clean uninstall. `pyALDIC-macOS.dmg` holds `pyALDIC.app` for Apple silicon
  Macs on macOS 14 or later; it is not notarized, so a downloaded copy's first
  launch goes through *System Settings > Privacy & Security > Open Anyway*,
  which a note on the disk image explains. The portable zip stays, now named
  `pyALDIC-Windows-Portable.zip`. Release file names no longer carry the
  version, so `releases/latest/download/<name>` always fetches the newest.
- **Post-processing analysis.** A second tab in the strain window.
  - Point, line and region probes are placed, selected, moved and reshaped on
    the field they measure. The chart follows the frame on show, and clicking
    the chart jumps to a frame.
  - Three views: a quantity over time; the field along a line, with the other
    frames in grey behind the current one; and a kymograph, distance against
    frame.
  - A line is also a **virtual extensometer** -- engineering strain, true
    strain, elongation -- and a **crack gauge**: opening, sliding and total
    separation. The Extensometer and Crack gauge tools place one and switch
    the chart to its reading.
  - Region statistics (mean, median, maximum, minimum, standard deviation)
    are area-weighted. Every frame reports the fraction of the probe that was
    reliable, and a line or region below an adjustable threshold (0.5 by
    default) is left blank rather than averaged over what is left.
  - A probe that a growing crack cuts keeps reading the material that
    remains, and says so: *crack from frame N*.
  - **Load data.** Import a testing machine's record (CSV with comma,
    semicolon or tab; decimal commas; a units row), match it to the frames by
    frame number or by time with an offset, and plot against load or stress
    (N or kN; stress from the initial cross-section) -- including
    **stress-strain curves**.
  - Charts export for a page -- PNG, SVG or PDF, light theme, 16 x 10 cm at
    300 dpi, text left editable -- or copy to the clipboard; the numbers export
    as self-describing CSV.
  - Probes and the load record are saved with the session; sessions from 0.8.0
    open unchanged.
  - All of it is scriptable: `from al_dic.analysis import AnalysisEngine,
    Probe, LoadData, extract_series`.
- **Display geometry and background, chosen separately.** *Show on*
  (deformed / reference) and *Show background image* replace the single
  *Show on deformed frame* checkbox, in both windows and the export dialog, so
  the field can be shown or exported on its own, in either geometry.
- **The black_rainbow colormap**, ported from MATLAB: jet with the green band
  replaced by black, so the middle of the range reads as background.

### Changed

- **One term per concept in all eight languages.** The translation glossary
  contradicted itself -- Traditional Chinese gave "frame" as 影格 in one row
  and 幀 in the next, and "Field" was rendered as a form field -- and 449
  translations followed whichever row their translator read. Each term now
  reads the same everywhere, and `tests/test_i18n_glossary_terms.py` keeps it
  so. The Starting Points search label reads **Starting Point Search** (was
  "Initial Seed Search"), like the rest of the interface.
- **`pip install al-dic` is lighter.** scikit-image is no longer a dependency:
  nothing imported it. Python 3.13 and 3.14 are now tested.
- **`examples/` is split by audience.** `examples/quickstart/` holds the image
  sequences for running the software; `examples/scripting/` holds the batch and
  plotting examples for driving it from Python. Scripts move to
  `examples/scripting/batch_process.py` and
  `examples/scripting/plot_results.py`; the example configs move with them and
  their sample paths are now relative to the repository root.
- The desktop app keeps its logs in each platform's own place:
  `%LOCALAPPDATA%\pyALDIC\logs` on Windows, `~/Library/Application
  Support/pyALDIC/logs` on macOS.

### Fixed

- **A fixed colour range was scaled twice on export** with physical units on:
  a range typed as +-200 um exported as +-60000 um.
- **The hidden-background fill could not be set**, and once it could, the
  strain window reverted it on the next frame change.
- **Clipped and untranslated labels.** "Show background image" in German, the
  colorbar's top label and scientific-notation ticks, and the parameter labels
  in French and Spanish ("Niveau de raffineme...") no longer lose their tails;
  the Region of Interest *Add* and *Cut* shape menus were English in every
  language.

## [0.8.0] — 2026-08-25

pyALDIC now ships as a Windows application you unzip and double-click, and a
round of export bugs that were losing data without saying so.

### Added

- **A portable Windows bundle.** Download `pyALDIC-<version>-win64.zip` from
  the release page, unzip it, and double-click `pyALDIC.exe`. No Python, no
  installer, no administrator rights. A `pyALDIC-console.exe` sits beside it
  for when you need to see a traceback. The wheel and source distribution are
  unchanged and still published to PyPI.
- **Compute kernels now compile in the background at startup.** The solver is
  JIT-compiled on first use and cached, so the cost is paid once per
  installation — but it used to land on the first click of *Run DIC Analysis*,
  freezing the interface for tens of seconds with nothing to explain it. On a
  24-thread workstation the first correlation took 32.1 s without a warm-up and
  0.1 s with one. Compilation now starts shortly after the window opens and
  says so in the log; the interface is a little less responsive while it runs,
  which is a better trade than a dead window after a button press.
- **A log file for frozen builds**, at
  `%LOCALAPPDATA%\pyALDIC\logs\pyALDIC.log`, and a dialog that reports an
  unhandled error and names that file. A windowed build has no console, so
  previously a startup failure produced no window and no message of any kind.

### Fixed

- **Exporting images from a folder with non-ASCII characters in its path wrote
  nothing, and reported success.** `cv2.imwrite` cannot open such a path on
  Windows; it returns `False` rather than raising, and the return value was
  discarded, so the dialog counted files that were never written. The folder
  did not have to be non-ASCII itself — the filename prefix comes from the name
  of your image folder. Reading was affected the same way, giving every
  exported PNG, JPEG, MP4 and GIF a blank background. Both now go through
  `cv2.imencode`/`imdecode`, the approach used elsewhere in the codebase.
- **Export errors were invisible.** Every handler that reports a failed image
  or animation export referenced a palette colour that does not exist, so the
  handler itself raised `AttributeError` before it could display anything. A
  test now asserts that every palette token referenced anywhere is defined.
- **A failed animation export was reported as a success**, in green, reading
  "Exported 0 animation(s)". A video encoder that cannot open — a missing
  FFmpeg backend, a read-only destination — now raises and is shown as an
  error.
- **Opening a session required write permission on the folder containing it**,
  because the read path created its temporary file next to the input. Sessions
  on read-only shares, mounted images and write-protected folders now open.
- **The `.aldic` file association is registered correctly from a frozen
  build.** It previously wrote a command containing `-m al_dic`, which only a
  Python interpreter can honour, and reported an association as valid even when
  it pointed at an executable that no longer exists.
- Two file dialogs (save Region of Interest mask, import mask image) appeared
  in English regardless of the selected language. The mask dialog also proposed
  a bare relative filename, which resolves against the working directory —
  which for an application launched from a shortcut is not your project folder.

## [0.7.2] — 2026-07-27

Session save/load fidelity, and a crash on closing the export dialog. Both came
from user reports. No algorithm changes — results are identical to 0.7.1.

### Fixed

- **Closing the export dialog while an export was running killed the whole
  application** — every window vanished and Python exited. The export workers
  are threads owned by the dialog and nothing stopped them on close, so a
  running thread was torn down under Qt, which aborts the process. Close, Esc,
  the window X and OK now stop the export and wait for it; a worker that
  outlasts the wait is detached and left to finish on its own rather than
  freezing the interface. Affects image *and* animation export, from both the
  main window and the strain window.
- **A reopened session behaved as if it had no Region of Interest.** Save
  reported an empty mask while Run computed happily, and — worse — editing
  silently destroyed the restored ROI: Invert replaced it with the whole image,
  and drawing a stroke replaced it with just that stroke. The editing buffer is
  now filled from the restored state as soon as a session is applied.
- **The refinement brush was never saved.** A session with painted refinement
  zones came back with the refinement settings intact but the zones gone, so a
  re-run built a different mesh and produced different results without saying
  so. Brush zones are now stored with the session; older sessions still load.
- `show_subset_window` was written into a session but never read back — the
  restore path kept its own copy of the key list and had drifted from the save
  side. Both now derive from one list.
- **A frame with no Region of Interest of its own now says so.** A session
  restores the frame that was on screen when it was saved, and the ROI toolbar
  acts on the current frame while the solver falls back to frame 1's ROI — so
  on any other frame the toolbar looked broken and blamed the mask. It now
  names the frame, says frame 1's Region of Interest is what gets computed, and
  offers both ways forward. Invert refuses there instead of handing that frame
  the whole image as a new per-frame Region of Interest.
- ROI toolbar messages are translated; they previously appeared in English
  inside an otherwise translated interface.

## [0.7.1] — 2026-07-26

Scripted batch processing, without the GUI. No changes to the DIC or strain
algorithms, and none to the interface — results are identical to 0.7.0.

### Added

- **`examples/batch_process.py`** — run several samples end to end from one
  JSON or YAML config: per-sample images, ROI, parameters and export formats.
  A failing sample is logged and the batch continues; samples that already
  have results are skipped, so an interrupted run resumes where it stopped.
  ROI is optional (whole image), one mask for every frame, or one mask per
  frame for a moving or growing region. YAML needs PyYAML; JSON works with
  the standard library, so the package itself gains no dependency.
- **`examples/plot_results.py`** — replot any field from an exported `.npz`
  after the fact. A batch can export data only (faster, smaller, and free of
  any Qt import) and plotting stays a later, repeatable decision.
- Both example configs document the subset-size convention: the GUI shows an
  odd subset size (DIC convention, one centre pixel) while `DICPara.winsize`
  is the even internal value — a GUI subset of 31 px is `winsize: 32`.

### Fixed

- Field-image export raised `ufunc 'bitwise_and' not supported` when the ROI
  mask needed no resizing: `_resize_mask` returned its input unconverted, and
  masks are `float64` because that is what `run_aldic` consumes. It now always
  returns `bool`, so the same mask array works in both calls.
- The README's programmatic-API example failed as written — it left
  `gridxy_roi_range` at its zero-size default (the GUI fills that in from the
  ROI you draw) and raised `No grid points generated`. It now derives the ROI
  from the mask bounding box and notes the power-of-2 `winstepsize` rule.
- An empty node grid blamed `winsize` unconditionally, reporting
  `winsize (32) too large for image (800x800)` for a 32 px subset on an
  800×800 image whose real problem was an unset ROI. The message now
  distinguishes an empty ROI from a genuinely oversized subset and says what
  to set.
- The README's adaptive-refinement animation showed an unbalanced quadtree —
  finest cells directly abutting base cells, which would put several hanging
  nodes on one edge. It now keeps a one-level transition ring (2:1 balance).

## [0.7.0] — 2026-07-20

Two themes: **crack-aware DIC end to end** — the mesh, the plane-fit
neighbours, the cumulative transform, edge-trim and field rendering all
respect a crack (including one that grows across frames) instead of smearing
displacement / strain across the discontinuity — and a **~10× faster strain
step** on dense meshes and long sequences.

### Performance

- **Plane-fit strain is ~10× faster on dense meshes / many frames.** The
  per-node weighted least squares (previously `np.linalg.lstsq` / SVD per node)
  is now a parallel Numba kernel solving the 3×3 normal equations; the
  geometric neighbour search is built once per sequence and reused every frame
  (coordinates are frame-invariant — strain is total-Lagrangian); and the
  per-frame boundary distance is evaluated at the nodes via a boundary-pixel
  KD-tree instead of two full-image Euclidean distance transforms. Numerically
  identical to before (~1e-15). On a 50k-node, 5 MP, 300-frame run the strain
  step drops from ~10 s/frame to ~1 s/frame.

### Added

- **Crack-aware everything.** For cracked / notched specimens: the FEM mesh is
  cut at thin continuous mask barriers; plane-fit neighbours and the cumulative
  transform never span an open crack; field rendering draws the two crack faces
  independently instead of interpolating a smooth ramp across the
  discontinuity; and a growing crack is re-trimmed at its current position every
  frame (warped back to the reference configuration).
- **"Fill trimmed edges (display only)" toggle** in the strain visualization
  controls (off by default). Fills the trimmed edge band back in from reliable
  interior nodes — interpolated in the interior, extrapolated out to the ROI
  edge, and recovered face-by-face along a crack inner edge (each side filled
  from its own side only, so the crack stays a sharp line). Affects the on-canvas
  view and exported **images / animations** only; exported **data** files
  (NPZ / MAT / CSV) always keep the trimmed edge as `NaN`.
- **"Strain window ≈ N×N nodes" readout** under the VSG size control: shows how
  many mesh data points the plane fit spans on a uniform mesh
  (2·⌊radius / step⌋ + 1 per axis), updated live as VSG or the DIC step change.
- **Cancel button for strain computation**, and **DIC Cancel now keeps the
  already-computed frames** (partial results) instead of discarding them.

### Changed

- **Plane fitting is the primary default strain method** (local weighted plane
  fitting); edge-trim validity is produced by plane fitting.
- **Reference vs deformed geometry is now frame-consistent** in the strain
  window: *not deformed* uses frame-0 geometry (matching the main window's
  displacement overlay), *show on deformed* uses the current frame's geometry —
  so edge-trim, masking and fill all follow the displayed frame.
- **Default initial-guess method is now FFT.**
- **BREAKING — exported variable structure unified across NPZ / MAT / CSV.**
  Displacement / strain arrays now share one consistent naming and world-
  coordinate convention across all three formats; scripts that parsed the old
  per-format layout need updating (see the export section of the manual).

### Fixed

- Growing / per-frame cracks: displacement and strain no longer smear across an
  open crack (crack-aware cumulative transform); the edge-trim shows in the
  display and uses the correct reference mask; a crack thinner than one element
  now cuts the mesh.
- Strain export no longer silently yields an all-NaN field — it raises with an
  actionable message when edge-trim removes every node; edge-trimmed strain is
  blanked (not back-filled) in exported images / animations; NaN deformed node
  positions are guarded in PNG / animation export.
- Session robustness: **Export is enabled immediately after opening a session**
  that already contains results (no recompute); moved image files are relocated
  on session open (auto-find + prompt).
- Seed propagation auto-rescues unseeded regions and keeps partial results on
  failure. GUI: minimum window size lowered for smaller screens; the Export and
  Batch-Import dialogs are kept on screen.

## [0.6.0] — 2026-07-02

This release is about surviving big jobs: full session persistence
(computed results included), an export pipeline that is ~35× smaller
and ~5× faster, and streaming memory management for large image sets.

### Added

- **Full session persistence — results included.** `File → Save
  Session` now writes the whole project — parameters, per-frame ROIs,
  view state, **and computed results** — into a single `.aldic`
  bundle (results deduplicated, stored pickle-free). `File → Open
  Session` restores the exact page — same frame, field, colormap and
  ranges — with no recompute, so hours of computation survive closing
  the GUI. Save / load run asynchronously behind a progress dialog
  with per-array progress for GB-scale writes; legacy v0.5 session
  JSON files still load.
- **Windows `.aldic` double-click association (optional).**
  `File → Associate .aldic files` registers the extension so a saved
  session opens straight into pyALDIC.
- **"Preview & Colorbar" tab in the Export dialog.** A WYSIWYG
  preview rendered through the real export path, plus colorbar
  styling: position (right / left / top / bottom), font size, font
  family, bar thickness, background (black / white), and outward
  margin. Field appearance (colormap / range / opacity) is two-way
  synced with the Images tab, an **Apply to all fields** button
  copies the styling across fields, and labels respect the
  configured physical units.
- **Export resolution presets & formats.** Long-edge presets
  512 / 768 / 1024 / 1536 / 2048 / native; **JPEG is the new
  default** with a quality control, PNG / TIFF still available.
- **GIF / MP4 frame-step decimation.** Keep every Nth frame while
  preserving the playback duration.

### Changed

- **Export is ~35× smaller and ~5× faster per image** — LUT colormap,
  binary-alpha compositing, and rendering at the output resolution.
  A 90-image 4K batch drops from **3.3 min / 2.8 GB** to
  **40 s / 80 MB**.
- **Streaming animation writer.** GIF / MP4 frames encode one at a
  time instead of accumulating in RAM; GIFs come out ~8× smaller.
- **Frames stream on demand.** A `FrameProvider` replaces the four
  full float64 image stacks previously materialised at Run click —
  about **40 GB less RAM** on a 300-image 4K dataset.
- **LRU-bounded caches.** `ref_cache`, the IC-GN / subproblem-1
  precompute caches, and the RGB preview cache are now bounded, so
  incremental-mode memory stays flat instead of growing O(N) with
  frame count. The compute worker also drops a redundant image copy
  and stores ROI masks as uint8.
- **Color range mode is an explicit choice.** An (o) Auto / ( ) Fixed
  radio pair replaces the lone "Auto" checkbox in the main window,
  the strain window, and the export dialog.

### Fixed

- **Numeric input is OS-locale-proof.** On comma-decimal systems
  (de / fr / es / pt / it / ru), typing `0.070` into a range box was
  silently parsed as `70`. Both `0.07` and `0,07` now parse as 0.07
  everywhere.
- **Strain window opens fitted to the screen.** It was a fixed
  1280×800, which overflowed small laptops and could not be shrunk
  on macOS.
- **Run progress bar no longer jumps 90 → 100** at the end of a run.

### i18n

- All 8 locales (en, zh_CN, zh_TW, ja, ko, de, fr, es) at **100%
  coverage**, including every new session / export / range string.

## [0.5.0] — 2026-06-21

### Added

- **Three-point circle ROI tool.** In addition to the center+radius circle,
  an ROI circle can now be defined by clicking **three points on the circle's
  edge** (the circumcircle), available in both the **Add** and **Cut** shape
  menus. After the first two points a live preview circle follows the cursor;
  the third click commits. Collinear or near-collinear points are rejected
  with a warning instead of producing a degenerate (infinite-radius) circle.
- **Low-confidence strain edge trimming (plane-fit method).** A new
  **Trim low-confidence edges** control in the Strain panel hides strain at
  ROI / hole edges where the VSG window crosses the boundary and the local
  plane fit becomes one-sided and unreliable. A coefficient α (0–1, default
  **0.70**) sets the trimmed band width (≈ α × VSG radius); **0** disables
  trimming (legacy behaviour). A live *"Trimmed: N nodes (M%)"* readout shows
  how many nodes are removed. The trim is applied consistently to the
  on-screen field and to every export (PNG / MAT / NPZ / CSV / report /
  animation). The default 0.70 is calibrated against a synthetic benchmark:
  plane-fit error rises sharply within the boundary band and returns to the
  interior baseline at ≈ 0.7 × VSG radius.

## [0.4.3] — 2026-06-15

Publication release accompanying the *SoftwareX* submission. This is the
citable snapshot archived to Zenodo and referenced from the manuscript;
it is functionally identical to 0.4.2 at the API and GUI level.

### Fixed

- **CI:** corrected a YAML syntax error in the i18n workflow's Gate B
  heredoc. Internal only — no user-facing behaviour change.

## [0.4.2] — 2026-05-20

This release focuses on the **batch ROI import** workflow used by
incremental tracking, plus a layout fix for the strain
post-processing window on small laptop screens.

### Added

- **Live preview panel in the batch-import dialog.** A third column
  in `BatchImportDialog` shows the currently focused frame's image,
  optionally with the assigned mask drawn as a coloured semi-
  transparent overlay. Three view modes (image only, image + mask,
  mask only), an α slider 0–100%, and four overlay colours
  (Blue / Red / Green / Yellow). Scroll wheel zooms (cursor anchored)
  and left-drag pans. Image and mask buffers are LRU-cached so
  flipping through frames is responsive.
- **1:N mask-to-frame broadcast.** In the batch-import dialog,
  selecting one mask plus several frames now assigns the same mask
  to every selected frame. The reverse (multiple masks → one frame)
  is rejected with a warning dialog because the relationship is
  one-mask-per-frame only.
- **Frame-list filtering by reference schedule.** The batch-import
  dialog now lists only the frames the pipeline will actually
  consume a mask for: `{0}` in accumulative mode, and
  `FrameSchedule.ref_frame_set` (under your Reference Update
  setting) in incremental mode. Non-reference frames in every-N /
  custom incremental modes no longer appear, because the solver
  only reads `para.img_ref_mask = masks[ref_idx]` per pair.
- **Pre-flight mask size check.** After picking a mask folder, every
  file is pre-scanned for size match against the loaded images.
  Mismatched files are disabled in the list (greyed, unselectable,
  tooltip explains the mismatch) and excluded from every assignment
  path. A warning bar above the panels summarises the count.
- **0/1 mask encoding support.** Mask files written as NumPy bool
  arrays cast to uint8 (values in `{0, 1}`) are now correctly
  binarised. Previously the hardcoded `> 127` threshold reduced
  every nonzero pixel to background, silently producing an all-False
  ROI. Both 0/1 and 0/255 encodings are now auto-detected.
- **Collapsible right column in the strain window.** Each of the
  five panels (`Strain Parameters`, `Field`, `Visualization`,
  `Physical Units`, `Log`) is now a `CollapsibleSection`. Default
  expansion: parameters / field / visualization expanded; physical
  units and log collapsed. Action buttons (Compute Strain, Export
  Results) sit outside the sections so they stay accessible no
  matter which sections are folded.

### Fixed

- **Strain window no longer overflows small laptop screens.** The
  right column previously stacked all five panels in a flat
  ~950 px pile, and the bottom panels (`Log`, `Physical Units`)
  sat off-screen on 1366×768 / 1280×800 laptops. The column is now
  wrapped in a `QScrollArea` and the default window size is reduced
  from 1340×960 to 1280×800, so the most-used controls remain on
  screen and a scrollbar takes over for any smaller resolution.
- **Mismatched mask sizes no longer silently resized at batch-import
  time.** Previously a mask whose on-disk dimensions did not match
  the loaded images was silently `cv2.resize`-d (nearest neighbour)
  to the image shape. This could turn a 256×256 mask into a 2048×2048
  ROI without warning. The dialog now flags every mismatched mask
  up front and excludes them from every assignment path.
- **i18n leaks in batch-import success log.** The per-frame
  "Imported mask for frame …" notice and the loaded-count summary
  now route through `tr()` / `tr_args` instead of bare f-strings,
  so localisations display the translated text.
- **`test_perf_init_mode_compare[seed_propagation]` no longer fails
  on bare synthetic data.** The graceful-skip path was broken when
  the `seed_set` requirement check moved from solver runtime into
  `validate_dicpara`, which fires inside `dicpara_default()` and
  raised before the test's `try/except` could catch it. The
  `_build_para()` call now sits inside the guarded block, restoring
  the original skip-on-missing-seed behaviour.

### i18n

- **15 new GUI strings**, all filled for `zh_CN` (100% coverage).
  Other six locales (`zh_TW`, `ja`, `ko`, `de`, `fr`, `es`) leave
  the new strings as `<translation type="unfinished">` per the
  project's i18n contract; Qt falls back to the English source so
  the UI remains usable in every locale.

## [0.4.1] — 2026-04-22

### Fixed

- **Plane-fit strain no longer silently returns a field of zeros when
  the VSG radius is too small.** Previously, if the VSG size was set
  below the DIC node spacing (e.g. VSG = 3 px with subset_step = 8 px,
  giving VSG radius = 1 px), every node's plane fit would find fewer
  than 3 valid neighbours inside the radius; `comp_def_grad` would
  return an all-NaN deformation gradient; `fill_nan_idw` would hit
  its all-NaN branch and quietly return a zero-filled array; and
  downstream strain computation would produce a "converged" result
  of **εxx = εyy = εxy = 0 everywhere** — indistinguishable from a
  legitimate no-strain run. The FEM nodal method was not affected
  because it integrates over element connectivity, not a radius.

### Added

- **`fill_nan_idw` opt-in `on_all_nan="raise"`.** New keyword-only
  argument; default `"zeros"` preserves the previous behaviour
  (warn + return zeros) for backward compatibility. The strain
  plane-fit path now passes `"raise"` so the GUI surfaces an
  actionable error in the LOG panel and a "Strain compute failed"
  dialog, instead of masquerading a bad configuration as a valid
  zero-strain result.
- **`StrainParamPanel` inline warning.** When Method = Plane
  fitting AND VSG radius < DIC node spacing, an amber warning
  below the VSG spin box now reads: *"⚠ VSG radius (X px) < DIC
  node spacing (Y px); plane fit will fail. Use VSG ≥ Z px or
  switch Method to FEM nodal."* Updates live when either VSG or
  subset_step changes.

## [0.4.0] — 2026-04-20

### Added

- **Multi-language support — 8 locales at 100% coverage.** pyALDIC's
  GUI now ships with full translations for English (source),
  简体中文 (zh_CN), 繁體中文 (zh_TW), 日本語 (ja), 한국어 (ko),
  Deutsch (de), Français (fr), and Español (es) — 273 user-visible
  strings per language. Choose a language from the new
  **Settings → Language** menu; the choice is persisted via
  `QSettings` and applied at next startup. System locale is picked
  up automatically on first launch with a fallback chain for
  regional variants (zh_HK → zh_TW, pt_BR → es, en_GB → en, etc.).
- **`al_dic.i18n` runtime package** — `LanguageManager` wraps
  `QTranslator` installation / persistence, plus a dev-only pseudo
  locale (`"pseudo"`) that wraps every `tr()` string in `⟦…~~~⟧`
  to surface missed wrappers and widgets that can't absorb +30%
  text expansion.
- **`al_dic.utils.locale_format`** — `QLocale`-based helpers for
  numbers, dates, file sizes, and durations so UI output respects
  the active language's conventions (e.g. `3,14` vs `3.14`).
- **`al_dic.utils.matplotlib_fonts`** — configures a CJK-aware
  font fallback chain at app startup so Chinese / Japanese / Korean
  axis labels and titles embedded in matplotlib figures render with
  real glyphs instead of tofu blocks.
- **CJK-aware Qt stylesheet.** The global `font-family` chain in
  `gui/theme.py` now includes Microsoft YaHei UI, PingFang SC,
  Hiragino Sans GB, Source Han Sans SC, Noto Sans CJK SC,
  Yu Gothic UI, and Malgun Gothic, so CJK strings render with a
  modern sans face on every major platform.
- **Translator workflow + docs** — `tools/i18n.py` (extract /
  compile / stats / add-lang CLI), `docs/i18n/glossary.md` with
  fixed DIC-term translations per language, and CLAUDE.md §i18n
  rules R1–R8 covering placeholder formatting, `&` mnemonics,
  context disambiguation, no `setFixedWidth`, and locale-aware
  number formatting.
- **End-to-end demo video** — `assets/pyALDIC_demo.gif` (78 s,
  800×586, 11 MB, README-embedded) and `assets/videos/pyALDIC_demo.mp4`
  (1920×1080 @ 30 fps, 14 MB) walk through the entire workflow:
  import → pick workflow → draw / batch-import ROI → refine mesh →
  run DIC → displacement / strain fields. Step 2 and step 5 use a
  2× centered zoom with frosted-glass background to highlight the
  sidebar and Run / Progress panel respectively.
- **`docs/i18n/glossary.md`** — DIC terminology table spanning en,
  zh_CN, zh_TW, ja, ko, plus a list of proper nouns / acronyms
  (pyALDIC, AL-DIC, IC-GN, ADMM, FFT, NCC, FEM, MAT, NPZ, CSV, PNG,
  GIF, MP4, PDF) that must stay English in every locale.
- **`tools/demo_video/build_demo.py`** — reproducible pipeline that
  assembles the README demo from per-step recordings: title card
  with pyALDIC icon + wordmark, step cards with fade in/out
  transitions, persistent top-bar "Step N — title" overlay,
  10× speed-up on the compute step, and step-4 center-zoom finale.

### Changed

- **README landing page overhauled.** The comparison-with-DIC-tools
  table was rewritten for academic honesty (dropped the algorithm /
  regularization / export-format rows that overstated pyALDIC's
  edge, added a technical 3D-stereo / subset-shape-function row
  where pyALDIC is *not* the winner, footnotes on the Ncorr
  MATLAB-license dependency and on the public-web-sourced release
  info). Green highlights now wrap the pyALDIC column with `<mark>`
  to read as *"this is the pyALDIC column"* rather than *"this is
  where we win"*. The **Accuracy** section now points readers to
  the two peer-reviewed AL-DIC papers (Yang & Bhattacharya 2019,
  Tong et al. 2025) and the DIC Challenge 2.0 community benchmark
  instead of reporting self-graded RMSE numbers. A new
  **About the Authors** section records iDICs *Good Practices
  Guide for Digital Image Correlation* editorial roles separately
  from the algorithmic accuracy claims.
- **README banner embeds the pyALDIC app icon** alongside the
  wordmark so the README, the installed desktop icon, and the
  taskbar/Alt-Tab identity all read as the same brand.
- **Widget combobox state sync uses `userData` instead of
  `currentText`.** Workflow-type, solver, and reference-update
  dropdowns now store stable English codes (`"incremental"`,
  `"accumulative"`, `"aldic"`, `"local"`, `"every_frame"`, …) as
  item userData while the visible text is localised. This
  decouples display from state-sync so translating the dropdown
  labels never breaks backend behaviour.
- **Default "Save Session" / "Open Session" dialog filters** are
  translated. The `*.aldic.json` glob stays invariant (rule R6),
  only the description text (*"pyALDIC Session"*, *"All Files"*)
  is localised.
- **GitHub repo description** updated from the outdated
  *"refactored and optimized STAQ-DIC implementation"* to
  *"pyALDIC: Augmented Lagrangian Digital Image Correlation in
  Python"*.

### Fixed

- **`AttributeError: 'str' object has no attribute 'arg'`** when
  any widget with a `%1 / %2`-style translated string rendered.
  PySide6's `self.tr()` returns plain `str`, not Qt's `QString`,
  so the `.arg()` chaining idiom from C++ Qt crashes at runtime.
  Introduced `al_dic.i18n.tr_args(text, *values)` to perform the
  same placeholder substitution on a Python `str`, and updated
  every call site (`app.py`, `right_sidebar.py`,
  `init_guess_widget.py`, `param_panel.py`, `roi_hint.py`).
- **`AttributeError: 'PipelineController' object has no attribute
  'tr'`** during `log_message.emit(...)` in the pipeline
  controller. Non-`QObject` classes do not inherit `tr()`; switched
  to `QCoreApplication.translate("PipelineController", …)`.
- **`pyside6-lupdate` dropped backslashes from `\uXXXX` escape
  sequences** inside `tr()` string literals, producing corrupt
  `"Open Sessionu2026"` entries in the `.ts` file. Every GUI
  source file now uses literal Unicode characters (`…`, `—`,
  `×`, `▶`, `⏸`, `⚠`) instead of escape sequences, matching
  what lupdate can actually extract.
- **Ugly CJK rendering on Windows.** Falling back on the default
  system CJK face produced an inconsistent, low-weight look; the
  new font-family chain picks up Microsoft YaHei UI and renders
  modern, consistent Chinese / Japanese / Korean glyphs.
- **`pyside6-lupdate` does not recurse directories for Python
  sources.** `tools/i18n.py extract` now enumerates every `.py`
  file under `src/al_dic/gui/` explicitly before passing them to
  lupdate, instead of relying on directory-walking that silently
  returned zero strings.

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
