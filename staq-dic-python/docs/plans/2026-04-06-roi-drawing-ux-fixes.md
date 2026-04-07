# ROI Drawing UX Fixes (Q1-Q6) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix six interrelated ROI drawing UX bugs by eliminating the "three-frame drift"
between `state.current_frame`, `state.roi_editing_frame`, and `self._roi_ctrl.mask`,
unifying them on `current_frame` as the single source of truth.

**Architecture:**
The ROI editing workflow currently has three independent "current frame" pointers
that can drift apart, causing the user to draw on the wrong frame, see stale overlays,
or have the progress bar / image list desync from the canvas. We delete the
`roi_editing_frame` field entirely and route every ROI write through
`state.current_frame`. The blue ROI overlay becomes a function of
`per_frame_rois[current_frame]` instead of `roi_ctrl.mask`, so external mutations
(batch import, context-menu import) are visible immediately. We also fix the mesh
overlay to enter the *preview* path during ROI editing, hide it for non-frame-0
edits, and clear stale pipeline results when image lists are mutated.

**Tech Stack:**
- PySide6 (Qt6) — `QGraphicsScene` / `QGraphicsView` layered canvas
- AppState singleton + Qt Signals (no direct panel-to-panel coupling)
- pytest + monkeypatch for GUI tests using a `_make_canvas_shim` fixture pattern
- Existing tests live in `tests/test_gui_*.py` and `tests/test_gui/test_*.py`

---

## Background: The Six Questions

| ID | Question                                                                                 | Decision                                                                                                      |
|----|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| Q1 | When the user clicks an image-list "Edit" button, does `current_frame` follow?           | YES — `current_frame`, `roi_editing_frame`, and `roi_ctrl.mask` must all sync.                                 |
| Q2 | When the user clicks the toolbar "Draw" button, which frame is drawn on?                 | The currently-displayed frame (i.e. `current_frame`).                                                          |
| Q3 | What does the blue overlay show after a *batch* import?                                  | `per_frame_rois[current_frame]` — immediately, no merge with the previous in-memory ROI controller buffer.    |
| Q4 | During ROI editing with results loaded, does the mesh come from results or from preview? | Preview — the user is editing the ROI, not inspecting the result.                                              |
| Q5 | Should the mesh overlay be visible while editing a *non-frame-0* ROI?                    | NO — preview mesh is only meaningful on frame 0; hide for K≠0.                                                 |
| Q6 | When does pipeline state get cleared?                                                    | Image order/count change (load, delete, reorder) → clear results. Tracking-mode change → keep results.        |

## Architectural Fix: Eliminate `roi_editing_frame`

**Before:**
```
state.current_frame      ← image_list selection / progress bar
state.roi_editing_frame  ← roi toolbar write target (drifts!)
self._roi_ctrl.mask      ← canvas blue overlay source (drifts!)
```

**After:**
```
state.current_frame      ← single source of truth
                            • image_list selection
                            • progress bar
                            • canvas display target
                            • ROI write target
state.per_frame_rois[k]  ← persisted boolean masks (single source for the blue overlay)
self._roi_ctrl           ← stamping buffer for in-progress draws
                            (loaded from per_frame_rois[current_frame] on entry / frame change;
                             saved back after each shape commit)
```

The single rule: **the user is always editing `current_frame`'s ROI**. To edit
frame K, navigate to frame K (whether via image-list click, "Edit" button, or
arrow keys). The toolbar Draw/Clear/Import buttons always operate on
`current_frame`. The image-list "Edit" button is just a convenience that switches
the frame *and* opens ROI editing in one click.

---

## Test File Layout

All Q1-Q6 fixes share a single new test module:

- `tests/test_gui/test_roi_drawing_ux.py` — new file, covers Q1-Q6 contracts.

We reuse the existing helpers from `tests/test_gui_pipeline_incomplete_roi_dialog.py`
and `tests/test_gui_canvas_results_mesh.py`:
- `_safe_disconnect`, `qapp`, `_reset_singleton` fixtures
- `_make_canvas_shim` pattern for unit-testing canvas methods without a real window
- Direct `AppState.instance()` mutation + signal emission

---

## Task 1 — Q1/Q2: Pin "image-list Edit click syncs current_frame" (RED)

**Files:**
- Create: `tests/test_gui/test_roi_drawing_ux.py`

**Step 1: Write the failing test**

```python
"""Q1-Q6 ROI drawing UX contracts.

These tests pin the post-fix invariant that ``state.current_frame`` is the
single source of truth for "which frame's ROI am I editing".  See
``docs/plans/2026-04-06-roi-drawing-ux-fixes.md``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from staq_dic.gui.app_state import AppState


def _safe_disconnect(signal) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            signal.disconnect()
        except (RuntimeError, TypeError):
            pass


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _reset_singleton():
    state = AppState.instance()
    for sig in (
        state.images_changed, state.current_frame_changed,
        state.roi_changed, state.params_changed,
        state.run_state_changed, state.progress_updated,
        state.results_changed, state.display_changed,
        state.log_message,
    ):
        _safe_disconnect(sig)
    state.reset()
    yield


class _StubImageController:
    def __init__(self, shape: tuple[int, int]) -> None:
        self._img_rgb = np.zeros((*shape, 3), dtype=np.uint8)

    def read_image_rgb(self, _idx: int) -> np.ndarray:
        return self._img_rgb.copy()

    def read_image(self, _idx: int) -> np.ndarray:
        return np.zeros(self._img_rgb.shape[:2], dtype=np.float64)


def _make_main_window(qapp):
    """Build a real MainWindow for end-to-end signal-flow tests."""
    from staq_dic.gui.app import MainWindow
    win = MainWindow()
    win._image_ctrl = _StubImageController((128, 128))
    state = AppState.instance()
    state.image_files = [f"/fake/img{i}.tif" for i in range(5)]
    state.images_changed.emit()  # triggers _init_roi_controller via stub
    return win


class TestQ1Q2_FrameSync:
    def test_image_list_edit_button_syncs_current_frame(self, qapp):
        """Clicking 'Edit' on frame K must set current_frame=K."""
        win = _make_main_window(qapp)
        state = AppState.instance()
        assert state.current_frame == 0

        # Simulate clicking "Edit" on frame 3
        win._on_roi_edit_for_frame(3)

        assert state.current_frame == 3
        assert state.roi_editing is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ1Q2_FrameSync::test_image_list_edit_button_syncs_current_frame -v`

Expected: FAIL — `_on_roi_edit_for_frame` currently sets `roi_editing_frame` but
not `current_frame`.

**Step 3: Implement the fix**

Edit `src/staq_dic/gui/app.py` `_on_roi_edit_for_frame`:

```python
def _on_roi_edit_for_frame(self, frame: int) -> None:
    """Enter ROI editing mode for a specific frame.

    Single source of truth: ``current_frame`` *is* the editing frame.
    Switching frames here keeps the image list selection, progress bar,
    and canvas display all in sync with what the user is editing.
    """
    state = self._state
    if not state.image_files:
        return
    if self._roi_ctrl is None:
        self._init_roi_controller()
    if self._roi_ctrl is None:
        return
    state.set_current_frame(frame)
    self._load_roi_buffer_for_current_frame()
    state.roi_editing = True
    state.display_changed.emit()
```

Add a new helper alongside it:

```python
def _load_roi_buffer_for_current_frame(self) -> None:
    """Mirror per_frame_rois[current_frame] into the ROI controller buffer.

    Called on entry to ROI editing and whenever current_frame changes
    during editing.
    """
    if self._roi_ctrl is None:
        return
    state = self._state
    existing = state.per_frame_rois.get(state.current_frame)
    if existing is not None:
        self._roi_ctrl.mask = existing.copy()
    else:
        self._roi_ctrl.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ1Q2_FrameSync::test_image_list_edit_button_syncs_current_frame -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_gui/test_roi_drawing_ux.py src/staq_dic/gui/app.py
git commit -m "test+fix(gui): Edit-button click syncs current_frame (Q1)"
```

---

## Task 2 — Q1/Q2: Pin "draw button targets the displayed frame" (RED)

**Files:**
- Modify: `tests/test_gui/test_roi_drawing_ux.py` (add to TestQ1Q2_FrameSync)
- Modify: `src/staq_dic/gui/app.py` (`_on_draw_requested`)

**Step 1: Add the failing test**

```python
    def test_draw_button_targets_current_frame(self, qapp):
        """After navigating to frame 2 and clicking Draw Rect, drawing must
        commit to frame 2 — not the previous edit target or frame 0.
        """
        win = _make_main_window(qapp)
        state = AppState.instance()

        # User clicks frame 2 row in image list
        state.set_current_frame(2)

        # User clicks "Draw Rect" toolbar button
        win._on_draw_requested("rect", "add")

        # Now whatever the user draws should commit to frame 2.
        # Simulate by stamping a rect into the controller and finishing.
        win._roi_ctrl.add_rectangle(10, 10, 50, 50, "add")
        win._canvas_area.canvas._finish_drawing()

        assert 2 in state.per_frame_rois
        # And no other frame got the mask
        assert 0 not in state.per_frame_rois
        assert 1 not in state.per_frame_rois
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ1Q2_FrameSync::test_draw_button_targets_current_frame -v`

Expected: FAIL — without the fix, `_finish_drawing` reads `state.roi_editing_frame`
which still holds 0.

**Step 3: Implement**

This task depends on Task 5 (delete `roi_editing_frame`), but we can make it pass
now by having `_on_draw_requested` first sync the buffer to the current frame:

```python
def _on_draw_requested(self, shape: str, mode: str) -> None:
    """Activate one-shot drawing mode on the canvas.

    The toolbar Draw button always edits the *currently displayed* frame.
    To edit a different frame, navigate there first (image list click,
    arrow keys, or per-frame Edit button).
    """
    self._load_roi_buffer_for_current_frame()
    self._enter_roi_editing()
    canvas = self._canvas_area.canvas
    canvas.set_drawing_mode(mode)
    canvas.set_tool(shape)
```

**Step 4: Run test**

Expected: still FAIL because `_finish_drawing` writes to `state.roi_editing_frame`,
not `state.current_frame`. We will fix that fully in Task 5. For now, mark this
test with `pytest.mark.xfail(reason="awaits Task 5: drop roi_editing_frame")` and
commit. We will remove the xfail in Task 5.

**Step 5: Commit**

```bash
git add tests/test_gui/test_roi_drawing_ux.py src/staq_dic/gui/app.py
git commit -m "test+wip(gui): draw button reloads buffer for current_frame (Q2)"
```

---

## Task 3 — Q1/Q2: Drop `roi_editing_frame` field entirely (RED → GREEN)

**Files:**
- Modify: `src/staq_dic/gui/app_state.py` (remove field, update `_init_state`, `set_image_files`)
- Modify: `src/staq_dic/gui/app.py` (replace all reads with `state.current_frame`)
- Modify: `src/staq_dic/gui/panels/canvas_area.py` (replace all reads)
- Modify: `tests/test_gui/test_app_state.py` (delete the two `roi_editing_frame` tests)
- Modify: `tests/test_gui/test_roi_drawing_ux.py` (remove xfail from Task 2 test)

**Step 1: Search for every reference**

Run: `grep -rn "roi_editing_frame" src tests`

Expected hits (post-Task 2):
- `src/staq_dic/gui/app_state.py` — field declaration + `set_image_files` reset
- `src/staq_dic/gui/app.py` — `_on_roi_clear`, `_on_roi_import`, `_on_roi_invert`
  (no longer in `_on_roi_edit_for_frame` after Task 1)
- `src/staq_dic/gui/panels/canvas_area.py` — 3 reads at lines 483, 729, 765
- `tests/test_gui/test_app_state.py` — 2 tests

**Step 2: Edit `app_state.py`**

Delete line 54 (`self.roi_editing_frame: int = 0`) and line 113
(`self.roi_editing_frame = 0` in `set_image_files`).

**Step 3: Edit `app.py` — replace `state.roi_editing_frame` with `state.current_frame`**

In `_on_roi_clear`:
```python
state.set_frame_roi(state.current_frame, None)
```

In `_on_roi_import`:
```python
state.set_frame_roi(state.current_frame, self._roi_ctrl.mask.copy())
```

In `_on_roi_invert`:
```python
state.set_frame_roi(state.current_frame, self._roi_ctrl.mask.copy())
```

**Step 4: Edit `canvas_area.py`**

`_finish_drawing` (line ~482):
```python
state.set_frame_roi(state.current_frame, self._roi_ctrl.mask.copy())
```

`_update_background` (line ~729):
```python
if state.roi_editing:
    self._load_frame(state.current_frame)
```

`_refresh_overlay` (line ~765):
```python
self.set_roi_editing_banner(state.current_frame)
```

**Step 5: Delete the obsolete app_state tests**

In `tests/test_gui/test_app_state.py`, delete:
- `test_roi_editing_frame_default` (lines 88-89)
- the `roi_editing_frame` portions of `test_reset_clears_per_frame` (lines 97, 103)

**Step 6: Remove the xfail in `test_roi_drawing_ux.py`**

Delete the `@pytest.mark.xfail(...)` decorator on `test_draw_button_targets_current_frame`
(if you added it in Task 2).

**Step 7: Run targeted tests**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py tests/test_gui/test_app_state.py -v`

Expected: PASS

**Step 8: Commit**

```bash
git add src/staq_dic/gui/app_state.py src/staq_dic/gui/app.py \
        src/staq_dic/gui/panels/canvas_area.py \
        tests/test_gui/test_app_state.py \
        tests/test_gui/test_roi_drawing_ux.py
git commit -m "refactor(gui): drop roi_editing_frame; current_frame is the editing frame"
```

---

## Task 4 — Q3: Pin "blue overlay reads per_frame_rois[current_frame]" (RED)

**Files:**
- Modify: `tests/test_gui/test_roi_drawing_ux.py` (add `TestQ3_OverlaySource`)

**Step 1: Add the failing test**

```python
class TestQ3_OverlaySource:
    """The blue ROI overlay must reflect per_frame_rois[current_frame],
    not the in-memory roi_ctrl.mask buffer.
    """

    def test_external_mutation_of_per_frame_rois_repaints_overlay(
        self, qapp, monkeypatch
    ):
        """Simulating a batch import that writes to per_frame_rois[3]
        directly must update the canvas overlay when current_frame=3.
        """
        win = _make_main_window(qapp)
        state = AppState.instance()
        state.set_current_frame(3)

        canvas = win._canvas_area.canvas
        captured: dict[str, object] = {}

        # Spy on update_roi_overlay to record what it reads
        original = canvas.update_roi_overlay
        def spy():
            captured["called"] = True
            original()
        monkeypatch.setattr(canvas, "update_roi_overlay", spy)

        # External mutation (simulates batch import)
        new_mask = np.ones((128, 128), dtype=bool)
        state.per_frame_rois[3] = new_mask
        state.roi_changed.emit()

        # The overlay must have been refreshed
        assert captured.get("called") is True

        # And it must read its data from per_frame_rois[3], not roi_ctrl.mask
        # (verified in next test)

    def test_overlay_data_source_is_per_frame_rois(self, qapp):
        """Direct contract test: update_roi_overlay must use
        per_frame_rois[current_frame] as its data source.
        """
        win = _make_main_window(qapp)
        state = AppState.instance()

        # Put divergent masks in roi_ctrl vs per_frame_rois
        state.set_current_frame(2)
        own_mask = np.zeros((128, 128), dtype=bool)
        own_mask[0:10, 0:10] = True
        state.per_frame_rois[2] = own_mask

        # Pollute the controller buffer with something different
        win._roi_ctrl.mask = np.ones((128, 128), dtype=bool)

        canvas = win._canvas_area.canvas
        canvas.update_roi_overlay()

        # If the overlay reads roi_ctrl.mask: pixmap is fully filled (~16384 pix).
        # If it reads per_frame_rois[2]: pixmap shows only the 10x10 square.
        # Sample one corner pixel that's outside per_frame_rois[2]:
        pixmap = canvas._roi_item.pixmap()
        assert not pixmap.isNull()
        img = pixmap.toImage()
        # Pixel (50, 50) should be transparent under the per_frame_rois reading
        assert img.pixelColor(50, 50).alpha() == 0
```

**Step 2: Run — expect FAIL**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ3_OverlaySource -v`

Expected: FAIL — current `update_roi_overlay` reads `self._roi_ctrl.mask`.

---

## Task 5 — Q3: Implement overlay reads from per_frame_rois (GREEN)

**Files:**
- Modify: `src/staq_dic/gui/panels/canvas_area.py` (`update_roi_overlay`)

**Step 1: Edit canvas_area.py**

Replace `update_roi_overlay` (lines 199-218):

```python
def update_roi_overlay(self) -> None:
    """Refresh the ROI overlay from per_frame_rois[current_frame].

    Single source of truth: the blue overlay always reflects what's
    persisted in app state for the current frame.  External mutations
    (batch import, context-menu import) become visible immediately
    via the roi_changed signal.
    """
    state = AppState.instance()
    mask = state.per_frame_rois.get(state.current_frame)
    if mask is None or not mask.any():
        self._roi_item.setPixmap(QPixmap())
        return
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, :] = [59, 130, 246, 80]
    rgba = np.ascontiguousarray(rgba)
    bytes_per_line = w * 4
    qimg = QImage(
        rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
    ).copy()
    self._roi_item.setPixmap(QPixmap.fromImage(qimg))
    self._roi_item.setPos(0, 0)
```

The `_roi_ctrl` reference is no longer needed inside `update_roi_overlay`, but
keep `set_roi_controller` for the drawing code paths (`_handle_draw_release`,
`_finalize_polygon`) that still need to stamp shapes into the buffer.

**Step 2: Wire `roi_changed` to repaint the overlay**

Make sure `_on_roi_changed` in `CanvasArea` (line ~705) already calls
`_refresh_overlay` which calls `update_roi_overlay`. It does — verify:

```python
def _on_roi_changed(self) -> None:
    self._viz_ctrl.invalidate_masks()
    self._refresh_overlay()
    self._refresh_mesh_overlay()
```

`_refresh_overlay` calls `self._canvas.update_roi_overlay()` in its
ROI-editing / no-results branches. Verify both branches still call it.

**Step 3: Run targeted test**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ3_OverlaySource -v`

Expected: PASS

**Step 4: Commit**

```bash
git add src/staq_dic/gui/panels/canvas_area.py tests/test_gui/test_roi_drawing_ux.py
git commit -m "fix(gui): blue overlay reads per_frame_rois[current_frame] (Q3)"
```

---

## Task 6 — Q1: Pin "frame change reloads ROI controller buffer" (RED)

**Files:**
- Modify: `tests/test_gui/test_roi_drawing_ux.py`

**Step 1: Add the failing test**

```python
class TestQ1_BufferFollowsFrame:
    def test_navigating_during_roi_editing_reloads_buffer(self, qapp):
        """Editing frame 0, then arrow-key to frame 3, must reload the
        ROI controller buffer from per_frame_rois[3].
        """
        win = _make_main_window(qapp)
        state = AppState.instance()

        # Seed two different masks
        m0 = np.zeros((128, 128), dtype=bool); m0[0:5, 0:5] = True
        m3 = np.zeros((128, 128), dtype=bool); m3[100:110, 100:110] = True
        state.per_frame_rois[0] = m0
        state.per_frame_rois[3] = m3

        # Enter editing on frame 0
        win._on_roi_edit_for_frame(0)
        assert win._roi_ctrl.mask[0, 0] == True
        assert win._roi_ctrl.mask[100, 100] == False

        # Navigate to frame 3 (without re-clicking Edit)
        state.set_current_frame(3)

        # Buffer must now hold m3, not m0
        assert win._roi_ctrl.mask[0, 0] == False
        assert win._roi_ctrl.mask[100, 100] == True
```

**Step 2: Run — expect FAIL**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ1_BufferFollowsFrame -v`

Expected: FAIL — nothing currently reloads the buffer on `current_frame_changed`.

---

## Task 7 — Q1: Reload ROI buffer when current_frame changes during editing (GREEN)

**Files:**
- Modify: `src/staq_dic/gui/app.py` (connect `current_frame_changed`)

**Step 1: Wire the signal**

Add to `MainWindow.__init__` (after `_init_roi_controller` is connected):

```python
# When the user navigates frames during ROI editing, reload the
# stamping buffer so the next draw operation targets the new frame.
self._state.current_frame_changed.connect(self._on_frame_changed_for_roi)
```

Add the handler:

```python
def _on_frame_changed_for_roi(self, _frame: int) -> None:
    """Reload the ROI controller buffer for the new current frame.

    Only matters during ROI editing — the canvas always paints the
    overlay from per_frame_rois[current_frame] regardless.  But the
    in-memory buffer must mirror the new frame so the next stamp
    operation (draw / invert / clear) starts from the right base.
    """
    if self._state.roi_editing:
        self._load_roi_buffer_for_current_frame()
```

**Step 2: Run test**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ1_BufferFollowsFrame -v`

Expected: PASS

**Step 3: Commit**

```bash
git add src/staq_dic/gui/app.py tests/test_gui/test_roi_drawing_ux.py
git commit -m "fix(gui): reload ROI buffer when current_frame changes during editing (Q1)"
```

---

## Task 8 — Q4: Pin "ROI editing → preview mesh path" (RED)

**Files:**
- Modify: `tests/test_gui/test_roi_drawing_ux.py`

**Step 1: Add the failing test**

```python
class TestQ4_MeshDuringEditing:
    def test_roi_editing_routes_to_preview_mesh(self, qapp, monkeypatch):
        """Even when results exist, entering ROI editing must show the
        preview mesh (which reflects the in-progress ROI), not the
        results mesh (which reflects whatever DIC ran on).
        """
        from staq_dic.gui.panels import canvas_area as ca_module

        win = _make_main_window(qapp)
        state = AppState.instance()

        # Stub results so the "results path" branch would normally win
        from staq_dic.core.data_structures import (
            DICMesh, FrameResult, PipelineResult,
        )
        from staq_dic.core.config import dicpara_default

        coords = np.array([[10, 10], [20, 10], [20, 20], [10, 20]],
                          dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=np.int64)
        mesh = DICMesh(coordinates_fem=coords, elements_fem=elements)
        result = PipelineResult(
            dic_para=dicpara_default(),
            dic_mesh=mesh,
            result_disp=[FrameResult(U=np.zeros(8), U_accum=np.zeros(8))],
            result_def_grad=[],
            result_strain=[],
            result_fe_mesh_each_frame=[mesh],
        )
        state.set_results(result)

        # Seed an ROI on frame 0 and enter editing
        state.per_frame_rois[0] = np.ones((128, 128), dtype=bool)
        state.roi_editing = True
        state.set_current_frame(0)

        canvas_area = win._canvas_area
        called = {"results": False, "preview": False}
        monkeypatch.setattr(
            canvas_area, "_show_results_mesh",
            lambda: called.__setitem__("results", True),
        )
        monkeypatch.setattr(
            canvas_area, "_generate_preview_mesh",
            lambda: called.__setitem__("preview", True),
        )

        canvas_area._refresh_mesh_overlay()

        assert called["preview"] is True
        assert called["results"] is False
```

**Step 2: Run — expect FAIL**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ4_MeshDuringEditing -v`

Expected: FAIL — current `_refresh_mesh_overlay` checks `state.results is not None`
first and goes to results.

---

## Task 9 — Q4 + Q5: Route mesh through preview during editing & hide for K≠0 (GREEN)

**Files:**
- Modify: `src/staq_dic/gui/panels/canvas_area.py` (`_refresh_mesh_overlay`)
- Modify: `tests/test_gui/test_roi_drawing_ux.py` (add Q5 test)

**Step 1: Add the Q5 red test first**

```python
class TestQ5_MeshHiddenForNonRefEditing:
    def test_mesh_hidden_when_editing_non_frame_zero_roi(self, qapp):
        """The preview mesh only models frame-0 geometry.  When the
        user is editing a per-frame ROI for K != 0, hide the mesh
        entirely so they don't get a misleading overlay.
        """
        from staq_dic.gui.panels.canvas_area import CanvasArea

        win = _make_main_window(qapp)
        state = AppState.instance()
        state.per_frame_rois[3] = np.ones((128, 128), dtype=bool)
        state.set_current_frame(3)
        state.roi_editing = True

        canvas_area = win._canvas_area
        canvas_area._refresh_mesh_overlay()

        assert canvas_area._mesh_overlay.isVisible() is False
```

**Step 2: Run — expect FAIL**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ5_MeshHiddenForNonRefEditing -v`

Expected: FAIL.

**Step 3: Implement both Q4 and Q5 in `_refresh_mesh_overlay`**

Edit `src/staq_dic/gui/panels/canvas_area.py` `_refresh_mesh_overlay` (line ~923):

```python
def _refresh_mesh_overlay(self) -> None:
    """Rebuild mesh overlay data (paths + transform).

    Routing rules:
    - User toggled mesh off            → hide
    - ROI editing on frame K != 0      → hide (preview mesh only models
                                          frame-0 geometry, would be
                                          misleading)
    - ROI editing on frame 0           → preview mesh from current params
                                          (Q4: ignore results, the user
                                          is editing the ROI not inspecting)
    - Otherwise + results              → results mesh
    - Otherwise + frame-0 ROI          → preview mesh
    - Else                             → hide
    """
    state = self._state
    if not state.show_mesh:
        self._mesh_overlay.setVisible(False)
        return

    if state.roi_editing:
        if state.current_frame != 0:
            self._mesh_overlay.set_mesh(None, None)
            self._mesh_overlay.setVisible(False)
            return
        # Frame-0 editing: always go through preview path
        if state.roi_mask is not None:
            self._mesh_preview_timer.start()
        else:
            self._mesh_overlay.set_mesh(None, None)
            self._mesh_overlay.setVisible(False)
        return

    if state.results is not None:
        self._show_results_mesh()
    elif state.roi_mask is not None:
        self._mesh_preview_timer.start()
    else:
        self._mesh_overlay.set_mesh(None, None)
        self._mesh_overlay.setVisible(False)
```

**Step 4: Run both Q4 and Q5 tests**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ4_MeshDuringEditing tests/test_gui/test_roi_drawing_ux.py::TestQ5_MeshHiddenForNonRefEditing -v`

Expected: PASS (Q5) + still FAIL on Q4 because `_mesh_preview_timer.start()`
defers to a real timer that won't fire in the test.

**Step 5: Make Q4 test deterministic**

Update the Q4 test to trigger the preview generation directly:

```python
        # Force the timer-driven path to fire synchronously
        canvas_area._mesh_preview_timer.stop()
        canvas_area._refresh_mesh_overlay()
        canvas_area._generate_preview_mesh()
```

Or simpler: make the assertion check that `_mesh_preview_timer.start` was called
*instead* of `_show_results_mesh`. Use a `monkeypatch` that records timer starts.

```python
        timer_started = {"v": False}
        monkeypatch.setattr(
            canvas_area._mesh_preview_timer, "start",
            lambda *a, **kw: timer_started.__setitem__("v", True),
        )
        canvas_area._refresh_mesh_overlay()
        assert timer_started["v"] is True
        # And the results path was *not* taken
        assert called["results"] is False
```

Drop the original `called["preview"]` assertion (it relied on us monkeypatching
`_generate_preview_mesh`, which is no longer called directly from
`_refresh_mesh_overlay`).

**Step 6: Re-run**

Expected: PASS

**Step 7: Commit**

```bash
git add src/staq_dic/gui/panels/canvas_area.py tests/test_gui/test_roi_drawing_ux.py
git commit -m "fix(gui): mesh overlay routing during ROI editing (Q4+Q5)"
```

---

## Task 10 — Q6: Pin "image deletion clears results" (RED)

**Files:**
- Modify: `tests/test_gui/test_roi_drawing_ux.py`

**Step 1: Add the failing test**

```python
class TestQ6_PipelineLifecycle:
    def test_image_deletion_clears_results(self, qapp):
        """Deleting an image makes any prior pipeline result stale —
        clear it.
        """
        from staq_dic.core.data_structures import (
            DICMesh, FrameResult, PipelineResult,
        )
        from staq_dic.core.config import dicpara_default

        win = _make_main_window(qapp)
        state = AppState.instance()

        coords = np.array([[10, 10], [20, 10], [20, 20], [10, 20]],
                          dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=np.int64)
        mesh = DICMesh(coordinates_fem=coords, elements_fem=elements)
        state.set_results(PipelineResult(
            dic_para=dicpara_default(),
            dic_mesh=mesh,
            result_disp=[FrameResult(U=np.zeros(8), U_accum=np.zeros(8))],
            result_def_grad=[],
            result_strain=[],
            result_fe_mesh_each_frame=[mesh],
        ))
        assert state.results is not None

        # Simulate image-list deletion of frames 1 and 2
        image_list = win._left_sidebar._image_list
        # Pretend the user selected those rows and triggered delete
        image_list._tree.topLevelItem(1).setSelected(True)
        image_list._tree.topLevelItem(2).setSelected(True)
        image_list._delete_selected()

        assert state.results is None

    def test_tracking_mode_change_keeps_results(self, qapp):
        """Changing tracking mode is just a UI change — must NOT
        clear stale-but-valid results.
        """
        from staq_dic.core.data_structures import (
            DICMesh, FrameResult, PipelineResult,
        )
        from staq_dic.core.config import dicpara_default

        win = _make_main_window(qapp)
        state = AppState.instance()

        coords = np.array([[10, 10], [20, 10], [20, 20], [10, 20]],
                          dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=np.int64)
        mesh = DICMesh(coordinates_fem=coords, elements_fem=elements)
        state.set_results(PipelineResult(
            dic_para=dicpara_default(),
            dic_mesh=mesh,
            result_disp=[FrameResult(U=np.zeros(8), U_accum=np.zeros(8))],
            result_def_grad=[],
            result_strain=[],
            result_fe_mesh_each_frame=[mesh],
        ))

        state.set_param("tracking_mode", "incremental")

        assert state.results is not None
```

**Step 2: Run — expect FAIL on the first test, PASS on the second**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ6_PipelineLifecycle -v`

Expected:
- `test_image_deletion_clears_results` FAIL (currently `_delete_selected` does
  not clear results)
- `test_tracking_mode_change_keeps_results` PASS (already correct)

---

## Task 11 — Q6: Clear results in `_delete_selected` (GREEN)

**Files:**
- Modify: `src/staq_dic/gui/widgets/image_list.py` (`_delete_selected`)

**Step 1: Edit `_delete_selected`**

After re-keying `per_frame_rois` and before the `images_changed` emission, add:

```python
        # Q6: image-list mutation invalidates pipeline results
        had_results = self._state.results is not None
        if had_results:
            self._state.results = None
            self._state.deformed_masks = None
            from staq_dic.gui.app_state import RunState
            self._state.run_state = RunState.IDLE
            self._state.show_deformed = False

        # Clear image caches for removed images
        self._image_ctrl.clear_cache()

        # Update state (triggers _rebuild_list via images_changed signal)
        self._state.image_files = files
        self._state.current_frame = min(
            self._state.current_frame, len(files) - 1
        )
        if had_results:
            self._state.results_changed.emit()
            self._state.run_state_changed.emit(RunState.IDLE)
        self._state.images_changed.emit()
```

**Step 2: Run targeted test**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestQ6_PipelineLifecycle -v`

Expected: both PASS.

**Step 3: Commit**

```bash
git add src/staq_dic/gui/widgets/image_list.py tests/test_gui/test_roi_drawing_ux.py
git commit -m "fix(gui): clear pipeline results when images are deleted (Q6)"
```

---

## Task 12 — Audit `_on_batch_import` and `_on_roi_import_for_frames`

**Files:**
- Modify: `src/staq_dic/gui/app.py`

These two methods write directly to `state.per_frame_rois[k]` and emit
`roi_changed`. After Task 5, the canvas overlay reads from `per_frame_rois`, so
the displayed overlay updates correctly. But the in-memory `_roi_ctrl.mask`
buffer is now stale for the *current* frame if it happened to be one of the
imported frames.

**Step 1: Add a regression test**

```python
class TestBatchImportConsistency:
    def test_batch_import_to_current_frame_reloads_buffer(self, qapp):
        """If the batch import target includes current_frame, the
        ROI controller buffer must mirror the imported mask so the
        next draw stamp starts from the imported base.
        """
        win = _make_main_window(qapp)
        state = AppState.instance()
        state.set_current_frame(2)

        # Simulate the inner work of _on_batch_import for frame 2
        new_mask = np.zeros((128, 128), dtype=bool)
        new_mask[40:60, 40:60] = True
        win._on_roi_import_for_frames({2: None})  # path stub below

        # We'll verify with a direct mutation flow:
        state.per_frame_rois[2] = new_mask
        state.roi_changed.emit()

        # Now check the buffer would have been refreshed
        # (if roi_editing) or would be refreshed on next entry
        win._on_roi_edit_for_frame(2)
        assert win._roi_ctrl.mask[50, 50] == True
```

This test passes naturally with Task 1's implementation: `_on_roi_edit_for_frame`
calls `_load_roi_buffer_for_current_frame` which reads `per_frame_rois[2]`. We
add it as a guardrail.

**Step 2: Run**

Run: `pytest tests/test_gui/test_roi_drawing_ux.py::TestBatchImportConsistency -v`

Expected: PASS.

**Step 3: Connect `roi_changed` to buffer reload during editing**

In `MainWindow.__init__`:

```python
self._state.roi_changed.connect(self._on_roi_changed_reload)
```

Add the handler:

```python
def _on_roi_changed_reload(self) -> None:
    """When per_frame_rois mutates externally, refresh the working
    buffer if we're editing the affected frame.
    """
    if self._state.roi_editing and self._roi_ctrl is not None:
        self._load_roi_buffer_for_current_frame()
```

This is defensive: it guarantees that an `_on_batch_import` that touches
`per_frame_rois[current_frame]` while the user is in editing mode will
re-mirror the buffer before the next draw stamp.

**Step 4: Commit**

```bash
git add src/staq_dic/gui/app.py tests/test_gui/test_roi_drawing_ux.py
git commit -m "fix(gui): refresh ROI buffer on roi_changed during editing"
```

---

## Task 13 — Verify the full GUI suite

**Step 1: Run the full test suite**

Run: `pytest tests/ -x --timeout=120`

Expected: all 600+ tests PASS.

If a previously passing test now fails, it almost certainly references
`state.roi_editing_frame`. Update it to use `state.current_frame` and
re-run.

**Step 2: Manual smoke test**

```bash
python -m staq_dic.gui.app
```

Walk through:
1. Load `examples/bubble_30_150_30/images` (or any folder)
2. Click frame 3 in image list → progress bar / canvas show frame 3
3. Click "Draw Rect" → draw a rectangle → frame 3 row in image list shows
   green "Edit" badge
4. Click frame 0 row → canvas shows frame 0 (no ROI yet)
5. Click "Draw Rect" → draw a rectangle → frame 0 row shows green
6. Right-click frame 1 → "Import ROI" → pick a mask file → blue overlay
   appears immediately (without re-clicking Edit)
7. Click frame 1 → blue overlay shows the imported mask
8. With ROI on frame 0, run DIC, then click frame 0's "Edit" button →
   mesh overlay should show the *preview* mesh (responsive to subset
   step changes), not the results mesh
9. Click frame 1's "Edit" button → mesh overlay should be hidden
10. Delete frame 4 from image list → results clear, run controls reset

**Step 3: No commit if everything works** — proceed to wrap-up.

---

## Wrap-up

After Task 13 passes:

1. Run `pytest tests/ --timeout=120` one more time and confirm green.
2. Read `README.md` to verify it doesn't reference `roi_editing_frame`
   anywhere; update if it does.
3. Push the branch:
   ```bash
   git push -u origin feature/roi-drawing-ux
   ```
4. Open a PR titled `fix(gui): unify ROI editing on current_frame (Q1-Q6)`.

## Files Touched (Summary)

| File                                                       | Type    | Changes                                                                            |
|------------------------------------------------------------|---------|------------------------------------------------------------------------------------|
| `src/staq_dic/gui/app_state.py`                            | edit    | drop `roi_editing_frame` field                                                     |
| `src/staq_dic/gui/app.py`                                  | edit    | new `_load_roi_buffer_for_current_frame`, frame-sync in `_on_roi_edit_for_frame`, signal wiring for buffer reload |
| `src/staq_dic/gui/panels/canvas_area.py`                   | edit    | `update_roi_overlay` reads `per_frame_rois`, `_refresh_mesh_overlay` Q4+Q5 routing |
| `src/staq_dic/gui/widgets/image_list.py`                   | edit    | `_delete_selected` clears results                                                  |
| `tests/test_gui/test_app_state.py`                         | edit    | drop two `roi_editing_frame` tests                                                 |
| `tests/test_gui/test_roi_drawing_ux.py`                    | create  | full Q1-Q6 test coverage                                                           |
