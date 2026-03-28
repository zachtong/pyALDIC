# Masked Subset IC-GN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix boundary-node accuracy by making IC-GN only use mask-interior pixels — instead of zeroing the entire reference image with `f_img * mask` (which creates artificial gradient edges), pass the raw image and use the mask purely for pixel selection.

**Architecture:** Three-layer change: (1) pipeline passes raw image alongside masked image, (2) gradient computation uses raw image, (3) IC-GN precomputation samples from raw image and uses mask + connected-component for pixel selection. The iteration kernels require no changes — they already operate on precomputed masked data.

**Tech Stack:** NumPy, SciPy (`ndimage.label`, `map_coordinates`), Numba (parallel kernels)

---

## Problem Analysis

**Current flow** (buggy at boundaries):
```
pipeline:     f_img = img * mask       ← zeros pixels outside mask
gradient:     df_dx = gradient(f_img)  ← HUGE artificial gradient at mask edge
precompute:   ref_patch = f_img[y:y+S, x:x+S] * mask_patch
              → boundary pixels: gradient = -(speckle_value)/dx ≈ -75
              → Hessian biased, IC-GN converges to wrong displacement
```

**Fixed flow:**
```
pipeline:     f_img_raw = img          ← NO masking
              f_img = img * mask       ← kept for integer_search backward compat
gradient:     df_dx = gradient(f_img_raw)  ← natural speckle gradient everywhere
precompute:   ref_patch = f_img_raw[y:y+S, x:x+S]
              bw = connected_center_mask(mask_patch)
              ref_sub = ref_patch * bw  ← mask selects pixels, values are raw
              → boundary pixels: gradient = natural speckle gradient ≈ -1
              → Hessian clean, IC-GN converges correctly
```

**What ALREADY works** (no changes needed):
- `_connected_center_mask` — connected component containing center pixel
- `mask_all` array — per-node valid pixel mask stored in precompute
- Iteration masking: `combined = (mask_all > 0.5) & g_valid` — dynamic pixel filtering
- ZNSSD statistics: mean/std computed only from valid pixels
- Gradient/Hessian assembly: only valid pixels contribute

**What needs to change:**
- Pipeline: pass raw image to gradient + precompute
- `compute_image_gradient`: compute from raw image
- `precompute_subsets_*`: sample from raw image, use mask directly (not `abs(val) < 1e-10` proxy)
- Numba kernels: same logic changes
- Validity threshold: use mask directly, make configurable
- Hessian conditioning: add check for ill-conditioned Hessians (thin strips of valid pixels)

---

## Task 1: Add `img_ref_raw` parameter to `compute_image_gradient`

**Files:**
- Modify: `src/staq_dic/io/image_ops.py:67-122`
- Test: `tests/test_io/test_image_ops.py` (new test)

**Step 1: Write failing test**

```python
# tests/test_io/test_image_ops.py — add to existing or create
def test_gradient_from_raw_image():
    """Gradient from raw image should NOT have artificial edge at mask boundary."""
    h, w = 64, 64
    rng = np.random.default_rng(42)
    img = rng.random((h, w)) * 200 + 20  # smooth speckle-like

    # Circular mask
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx - 32)**2 + (yy - 32)**2 < 20**2).astype(np.float64)

    # Old way: gradient from masked image
    Df_old = compute_image_gradient(img * mask, mask)

    # New way: gradient from raw image
    Df_new = compute_image_gradient(img * mask, mask, img_raw=img)

    # At boundary pixels: old gradient has large artificial edge, new doesn't
    from scipy.ndimage import binary_erosion
    boundary = mask.astype(bool) & ~binary_erosion(mask.astype(bool), iterations=1)
    boundary_idx = np.where(boundary)

    old_grad_mag = np.sqrt(Df_old.df_dx[boundary_idx]**2 + Df_old.df_dy[boundary_idx]**2)
    new_grad_mag = np.sqrt(Df_new.df_dx[boundary_idx]**2 + Df_new.df_dy[boundary_idx]**2)

    # New gradient at boundary should be MUCH smaller (no artificial edge)
    assert np.median(new_grad_mag) < np.median(old_grad_mag) * 0.5
```

**Step 2: Implement**

In `image_ops.py`, add optional `img_raw` parameter:
```python
def compute_image_gradient(img_ref, img_ref_mask=None, img_raw=None):
    """Compute image gradients.

    Args:
        img_ref: (H, W) reference image (possibly masked).
        img_ref_mask: (H, W) binary mask.
        img_raw: (H, W) unmasked raw image. If provided, gradients are
                 computed from this instead of img_ref. This avoids
                 artificial gradient edges at mask boundaries.
    """
    source = img_raw if img_raw is not None else img_ref
    # ... compute gradients from `source` ...
    # ... mask gradients with img_ref_mask ...
```

**Step 3: Run tests, commit**

---

## Task 2: Pipeline passes raw image through ref_cache

**Files:**
- Modify: `src/staq_dic/core/pipeline.py:489-496` (ref image setup)
- Modify: `src/staq_dic/core/pipeline.py:~530` (precompute_subpb1 call)
- Modify: `src/staq_dic/core/pipeline.py:~560` (local_icgn call)

**Step 1: Change ref_cache to store raw image**

```python
# OLD (line ~494):
f_img = img_normalized[ref_idx] * f_mask
Df = compute_image_gradient(f_img, f_mask)
ref_cache[ref_idx] = (f_img, f_mask, Df)

# NEW:
f_img_raw = img_normalized[ref_idx].copy()
f_img = f_img_raw * f_mask    # still used by integer_search
Df = compute_image_gradient(f_img, f_mask, img_raw=f_img_raw)
ref_cache[ref_idx] = (f_img, f_img_raw, f_mask, Df)
```

**Step 2: Pass raw image to IC-GN precompute and solver calls**

All precompute/solver calls receive `f_img_raw`:
```python
pre_subpb1 = precompute_subpb1(coords, Df, f_img_raw, para)
# ...
U_subpb1, F_subpb1 = local_icgn(coords, f_img_raw, g_img, Df, para, U0=...)
```

**Step 3: Unpack ref_cache correctly everywhere it's used**

Search for all `ref_cache[ref_idx]` unpacking and update to 4-tuple.

**Step 4: Run full test suite, commit**

---

## Task 3: Modify `precompute_subsets_6dof` — raw image + mask-based validity

**Files:**
- Modify: `src/staq_dic/solver/icgn_batch.py:24-150`
- Test: `tests/test_solver/test_icgn_batch.py` (new test)

**Step 1: Write failing test**

```python
def test_precompute_6dof_boundary_gradient():
    """Boundary-node gradient should reflect raw speckle, not artificial edge."""
    h, w = 64, 64
    rng = np.random.default_rng(42)
    img_raw = rng.random((h, w)) * 200 + 20
    mask = np.ones((h, w), dtype=np.float64)
    mask[:, 32:] = 0.0  # right half masked

    # Node at x=30 (2px from boundary): subset extends into masked region
    coords = np.array([[30.0, 32.0]])

    from staq_dic.io.image_ops import compute_image_gradient
    Df = compute_image_gradient(img_raw * mask, mask, img_raw=img_raw)

    pre = precompute_subsets_6dof(
        coords, img_raw, Df.df_dx, Df.df_dy, mask, winsize=16,
    )
    assert pre["valid"][0]  # should be valid (center is in mask)

    # Check that gradient values in the valid region are reasonable
    bw = pre["mask_all"][0]
    gx = pre["gx_all"][0]
    valid_gx = gx[bw > 0.5]
    assert np.max(np.abs(valid_gx)) < 50  # no artificial edge (would be ~75+)
```

**Step 2: Change function signature**

```python
def precompute_subsets_6dof(
    coords, img_ref, df_dx, df_dy, img_ref_mask, winsize,
    min_valid_ratio=0.5,
):
```

Note: `img_ref` is now expected to be the RAW (unmasked) image. The pipeline change in Task 2 ensures this.

**Step 3: Replace pixel-value-based validity with mask-based validity**

```python
# OLD (line 103-109):
mask_patch = img_ref_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
ref_patch = img_ref[y_lo:y_hi + 1, x_lo:x_hi + 1] * mask_patch
n_masked = np.sum(np.abs(ref_patch) < 1e-10)
if n_masked > 0.4 * ref_patch.size:
    mark_hole[i] = True
    continue

# NEW:
mask_patch = img_ref_mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
ref_patch_raw = img_ref[y_lo:y_hi + 1, x_lo:x_hi + 1]  # raw pixels

# Connected component containing center
bw = _connected_center_mask(mask_patch > 0.5)
n_connected = np.sum(bw > 0.5)
if n_connected < min_valid_ratio * mask_patch.size:
    mark_hole[i] = True
    continue

# Apply connected component mask to raw pixels
ref_sub = ref_patch_raw * bw
gx_sub = df_dx[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw
gy_sub = df_dy[y_lo:y_hi + 1, x_lo:x_hi + 1] * bw
```

**Step 4: Fix statistics to use mask, not pixel-value proxy**

```python
# OLD (line 124-131):
nz = np.abs(ref_sub) > 1e-10
n_valid = nz.sum()
if n_valid < 4:
    continue
meanf = np.mean(ref_sub[nz])

# NEW:
valid_px = bw > 0.5
n_valid = int(np.sum(valid_px))
if n_valid < max(4, int(min_valid_ratio * mask_patch.size)):
    continue
meanf = np.mean(ref_sub[valid_px])
varf = np.var(ref_sub[valid_px])
bottomf = np.sqrt(max((n_valid - 1) * varf, 1e-30))
```

**Step 5: Add Hessian conditioning check**

```python
# After H = _build_hessian_6dof(XX * bw, YY * bw, gx_sub, gy_sub):
# Check if Hessian is invertible (thin strip → ill-conditioned)
try:
    np.linalg.cholesky(H + 1e-12 * np.eye(6))
except np.linalg.LinAlgError:
    mark_hole[i] = True
    continue
```

**Step 6: Run tests, commit**

---

## Task 4: Same changes for `precompute_subsets_2dof`

**Files:**
- Modify: `src/staq_dic/solver/icgn_batch.py:153-286` (2-DOF precompute)

Apply identical changes as Task 3 to the 2-DOF variant:
- Raw image sampling
- Mask-based validity (not pixel-value proxy)
- `min_valid_ratio` parameter
- Hessian conditioning check (2x2 matrix)

---

## Task 5: Update Numba kernels

**Files:**
- Modify: `src/staq_dic/solver/numba_kernels.py:~600-840`

**Step 1: `_precompute_one_6dof`** (line ~600):
```python
# OLD:
val = img_ref[y_lo + iy, x_lo + ix] * m
if abs(val) < 1e-10:
    n_masked += 1

# NEW:
if m < 0.5:
    n_masked += 1
```

And for pixel extraction:
```python
# OLD:
r = img_ref[y_lo + iy, x_lo + ix] * b

# NEW: (img_ref is now raw, b handles masking)
r = img_ref[y_lo + iy, x_lo + ix] * b
# (same code, but img_ref is now unmasked — change is in the caller)
```

**Step 2: `_precompute_one_2dof`** (line 748):
Same changes.

**Step 3: Statistics using mask not pixel value**
```python
# OLD:
if abs(r) > 1e-10:
    n_valid += 1

# NEW:
if b > 0.5:
    n_valid += 1
```

**Step 4: Run tests, commit**

---

## Task 6: Update `subpb1_solver` and `local_icgn` signatures

**Files:**
- Modify: `src/staq_dic/solver/subpb1_solver.py:31-65, 68-162`
- Modify: `src/staq_dic/solver/local_icgn.py:~20-90`

**Changes:**
- `precompute_subpb1(coords, Df, f_img, para)` → the `f_img` passed is now raw (from pipeline change)
- `local_icgn(coords, f_img, ...)` → `f_img` is now raw
- No signature changes needed if pipeline already passes raw image in place of masked image

**Important:** The masked `f_img` is ONLY kept for `integer_search_pyramid` which uses NCC template matching and expects the masked image.

---

## Task 7: Add `min_valid_ratio` to DICPara

**Files:**
- Modify: `src/staq_dic/core/data_structures.py` (DICPara)
- Modify: `src/staq_dic/core/config.py` (default value)

Add field:
```python
@dataclass(frozen=True)
class DICPara:
    # ... existing fields ...
    min_valid_ratio: float = 0.5  # min fraction of subset pixels that must be valid
```

Thread this through to precompute calls.

---

## Task 8: Unit tests for masked subset behavior

**Files:**
- Create: `tests/test_solver/test_masked_subset.py`

**Test cases:**
1. **Full mask (no boundary):** precompute with raw image == precompute with masked image (all pixels interior → identical results)
2. **Half-mask node:** node near straight boundary, verify only connected-component pixels used, gradient is natural (not artificial edge)
3. **Two-island window:** mask creates two disconnected regions in subset, verify only center-containing region kept
4. **Too-few-pixels rejection:** node mostly outside mask, verify `mark_hole = True`
5. **Thin strip check:** node at corner of mask, valid pixels form thin L-shape, verify Hessian conditioning rejects it
6. **End-to-end RMSE:** run DIC with annular mask, verify boundary RMSE improves vs old method

---

## Task 9: Integration benchmark — before/after comparison

**Files:**
- Modify: `scripts/benchmark_quadtree.py`

Add a comparison mode that runs the same 48 benchmark cases and produces a "before vs after" RMSE table. The "before" results are the ones already saved from the current run.

**Expected outcome:**
- Full mask (no holes): identical RMSE (no boundary nodes)
- Annular/hole masks: significant RMSE improvement at boundary nodes
- Quadtree refinement should now IMPROVE accuracy (more boundary nodes = more good measurements, not more bad ones)

---

## Implementation Order and Dependencies

```
Task 1 (gradient)
  ↓
Task 2 (pipeline)
  ↓
Task 3 (6dof precompute) ─── Task 4 (2dof precompute) ─── Task 5 (numba)
  ↓                           ↓                             ↓
Task 6 (solver signatures) ←──┘                             │
  ↓                                                          │
Task 7 (DICPara) ←──────────────────────────────────────────┘
  ↓
Task 8 (unit tests)
  ↓
Task 9 (benchmark)
```

**Total estimated changes:** ~150 lines modified across 6 files + ~100 lines new tests

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Numba kernel changes break JIT compilation | Run Numba tests first, fall back to Python path |
| Hessian conditioning check too aggressive | Start with permissive threshold, tune with benchmark |
| `integer_search` still uses masked image | Intentional — NCC template matching is less sensitive to boundary zeros |
| Real experiment images have no speckle outside mask | Correct — raw image outside mask is naturally dark, gradient is physically correct |
