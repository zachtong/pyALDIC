"""Export pipeline results as per-frame PNG images.

Rendering pipeline (no Qt dependency):
    1. Scatter node values -> regular grid via LinearNDInterpolator.
    2. Apply ROI mask with 3-tier priority (matching GUI VizController):
       a. Explicit deformed mask (per-frame ROI) -- direct lookup.
       b. Inverse-warped reference mask -- deformed grid -> ref coords.
       c. Reference mask -- direct lookup (non-deformed mode).
    3. Normalize + apply matplotlib colormap -> RGBA (NaN -> alpha=0).
    4. Composite RGBA overlay over background image at field_opacity.
    5. cv2.imwrite -> PNG file.

Background modes:
    "ref_frame"     -- use the first image file (frame 0) for every frame.
    "current_frame" -- use the frame's own image file as background.

Directory structure:
    dest_dir/
      {prefix}_images_{timestamp}/
        {field_name}/
          frame_001.png
          frame_002.png
          ...
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from al_dic.core.data_structures import PipelineResult, split_uv
from al_dic.export.export_utils import ensure_dir, frame_tag


# ---------------------------------------------------------------------------
# Colorbar + physical-units support (used by export_png and export_animation)
# ---------------------------------------------------------------------------

_COLORBAR_FIELD_LABELS: dict[str, str] = {
    "disp_u": "U",
    "disp_v": "V",
    "disp_magnitude": "Magnitude",
    "strain_exx": "\u03b5xx",
    "strain_eyy": "\u03b5yy",
    "strain_exy": "\u03b5xy",
    "strain_principal_max": "\u03b5\u2081",
    "strain_principal_min": "\u03b5\u2082",
    "strain_maxshear": "\u03b3 max",
    "strain_von_mises": "von Mises",
    "strain_rotation": "\u03c9 rot",
}

_DISPLACEMENT_FIELDS = {"disp_u", "disp_v", "disp_magnitude"}


def colorbar_label(
    field_name: str,
    use_physical: bool = False,
    pixel_unit: str = "mm",
) -> str:
    """Build a colorbar label string with appropriate unit suffix.

    Displacement fields show ``px`` or the chosen *pixel_unit*.
    Strain fields are dimensionless — no unit suffix.
    """
    base = _COLORBAR_FIELD_LABELS.get(field_name, field_name)
    if field_name in _DISPLACEMENT_FIELDS:
        return f"{base} ({pixel_unit})" if use_physical else f"{base} (px)"
    return base


def scale_field_values(
    values: NDArray,
    field_name: str,
    pixel_size: float,
) -> NDArray:
    """Scale raw pixel-unit values to physical units (displacement only).

    Strain fields are dimensionless and returned unchanged.
    """
    if field_name in _DISPLACEMENT_FIELDS and pixel_size != 1.0:
        return values * pixel_size
    return values


def render_colorbar_strip(
    height: int,
    cmap_name: str,
    vmin: float,
    vmax: float,
    label: str,
    dpi: int = 150,
) -> NDArray:
    """Render a vertical colorbar as a BGR uint8 image matching *height*.

    Uses matplotlib for high-quality tick labels and gradient rendering.
    The strip has a black background with white text to match DIC imagery.

    Returns:
        (height, strip_width, 3) BGR uint8 array.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig_h = max(2.0, height / dpi)
    fig_w = 1.2  # inches

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("black")

    norm = Normalize(vmin=vmin, vmax=vmax)
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        cmap = plt.get_cmap("jet")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax)

    cb.set_label(label, fontsize=9, color="white")
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.outline.set_edgecolor("white")
    cb.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.15, right=0.55, top=0.97, bottom=0.03)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="black")
    plt.close(fig)
    buf.seek(0)

    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is not None and bgr.shape[0] != height:
        scale_f = height / bgr.shape[0]
        new_w = max(1, int(bgr.shape[1] * scale_f))
        bgr = cv2.resize(bgr, (new_w, height), interpolation=cv2.INTER_AREA)

    return bgr


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _to_bgr(image: NDArray, H: int, W: int) -> NDArray:
    """Convert any uint8 image to (H, W, 3) BGR, resizing if needed."""
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()
    if bgr.shape[:2] != (H, W):
        bgr = cv2.resize(bgr, (W, H))
    return bgr


def _load_frame_image(
    image_files: list[str], frame: int, bg_mode: str
) -> NDArray | None:
    """Load a background image for the given frame index.

    Args:
        image_files: Ordered list of source image paths.
        frame:       Current frame index (0-based).
        bg_mode:     "ref_frame" -> always load index 0;
                     "current_frame" -> load the matching frame.
    Returns:
        (H, W) uint8 grayscale array, or None if unavailable.
    """
    if not image_files:
        return None
    idx = 0 if bg_mode == "ref_frame" else min(frame, len(image_files) - 1)
    try:
        img = cv2.imread(image_files[idx], cv2.IMREAD_GRAYSCALE)
        return img  # (H, W) uint8, or None if imread fails
    except Exception:
        return None


def render_field_frame(
    coords: NDArray,
    values: NDArray,
    image_shape: tuple[int, int],
    bg_image: NDArray | None,
    field_cfg: "FieldImageConfig",
    roi_mask: NDArray | None = None,
    deformed_coords: NDArray | None = None,
    deformed_mask: NDArray | None = None,
) -> NDArray:
    """Render a single field frame to a BGR uint8 image.

    Composite pipeline (matches GUI VizController rendering):
    1. Interpolate scatter -> regular grid (NaN outside convex hull).
    2. Apply ROI mask (3-tier priority, matching GUI):
       a. *deformed_mask* provided -> direct lookup (per-frame ROI or
          pre-computed inverse-warped mask).
       b. *roi_mask* provided (non-deformed mode) -> direct lookup.
       Caller is responsible for computing the warped mask when in
       deformed mode and no explicit deformed_mask is given.
    3. Normalize, apply matplotlib colormap -> RGBA (NaN -> alpha=0).
    4. Composite RGBA over background:
         alpha_w = (rgba_alpha / 255) x field_opacity   (per pixel)
         result  = bg x (1 - alpha_w) + field_bgr x alpha_w
       field_opacity = field_cfg.bg_alpha (0=invisible, 1=fully opaque).
       Matches GUI's setOpacity(overlay_alpha) behaviour.

    Args:
        coords:          (N, 2) reference node positions (x, y) in pixels.
        values:          (N,) field values at each node.
        image_shape:     (H, W) output image size in pixels.
        bg_image:        Optional background (H, W) or (H, W, 3) uint8.
                         Black background is used when None.
        field_cfg:       Per-field colour/range/opacity settings.
        roi_mask:        Optional (H, W) bool mask (reference coords);
                         True = inside ROI.  Used in non-deformed mode.
        deformed_coords: If given, use as rendering positions instead of coords.
        deformed_mask:   Optional (H, W) bool mask in deformed coords.
                         When provided, takes priority over *roi_mask*.

    Returns:
        (H, W, 3) BGR uint8 image.
    """
    from matplotlib import colormaps

    H, W = image_shape
    render_coords = deformed_coords if deformed_coords is not None else coords

    # Prepare background (black when no image provided)
    bg_bgr = _to_bgr(bg_image, H, W) if bg_image is not None else np.zeros((H, W, 3), dtype=np.uint8)

    # Interpolate scatter -> regular grid
    gx = np.linspace(0, W - 1, W)
    gy = np.linspace(0, H - 1, H)
    grid_x, grid_y = np.meshgrid(gx, gy)

    valid = np.isfinite(values)
    if valid.sum() < 3:
        return bg_bgr.copy()

    try:
        interp = LinearNDInterpolator(
            render_coords[valid], values[valid], fill_value=np.nan
        )
        grid_vals = interp(grid_x, grid_y)  # (H, W), NaN outside hull
    except Exception:
        return bg_bgr.copy()

    # Apply ROI mask: outside the mask -> NaN -> transparent.
    # Priority: deformed_mask (per-frame / warped) > roi_mask (reference).
    if deformed_mask is not None:
        grid_vals[~deformed_mask] = np.nan
    elif roi_mask is not None:
        grid_vals[~roi_mask] = np.nan

    # Normalise to [0, 1]
    if field_cfg.auto_range:
        finite = grid_vals[np.isfinite(grid_vals)]
        vmin = float(finite.min()) if len(finite) > 0 else 0.0
        vmax = float(finite.max()) if len(finite) > 0 else 1.0
    else:
        vmin, vmax = field_cfg.vmin, field_cfg.vmax
    span = (vmax - vmin) or 1.0

    # Apply matplotlib colormap -> RGBA; NaN pixels get alpha=0 (transparent)
    try:
        cm = colormaps[field_cfg.colormap]
    except KeyError:
        cm = colormaps["jet"]
    nan_mask = np.isnan(grid_vals)
    normed = np.where(nan_mask, 0.0, np.clip((grid_vals - vmin) / span, 0.0, 1.0))
    rgba = (cm(normed) * 255).astype(np.uint8)  # (H, W, 4) RGBA
    rgba[nan_mask, 3] = 0  # transparent outside hull / mask

    # Composite over background
    # bg_alpha is re-purposed as field opacity (0=invisible, 1=fully opaque),
    # matching GUI overlay_alpha semantics.
    field_opacity = float(field_cfg.bg_alpha)
    alpha_w = rgba[:, :, 3:4].astype(np.float32) / 255.0 * field_opacity  # (H, W, 1)

    field_bgr = rgba[:, :, [2, 1, 0]].astype(np.float32)  # RGB -> BGR
    result = (
        bg_bgr.astype(np.float32) * (1.0 - alpha_w) + field_bgr * alpha_w
    ).astype(np.uint8)

    return result


def _compute_warped_mask(
    coords: NDArray,
    deformed_coords: NDArray,
    roi_mask: NDArray,
    image_shape: tuple[int, int],
) -> NDArray:
    """Warp reference-frame ROI mask to deformed coords via inverse displacement.

    For each pixel (x, y) in deformed space, compute the corresponding
    reference position (x - u, y - v) and look up the reference mask.
    This reproduces the GUI's VizController._warp_cache logic.

    Returns:
        (H, W) bool array — True = inside ROI in deformed space.
    """
    H, W = image_shape
    u_disp = deformed_coords[:, 0] - coords[:, 0]
    v_disp = deformed_coords[:, 1] - coords[:, 1]

    # Share one Delaunay triangulation for both displacement components
    tri = Delaunay(deformed_coords)
    u_interp = LinearNDInterpolator(tri, u_disp, fill_value=np.nan)
    v_interp = LinearNDInterpolator(tri, v_disp, fill_value=np.nan)

    gx = np.arange(W, dtype=np.float64)
    gy = np.arange(H, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(gx, gy)

    u_grid = u_interp(grid_x, grid_y)
    v_grid = v_interp(grid_x, grid_y)

    # Reference positions of deformed grid points
    xr = grid_x - u_grid
    yr = grid_y - v_grid

    nan_warp = np.isnan(xr) | np.isnan(yr)
    xr_safe = np.nan_to_num(xr, nan=0.0)
    yr_safe = np.nan_to_num(yr, nan=0.0)
    xi = np.clip(np.round(xr_safe).astype(int), 0, W - 1)
    yi = np.clip(np.round(yr_safe).astype(int), 0, H - 1)

    return roi_mask[yi, xi] & ~nan_warp


def export_png(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    configs: list,
    image_files: list[str],
    bg_mode: str,
    roi_mask: NDArray | None,
    dpi: int,
    show_deformed: bool,
    frame_start: int,
    frame_end: int,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    per_frame_rois: dict[int, NDArray] | None = None,
    include_colorbar: bool = False,
    use_physical_units: bool = False,
    pixel_size: float = 1.0,
    pixel_unit: str = "mm",
) -> list[Path]:
    """Render and save PNG images for each enabled field and frame.

    Args:
        dest_dir:    Parent output directory.
        prefix:      Filename prefix.
        timestamp:   14-digit timestamp string.
        results:     Full pipeline results.
        configs:     List of FieldImageConfig, one per field.
        image_files: Ordered list of source image file paths.
        bg_mode:     "ref_frame" (always use frame 0 image) or
                     "current_frame" (use each frame's own image).
        roi_mask:    Optional (H, W) bool mask for field trimming
                     (reference coords).
        dpi:         Output DPI (PNG metadata only; does not affect pixels).
        show_deformed: Shift node positions by accumulated displacement.
        frame_start: First frame index (inclusive).
        frame_end:   Last frame index (inclusive; -1 = last frame).
        stop_event:  If set, stop after the current frame.
        progress_callback: Called with (frames_done, total_frames).
        per_frame_rois: Optional mapping of image-file index → (H, W) bool
                     mask.  Keys use the same scheme as AppState.per_frame_rois
                     (0 = reference frame, 1..N = deformed frames).  When a
                     per-frame mask exists for the current frame, it overrides
                     the inverse-warped reference mask in deformed mode.
        include_colorbar: Append a matplotlib-rendered colorbar strip to the
                     right of each exported image.
        use_physical_units: Scale displacement values by *pixel_size* and
                     show physical units on colorbar labels.
        pixel_size:  Physical size of one pixel (e.g. mm/px).
        pixel_unit:  Unit string shown on colorbar labels (e.g. "mm").

    Returns:
        List of Paths to written PNG files.
    """
    from al_dic.gui.dialogs.export_dialog import FieldImageConfig  # avoid circular

    images_dir = dest_dir / f"{prefix}_images_{timestamp}"
    n_frames = len(results.result_disp)
    if frame_end < 0 or frame_end >= n_frames:
        frame_end = n_frames - 1

    enabled_configs = [c for c in configs if c.enabled]
    if not enabled_configs:
        return []

    coords = results.dic_mesh.coordinates_fem
    img_shape = results.dic_para.img_size
    if img_shape == (0, 0):
        img_shape = (256, 256)  # fallback for tests

    total_frames = frame_end - frame_start + 1
    frames_done = 0
    paths: list[Path] = []

    for t in range(frame_start, frame_end + 1):
        if stop_event is not None and stop_event.is_set():
            break

        tag = frame_tag(t, n_frames)
        fr = results.result_disp[t]

        # Deformed node positions (optional)
        if show_deformed and fr.U_accum is not None:
            u, v = split_uv(fr.U_accum)
            deformed_coords: NDArray | None = coords + np.column_stack([u, v])
        elif show_deformed and fr.U is not None:
            u, v = split_uv(fr.U)
            deformed_coords = coords + np.column_stack([u, v])
        else:
            deformed_coords = None

        # Load background image for this frame.
        # result_disp[t] corresponds to image_files[t + 1] (deformed frame),
        # so use t + 1 to match the per-frame ROI indexing.
        bg_image = _load_frame_image(image_files, t + 1, bg_mode)

        # Resolve deformed mask (computed once per frame, shared across fields).
        # Priority matches GUI VizController:
        #   1. Explicit per-frame ROI -> direct lookup in deformed coords.
        #   2. Inverse-warped reference mask -> map deformed grid -> ref coords.
        #   3. (Non-deformed mode) reference mask applied directly.
        deformed_mask: NDArray | None = None
        if show_deformed and deformed_coords is not None:
            # per_frame_rois keys are image-file indices:
            #   0 = reference frame, 1..N = deformed frames.
            # result_disp[t] corresponds to image index t + 1.
            img_idx = t + 1
            if per_frame_rois is not None:
                pfr = per_frame_rois.get(img_idx)
                if pfr is not None:
                    deformed_mask = pfr

            # Fallback: inverse-warp the reference mask
            if deformed_mask is None and roi_mask is not None:
                deformed_mask = _compute_warped_mask(
                    coords, deformed_coords, roi_mask, img_shape
                )

        for cfg in enabled_configs:
            raw_values = _extract_field_values(cfg.field_name, t, results, fr)
            if raw_values is None:
                continue

            # Physical-unit scaling (displacement only; strain is dimensionless)
            values = (scale_field_values(raw_values, cfg.field_name, pixel_size)
                      if use_physical_units else raw_values)

            # Pre-compute actual vmin/vmax when colorbar is requested, or
            # when physical-unit scaling changes the value range (displacement only).
            is_scaled = use_physical_units and cfg.field_name in _DISPLACEMENT_FIELDS
            need_precompute = include_colorbar or is_scaled
            if need_precompute:
                finite = values[np.isfinite(values)]
                if cfg.auto_range:
                    actual_vmin = float(finite.min()) if len(finite) > 0 else 0.0
                    actual_vmax = float(finite.max()) if len(finite) > 0 else 1.0
                else:
                    actual_vmin, actual_vmax = cfg.vmin, cfg.vmax
                    # Scale user-entered fixed range to physical units
                    if use_physical_units and cfg.field_name in _DISPLACEMENT_FIELDS:
                        actual_vmin *= pixel_size
                        actual_vmax *= pixel_size
                render_cfg = replace(cfg, auto_range=False,
                                     vmin=actual_vmin, vmax=actual_vmax)
            else:
                render_cfg = cfg

            img = render_field_frame(
                coords=coords,
                values=values,
                image_shape=img_shape,
                bg_image=bg_image,
                field_cfg=render_cfg,
                roi_mask=roi_mask,
                deformed_coords=deformed_coords,
                deformed_mask=deformed_mask,
            )

            # Append colorbar strip to the right of the image
            if include_colorbar:
                cb_lbl = colorbar_label(
                    cfg.field_name, use_physical_units, pixel_unit)
                cb = render_colorbar_strip(
                    img.shape[0], cfg.colormap,
                    actual_vmin, actual_vmax, cb_lbl, dpi)
                if cb is not None:
                    img = np.hstack([img, cb])

            field_dir = images_dir / cfg.field_name
            ensure_dir(field_dir)
            out = field_dir / f"{tag}.png"
            cv2.imwrite(str(out), img)
            paths.append(out)

        frames_done += 1
        if progress_callback is not None:
            progress_callback(frames_done, total_frames)

    return paths


def _extract_field_values(
    field_name: str,
    frame: int,
    results: PipelineResult,
    fr,
) -> NDArray | None:
    """Extract the (N,) array for *field_name* at *frame*."""
    U = fr.U_accum if fr.U_accum is not None else fr.U
    if U is None:
        return None
    u, v = split_uv(U)

    if field_name == "disp_u":
        return u
    if field_name == "disp_v":
        return v
    if field_name == "disp_magnitude":
        return np.sqrt(u ** 2 + v ** 2)

    if not results.result_strain or frame >= len(results.result_strain):
        return None
    sr = results.result_strain[frame]
    return getattr(sr, field_name, None)
