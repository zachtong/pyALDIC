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

import functools
import logging
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

from al_dic.core.data_structures import PipelineResult, split_uv
from al_dic.export.export_utils import ensure_dir, frame_tag
from al_dic.utils.crack_barrier import (
    crack_aware_nearest_fill,
    cross_crack_cell_mask,
)
# Colorbar rendering lives in its own module (no cycle); re-export
# render_colorbar_strip here for backward-compatible imports.
from al_dic.export.colorbar import (  # noqa: F401
    ColorbarStyle, add_margin, attach_colorbar, render_colorbar_strip,
)


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


logger = logging.getLogger(__name__)

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
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    return bgr


def _resize_mask(mask: NDArray, H: int, W: int) -> NDArray:
    """Resize a mask to (H, W) as bool, keeping a hard (nearest) edge.

    Always returns bool: callers combine the result with ``&=``, and the
    same mask arrays are handed to ``run_aldic`` as float64, so a float
    (or uint8) mask that happened to need no resizing used to pass through
    unconverted and raise a bitwise-and TypeError.
    """
    if mask.shape == (H, W):
        return mask.astype(bool, copy=False)
    return cv2.resize(mask.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def output_shape_for(image_shape: tuple[int, int], max_dim: int) -> tuple[int, int]:
    """Scale *image_shape* (H, W) so its long edge is <= *max_dim*.

    Returns *image_shape* unchanged when *max_dim* is 0/negative or already
    within the cap.  Aspect ratio preserved.
    """
    H, W = image_shape
    if max_dim <= 0 or max(H, W) <= max_dim:
        return image_shape
    s = max_dim / max(H, W)
    return (max(1, round(H * s)), max(1, round(W * s)))


def encode_params_for(ext: str, jpeg_quality: int) -> list[int]:
    """cv2.imwrite params for a file extension: JPEG quality, fast PNG."""
    e = ext.lower()
    if e in (".jpg", ".jpeg"):
        return [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    if e == ".png":
        # Level 1 = fast; higher levels are far slower for marginal size gain
        # on speckle-heavy DIC frames (measured ~5x slower for ~5% smaller).
        return [cv2.IMWRITE_PNG_COMPRESSION, 1]
    if e == ".webp":
        return [cv2.IMWRITE_WEBP_QUALITY, int(jpeg_quality)]
    return []


#: Fill used in place of a background image when the user hides it.
#: "transparent" is honoured only by formats with an alpha channel; the
#: exporters downgrade it to white for JPEG, GIF and MP4 rather than refusing.
HIDDEN_BG_COLORS: tuple[str, ...] = ("white", "black", "transparent")
_HIDDEN_BG_FILL = {"white": 255, "black": 0, "transparent": 0}


def hidden_bg_fill(color: str) -> int:
    """Grey level painted where the background image would have been."""
    return _HIDDEN_BG_FILL.get(color, 0)


def _load_frame_image(
    image_files: list[str], frame: int, bg_mode: str
) -> NDArray | None:
    """Load a background image for the given frame index.

    Args:
        image_files: Ordered list of source image paths.
        frame:       Current frame index (0-based).
        bg_mode:     "ref_frame" -> always load index 0;
                     "current_frame" -> load the matching frame;
                     "none" -> the user hid the background.
    Returns:
        (H, W) uint8 grayscale array, or None if unavailable or hidden.
    """
    if bg_mode == "none" or not image_files:
        return None
    idx = 0 if bg_mode == "ref_frame" else min(frame, len(image_files) - 1)
    # cv2.imread cannot open a non-ASCII path on Windows -- it returns None,
    # which here silently produced a blank background for every export made
    # from a folder named in Chinese, Japanese or Korean. Decode from bytes we
    # read ourselves, the idiom used everywhere else in this codebase.
    try:
        buf = np.fromfile(image_files[idx], dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except OSError as exc:
        logger.warning("Background image %s could not be read: %s",
                       image_files[idx], exc)
        return None
    if img is None:
        logger.warning("Background image %s could not be decoded",
                       image_files[idx])
    return img  # (H, W) uint8, or None


@functools.lru_cache(maxsize=32)
def _colormap_bgr_lut(cmap_name: str) -> NDArray:
    """256-entry BGR uint8 LUT sampled from a matplotlib colormap.

    Indexing this LUT with quantised field values reproduces
    ``colormaps[name](x)`` to within one grey level for the standard
    256-entry colormaps (jet, viridis, ...), but is ~7x faster than calling
    matplotlib per pixel over a full-resolution grid.  Cached per colormap.
    """
    from al_dic.core.colormaps import resolve

    cm = resolve(cmap_name)
    rgba = (cm(np.linspace(0.0, 1.0, 256)) * 255).astype(np.uint8)  # (256,4)
    return np.ascontiguousarray(rgba[:, [2, 1, 0]])  # RGB -> BGR, (256, 3)


def render_field_frame(
    coords: NDArray,
    values: NDArray,
    image_shape: tuple[int, int],
    bg_image: NDArray | None,
    field_cfg: "FieldImageConfig",
    roi_mask: NDArray | None = None,
    deformed_coords: NDArray | None = None,
    deformed_mask: NDArray | None = None,
    tri_cache: dict | None = None,
    render_max_dim: int = 0,
    output_shape: tuple[int, int] | None = None,
    blank_invalid_nodes: bool = True,
    hidden_bg_color: str = "black",
) -> NDArray:
    """Render a single field frame to a BGR (or BGRA) uint8 image.

    *hidden_bg_color* only applies when *bg_image* is None. "transparent"
    returns four channels; the default stays black so that existing callers,
    which pass no background at all in tests, keep their current output.

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
        render_max_dim:  Downsample the internal render grid so its long edge
                         is <= this (0 = full output resolution).
        output_shape:    (Ho, Wo) size of the produced image.  Defaults to
                         *image_shape*.  When smaller, the field is sampled
                         over the full coordinate span but rendered, composited
                         and returned at this size -- cutting file size and
                         encode time without upscaling back to native.
        blank_invalid_nodes: When True (default), blank grid cells whose
                         nearest node carries NaN (edge-trimmed strain), so the
                         trim is visible.  When False, skip the blanking and let
                         the interpolator re-fill the trimmed band from reliable
                         interior nodes ("fill trimmed edges").  No-op for
                         fields without trimmed nodes (e.g. displacement).

    Returns:
        (Ho, Wo, 3) BGR uint8 image (Ho, Wo = *output_shape* or *image_shape*).
    """
    H, W = image_shape                                  # coordinate span
    Ho, Wo = output_shape if output_shape is not None else image_shape
    render_coords = deformed_coords if deformed_coords is not None else coords

    # Prepare background at the OUTPUT resolution.  With no image the user
    # picked the fill; "transparent" is carried as alpha at the end, so the
    # colour under it only matters for pixels nothing else covers.
    hidden = bg_image is None
    if hidden:
        bg_bgr = np.full((Ho, Wo, 3), hidden_bg_fill(hidden_bg_color),
                         dtype=np.uint8)
    else:
        bg_bgr = _to_bgr(bg_image, Ho, Wo)

    # A node contributes only when BOTH its value and its render position are
    # finite.  Crack-destroyed nodes carry NaN deformed positions (crack-aware
    # cumulative transform), and Delaunay/qhull rejects non-finite points --
    # filter them here so the triangulation below never sees a NaN.  Bit-exact
    # when every position is finite.
    valid = np.isfinite(values) & np.isfinite(render_coords).all(axis=1)
    if valid.sum() < 3:
        return bg_bgr.copy()

    # Coarse render grid derived from the OUTPUT size: the field is a linear
    # interpolation of the mesh nodes, so its detail is bounded by the node
    # spacing.  Evaluating on a downsampled grid then upscaling is visually
    # lossless in the interior while cutting the O(pixels) interpolate +
    # colormap cost by scale^2.  The grid always spans the full coordinate
    # range so node positions map correctly regardless of output size.
    if render_max_dim and max(Ho, Wo) > render_max_dim:
        scale = int(np.ceil(max(Ho, Wo) / render_max_dim))
    else:
        scale = 1
    rh = max(2, int(np.ceil(Ho / scale)))
    rw = max(2, int(np.ceil(Wo / scale)))

    gx = np.linspace(0, W - 1, rw)
    gy = np.linspace(0, H - 1, rh)
    grid_x, grid_y = np.meshgrid(gx, gy)

    try:
        # Reuse the Delaunay across frames/fields sharing the same valid
        # mask. The caller scopes tri_cache so render_coords is constant
        # within it (non-deformed: shared across frames; deformed: per
        # frame). Always build an explicit Delaunay (byte-identical to letting
        # LinearNDInterpolator build it internally) so the crack-aware pass
        # below can reuse the same triangulation via find_simplex.
        pts = render_coords[valid]
        tri = None
        if tri_cache is not None:
            key = valid.tobytes()
            tri = tri_cache.get(key)
        if tri is None:
            tri = Delaunay(pts)
            if tri_cache is not None:
                tri_cache[key] = tri
        interp = LinearNDInterpolator(tri, values[valid], fill_value=np.nan)
        grid_vals = interp(grid_x, grid_y)  # (rh, rw), NaN outside hull

        # Blank cells whose nearest node was edge-trimmed (NaN value).  The
        # interpolator above drops those nodes and back-fills their cells from
        # valid neighbours, which would hide the strain edge-trim in the
        # exported image/animation -- unlike the on-screen view, where
        # VizController blanks them.  Rasterise node finite-ness by nearest
        # neighbour (matching VizController._invalid_node_grid) and NaN the
        # trimmed cells so display and export agree.  No-op when nothing is
        # trimmed (e.g. displacement fields), so those exports are unchanged.
        # Skipped when *blank_invalid_nodes* is False ("fill trimmed edges"):
        # the back-fill above then becomes the desired re-interpolation.
        node_ok = np.isfinite(render_coords).all(axis=1)
        has_trim = node_ok.any() and not bool(valid[node_ok].all())
        if blank_invalid_nodes and has_trim:
            nn = NearestNDInterpolator(
                render_coords[node_ok], valid[node_ok].astype(np.float64)
            )
            keep = nn(grid_x, grid_y) >= 0.5
            grid_vals = np.where(keep, grid_vals, np.nan)

        # Crack-aware rendering: the fresh Delaunay above reconnects nodes on
        # opposite sides of a crack that the mesh split, smearing the
        # discontinuity.  Blank grid cells inside a triangle whose edge crosses
        # the barrier (crack / hole) so the render respects it like the mesh
        # does.  Barrier is in the same coordinate space as render_coords:
        # deformed_mask in deformed mode, else roi_mask.  Bit-exact (returns
        # None) when no triangle crosses a barrier -- crack-free exports are
        # unchanged.
        barrier = deformed_mask if deformed_coords is not None else roi_mask
        crack_blank = cross_crack_cell_mask(tri, pts, grid_x, grid_y, barrier)
        if crack_blank is not None:
            grid_vals = np.where(crack_blank, np.nan, grid_vals)

        # "Fill trimmed edges" (fill ON): recover every trimmed cell from
        # reliable nodes -- the ROI OUTER edge (beyond the shrunken hull) AND
        # the crack INNER edge (just blanked above).  Runs AFTER the crack pass
        # and fills each cell from the nearest node reachable WITHOUT crossing
        # the crack, so the two crack faces are filled from their own side and
        # the crack stays a sharp line.  Restricted to inside the barrier ROI;
        # the crack void itself (barrier < 0.5) is left transparent.
        if not blank_invalid_nodes and has_trim:
            need = np.isnan(grid_vals)
            if barrier is not None:
                b = np.asarray(barrier)
                bx = np.clip(np.round(grid_x).astype(np.int64), 0, b.shape[1] - 1)
                by = np.clip(np.round(grid_y).astype(np.int64), 0, b.shape[0] - 1)
                need &= b[by, bx] > 0.5
            if need.any():
                fill = crack_aware_nearest_fill(
                    grid_x, grid_y, need, pts, values[valid], barrier,
                )
                grid_vals = np.where(need, fill, grid_vals)
    except Exception:
        return bg_bgr.copy()

    # Normalise to [0, 1] (auto-range from the coarse grid is within a grey
    # level of the full-res range for a smooth field).
    if field_cfg.auto_range:
        finite = grid_vals[np.isfinite(grid_vals)]
        vmin = float(finite.min()) if len(finite) > 0 else 0.0
        vmax = float(finite.max()) if len(finite) > 0 else 1.0
    else:
        vmin, vmax = field_cfg.vmin, field_cfg.vmax
    span = (vmax - vmin) or 1.0

    # Colormap via a precomputed BGR LUT (fast, matplotlib-accurate).
    nan_mask = np.isnan(grid_vals)
    idx = np.clip((grid_vals - vmin) / span * 255.0, 0.0, 255.0)
    idx[nan_mask] = 0.0
    lut = _colormap_bgr_lut(field_cfg.colormap)
    field_small = lut[idx.astype(np.uint8)]                   # (rh, rw, 3) BGR
    valid_small = np.where(nan_mask, np.uint8(0), np.uint8(255))  # (rh, rw)

    # Upscale colour + hull validity to output resolution.
    if (rh, rw) != (Ho, Wo):
        field_bgr = cv2.resize(field_small, (Wo, Ho), interpolation=cv2.INTER_LINEAR)
        valid_full = cv2.resize(valid_small, (Wo, Ho), interpolation=cv2.INTER_LINEAR)
    else:
        field_bgr = field_small
        valid_full = valid_small

    # "inside" = inside the mesh hull AND inside the ROI mask.  The ROI mask is
    # applied at the output resolution (resized nearest) so its edge stays as
    # crisp as the output allows.  Priority: deformed_mask > roi_mask.
    inside = valid_full >= 128
    if deformed_mask is not None:
        inside &= _resize_mask(deformed_mask, Ho, Wo)
    elif roi_mask is not None:
        inside &= _resize_mask(roi_mask, Ho, Wo)

    # Composite over background.  bg_alpha is re-purposed as field opacity
    # (0=invisible, 1=fully opaque), matching GUI overlay_alpha semantics.
    # The colormap alpha is binary here (opaque inside the hull, transparent
    # outside), so a single SIMD blend + restore-background reproduces the
    # per-pixel formula bg*(1-op) + field*op exactly, ~4x faster than the
    # float64 numpy path.
    op = float(field_cfg.bg_alpha)

    if hidden and hidden_bg_color == "transparent":
        # Nothing to blend toward: a 70% field over black is just a darker
        # field, which is not what the opacity control means here.  Spend the
        # opacity on the alpha channel instead and leave the colour alone.
        alpha = (inside * (op * 255.0)).clip(0, 255).astype(np.uint8)
        return np.dstack([field_bgr, alpha])

    blended = cv2.addWeighted(bg_bgr, 1.0 - op, field_bgr, op, 0.0)
    result = np.where(inside[:, :, None], blended, bg_bgr)

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

    # Crack-destroyed nodes carry NaN deformed positions (crack-aware cumulative
    # transform); Delaunay/qhull rejects non-finite points, so build the warp
    # triangulation from the live nodes only.  Bit-exact when all are finite.
    finite = np.isfinite(deformed_coords).all(axis=1)
    if finite.sum() < 3:
        return np.zeros((H, W), dtype=bool)

    # Share one Delaunay triangulation for both displacement components
    tri = Delaunay(deformed_coords[finite])
    u_interp = LinearNDInterpolator(tri, u_disp[finite], fill_value=np.nan)
    v_interp = LinearNDInterpolator(tri, v_disp[finite], fill_value=np.nan)

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
    image_format: str = "png",
    render_max_dim: int = 1536,
    output_max_dim: int = 0,
    jpeg_quality: int = 92,
    colorbar_style: ColorbarStyle | None = None,
    margin_ratio: float = 0.0,
    margin_color: str = "white",
    fill_trimmed_edges: bool = False,
    hidden_bg_color: str = "black",
) -> list[Path]:
    """Render and save images for each enabled field and frame.

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
        fill_trimmed_edges: When True, re-interpolate the edge-trimmed strain
                     band from reliable interior nodes instead of blanking it
                     (matches the strain window's toggle).  Default False.

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

    # Map the selected image format to a file extension; cv2.imwrite picks
    # the encoder from the extension (.png / .jpg / .tif). Previously the
    # format combo was ignored and everything was written as .png.
    ext = {
        "png": ".png", "jpeg": ".jpg", "jpg": ".jpg",
        "tiff": ".tif", "tif": ".tif",
    }.get(image_format.lower(), ".png")

    # JPEG has no alpha channel.  Downgrading to a white fill loses the
    # transparency but still produces the file the user asked for; encoding a
    # four-channel array would simply fail.
    if hidden_bg_color == "transparent" and ext == ".jpg":
        hidden_bg_color = "white"

    coords = results.dic_mesh.coordinates_fem
    img_shape = results.dic_para.img_size
    if img_shape == (0, 0):
        img_shape = (256, 256)  # fallback for tests

    # Output resolution cap (file size + encode time) and encode parameters.
    out_shape = output_shape_for(img_shape, output_max_dim)
    enc_params = encode_params_for(ext, jpeg_quality)
    cb_style = colorbar_style if colorbar_style is not None else ColorbarStyle()

    total_frames = frame_end - frame_start + 1
    frames_done = 0
    paths: list[Path] = []

    # Reuse the field triangulation across frames/fields. In non-deformed
    # mode render_coords == coords is constant, so one Delaunay per valid
    # mask serves the whole export; deformed frames get a fresh cache each
    # (their render coordinates change per frame).
    shared_tri_cache: dict = {}

    # In "ref_frame" mode every frame uses image 0 as background; decode it
    # once instead of re-reading the (potentially 4K) file on every frame.
    ref_bg = (
        _load_frame_image(image_files, 0, "ref_frame")
        if bg_mode == "ref_frame" else None
    )

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
        # so use t + 1 to match the per-frame ROI indexing.  In "ref_frame"
        # mode reuse the single pre-decoded reference image.
        bg_image = ref_bg if ref_bg is not None else _load_frame_image(
            image_files, t + 1, bg_mode)

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

        # Deformed frames each carry their own render coordinates -> per-frame
        # cache; non-deformed frames share the constant reference cache.
        frame_tri_cache = {} if deformed_coords is not None else shared_tri_cache

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
                    # A fixed range is already in the units being exported:
                    # the window stores what the user typed against the unit
                    # label it was showing, and flipping that label does not
                    # convert it. Scaling here made a +-200 um range export as
                    # +-60000 um at 300 um/px.
                    actual_vmin, actual_vmax = cfg.vmin, cfg.vmax
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
                tri_cache=frame_tri_cache,
                render_max_dim=render_max_dim,
                output_shape=out_shape,
                blank_invalid_nodes=not fill_trimmed_edges,
                hidden_bg_color=hidden_bg_color,
            )

            # Append the styled colorbar (position/font/thickness/background)
            if include_colorbar:
                cb_lbl = colorbar_label(
                    cfg.field_name, use_physical_units, pixel_unit)
                img = attach_colorbar(
                    img, cb_style, cfg.colormap,
                    actual_vmin, actual_vmax, cb_lbl, dpi)
            # Expand the canvas outward with a margin (publication layouts)
            img = add_margin(img, margin_ratio, margin_color)

            field_dir = images_dir / cfg.field_name
            ensure_dir(field_dir)
            out = field_dir / f"{tag}{ext}"
            # Not cv2.imwrite: it fails on non-ASCII paths and returns False
            # rather than raising, so the caller counted images it never wrote.
            ok, buf = cv2.imencode(ext, img, enc_params)
            if not ok:
                raise IOError(f"Failed to encode image {out}")
            buf.tofile(str(out))
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
    return sr.trimmed_field(field_name)  # edge-trim applied (NaN at edges)
