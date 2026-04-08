"""Export pipeline results as per-frame PNG images.

Rendering pipeline (no Qt dependency):
    1. Scatter node values → regular grid via LinearNDInterpolator.
    2. Normalize to [0, 255] uint8.
    3. cv2.applyColorMap → pseudocolour BGR image.
    4. (optional) cv2.addWeighted with reference frame image as background.
    5. cv2.imwrite → PNG file.

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
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator

from staq_dic.core.data_structures import PipelineResult, split_uv
from staq_dic.export.export_utils import ensure_dir, frame_tag

# Mapping from matplotlib-style colormap names to OpenCV colormap codes
_CMAP_MAP: dict[str, int] = {
    "jet":      cv2.COLORMAP_JET,
    "viridis":  cv2.COLORMAP_VIRIDIS,
    "turbo":    cv2.COLORMAP_TURBO,
    "plasma":   cv2.COLORMAP_PLASMA,
    "inferno":  cv2.COLORMAP_INFERNO,
    "coolwarm": cv2.COLORMAP_COOL,
    "RdBu_r":   cv2.COLORMAP_RD_BU if hasattr(cv2, "COLORMAP_RD_BU") else cv2.COLORMAP_COOL,
    "seismic":  cv2.COLORMAP_COOL,
}
_DEFAULT_CMAP = cv2.COLORMAP_JET


def _get_cv2_cmap(name: str) -> int:
    return _CMAP_MAP.get(name, _DEFAULT_CMAP)


def render_field_frame(
    coords: NDArray,
    values: NDArray,
    image_shape: tuple[int, int],
    ref_image: NDArray | None,
    field_cfg: "FieldImageConfig",
    deformed_coords: NDArray | None = None,
) -> NDArray:
    """Render a single field frame to a BGR uint8 image.

    Args:
        coords: (N, 2) node positions in pixel space (x, y).
        values: (N,) field values at each node.
        image_shape: (H, W) output image size in pixels.
        ref_image: Optional (H, W) uint8 greyscale reference image.
        field_cfg: Per-field color and range settings.
        deformed_coords: If given, use these as rendering positions instead.

    Returns:
        (H, W, 3) BGR uint8 image.
    """
    from staq_dic.gui.dialogs.export_dialog import FieldImageConfig  # local import to avoid circular

    H, W = image_shape
    render_coords = deformed_coords if deformed_coords is not None else coords

    # --- Interpolate scatter → regular grid ---
    # Build grid in image pixel coordinates (x=col, y=row)
    gx = np.linspace(0, W - 1, W)
    gy = np.linspace(0, H - 1, H)
    grid_x, grid_y = np.meshgrid(gx, gy)  # both (H, W)

    valid_mask = np.isfinite(values)
    if valid_mask.sum() < 3:
        # Not enough points — return a blank image
        return np.zeros((H, W, 3), dtype=np.uint8)

    try:
        interp = LinearNDInterpolator(
            render_coords[valid_mask],
            values[valid_mask],
            fill_value=np.nan,
        )
        grid_vals = interp(grid_x, grid_y)  # (H, W), NaN outside convex hull
    except Exception:
        # Degenerate geometry (e.g. collinear nodes) — return blank
        return np.zeros((H, W, 3), dtype=np.uint8)

    # --- Normalise to [0, 255] ---
    if field_cfg.auto_range:
        finite = grid_vals[np.isfinite(grid_vals)]
        vmin = float(finite.min()) if len(finite) > 0 else 0.0
        vmax = float(finite.max()) if len(finite) > 0 else 1.0
    else:
        vmin, vmax = field_cfg.vmin, field_cfg.vmax

    span = vmax - vmin
    if span == 0:
        span = 1.0

    norm = np.clip((grid_vals - vmin) / span, 0.0, 1.0)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    gray = (norm * 255).astype(np.uint8)

    # --- Apply colormap ---
    cmap_code = _get_cv2_cmap(field_cfg.colormap)
    coloured = cv2.applyColorMap(gray, cmap_code)  # (H, W, 3) BGR

    # --- Blend with reference image background ---
    if ref_image is not None and field_cfg.bg_alpha < 1.0:
        if ref_image.ndim == 2:
            bg_bgr = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)
        else:
            bg_bgr = ref_image.copy()
        if bg_bgr.shape[:2] != (H, W):
            bg_bgr = cv2.resize(bg_bgr, (W, H))
        # bg_alpha controls background opacity; field is (1-bg_alpha)
        alpha = float(field_cfg.bg_alpha)
        coloured = cv2.addWeighted(bg_bgr, alpha, coloured, 1.0 - alpha, 0)

    return coloured


def export_png(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    configs: list,
    ref_image: NDArray | None,
    dpi: int,
    show_deformed: bool,
    show_background: bool,
    frame_start: int,
    frame_end: int,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Render and save PNG images for each enabled field and frame.

    Args:
        dest_dir: Parent output directory.
        prefix: Filename prefix.
        timestamp: 14-digit timestamp string.
        results: Full pipeline results.
        configs: List of FieldImageConfig, one per field.
        ref_image: Optional reference greyscale image (H, W) uint8.
        dpi: Output DPI (stored in PNG metadata; does not affect pixel size).
        show_deformed: If True, shift node positions by accumulated displacement.
        show_background: If True and ref_image provided, blend background.
        frame_start: First frame index (inclusive).
        frame_end: Last frame index (inclusive, -1 = last frame).
        stop_event: If set, stop after the current frame.
        progress_callback: Called with (frames_done, total_frames).

    Returns:
        List of Paths to written PNG files.
    """
    from staq_dic.gui.dialogs.export_dialog import FieldImageConfig  # avoid circular

    images_dir = dest_dir / f"{prefix}_images_{timestamp}"

    n_frames = len(results.result_disp)
    if frame_end < 0 or frame_end >= n_frames:
        frame_end = n_frames - 1

    enabled_configs = [c for c in configs if c.enabled]
    if not enabled_configs:
        return []

    coords = results.dic_mesh.coordinates_fem
    img_shape = results.dic_para.img_size  # (H, W)
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

        # Deformed node positions
        if show_deformed and fr.U_accum is not None:
            u, v = split_uv(fr.U_accum)
            deformed_coords = coords + np.column_stack([u, v])
        elif show_deformed and fr.U is not None:
            u, v = split_uv(fr.U)
            deformed_coords = coords + np.column_stack([u, v])
        else:
            deformed_coords = None

        # Select appropriate reference image for background
        bg_image = ref_image if show_background else None

        for cfg in enabled_configs:
            values = _extract_field_values(cfg.field_name, t, results, fr)
            if values is None:
                continue

            img = render_field_frame(
                coords=coords,
                values=values,
                image_shape=img_shape,
                ref_image=bg_image,
                field_cfg=cfg,
                deformed_coords=deformed_coords,
            )

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
    # Displacement fields
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

    # Strain fields
    if not results.result_strain or frame >= len(results.result_strain):
        return None
    sr = results.result_strain[frame]
    return getattr(sr, field_name, None)
