"""Export pipeline results as animated GIF or MP4.

Rendering pipeline:
    1. For each frame, call render_field_frame() (reused from export_png).
    2. Collect rendered BGR uint8 images.
    3. GIF:  convert BGR→RGB, pass frames to imageio.mimwrite.
    4. MP4:  write frames via cv2.VideoWriter (fourcc mp4v; falls back to avi).

Directory structure:
    dest_dir/
      {prefix}_animation_{timestamp}/
        {field_name}.mp4   (or .gif)
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from al_dic.core.data_structures import PipelineResult, split_uv
from al_dic.export.export_utils import ensure_dir, frame_tag


def export_animation(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
    configs: list,
    image_files: list[str],
    bg_mode: str,
    roi_mask: NDArray | None,
    fmt: str,
    fps: int,
    show_deformed: bool,
    frame_start: int,
    frame_end: int,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    per_frame_rois: dict[int, NDArray] | None = None,
    include_colorbar: bool = False,
    use_physical_units: bool = False,
    pixel_size: float = 1.0,
    pixel_unit: str = "mm",
) -> list[Path]:
    """Export one animation file per enabled field.

    Args:
        dest_dir: Parent output directory.
        prefix: Filename prefix.
        timestamp: 14-digit timestamp string.
        results: Full pipeline results.
        configs: List of FieldImageConfig from export_dialog, one per field.
        image_files: Ordered list of source image file paths.
        bg_mode: "ref_frame" or "current_frame".
        roi_mask: Optional (H, W) bool mask for field trimming
                  (reference coords).
        fmt: Animation format: "mp4" or "gif".
        fps: Frames per second.
        show_deformed: Shift node positions by accumulated displacement.
        frame_start: First frame index (inclusive).
        frame_end: Last frame index (inclusive; -1 = last frame).
        stop_event: If set, stop after the current frame.
        progress_callback: Called with (frames_done, total_frames, field_name).
        per_frame_rois: Optional mapping of image-file index → (H, W) bool
                     mask.  Overrides inverse-warped mask in deformed mode.
                     Keys: 0 = reference, 1..N = deformed frames.
        include_colorbar: Append a matplotlib-rendered colorbar strip to the
                     right of each frame.
        use_physical_units: Scale displacement values by *pixel_size*.
        pixel_size:  Physical size of one pixel (e.g. mm/px).
        pixel_unit:  Unit string shown on colorbar labels.

    Returns:
        List of Paths to written animation files (one per enabled field).
    """
    from al_dic.export.export_png import (
        render_field_frame, _extract_field_values, _load_frame_image,
        _compute_warped_mask, scale_field_values, colorbar_label,
        render_colorbar_strip, _DISPLACEMENT_FIELDS,
    )

    anim_dir = dest_dir / f"{prefix}_animation_{timestamp}"
    ensure_dir(anim_dir)

    n_frames = len(results.result_disp)
    if frame_end < 0 or frame_end >= n_frames:
        frame_end = n_frames - 1

    enabled_configs = [c for c in configs if c.enabled]
    if not enabled_configs:
        return []

    coords = results.dic_mesh.coordinates_fem
    img_shape = results.dic_para.img_size  # (H, W)
    if img_shape == (0, 0):
        img_shape = (256, 256)

    total_frames = frame_end - frame_start + 1
    fmt = fmt.lower()
    paths: list[Path] = []

    for cfg in enabled_configs:
        frames_done = 0
        rendered: list[NDArray] = []

        for t in range(frame_start, frame_end + 1):
            if stop_event is not None and stop_event.is_set():
                break

            fr = results.result_disp[t]

            # Deformed node positions
            if show_deformed and fr.U_accum is not None:
                u, v = split_uv(fr.U_accum)
                deformed_coords: NDArray | None = coords + np.column_stack([u, v])
            elif show_deformed and fr.U is not None:
                u, v = split_uv(fr.U)
                deformed_coords = coords + np.column_stack([u, v])
            else:
                deformed_coords = None

            # result_disp[t] corresponds to image_files[t + 1] (deformed frame)
            bg_image = _load_frame_image(image_files, t + 1, bg_mode)

            # Resolve deformed mask (same logic as export_png)
            deformed_mask: NDArray | None = None
            if show_deformed and deformed_coords is not None:
                img_idx = t + 1
                if per_frame_rois is not None:
                    pfr = per_frame_rois.get(img_idx)
                    if pfr is not None:
                        deformed_mask = pfr
                if deformed_mask is None and roi_mask is not None:
                    deformed_mask = _compute_warped_mask(
                        coords, deformed_coords, roi_mask, img_shape
                    )

            raw_values = _extract_field_values(cfg.field_name, t, results, fr)
            if raw_values is None:
                # Insert a blank frame to keep timing consistent
                rendered.append(np.zeros((*img_shape, 3), dtype=np.uint8))
            else:
                # Physical-unit scaling
                values = (scale_field_values(raw_values, cfg.field_name, pixel_size)
                          if use_physical_units else raw_values)

                # Pre-compute range when colorbar is requested, or
                # when physical-unit scaling changes the value range.
                is_scaled = use_physical_units and cfg.field_name in _DISPLACEMENT_FIELDS
                need_precompute = include_colorbar or is_scaled
                if need_precompute:
                    finite = values[np.isfinite(values)]
                    if cfg.auto_range:
                        actual_vmin = float(finite.min()) if len(finite) > 0 else 0.0
                        actual_vmax = float(finite.max()) if len(finite) > 0 else 1.0
                    else:
                        actual_vmin, actual_vmax = cfg.vmin, cfg.vmax
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

                # Append colorbar strip
                if include_colorbar:
                    cb_lbl = colorbar_label(
                        cfg.field_name, use_physical_units, pixel_unit)
                    cb = render_colorbar_strip(
                        img.shape[0], cfg.colormap,
                        actual_vmin, actual_vmax, cb_lbl)
                    if cb is not None:
                        img = np.hstack([img, cb])

                rendered.append(img)

            frames_done += 1
            if progress_callback is not None:
                progress_callback(frames_done, total_frames, cfg.field_name)

        if not rendered:
            continue

        # Determine output frame size from first rendered frame
        out_shape = (rendered[0].shape[0], rendered[0].shape[1])

        if fmt == "gif":
            out = anim_dir / f"{cfg.field_name}.gif"
            _write_gif(rendered, out, fps)
        else:
            out = anim_dir / f"{cfg.field_name}.mp4"
            if not _write_mp4(rendered, out, fps, out_shape):
                # Fallback to AVI if mp4v fails
                out = anim_dir / f"{cfg.field_name}.avi"
                _write_avi(rendered, out, fps, out_shape)

        paths.append(out)

    return paths


def _write_gif(frames: list[NDArray], out: Path, fps: int) -> None:
    """Write BGR frames as an animated GIF via imageio."""
    import imageio

    # imageio expects RGB uint8
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    duration = 1.0 / max(fps, 1)
    imageio.mimwrite(str(out), rgb_frames, format="GIF", duration=duration, loop=0)


def _write_mp4(frames: list[NDArray], out: Path, fps: int,
               img_shape: tuple[int, int]) -> bool:
    """Write BGR frames as MP4 via cv2.VideoWriter. Returns True on success."""
    H, W = img_shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (W, H))
    if not writer.isOpened():
        return False
    for frame in frames:
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        writer.write(frame)
    writer.release()
    return out.exists() and out.stat().st_size > 0


def _write_avi(frames: list[NDArray], out: Path, fps: int,
               img_shape: tuple[int, int]) -> None:
    """Fallback: write BGR frames as AVI (XVID)."""
    H, W = img_shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (W, H))
    for frame in frames:
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        writer.write(frame)
    writer.release()
