"""Preview how the demo video will look when step 2 and step 5 are zoomed.

For each step, take a representative frame, crop the user-specified region,
scale it 2x, then paste the enlarged crop into the center of the original
frame with a red border. A faint red outline is drawn at the source crop
location so the source-to-zoom mapping is visually obvious.

Crops are in GIF pixel coords (2124x1482). Out-of-bounds coords are clamped.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GIF_DIR = PROJECT_ROOT / "assets" / "videos"
OUT_DIR = PROJECT_ROOT / "reports" / "preview" / "zoom_proposals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZOOM_SCALE = 2.0
FROST_BLUR_RADIUS = 8
FROST_DIM_ALPHA = 40  # 0 = no dim, 255 = black

PROPOSALS = {
    "step2_select_your_workflow": {
        "crop": (0, 600, 500, 1200),
        "label": "Step 2 zoom x2 -> center  (crop 0,600 -> 500,1200)",
        "frame_idx": 40,
    },
    "step5_computing": {
        "crop": (1700, 80, 2150, 520),
        "label": "Step 5 zoom x2 -> center  (crop 1700,80 -> 2150,520)",
        "frame_idx": 100,
    },
}


def _clamp_crop(crop: tuple[int, int, int, int],
                w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = crop
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _compose(frame: Image.Image, crop: tuple[int, int, int, int],
             label: str) -> Image.Image:
    original = frame.convert("RGB").copy()
    w, h = original.size
    x0, y0, x1, y1 = _clamp_crop(crop, w, h)

    # 1) Frosted background: blur + slight dim, then restore the source
    #    crop region back to sharp.
    frosted = original.filter(ImageFilter.GaussianBlur(FROST_BLUR_RADIUS))
    dim = Image.new("RGBA", (w, h), (0, 0, 0, FROST_DIM_ALPHA))
    frosted = frosted.convert("RGBA")
    frosted.alpha_composite(dim)
    frosted = frosted.convert("RGB")
    frosted.paste(original.crop((x0, y0, x1, y1)), (x0, y0))
    base = frosted

    # 2) Zoomed crop.
    region = original.crop((x0, y0, x1, y1))
    zw = int(round((x1 - x0) * ZOOM_SCALE))
    zh = int(round((y1 - y0) * ZOOM_SCALE))
    zoomed = region.resize((zw, zh), Image.LANCZOS)

    # 3) Center paste coordinates.
    cx = (w - zw) // 2
    cy = (h - zh) // 2

    # 4) Red outline around the source crop region (on the sharp patch).
    draw = ImageDraw.Draw(base, "RGBA")
    draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64, 255), width=6)

    # 5) Paste zoomed crop + red border.
    base.paste(zoomed, (cx, cy))
    draw = ImageDraw.Draw(base, "RGBA")
    draw.rectangle((cx, cy, cx + zw, cy + zh),
                   outline=(255, 64, 64, 255), width=10)

    # 5) Header strip with crop description.
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    pad = 18
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.rectangle((0, 0, tw + 2 * pad, th + 2 * pad),
                   fill=(20, 20, 20, 220))
    draw.text((pad, pad), label, fill=(255, 255, 255, 255), font=font)
    return base


def main() -> None:
    for stem, cfg in PROPOSALS.items():
        gif_path = GIF_DIR / f"{stem}.gif"
        im = Image.open(gif_path)
        n = getattr(im, "n_frames", 1)
        idx = min(cfg["frame_idx"], n - 1)
        im.seek(idx)
        preview = _compose(im, cfg["crop"], cfg["label"])
        out = OUT_DIR / f"{stem}_zoom_center_x2.png"
        preview.save(out, "PNG")
        print(f"wrote {out.relative_to(PROJECT_ROOT)}  "
              f"(frame {idx}/{n - 1}, size={preview.size})")


if __name__ == "__main__":
    main()
