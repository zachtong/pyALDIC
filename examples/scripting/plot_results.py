"""Plot fields from an exported .npz — after the fact, without re-running DIC.

A batch can export data only (drop "images" from the config's ``export`` list):
that runs faster, writes far less, and imports no Qt. Then come back here and
plot whenever you like, as often as you like, with whatever colormap and range
you want — the correlation is never repeated.

    # what's in the file?
    python examples/scripting/plot_results.py results/specimen_01/specimen_01_results_2026...npz

    # one field, all frames -> PNGs next to the npz
    python examples/scripting/plot_results.py <file.npz> --field strain_eyy

    # last frame only, fixed colour range, custom colormap, on screen
    python examples/scripting/plot_results.py <file.npz> --field disp_u --frame -1 \
        --vmin -2 --vmax 2 --cmap turbo --show

The .npz holds ``coordinates`` (N, 2) plus one (N, T) matrix per field — N mesh
nodes, T frames — so anything below is plain matplotlib on those arrays. Use it
as a starting point for your own figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def describe(data) -> None:
    """Print what the archive contains."""
    coords = data["coordinates"]
    fields = [k for k in data.files if k != "coordinates"]
    n_frames = data[fields[0]].shape[1] if fields else 0
    print(f"nodes  : {coords.shape[0]}")
    print(f"frames : {n_frames}")
    print("fields :")
    for f in sorted(fields):
        arr = data[f]
        finite = np.isfinite(arr).mean()
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        print(f"   {f:22s} {str(arr.shape):12s} finite={finite:6.1%}  "
              f"range=[{lo:+.4g}, {hi:+.4g}]")


def plot_frame(coords, values, title, cmap, vmin, vmax, levels):
    """Filled contour of one field on the mesh nodes. Returns the figure."""
    import matplotlib.pyplot as plt

    good = np.isfinite(values)
    if not good.any():
        raise ValueError(f"{title}: every node is NaN, nothing to plot")

    fig, ax = plt.subplots(figsize=(7, 6))
    tpc = ax.tricontourf(
        coords[good, 0], coords[good, 1], values[good],
        levels=levels, cmap=cmap, vmin=vmin, vmax=vmax,
    )
    fig.colorbar(tpc, ax=ax, label=title)
    ax.set_aspect("equal")
    ax.invert_yaxis()             # image convention: y grows downward
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz", type=Path, help="exported *_results_*.npz")
    ap.add_argument("--field", help="field to plot (omit to just list contents)")
    ap.add_argument("--frame", type=int, help="frame index; -1 = last; omit = all")
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--vmin", type=float, help="fixed colour range minimum")
    ap.add_argument("--vmax", type=float, help="fixed colour range maximum")
    ap.add_argument("--levels", type=int, default=40, help="contour levels")
    ap.add_argument("--out", type=Path, help="output folder (default: next to npz)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--show", action="store_true", help="display instead of saving")
    args = ap.parse_args(argv)

    if not args.npz.is_file():
        print(f"not found: {args.npz}")
        return 2

    data = np.load(args.npz)
    if not args.field:
        describe(data)
        print("\nPass --field <name> to plot one.")
        return 0

    if args.field not in data.files:
        print(f"'{args.field}' is not in this archive. Available: "
              f"{sorted(k for k in data.files if k != 'coordinates')}")
        return 2

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")     # headless: never needs a display
    import matplotlib.pyplot as plt

    coords = data["coordinates"]
    field = data[args.field]
    n_frames = field.shape[1]
    frames = range(n_frames) if args.frame is None else [args.frame % n_frames]

    # A shared colour range keeps frames comparable; per-frame autoscale would
    # make a growing field look static.
    vmin = args.vmin if args.vmin is not None else float(np.nanmin(field))
    vmax = args.vmax if args.vmax is not None else float(np.nanmax(field))

    out_dir = args.out or args.npz.parent / f"plots_{args.field}"
    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)

    for t in frames:
        fig = plot_frame(coords, field[:, t],
                         f"{args.field} — frame {t + 1}/{n_frames}",
                         args.cmap, vmin, vmax, args.levels)
        if args.show:
            plt.show()
        else:
            dest = out_dir / f"{args.field}_frame_{t + 1:03d}.png"
            fig.savefig(dest, dpi=args.dpi)
            print(f"wrote {dest}")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
