# Scripting examples

Using pyALDIC from Python, without the interface. Both scripts call the same
`run_aldic` entry point the GUI does, so results are identical to pressing Run.

## `batch_process.py` — many samples from one config

Describe your samples in a JSON or YAML file and run the lot:

```bash
python examples/scripting/batch_process.py examples/scripting/batch_config.example.json
```

The bundled config processes the three `quickstart/` cases, so it runs as-is
from the repository root. A sample that fails is logged and the batch carries
on to the next one; with `resume: true` an interrupted batch can simply be
re-run.

`batch_config.example.yaml` documents every supported key; the `.json` twin has
identical keys and needs no extra dependency.

## `plot_results.py` — figures after the fact

Plot fields from an exported `.npz` without repeating the correlation. Run it
with no options to see what a file contains:

```bash
python examples/scripting/plot_results.py results/specimen_01/..._results_....npz
```

then pick a field, frame, colour range and colormap.

Dropping `images` from the config's `export` list gives a data-only batch: it
runs faster, writes far less, and imports no Qt at all — useful on a machine
with no display. Plot later with this script instead.

## A parameter worth knowing about

`winsize` in a config file must be **even**. The interface shows the subset
size as an **odd** number, following the DIC convention that a subset has one
centre pixel — so the quickstart README's "subset 31 px" is `winsize: 32` here.
