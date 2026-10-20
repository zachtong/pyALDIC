# Benchmarks behind the paper's performance statements

Run from this folder with pyALDIC installed (`pip install al-dic`, or an
editable install of this repository):

```
python throughput.py --repeats 7     # points of interest per second, Local and AL-DIC
python peak_memory.py                # peak memory, 4096 x 4096 pixels, 3 frames
```

To measure an older release, check it out next to this one and point `--src`
at its `src` directory, e.g.

```
git worktree add ../pyALDIC-v0.4.1 v0.4.1
python peak_memory.py --src ../pyALDIC-v0.4.1/src
```

## What is measured

* **`throughput.py`**: synthetic speckle with a 5-pixel horizontal translation,
  two frames, 31 x 31-pixel subsets (`winsize=30`), 8-pixel spacing, FFT
  initial guess, three ADMM iterations for AL-DIC. Points of interest per second
  = nodes / wall time of the whole `run_aldic` call (initial guess, correlation
  and the cumulative transform included), after a warm-up run that compiles the
  Numba kernels; the median of the repeats is reported.
* **`peak_memory.py`**: 4096 x 4096-pixel synthetic speckle under uniform shear,
  three frames, 31 x 31-pixel subsets, 16-pixel spacing, FFT initial guess
  (the search region auto-expands from 16 to 70 pixels on this case).
  Peak memory is the highest traced allocation (`tracemalloc`, which includes
  NumPy arrays) during `run_aldic`. The same case runs in the test suite as
  `PYALDIC_RUN_PERF=1 pytest tests/test_perf_release.py -m "perf and large" -k xlarge_4096`.

## Results (2026-10-19)

Machine: Intel Core i7-13700F (Raptor Lake, 16 cores / 24 threads), 64 GB RAM,
Windows 11, Python 3.11.15, NumPy 2.4.6.

| pyALDIC | Image | Local DIC (POI/s) | AL-DIC (POI/s) |
|---|---|---:|---:|
| 1.0.3 | 512 x 512 | 13,465 | 3,370 |
| 1.0.3 | 1024 x 1024 | 14,019 | 3,303 |

| pyALDIC | Peak memory, 4096 x 4096 x 3 frames | Wall time |
|---|---:|---:|
| 0.4.1 | 36.3 GiB | not comparable (see below) |
| 1.0.3 | 11.6 GiB | 89.6 s |

Notes:

* Throughput depends on the machine's state (power plan, temperature,
  background load). On the same machine and day, v0.6.0 ran the 1024 x 1024
  Local case at about 14,700 POI/s against 13,300-13,800 for v1.0.3 and the
  AL-DIC case at the same rate; an earlier session had measured about 18,500
  POI/s for v0.6.0, which v0.6.0 itself no longer reaches here.
* An old checkout run from a long path may fail to write its Numba cache and
  recompile kernels on every call, which inflates its wall time; peak memory is
  unaffected.
