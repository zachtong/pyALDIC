"""The startup warm-up must reach the kernels it exists to compile.

The solver dispatches on problem size: ``icgn_batch`` only calls the Numba
precompute kernels once a frame has at least 50 nodes. A warm-up correlation
below that threshold compiles a different, much cheaper set of signatures and
buys the user nothing -- which is exactly what happened at the first attempt:
warm-up finished in a quick 4.9 s and left the first real correlation as slow
as it had been with no warm-up at all.

Nothing in the type system connects the warm-up's image size to that threshold,
so this file is the connection.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from al_dic.gui.kernel_warmup import WARMUP_IMAGE_PX

# Mirrors the `if N >= 50` dispatch guard in al_dic/solver/icgn_batch.py.
BATCH_BACKEND_MIN_NODES = 50

# The warm-up's own parameters, kept in step with _correlate_tiny_pair.
WARMUP_WINSIZE = 32
WARMUP_STEP = 16

# Every Numba kernel the first real correlation needs. Each was found by
# measurement: dropping any one of them leaves that much compilation on the
# user's first click.
REQUIRED_KERNELS = (
    "precompute_subsets_6dof_numba",
    "icgn_6dof_parallel",
    "precompute_subsets_2dof_numba",
    "icgn_2dof_parallel",
    # Second-order shape-function stage (Local DIC), in icgn2_kernels.
    "icgn2_refine_parallel",
)


def test_warmup_geometry_reaches_the_batch_backend():
    """The warm-up region must produce enough nodes to use the Numba backend."""
    per_axis = (WARMUP_IMAGE_PX - WARMUP_WINSIZE) // WARMUP_STEP + 1
    nodes = per_axis**2
    assert nodes >= BATCH_BACKEND_MIN_NODES, (
        f"a {WARMUP_IMAGE_PX} px warm-up region yields {nodes} nodes, below the "
        f"{BATCH_BACKEND_MIN_NODES}-node threshold at which icgn_batch uses the "
        "Numba precompute kernels. The warm-up would compile the wrong "
        "signatures and the user would still wait on the first real run."
    )


# The subprocess is not incidental. ``Dispatcher.signatures`` is process-global
# state, and by the time this file runs, other tests in the same session have
# long since compiled these kernels themselves -- so asserting in-process would
# pass no matter what the warm-up did.
_PROBE = textwrap.dedent(
    """
    import json
    from al_dic.solver import icgn2_kernels as nk2
    from al_dic.solver import numba_kernels as nk

    def kernel(name):
        return getattr(nk, name, None) or getattr(nk2, name)

    if not nk.HAS_NUMBA:
        print(json.dumps({"skip": "numba not installed"}))
        raise SystemExit(0)

    before = {n: len(kernel(n).signatures or []) for n in %(names)r}

    from al_dic.gui.kernel_warmup import _correlate_tiny_pair
    _correlate_tiny_pair()

    after = {n: len(kernel(n).signatures or []) for n in %(names)r}
    print(json.dumps({"before": before, "after": after}))
    """
)


@pytest.mark.slow
def test_warmup_compiles_every_kernel_the_first_run_needs():
    """In a fresh interpreter, the warm-up compiles every solver kernel."""
    import json

    proc = subprocess.run(
        [sys.executable, "-c", _PROBE % {"names": REQUIRED_KERNELS}],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    if "skip" in payload:
        pytest.skip(payload["skip"])

    assert not any(payload["before"].values()), (
        "the probe interpreter started with kernels already compiled, so this "
        f"test proves nothing: {payload['before']}"
    )
    missed = [name for name, count in payload["after"].items() if count == 0]
    assert not missed, (
        f"the warm-up did not compile {missed}, so that compilation still "
        "lands on the user's first Run"
    )
