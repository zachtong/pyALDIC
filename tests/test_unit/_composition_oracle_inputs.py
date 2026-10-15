"""Deterministic inputs for the cumulative-composition oracle.

``tests/fixtures/cumulative_composition_04834a8.npz`` holds only the OUTPUTS
(``U_accum`` per frame) that
``al_dic.core.pipeline._compute_cumulative_displacements_tree`` produced at
commit 04834a8 for the cases built here.  A test rebuilds the inputs with this
module, runs the (possibly refactored) composition on them and compares the
result with the recorded outputs::

    from ._composition_oracle_inputs import (
        CASE_NAMES, FIXTURE_PATH, build_case, fixture_key, input_fingerprint,
        run_case,
    )

    oracle = np.load(FIXTURE_PATH)
    for name in CASE_NAMES:
        case = build_case(name)
        assert input_fingerprint(case) == str(
            oracle[fixture_key(name, "input_sha256")])
        u_accum, has_u_accum = run_case(case, compose)
        np.testing.assert_array_equal(
            has_u_accum, oracle[fixture_key(name, "has_u_accum")])
        np.testing.assert_allclose(
            u_accum, oracle[fixture_key(name, "u_accum")],
            rtol=0, atol=1e-12, equal_nan=True)

Reproducibility rules.  Any edit that changes a built input invalidates the
fixture (``input_fingerprint`` catches it), so keep them when editing:

* No numpy RNG.  Draws come from the counter-based SplitMix64 below, which
  depends on nothing but integer arithmetic, so a numpy release cannot
  change the stream.
* Inputs use only IEEE-exact elementwise arithmetic (+, -, *, /, comparisons,
  slicing).  No transcendental functions, reductions or BLAS calls, whose
  last bit may differ between platforms.
* No Delaunay ties whose resolution matters.  The ``jitter_*`` cases use
  jittered (non-lattice) node sets, so their triangulations are unique.  The
  ``crack_*`` cases need real axis-aligned Q4 elements (the crack-aware
  element override skips any other shape), and a rectangle's corners are
  always co-circular.  Their fields are therefore additively separable,
  ``f(x, y) = g(x) + h(y)``: the four corner values of every lattice cell
  are coplanar, so both diagonal triangulations interpolate to the same
  value and qhull's tie-breaking cannot change the result.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from al_dic.core.data_structures import DICMesh, FrameResult, FrameSchedule

FIXTURE_COMMIT = "04834a8"
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / f"cumulative_composition_{FIXTURE_COMMIT}.npz"
)

CASE_NAMES: tuple[str, ...] = (
    "jitter_incremental",
    "jitter_tree_multilattice",
    "jitter_chain_break",
    "jitter_missing_first_pair",
    "crack_tree",
    "crack_accumulative",
)

# Fields stored per case in the fixture (see ``fixture_key``).
FIXTURE_FIELDS: tuple[str, ...] = ("u_accum", "has_u_accum", "input_sha256")

_STEP = 16.0
_JITTER = 0.3 * _STEP


@dataclass(frozen=True)
class CompositionCase:
    """One composition input set.  Arrays are never handed out directly:
    :meth:`call_args` returns fresh copies for every call."""

    name: str
    description: str
    result_disp: tuple[FrameResult | None, ...]
    result_fe_mesh: tuple[DICMesh | None, ...]
    n_frames: int
    schedule: FrameSchedule
    masks: tuple[NDArray[np.float64], ...] | None = None
    crack_radius: float | None = None

    @property
    def n_pairs(self) -> int:
        return self.n_frames - 1

    @property
    def n_nodes(self) -> int:
        return int(self.result_fe_mesh[0].coordinates_fem.shape[0])

    def call_args(self) -> tuple[tuple, dict]:
        """Positional and keyword arguments for one composition call.

        The composition replaces ``result_disp`` entries in place, so every
        call gets new lists holding copied arrays.
        """
        disp = [
            None if r is None else replace(r, U=r.U.copy())
            for r in self.result_disp
        ]
        meshes = [
            None if m is None else DICMesh(
                coordinates_fem=m.coordinates_fem.copy(),
                elements_fem=m.elements_fem.copy(),
            )
            for m in self.result_fe_mesh
        ]
        args = (disp, meshes, self.n_frames, self.schedule)
        if self.masks is None:
            return args, {}
        kwargs = {
            "masks": [m.copy() for m in self.masks],
            "crack_radius": self.crack_radius,
        }
        return args, kwargs


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def fixture_key(case_name: str, field: str) -> str:
    """Key of one per-case array in the oracle fixture."""
    if case_name not in CASE_NAMES:
        raise KeyError(f"unknown composition case {case_name!r}")
    if field not in FIXTURE_FIELDS:
        raise KeyError(f"unknown fixture field {field!r}")
    return f"{case_name}__{field}"


def build_case(name: str) -> CompositionCase:
    """Build the named case from scratch (deterministic)."""
    builders: dict[str, Callable[[], CompositionCase]] = {
        "jitter_incremental": _jitter_incremental,
        "jitter_tree_multilattice": _jitter_tree_multilattice,
        "jitter_chain_break": _jitter_chain_break,
        "jitter_missing_first_pair": _jitter_missing_first_pair,
        "crack_tree": _crack_tree,
        "crack_accumulative": _crack_accumulative,
    }
    if name not in builders:
        raise KeyError(f"unknown composition case {name!r}")
    return builders[name]()


def build_all_cases() -> tuple[CompositionCase, ...]:
    return tuple(build_case(name) for name in CASE_NAMES)


def run_case(
    case: CompositionCase,
    compose: Callable[..., Sequence[FrameResult | None]],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Run ``compose`` on fresh copies of the case inputs.

    Returns ``(u_accum, has_u_accum)``: ``u_accum`` is (n_pairs, 2*n_nodes)
    with an all-NaN row wherever the output frame is ``None`` or carries no
    ``U_accum``; ``has_u_accum`` marks the rows that do carry one.
    """
    args, kwargs = case.call_args()
    results = compose(*args, **kwargs)
    return stack_u_accum(results, case.n_pairs, case.n_nodes)


def stack_u_accum(
    results: Sequence[FrameResult | None],
    n_pairs: int,
    n_nodes: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Stack per-frame ``U_accum`` into one array (see :func:`run_case`)."""
    if len(results) != n_pairs:
        raise ValueError(f"expected {n_pairs} results, got {len(results)}")
    u_accum = np.full((n_pairs, 2 * n_nodes), np.nan)
    has_u_accum = np.zeros(n_pairs, dtype=bool)
    for i, res in enumerate(results):
        if res is None or res.U_accum is None:
            continue
        if res.U_accum.shape != (2 * n_nodes,):
            raise ValueError(
                f"frame {i + 1}: U_accum shape {res.U_accum.shape}, "
                f"expected {(2 * n_nodes,)}"
            )
        u_accum[i] = res.U_accum
        has_u_accum[i] = True
    return u_accum, has_u_accum


def input_fingerprint(case: CompositionCase) -> str:
    """SHA-256 over every input array, so a test can tell a changed builder
    apart from a changed composition."""
    digest = hashlib.sha256()

    def add(tag: str, arr: NDArray | None) -> None:
        digest.update(tag.encode())
        if arr is None:
            digest.update(b"<none>")
            return
        a = np.ascontiguousarray(arr)
        digest.update(f"{a.dtype.str}{a.shape}".encode())
        digest.update(a.tobytes())

    digest.update(f"{case.n_frames}{case.schedule.ref_indices}".encode())
    digest.update(repr(case.crack_radius).encode())
    pairs = zip(case.result_disp, case.result_fe_mesh)
    for i, (res, mesh) in enumerate(pairs):
        add(f"U{i}", None if res is None else res.U)
        add(f"ref{i}", None if res is None else np.array([res.ref_frame]))
        add(f"xy{i}", None if mesh is None else mesh.coordinates_fem)
        add(f"el{i}", None if mesh is None else mesh.elements_fem)
    for i, m in enumerate(case.masks or ()):
        add(f"mask{i}", m)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Deterministic draws (counter-based SplitMix64, no numpy RNG)
# ---------------------------------------------------------------------------

_SM_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SM_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SM_MUL2 = np.uint64(0x94D049BB133111EB)


def _splitmix64(seed: int, n: int) -> NDArray[np.uint64]:
    """Element i is SplitMix64 output i of ``seed``.

    uint64 array arithmetic wraps mod 2**64, as the algorithm requires.
    """
    z = np.uint64(seed) + np.arange(1, n + 1, dtype=np.uint64) * _SM_GAMMA
    z = (z ^ (z >> np.uint64(30))) * _SM_MUL1
    z = (z ^ (z >> np.uint64(27))) * _SM_MUL2
    return z ^ (z >> np.uint64(31))


def _uniform(seed: int, n: int, lo: float, hi: float) -> NDArray[np.float64]:
    """n draws in [lo, hi): the top 53 bits scaled by 2**-53 (exact)."""
    unit = (_splitmix64(seed, n) >> np.uint64(11)).astype(np.float64)
    unit = unit * (1.0 / 9007199254740992.0)
    return lo + (hi - lo) * unit


def _permutation(seed: int, n: int) -> NDArray[np.int64]:
    return np.argsort(_uniform(seed, n, 0.0, 1.0), kind="stable").astype(
        np.int64
    )


# ---------------------------------------------------------------------------
# Geometry and field builders
# ---------------------------------------------------------------------------


def _lattice(
    x_lo: float, nx: int, y_lo: float, ny: int,
) -> NDArray[np.float64]:
    """Regular nodes, index = ix * ny + iy (as in the pipeline meshes)."""
    xs = x_lo + _STEP * np.arange(nx, dtype=np.float64)
    ys = y_lo + _STEP * np.arange(ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _jittered_lattice(
    seed: int, x_lo: float, nx: int, y_lo: float, ny: int,
) -> NDArray[np.float64]:
    nodes = _lattice(x_lo, nx, y_lo, ny)
    jitter = _uniform(seed, nodes.size, -_JITTER, _JITTER).reshape(nodes.shape)
    return nodes + jitter


def _q4_elements(nx: int, ny: int) -> NDArray[np.int64]:
    """Q4 connectivity (midside columns -1) of ``_lattice(_, nx, _, ny)``."""
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    elems = np.full((len(ii), 8), -1, dtype=np.int64)
    elems[:, 0] = ii * ny + jj
    elems[:, 1] = (ii + 1) * ny + jj
    elems[:, 2] = (ii + 1) * ny + (jj + 1)
    elems[:, 3] = ii * ny + (jj + 1)
    return elems


def _smooth_uv(
    nodes: NDArray[np.float64],
    seed: int,
    shift: tuple[float, float],
    amp: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Translation plus a cubic polynomial (non-affine) in normalized x, y."""
    c = _uniform(seed, 12, -amp, amp)
    x = (nodes[:, 0] - 120.0) / 80.0
    y = (nodes[:, 1] - 104.0) / 64.0
    u = (shift[0] + c[0] * x + c[1] * y + c[2] * x * y + c[3] * x * x
         + c[4] * y * y + c[5] * x * x * y)
    v = (shift[1] + c[6] * x + c[7] * y + c[8] * x * y + c[9] * y * y
         + c[10] * x * x + c[11] * x * y * y)
    return u, v


def _frame_result(
    u: NDArray[np.float64], v: NDArray[np.float64], ref_frame: int,
) -> FrameResult:
    U = np.empty(2 * len(u), dtype=np.float64)
    U[0::2] = u
    U[1::2] = v
    return FrameResult(U=U, ref_frame=ref_frame)


def _bare_mesh(nodes: NDArray[np.float64]) -> DICMesh:
    """Node-only mesh: the mask-free composition never reads elements."""
    return DICMesh(
        coordinates_fem=nodes, elements_fem=np.empty((0, 8), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Jittered (mask-free) cases
# ---------------------------------------------------------------------------


def _jitter_incremental() -> CompositionCase:
    """One jittered lattice shared by every pair of an incremental chain.

    Translations of 18-30 px per pair walk the tracked points out of the
    fixed pair mesh, so most later queries take the nearest-neighbor
    extrapolation.  Pair 2 has NaN nodes shared by u and v (one reduced
    triangulation); pair 4 has NaN nodes that differ between u and v (the
    two-interpolant fallback).
    """
    nodes = _jittered_lattice(101, 40.0, 11, 40.0, 9)
    n = len(nodes)
    shifts = ((22.0, -4.0), (26.0, 3.0), (18.0, 6.0),
              (30.0, -2.0), (24.0, 5.0), (20.0, -6.0))
    pick = _permutation(301, n)
    results = []
    for k, shift in enumerate(shifts):
        u, v = _smooth_uv(nodes, 200 + k, shift, amp=2.5)
        if k == 2:
            u[pick[:6]] = np.nan
            v[pick[:6]] = np.nan
        if k == 4:
            u[pick[6:10]] = np.nan
            v[pick[10:14]] = np.nan
        results.append(_frame_result(u, v, ref_frame=k))
    n_frames = len(shifts) + 1
    return CompositionCase(
        name="jitter_incremental",
        description=_jitter_incremental.__doc__ or "",
        result_disp=tuple(results),
        result_fe_mesh=tuple(_bare_mesh(nodes.copy()) for _ in shifts),
        n_frames=n_frames,
        schedule=FrameSchedule.from_mode("incremental", n_frames),
    )


_TREE_REFS = (0, 1, 2, 3, 4, 5, 0, 2, 7, 1, 9, 3)


def _tree_lattice(ref: int) -> NDArray[np.float64]:
    """A different jittered lattice per reference frame (per-reference grids):
    origin, extent and jitter all depend on ``ref``."""
    return _jittered_lattice(
        400 + ref, 40.0 + 12.0 * ref, 11 + ref % 3, 40.0 - 4.0 * (ref % 2),
        9 + ref % 2,
    )


def _jitter_tree_multilattice() -> CompositionCase:
    """Tree schedule (0,1,2,3,4,5,0,2,7,1,9,3) whose pairs are solved on one
    lattice per reference frame: 8 distinct node sets (6 distinct
    triangulations are built), reused out of order -- frames 7, 8 and 10
    return to the lattices of frames 0, 2 and 1 after four or more others,
    so a 4-entry triangulation cache must evict and rebuild them -- with
    branches, shared path segments and re-rooting.  Pair 3 has shared NaN
    nodes; pair 8 (frame 9) is all-NaN, so frame 9 and its child frame 11
    lose every point; frame 12's result is missing (skipped).
    """
    results: list[FrameResult | None] = []
    meshes = []
    for i, ref in enumerate(_TREE_REFS):
        nodes = _tree_lattice(ref)
        u, v = _smooth_uv(nodes, 500 + i, (12.0, 1.0), amp=1.5)
        if i == 3:
            pick = _permutation(510, len(nodes))[:5]
            u[pick] = np.nan
            v[pick] = np.nan
        if i == 8:
            u[:] = np.nan
            v[:] = np.nan
        skipped = i == 11
        results.append(None if skipped else _frame_result(u, v, ref))
        meshes.append(_bare_mesh(nodes))
    return CompositionCase(
        name="jitter_tree_multilattice",
        description=_jitter_tree_multilattice.__doc__ or "",
        result_disp=tuple(results),
        result_fe_mesh=tuple(meshes),
        n_frames=len(_TREE_REFS) + 1,
        schedule=FrameSchedule(ref_indices=_TREE_REFS),
    )


def _jitter_chain_break() -> CompositionCase:
    """Incremental chain whose frame-2 result is missing: frames 3 and 4 hit
    the chain-break path and keep frame 1's cumulative value (pins the
    pre-existing behavior at 04834a8)."""
    nodes = _jittered_lattice(600, 40.0, 7, 40.0, 6)
    n_frames = 5
    results: list[FrameResult | None] = []
    for k in range(n_frames - 1):
        u, v = _smooth_uv(nodes, 610 + k, (3.0, -2.0), amp=1.0)
        results.append(None if k == 1 else _frame_result(u, v, ref_frame=k))
    return CompositionCase(
        name="jitter_chain_break",
        description=_jitter_chain_break.__doc__ or "",
        result_disp=tuple(results),
        result_fe_mesh=tuple(
            _bare_mesh(nodes.copy()) for _ in range(n_frames - 1)
        ),
        n_frames=n_frames,
        schedule=FrameSchedule.from_mode("incremental", n_frames),
    )


def _jitter_missing_first_pair() -> CompositionCase:
    """Frame 1's result is missing: the composition returns early and no
    frame gets a U_accum (pins the early-return behavior at 04834a8)."""
    nodes = _jittered_lattice(650, 40.0, 5, 40.0, 4)
    n_frames = 3
    u, v = _smooth_uv(nodes, 660, (2.0, 1.0), amp=1.0)
    return CompositionCase(
        name="jitter_missing_first_pair",
        description=_jitter_missing_first_pair.__doc__ or "",
        result_disp=(None, _frame_result(u, v, ref_frame=1)),
        result_fe_mesh=(_bare_mesh(nodes), _bare_mesh(nodes.copy())),
        n_frames=n_frames,
        schedule=FrameSchedule.from_mode("incremental", n_frames),
    )


# ---------------------------------------------------------------------------
# Crack cases (lattice Q4 meshes, per-frame masks, crack-aware override)
# ---------------------------------------------------------------------------

_CRACK_IMG = (144, 224)          # (h, w)
_CRACK_Y = 64                    # crack line, on node row y = 64
_CRACK_RADIUS = 2.0 * _STEP      # what run_aldic passes (2 * winstepsize)
# Half-width of the open band in frames 0..5 (frame 3 swallows rows 48/80,
# frame 4 is a narrower sibling, frame 5 is widest).
_CRACK_HALF_WIDTH = (2, 6, 10, 26, 12, 28)
_CRACK_LATTICE_NX = {0: 12, 1: 12, 2: 13, 3: 13}  # refs 2, 3 grow by one step


def _crack_mask(frame: int) -> NDArray[np.float64]:
    """Frame-``frame`` ROI mask: a through-crack band at y = 64.  Frame 0
    adds a 3x3 pocket whose center pixel (node 112, 96) stays material, so
    that node is not born dead but is killed by the parent-side check.
    Frame 5 adds a void around node (160, 112)'s deformed position."""
    h, w = _CRACK_IMG
    mask = np.ones((h, w), dtype=np.float64)
    hw = _CRACK_HALF_WIDTH[frame]
    mask[_CRACK_Y - hw:_CRACK_Y + hw + 1, :] = 0.0
    if frame == 0:
        mask[95:98, 111:114] = 0.0
        mask[96, 112] = 1.0
    if frame == 5:
        mask[108:127, 151:170] = 0.0
    return mask


def _cut_elements(
    nodes: NDArray[np.float64],
    elems: NDArray[np.int64],
    mask: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Drop elements a masked band crosses or that are mostly masked."""
    keep = np.ones(len(elems), dtype=bool)
    for e, corners in enumerate(elems[:, :4]):
        xs = nodes[corners, 0]
        ys = nodes[corners, 1]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        gap = mask[y0:y1 + 1, x0:x1 + 1] <= 0.5
        crossed = bool(gap.all(axis=1).any())
        mostly = int(gap.sum()) * 2 > gap.size
        keep[e] = not (crossed or mostly)
    return elems[keep]


def _crack_pair_mesh(ref: int, mask: NDArray[np.float64]) -> DICMesh:
    nx, ny = _CRACK_LATTICE_NX[ref], 8
    nodes = _lattice(16.0, nx, 16.0, ny)
    elems = _cut_elements(nodes, _q4_elements(nx, ny), mask)
    return DICMesh(coordinates_fem=nodes, elements_fem=elems)


def _crack_uv(
    nodes: NDArray[np.float64], seed: int, opening: float, slide: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Additively separable field g(x) + h(y): smooth cubic terms plus a
    step across the crack (faces open by +-opening and slide by +-slide)."""
    c = _uniform(seed, 10, -0.8, 0.8)
    x = (nodes[:, 0] - 112.0) / 96.0
    y = (nodes[:, 1] - 64.0) / 64.0
    yc = nodes[:, 1] - _CRACK_Y
    face = np.where(yc < 0.0, -1.0, np.where(yc > 0.0, 1.0, 0.0))
    stretch = 1.75  # pushes the x = 16 / x = 208 columns ~2 px outward
    u = (c[0] + stretch * x + c[1] * x * x + c[2] * y
         + c[3] * y * y * y + slide * face)
    v = (c[4] + c[5] * x + c[6] * x * x * x + c[7] * y
         + c[8] * y * y + c[9] * y * y * y + opening * face)
    return u, v


def _crack_case(
    name: str,
    description: str,
    refs: tuple[int, ...],
    openings: tuple[float, ...],
    slides: tuple[float, ...],
    seed: int,
) -> CompositionCase:
    masks = tuple(_crack_mask(f) for f in range(len(refs) + 1))
    results = []
    meshes = []
    for i, ref in enumerate(refs):
        mesh = _crack_pair_mesh(ref, masks[ref])
        u, v = _crack_uv(
            mesh.coordinates_fem, seed + i, openings[i], slides[i],
        )
        results.append(_frame_result(u, v, ref_frame=ref))
        meshes.append(mesh)
    return CompositionCase(
        name=name,
        description=description,
        result_disp=tuple(results),
        result_fe_mesh=tuple(meshes),
        n_frames=len(refs) + 1,
        schedule=FrameSchedule(ref_indices=refs),
        masks=masks,
        crack_radius=_CRACK_RADIUS,
    )


def _crack_tree() -> CompositionCase:
    """Crack-aware composition on schedule (0,1,2,2,3).  Covers born-dead
    nodes (row y = 64), the parent-side kill (frame-0 pocket node), element
    override next to the gap, unresolved near-gap survivors (the x = 16 /
    x = 208 columns pushed out of every element), child-side deaths when the
    band swallows rows 48/80 at frame 3 but not at sibling frame 4, a void
    killing a node at frame 5 (child of 3) on top of inherited deaths, and
    per-reference lattices (refs 2 and 3 are one step wider)."""
    return _crack_case(
        "crack_tree", _crack_tree.__doc__ or "", (0, 1, 2, 2, 3),
        openings=(1.0, 1.5, 2.0, 1.25, 1.0),
        slides=(0.4, -0.3, 0.5, 0.2, -0.25),
        seed=700,
    )


def _crack_accumulative() -> CompositionCase:
    """The same masks under the accumulative schedule: every pair is solved
    on frame 0's cut lattice and every query sits at a frame-0 node, so the
    parent-side kill repeats per frame and each frame's child-side check
    acts alone."""
    return _crack_case(
        "crack_accumulative", _crack_accumulative.__doc__ or "",
        (0, 0, 0, 0, 0),
        openings=(1.0, 2.5, 4.5, 3.0, 5.0),
        slides=(0.4, 0.1, 0.6, 0.3, 0.35),
        seed=800,
    )
