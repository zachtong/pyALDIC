"""The analysis engine: one triangulation per run, sampling as a gather.

Every expected number here comes from the physics of a hand-built field, not
from the implementation. Most tests pin a defect found in the 2026-09 audit of
the first implementation; the docstring says which.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from al_dic.analysis.engine import AnalysisEngine, SampleStatus
from al_dic.analysis.probes import AreaGeom, LineGeom, PointGeom, Probe
from al_dic.analysis.series import FrameStatus
from al_dic.core.data_structures import (
    DICMesh, FrameResult, PipelineResult, StrainResult,
)

STEP = 4.0


# ---------------------------------------------------------------------------
# Synthetic runs
# ---------------------------------------------------------------------------

def grid(extent: float = 44.0, step: float = STEP) -> np.ndarray:
    xs = np.arange(0.0, extent + step / 2, step)
    gx, gy = np.meshgrid(xs, xs)
    return np.column_stack([gx.ravel(), gy.ravel()])


def build(
    nodes: np.ndarray,
    disp,
    n_frames: int,
    *,
    dead: dict[int, np.ndarray] | None = None,
    strain=None,
    strain_valid: np.ndarray | None = None,
    img: tuple[int, int] = (48, 48),
) -> PipelineResult:
    """A run whose frame f displaces node (x, y) by ``disp(x, y, f)``.

    ``dead[f]`` lists nodes consumed by a crack at frame f and after -- the
    NaN ``U_accum`` the crack-aware cumulative transform writes.
    """
    x, y = nodes[:, 0], nodes[:, 1]
    frames, strains = [], []
    for f in range(1, n_frames + 1):
        u, v = disp(x, y, f)
        U = np.empty(2 * len(nodes))
        U[0::2], U[1::2] = u, v
        for k, idx in (dead or {}).items():
            if f >= k:
                U[2 * idx] = np.nan
                U[2 * idx + 1] = np.nan
        frames.append(FrameResult(U=U, U_accum=U))
        if strain is not None:
            e = strain(x, y, f)
            strains.append(StrainResult(
                disp_u=u, disp_v=v,
                strain_exx=e, strain_eyy=e, strain_exy=e,
                strain_principal_max=e, strain_principal_min=e,
                strain_maxshear=e, strain_von_mises=e, strain_rotation=e,
                strain_valid=strain_valid,
            ))
    return PipelineResult(
        dic_para=SimpleNamespace(winstepsize=STEP, img_size=img),
        dic_mesh=DICMesh(coordinates_fem=nodes,
                         elements_fem=np.zeros((0, 4), np.int64)),
        result_disp=frames,
        result_def_grad=[],
        result_strain=strains,
        result_fe_mesh_each_frame=[],
    )


def stretch(rate: float = 0.01):
    """Uniaxial: u = rate * f * x, v = 0 -- engineering strain rate * f."""
    return lambda x, y, f: (rate * f * x, np.zeros_like(y))


def probe(kind: str, geom) -> Probe:
    return Probe(id=1, kind=kind, geometry=geom, label="P1", color="#ef4444")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_a_point_is_barycentric_on_a_linear_field():
    r = build(grid(), lambda x, y, f: (0.1 * x + 0.02 * y, 0.0 * x), 1)
    eng = AnalysisEngine(r)
    s = eng.sample(eng.plan(PointGeom(13.0, 7.0)), "disp_u", 1)
    assert s.status[0] == SampleStatus.VALID
    assert s.values[0] == pytest.approx(0.1 * 13.0 + 0.02 * 7.0)


def test_a_line_carries_its_reference_arc_length():
    eng = AnalysisEngine(build(grid(), stretch(), 1))
    plan = eng.plan(LineGeom(4.0, 8.0, 34.0, 8.0))
    assert plan.distance[0] == 0.0
    assert plan.distance[-1] == pytest.approx(30.0)
    assert np.all(np.diff(plan.distance) > 0)


# ---------------------------------------------------------------------------
# Gauges -- audit C1: the extensometer was scaled by the pixel size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pixel_size", [1.0, 0.005, 300.0])
def test_extensometer_strain_does_not_depend_on_the_unit(pixel_size):
    """1 %/frame read 0, 3, 6, 9 at 300 um/px before the fix."""
    eng = AnalysisEngine(build(grid(), stretch(0.01), 3))
    ts = eng.series(probe("line", LineGeom(4.0, 20.0, 36.0, 20.0)),
                    None, "strain", pixel_size=pixel_size)
    np.testing.assert_allclose(ts.values, [0.0, 0.01, 0.02, 0.03], atol=1e-12)
    assert ts.unit == ""


def test_extensometer_reads_zero_under_a_rigid_rotation():
    theta = math.radians(4.0)
    c, s = math.cos(theta), math.sin(theta)
    rot = lambda x, y, f: ((c - 1) * x - s * y, s * x + (c - 1) * y)  # noqa: E731
    eng = AnalysisEngine(build(grid(), rot, 2))
    ts = eng.series(probe("line", LineGeom(8.0, 12.0, 30.0, 26.0)),
                    None, "strain", pixel_size=0.005)
    np.testing.assert_allclose(ts.values, 0.0, atol=1e-12)


def test_true_strain_and_elongation():
    eng = AnalysisEngine(build(grid(), stretch(0.1), 2))
    line = probe("line", LineGeom(4.0, 20.0, 24.0, 20.0))        # L0 = 20 px
    true = eng.series(line, None, "true_strain")
    elong = eng.series(line, None, "elongation", pixel_size=0.5,
                       length_unit="mm")
    np.testing.assert_allclose(true.values, np.log([1.0, 1.1, 1.2]), atol=1e-12)
    np.testing.assert_allclose(elong.values, [0.0, 1.0, 2.0], atol=1e-12)  # 20px*0.1*0.5mm
    assert elong.unit == "mm"


def test_crack_opening_components():
    """Opening along the gauge, sliding across it, in the length unit."""
    # Right half moves by (+2, +1) px per frame; left half stays.
    shift = lambda x, y, f: (np.where(x > 22, 2.0 * f, 0.0),   # noqa: E731
                             np.where(x > 22, 1.0 * f, 0.0))
    eng = AnalysisEngine(build(grid(), shift, 1))
    gauge = probe("line", LineGeom(12.0, 20.0, 32.0, 20.0))        # along +x
    opening = eng.series(gauge, None, "cod", pixel_size=0.5, length_unit="mm")
    sliding = eng.series(gauge, None, "cod_sliding", pixel_size=0.5,
                         length_unit="mm")
    magnitude = eng.series(gauge, None, "cod_magnitude", pixel_size=0.5,
                           length_unit="mm")
    assert opening.values[1] == pytest.approx(1.0)            # 2 px * 0.5
    assert sliding.values[1] == pytest.approx(0.5)            # 1 px * 0.5
    assert magnitude.values[1] == pytest.approx(0.5 * math.hypot(2, 1))


# ---------------------------------------------------------------------------
# Validity -- audit H3: displacement was cut by the strain edge trim
# ---------------------------------------------------------------------------

def test_a_displacement_probe_ignores_the_strain_trim():
    nodes = grid()
    trimmed = nodes[:, 0] < 10.0          # a band along the left edge
    r = build(nodes, stretch(), 2, strain=lambda x, y, f: 0.01 * f + 0 * x,
              strain_valid=~trimmed)
    eng = AnalysisEngine(r)
    ts = eng.series(probe("point", PointGeom(2.0, 20.0)), "disp_u", "value")
    assert np.all(np.isfinite(ts.values)), "displacement is not trimmed"


def test_a_strain_probe_respects_the_strain_trim():
    nodes = grid()
    trimmed = nodes[:, 0] < 10.0
    r = build(nodes, stretch(), 2, strain=lambda x, y, f: 0.01 * f + 0 * x,
              strain_valid=~trimmed)
    eng = AnalysisEngine(r)
    ts = eng.series(probe("point", PointGeom(2.0, 20.0)), "strain_eyy", "value")
    assert np.all(np.isnan(ts.values))
    # Beside a hole this is the usual case, and "no data" told users nothing.
    assert set(ts.status) == {FrameStatus.UNRELIABLE}


def test_strain_that_was_never_computed_says_so():
    """Frame 0 used to read 0 / "ok" for a strain nobody computed."""
    eng = AnalysisEngine(build(grid(), stretch(), 2))
    ts = eng.series(probe("point", PointGeom(20.0, 20.0)), "strain_eyy", "value")
    assert np.all(np.isnan(ts.values))
    assert set(ts.status) == {FrameStatus.NOT_COMPUTED}


# ---------------------------------------------------------------------------
# Cracks and holes -- audit H1 and the whole-probe verdict
# ---------------------------------------------------------------------------

def _hole_run(n_frames: int = 2):
    """Plate with a hole: no nodes inside it, a frame-0 mask that voids it."""
    nodes = grid()
    centre, radius = np.array([22.0, 22.0]), 7.0
    keep = np.hypot(*(nodes - centre).T) > radius
    yy, xx = np.mgrid[0:48, 0:48]
    mask = (np.hypot(xx - centre[0], yy - centre[1]) > radius).astype(float)
    return build(nodes[keep], stretch(), n_frames), mask


def test_a_section_from_a_hole_edge_is_measured_not_blanked():
    """Wired masks made every frame of this line NaN, reported as a crack."""
    r, mask = _hole_run()
    eng = AnalysisEngine(r, ref_mask=mask)
    line = probe("line", LineGeom(30.0, 22.0, 43.0, 22.0))
    ts = eng.series(line, "disp_u", "mean")
    assert np.all(np.isfinite(ts.values))
    assert set(ts.status) == {FrameStatus.OK}


def test_samples_inside_a_hole_are_not_material():
    r, mask = _hole_run()
    eng = AnalysisEngine(r, ref_mask=mask)
    s = eng.sample(eng.plan(LineGeom(10.0, 22.0, 34.0, 22.0)), "disp_u", 1)
    inside = np.abs(eng.plan(LineGeom(10.0, 22.0, 34.0, 22.0)).xy[:, 0] - 22.0) < 6.0
    assert np.all(s.status[inside] == SampleStatus.OUTSIDE)
    # ...so they do not count against the valid fraction.
    assert s.valid_fraction == pytest.approx(1.0)


def test_a_point_in_a_hole_reads_nothing():
    r, mask = _hole_run()
    eng = AnalysisEngine(r, ref_mask=mask)
    ts = eng.series(probe("point", PointGeom(22.0, 22.0)), "disp_u", "value")
    assert np.all(np.isnan(ts.values))


def test_a_growing_crack_drops_only_the_samples_it_consumes():
    """The line keeps a value; the frame is marked, not blanked."""
    nodes = grid()
    column = np.flatnonzero(np.isclose(nodes[:, 0], 20.0))   # a crack along x = 20
    r = build(nodes, stretch(), 4, dead={3: column})
    eng = AnalysisEngine(r)
    ts = eng.series(probe("line", LineGeom(4.0, 22.0, 40.0, 22.0)),
                    "disp_u", "mean")
    assert list(ts.status[:3]) == [FrameStatus.OK] * 3
    assert ts.status[3] is FrameStatus.CRACK
    assert np.isfinite(ts.values[3]), "enough of the line is still material"


def test_an_extensometer_keeps_reading_across_a_crack():
    """Elongation at break is a standard output; the frame is annotated."""
    nodes = grid()
    column = np.flatnonzero(np.isclose(nodes[:, 0], 20.0))
    r = build(nodes, stretch(), 3, dead={2: column})
    eng = AnalysisEngine(r)
    ts = eng.series(probe("line", LineGeom(4.0, 22.0, 40.0, 22.0)), None, "strain")
    assert np.all(np.isfinite(ts.values))
    assert ts.status[2] is FrameStatus.CRACK


def test_a_gauge_endpoint_on_consumed_material_is_reported():
    nodes = grid()
    column = np.flatnonzero(np.isclose(nodes[:, 0], 20.0))
    r = build(nodes, stretch(), 2, dead={1: column})
    eng = AnalysisEngine(r)
    ts = eng.series(probe("line", LineGeom(20.5, 22.0, 40.0, 22.0)), None, "cod")
    assert math.isnan(ts.values[1])
    assert ts.status[1] is FrameStatus.ENDPOINT_LOST


def test_the_masks_argument_still_acts_as_a_barrier():
    """The Python API kept a per-frame mask route for runs without dead nodes."""
    nodes = grid()
    r = build(nodes, stretch(), 2)
    crack = np.ones((48, 48))
    crack[:, 19:22] = 0.0
    eng = AnalysisEngine(r)
    ts = eng.series(probe("line", LineGeom(4.0, 22.0, 40.0, 22.0)),
                    "disp_u", "mean", masks=[np.ones((48, 48)), crack, crack])
    assert ts.status[1] is FrameStatus.CRACK


# ---------------------------------------------------------------------------
# Regions -- audit M1: statistics were unweighted over nodes
# ---------------------------------------------------------------------------

def test_a_region_mean_is_area_weighted_on_a_refined_mesh():
    """u = 0.01 x over [0, 44]: node mean was biased toward the refined side."""
    coarse = grid()
    fine_x = np.arange(0.0, 8.0, 0.5)
    fine_y = np.arange(0.0, 44.5, 0.5)
    fx, fy = np.meshgrid(fine_x, fine_y)
    fine = np.column_stack([fx.ravel(), fy.ravel()])
    nodes = np.unique(np.vstack([coarse, fine]), axis=0)
    r = build(nodes, lambda x, y, f: (0.01 * x, 0 * y), 1)
    eng = AnalysisEngine(r)
    ts = eng.series(probe("area", AreaGeom.rect(0.0, 0.0, 44.0, 44.0)),
                    "disp_u", "mean")
    assert ts.values[1] == pytest.approx(0.01 * 22.0, rel=0.02)


def test_a_region_maximum_reaches_the_nodes_inside_it():
    nodes = grid()
    r = build(nodes, lambda x, y, f: (np.exp(-((x - 20) ** 2 + (y - 20) ** 2) / 8),
                                      0 * y), 1)
    eng = AnalysisEngine(r)
    ts = eng.series(probe("area", AreaGeom.circle(20.0, 20.0, 6.0)),
                    "disp_u", "max")
    assert ts.values[1] == pytest.approx(1.0)         # the node at (20, 20)


def test_a_region_smaller_than_a_cell_still_measures():
    eng = AnalysisEngine(build(grid(), lambda x, y, f: (0.1 * x, 0 * y), 1))
    ts = eng.series(probe("area", AreaGeom.rect(9.0, 9.0, 10.0, 10.0)),
                    "disp_u", "mean")
    assert ts.values[1] == pytest.approx(0.95, abs=0.06)


# ---------------------------------------------------------------------------
# Statistics semantics -- audit, low severity
# ---------------------------------------------------------------------------

def test_valid_fraction_is_dimensionless_and_zero_when_empty():
    nodes = grid()
    everything = np.arange(len(nodes))
    r = build(nodes, stretch(), 2, dead={2: everything})
    eng = AnalysisEngine(r)
    ts = eng.series(probe("area", AreaGeom.rect(4.0, 4.0, 40.0, 40.0)),
                    "disp_u", "valid_fraction", length_unit="mm")
    assert ts.unit == ""
    assert ts.values[0] == pytest.approx(1.0)
    assert ts.values[2] == 0.0


def test_a_point_joins_mean_but_not_std():
    eng = AnalysisEngine(build(grid(), stretch(), 1))
    pt = probe("point", PointGeom(20.0, 20.0))
    assert eng.supports(pt, "mean")
    assert not eng.supports(pt, "std")
    ts = eng.series(pt, "disp_u", "mean")
    assert ts.values[1] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Line views
# ---------------------------------------------------------------------------

def test_profile_is_the_kymograph_row_for_that_frame():
    eng = AnalysisEngine(build(grid(), stretch(0.01), 3))
    line = probe("line", LineGeom(4.0, 20.0, 40.0, 20.0))
    kymo = eng.kymograph(line, "disp_u")
    prof = eng.profile(line, "disp_u", 2)
    assert kymo.values.shape == (4, len(prof.distance))
    np.testing.assert_allclose(prof.values, kymo.values[2])
    np.testing.assert_allclose(prof.values, 0.02 * (4.0 + prof.distance))


def test_the_kymograph_shows_the_crack_band():
    nodes = grid()
    column = np.flatnonzero(np.isclose(nodes[:, 0], 20.0))
    r = build(nodes, stretch(), 3, dead={2: column})
    eng = AnalysisEngine(r)
    kymo = eng.kymograph(probe("line", LineGeom(4.0, 22.0, 40.0, 22.0)), "disp_u")
    near = np.abs(eng.plan(LineGeom(4.0, 22.0, 40.0, 22.0)).xy[:, 0] - 20.0) < 3.0
    assert np.all(np.isfinite(kymo.values[1, near])), "before the crack"
    assert np.all(np.isnan(kymo.values[2, near])), "after the crack"
    assert np.all(kymo.status[2, near] == SampleStatus.CONSUMED)
    # Every element touching the dead column loses its data; beyond them,
    # nothing changes.
    beyond = np.abs(eng.plan(LineGeom(4.0, 22.0, 40.0, 22.0)).xy[:, 0] - 20.0) > 4.5
    assert np.all(np.isfinite(kymo.values[2, beyond])), "the rest is untouched"


# ---------------------------------------------------------------------------
# Performance -- audit C2: 1.29 s per frame at 20k nodes
# ---------------------------------------------------------------------------

def test_a_long_run_is_interactive():
    """One triangulation per run, not per frame of every curve."""
    xs = np.arange(0.0, 564.0, 4.0)                      # 141 x 141 = 19 881
    gx, gy = np.meshgrid(xs, xs)
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    r = build(nodes, stretch(1e-4), 100, img=(564, 564))
    t0 = time.perf_counter()
    eng = AnalysisEngine(r)
    ts = eng.series(probe("line", LineGeom(20.0, 280.0, 540.0, 280.0)),
                    "disp_u", "mean")
    elapsed = time.perf_counter() - t0
    assert np.all(np.isfinite(ts.values))
    assert elapsed < 3.0, f"{elapsed:.2f}s for 100 frames"


# ---------------------------------------------------------------------------
# Carried over from the first implementation's sampler and series tests
# ---------------------------------------------------------------------------

def test_a_point_outside_the_mesh_is_nan_not_clamped():
    """The reference implementation clamped to the edge: a plausible wrong number."""
    eng = AnalysisEngine(build(grid(), stretch(), 1))
    ts = eng.series(probe("point", PointGeom(60.0, 20.0)), "disp_u", "value")
    assert np.all(np.isnan(ts.values))
    assert set(ts.status) == {FrameStatus.NO_DATA}


def test_line_sampling_follows_the_mesh_step():
    eng = AnalysisEngine(build(grid(), stretch(), 1))
    # 32 px at a 4 px step, two samples per step, plus the far end.
    assert eng.line_sample_count(LineGeom(4.0, 8.0, 36.0, 8.0)) == 17


def test_a_line_beside_a_crack_is_bit_exact_against_no_crack():
    """Crack awareness costs nothing away from the crack."""
    nodes = grid()
    column = np.flatnonzero(np.isclose(nodes[:, 0], 20.0))
    plain = AnalysisEngine(build(nodes, stretch(), 3))
    cracked = AnalysisEngine(build(nodes, stretch(), 3, dead={2: column}))
    far = probe("line", LineGeom(28.0, 4.0, 40.0, 36.0))
    a = plain.series(far, "disp_u", "mean")
    b = cracked.series(far, "disp_u", "mean")
    np.testing.assert_array_equal(a.values, b.values)
    assert a.status == b.status


def test_a_polygon_region_reads_the_field_inside_it():
    eng = AnalysisEngine(build(grid(), lambda x, y, f: (0.01 * x, 0 * y), 1))
    triangle = AreaGeom.polygon([(8.0, 8.0), (32.0, 8.0), (8.0, 32.0)])
    ts = eng.series(probe("area", triangle), "disp_u", "mean")
    # Centroid x of the triangle is 16: u = 0.16 on a linear field.
    assert ts.values[1] == pytest.approx(0.16, abs=0.01)


def test_a_gauge_endpoint_off_the_mesh_is_lost_on_every_frame():
    eng = AnalysisEngine(build(grid(), stretch(), 2))
    ts = eng.series(probe("line", LineGeom(20.0, 20.0, 60.0, 20.0)), None, "strain")
    assert set(ts.status) == {FrameStatus.ENDPOINT_LOST}


def test_std_is_the_population_standard_deviation():
    eng = AnalysisEngine(build(grid(), lambda x, y, f: (0.01 * x, 0 * y), 1))
    line = probe("line", LineGeom(4.0, 20.0, 36.0, 20.0))
    plan = eng.plan(line.geometry)
    xs = plan.xy[:, 0]
    ts = eng.series(line, "disp_u", "std")
    assert ts.values[1] == pytest.approx(np.std(0.01 * xs))


def test_median_shrugs_off_a_spike_the_mean_does_not():
    nodes = grid()
    spike = np.isclose(nodes[:, 0], 20.0) & np.isclose(nodes[:, 1], 20.0)
    r = build(nodes, lambda x, y, f: (np.where(spike, 100.0, 0.0), 0 * y), 1)
    eng = AnalysisEngine(r)
    region = probe("area", AreaGeom.rect(4.0, 4.0, 40.0, 40.0))
    assert eng.series(region, "disp_u", "median").values[1] == pytest.approx(0.0)
    assert eng.series(region, "disp_u", "mean").values[1] > 0.0


def test_a_frame_below_the_threshold_is_left_blank():
    nodes = grid()
    left = np.flatnonzero(nodes[:, 0] < 24.0)          # most of the region
    r = build(nodes, stretch(), 2, dead={2: left})
    eng = AnalysisEngine(r)
    region = probe("area", AreaGeom.rect(4.0, 4.0, 40.0, 40.0))
    blank = eng.series(region, "disp_u", "mean", min_valid_fraction=0.5)
    kept = eng.series(region, "disp_u", "mean", min_valid_fraction=0.0)
    assert blank.status[2] is FrameStatus.BELOW_THRESHOLD
    assert math.isnan(blank.values[2])
    assert math.isfinite(kept.values[2]), "a threshold of 0 keeps what is left"


def test_crack_opening_does_not_depend_on_endpoint_order():
    shift = lambda x, y, f: (np.where(x > 22, 2.0 * f, 0.0), 0 * y)  # noqa: E731
    eng = AnalysisEngine(build(grid(), shift, 1))
    forward = eng.series(probe("line", LineGeom(12.0, 20.0, 32.0, 20.0)), None, "cod")
    backward = eng.series(probe("line", LineGeom(32.0, 20.0, 12.0, 20.0)), None, "cod")
    assert forward.values[1] == pytest.approx(2.0)
    assert backward.values[1] == pytest.approx(2.0), "opening is positive either way"


def test_an_unknown_statistic_is_an_error_not_a_default():
    eng = AnalysisEngine(build(grid(), stretch(), 1))
    with pytest.raises(ValueError):
        eng.series(probe("area", AreaGeom.rect(4.0, 4.0, 40.0, 40.0)),
                   "disp_u", "average")


def test_world_axes_flip_v_before_the_statistic():
    """Negating a reduced maximum gives the other axis's minimum, not its max."""
    r = build(grid(), lambda x, y, f: (0 * x, 0.01 * y - 0.1), 1)
    eng = AnalysisEngine(r)
    region = probe("area", AreaGeom.rect(4.0, 4.0, 40.0, 40.0))
    image_min = eng.series(region, "disp_v", "min").values[1]
    world_max = eng.series(region, "disp_v", "max", axes="world").values[1]
    assert world_max == pytest.approx(-image_min)
    image_std = eng.series(region, "disp_v", "std").values[1]
    world_std = eng.series(region, "disp_v", "std", axes="world").values[1]
    assert world_std == pytest.approx(image_std), "a spread has no sign"
    assert eng.series(region, "disp_v", "mean", axes="world").axes == "world"
