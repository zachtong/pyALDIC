"""Probe geometry, validation and serialisation.

Coordinates are frame-0 (reference) image pixels throughout, matching
``DICMesh.coordinates_fem``, so nothing here transforms between coordinate
systems. Geometry is stored in pixels even when the project displays physical
units, so that changing the unit setting cannot invalidate a saved probe.
"""

from __future__ import annotations

import dataclasses

import pytest

from al_dic.analysis.probes import (
    AreaGeom,
    LineGeom,
    PointGeom,
    Probe,
    ProbeSet,
    allowed_reductions,
    probe_from_dict,
    probe_to_dict,
)


# --- geometry ------------------------------------------------------------

def test_point_geometry():
    g = PointGeom(x=12.5, y=30.0)
    assert (g.x, g.y) == (12.5, 30.0)


def test_line_requires_two_distinct_endpoints():
    LineGeom(x0=0.0, y0=0.0, x1=10.0, y1=0.0)
    with pytest.raises(ValueError, match="zero length"):
        LineGeom(x0=5.0, y0=5.0, x1=5.0, y1=5.0)


def test_line_length_and_direction():
    g = LineGeom(x0=0.0, y0=0.0, x1=3.0, y1=4.0)
    assert g.length() == pytest.approx(5.0)
    ux, uy = g.direction()
    assert (ux, uy) == pytest.approx((0.6, 0.8))


def test_area_rect_is_normalised():
    """A rectangle dragged right-to-left is the same rectangle."""
    a = AreaGeom.rect(10.0, 20.0, 0.0, 5.0)
    b = AreaGeom.rect(0.0, 5.0, 10.0, 20.0)
    assert a.data == b.data


def test_area_circle_rejects_non_positive_radius():
    AreaGeom.circle(10.0, 10.0, 4.0)
    with pytest.raises(ValueError, match="radius"):
        AreaGeom.circle(10.0, 10.0, 0.0)


def test_area_polygon_needs_three_vertices():
    AreaGeom.polygon([(0, 0), (10, 0), (5, 8)])
    with pytest.raises(ValueError, match="three"):
        AreaGeom.polygon([(0, 0), (10, 0)])


def test_area_shape_spelling_is_polygon_not_poly():
    """One spelling only.

    The reference implementation validates "polygon" server-side but posts
    "poly" from the UI and branches on "poly" in its mask builder, so polygon
    probes are unusable end to end while its tests still pass.
    """
    assert AreaGeom.polygon([(0, 0), (1, 0), (0, 1)]).shape == "polygon"


# --- probe ---------------------------------------------------------------

def test_probe_is_immutable():
    p = Probe(id=1, kind="point", geometry=PointGeom(1.0, 2.0),
              label="P1", color="#FF0000")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.label = "renamed"
    assert dataclasses.replace(p, label="renamed").label == "renamed"


def test_probe_rejects_geometry_of_the_wrong_kind():
    with pytest.raises(ValueError, match="kind"):
        Probe(id=1, kind="line", geometry=PointGeom(1.0, 2.0),
              label="P1", color="#FF0000")


def test_probe_rejects_malformed_colour():
    with pytest.raises(ValueError, match="colour|color"):
        Probe(id=1, kind="point", geometry=PointGeom(0.0, 0.0),
              label="P1", color="red")


# --- reductions ----------------------------------------------------------

def test_a_point_is_a_one_sample_region():
    """It joins a chart of means, medians or extremes as its own value --
    the same quantity on the same axis -- but not a spread or a coverage,
    which one sample cannot have. The reference accepted any metric for a
    point and silently ignored it.
    """
    assert allowed_reductions("point") == frozenset(
        {"value", "mean", "median", "max", "min"}
    )


def test_line_reductions_include_gauge_measurements():
    red = allowed_reductions("line")
    assert {"mean", "median", "max", "min", "std", "valid_fraction"} <= red
    assert {"strain", "true_strain", "elongation",
            "cod", "cod_sliding", "cod_magnitude"} <= red


def test_area_reductions_exclude_gauge_measurements():
    """strain and cod come from two endpoints, which an area does not have."""
    red = allowed_reductions("area")
    assert {"mean", "median", "max", "min", "std", "valid_fraction"} <= red
    assert not ({"strain", "cod"} & red)


# --- serialisation -------------------------------------------------------

@pytest.mark.parametrize("probe", [
    Probe(id=1, kind="point", geometry=PointGeom(12.5, 30.25),
          label="tip", color="#FF0000"),
    Probe(id=2, kind="line", geometry=LineGeom(0.0, 0.0, 40.0, 10.0),
          label="gauge", color="#00FF00", visible=False),
    Probe(id=3, kind="area", geometry=AreaGeom.rect(0.0, 0.0, 20.0, 20.0),
          label="box", color="#0000FF"),
    Probe(id=4, kind="area", geometry=AreaGeom.circle(30.0, 30.0, 8.0),
          label="disc", color="#FFFF00"),
    Probe(id=5, kind="area", geometry=AreaGeom.polygon([(0, 0), (9, 1), (4, 7)]),
          label="tri", color="#FF00FF"),
])
def test_serialisation_round_trip(probe):
    restored = probe_from_dict(probe_to_dict(probe))
    assert restored == probe


def test_serialised_form_is_json_safe():
    import json

    p = Probe(id=7, kind="area", geometry=AreaGeom.polygon([(0, 0), (9, 1), (4, 7)]),
              label="tri", color="#FF00FF")
    payload = json.loads(json.dumps(probe_to_dict(p)))
    assert probe_from_dict(payload) == p


def test_from_dict_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        probe_from_dict({"id": 1, "kind": "blob", "geometry": {},
                         "label": "x", "color": "#FFFFFF", "visible": True})


# --- probe set -----------------------------------------------------------

def test_ids_are_unique_across_kinds():
    """One counter for every kind.

    The reference keeps three independent counters, so a point, a line and an
    area can all be id 1; deletion then needs id *and* type, and its chart
    colour map -- keyed on id alone -- is corrupted whenever ids collide.
    """
    s = ProbeSet()
    a = s.add("point", PointGeom(1.0, 1.0))
    b = s.add("line", LineGeom(0.0, 0.0, 5.0, 5.0))
    c = s.add("area", AreaGeom.rect(0.0, 0.0, 4.0, 4.0))
    assert len({a.id, b.id, c.id}) == 3


def test_ids_are_not_reused_after_removal():
    s = ProbeSet()
    first = s.add("point", PointGeom(1.0, 1.0))
    s.remove(first.id)
    second = s.add("point", PointGeom(2.0, 2.0))
    assert second.id != first.id


def test_remove_is_implemented():
    """The reference's remove_probe() is a bare `pass`."""
    s = ProbeSet()
    p = s.add("point", PointGeom(1.0, 1.0))
    s.remove(p.id)
    assert list(s) == []


def test_remove_unknown_id_raises():
    with pytest.raises(KeyError):
        ProbeSet().remove(99)


def test_default_labels_and_colours_are_assigned():
    s = ProbeSet()
    p = s.add("point", PointGeom(1.0, 1.0))
    assert p.label
    assert p.color.startswith("#") and len(p.color) == 7


def test_colours_cycle_without_immediate_repeats():
    s = ProbeSet()
    colours = [s.add("point", PointGeom(float(i), 0.0)).color for i in range(4)]
    assert len(set(colours)) == 4


def test_replace_updates_in_place_by_id():
    s = ProbeSet()
    p = s.add("point", PointGeom(1.0, 1.0))
    s.replace(dataclasses.replace(p, label="renamed"))
    assert s.get(p.id).label == "renamed"
    assert len(list(s)) == 1


def test_probe_set_round_trip():
    s = ProbeSet()
    s.add("point", PointGeom(1.0, 1.0), label="a")
    s.add("line", LineGeom(0.0, 0.0, 3.0, 4.0), label="b")
    restored = ProbeSet.from_list(s.to_list())
    assert [p.label for p in restored] == ["a", "b"]
    # The counter must survive, or a restored session reuses ids.
    assert restored.add("point", PointGeom(9.0, 9.0)).id == 3
