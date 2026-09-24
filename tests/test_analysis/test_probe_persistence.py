"""Probes in a session file: parsed before anything is touched, ids kept.

Pins three defects from the 2026-09 audit: a malformed probe cleared the open
project's results before it was even parsed; a non-dict entry escaped as an
AttributeError past ``except SessionError``; and the id counter was not saved,
so a deleted probe's id came back after a reload.
"""

from __future__ import annotations

import pytest

from al_dic.analysis.probes import LineGeom, PointGeom, ProbeSet
from al_dic.gui.session import SessionError, _parse_config


def _doc(probes) -> dict:
    return {"schema_version": 3, "image_files": [], "probes": probes}


def test_a_deleted_id_is_not_reused_after_a_round_trip():
    ps = ProbeSet()
    ps.add("point", PointGeom(1.0, 1.0))
    ps.add("point", PointGeom(2.0, 2.0))
    third = ps.add("line", LineGeom(0.0, 0.0, 5.0, 0.0))
    ps.remove(third.id)
    back = ProbeSet.from_payload(ps.to_payload())
    assert back.add("point", PointGeom(3.0, 3.0)).id == 4


def test_duplicate_ids_are_refused():
    ps = ProbeSet()
    ps.add("point", PointGeom(1.0, 1.0))
    items = ps.to_payload()["items"] * 2
    with pytest.raises(ValueError, match="duplicate"):
        ProbeSet.from_payload({"items": items, "next_id": 5})


def test_the_list_form_written_during_development_still_loads():
    ps = ProbeSet()
    ps.add("point", PointGeom(1.0, 1.0))
    back = ProbeSet.from_payload(ps.to_payload()["items"])
    assert len(back) == 1
    assert back.add("point", PointGeom(2.0, 2.0)).id == 2


def test_probes_are_parsed_when_the_file_is_read():
    """So apply_session can no longer fail halfway, after clearing results."""
    ps = ProbeSet()
    ps.add("point", PointGeom(1.0, 1.0), label="tip")
    session = _parse_config(_doc(ps.to_payload()))
    assert isinstance(session.probes, ProbeSet)
    assert [p.label for p in session.probes] == ["tip"]


@pytest.mark.parametrize("bad", [
    ["not-a-dict"],
    [{"id": 1, "kind": "point"}],                                   # no geometry
    [{"id": 1, "kind": "wedge", "geometry": {}, "label": "x", "color": "#000000"}],
    {"items": "nope", "next_id": 2},
    "nope",
])
def test_a_malformed_probe_block_is_a_session_error(bad):
    """Not an AttributeError that escapes the application's handler."""
    with pytest.raises(SessionError):
        _parse_config(_doc(bad))


def test_duplicate_ids_in_a_file_are_a_session_error():
    ps = ProbeSet()
    ps.add("point", PointGeom(1.0, 1.0))
    items = ps.to_payload()["items"] * 2
    with pytest.raises(SessionError):
        _parse_config(_doc({"items": items, "next_id": 3}))
