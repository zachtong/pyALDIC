"""Canonical names for the measured quantities a result carries.

One registry, so that a caller asking for a quantity by name gets an error for a
typo rather than a column of nothing. ``export_csv``, ``export_mat`` and
``export_npz`` each predate this module and carry their own identical copies;
``tests/test_analysis/test_fields.py`` fails if any of them drifts from here, so
the duplication is guarded until those modules are migrated deliberately.

Displacements carry a length unit and follow the project's physical-unit
setting. Strains are dimensionless and must never be scaled by it -- multiplying
a strain by a pixel-to-millimetre ratio is a silent, plausible-looking error.
"""

from __future__ import annotations

#: Displacement components, in the reference (frame-0) coordinate system.
DISP_FIELDS: frozenset[str] = frozenset([
    "disp_u", "disp_v", "disp_magnitude",
])

#: Strain components produced by the strain post-processing stage.
STRAIN_FIELDS: frozenset[str] = frozenset([
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
])

#: Everything a probe or an export may be asked for.
ALL_FIELDS: frozenset[str] = DISP_FIELDS | STRAIN_FIELDS

#: The one strain component that is not dimensionless.
_ANGLE_FIELDS: frozenset[str] = frozenset(["strain_rotation"])


def is_strain_field(name: str) -> bool:
    """True when *name* is a strain component rather than a displacement."""
    return name in STRAIN_FIELDS


def field_unit(name: str, length_unit: str) -> str:
    """Unit string for *name* given the project's configured *length_unit*.

    Displacements take the length unit; strains are dimensionless and return an
    empty string; rotation is an angle in degrees whatever the length unit is
    -- ``compute_strain`` stores it via ``np.degrees``. It was labelled "rad"
    here, so a 2 degree rotation read as 2 radians on the chart and in the CSV.
    """
    validate_field(name)
    if name in _ANGLE_FIELDS:
        return "°"
    if is_strain_field(name):
        return ""
    return length_unit


def validate_field(name: str) -> None:
    """Raise ``ValueError`` unless *name* is a known field.

    The message lists the near-misses, because the usual cause is a
    transposition (``strain_eyx`` for ``strain_exy``) and an unexplained
    rejection is barely better than a silent one.
    """
    if name in ALL_FIELDS:
        return
    prefix = name.split("_")[0] if "_" in name else name
    near = sorted(f for f in ALL_FIELDS if f.startswith(prefix))
    suggestion = ", ".join(near or sorted(ALL_FIELDS))
    raise ValueError(f"Unknown field {name!r}. Known fields: {suggestion}")
