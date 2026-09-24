"""The canonical field registry, and a guard on the copies that predate it.

``export_csv``, ``export_mat`` and ``export_npz`` each carry their own
``_DISP_FIELDS`` / ``_STRAIN_FIELDS`` frozenset. They are identical today. Rather
than add a fourth copy for the analysis package -- or migrate three modules with
a documented history of breaking changes -- the registry becomes the single
source of truth and this test fails the moment any copy drifts from it.
"""

from __future__ import annotations

import pytest

from al_dic.core.fields import (
    DISP_FIELDS,
    STRAIN_FIELDS,
    field_unit,
    is_strain_field,
    validate_field,
)


def test_registry_contents():
    assert DISP_FIELDS == frozenset(
        {"disp_u", "disp_v", "disp_magnitude"}
    )
    assert STRAIN_FIELDS == frozenset(
        {
            "strain_exx", "strain_eyy", "strain_exy",
            "strain_principal_max", "strain_principal_min",
            "strain_maxshear", "strain_von_mises", "strain_rotation",
        }
    )
    assert not (DISP_FIELDS & STRAIN_FIELDS)


@pytest.mark.parametrize("module_name", [
    "al_dic.export.export_csv",
    "al_dic.export.export_mat",
    "al_dic.export.export_npz",
])
def test_export_modules_have_not_drifted(module_name):
    """Every pre-existing copy still agrees with the registry."""
    import importlib

    mod = importlib.import_module(module_name)
    assert mod._DISP_FIELDS == DISP_FIELDS, (
        f"{module_name}._DISP_FIELDS has drifted from al_dic.core.fields"
    )
    assert mod._STRAIN_FIELDS == STRAIN_FIELDS, (
        f"{module_name}._STRAIN_FIELDS has drifted from al_dic.core.fields"
    )


def test_validate_field_accepts_known_names():
    for name in DISP_FIELDS | STRAIN_FIELDS:
        validate_field(name)


def test_validate_field_rejects_unknown_and_names_the_alternatives():
    """A typo must raise, not silently produce a column of nulls.

    The reference implementation looks strain components up in an open dict, so
    `strain_eyx` yields a series of nulls that looks like missing data.
    """
    with pytest.raises(ValueError) as exc:
        validate_field("strain_eyx")
    message = str(exc.value)
    assert "strain_eyx" in message
    assert "strain_exy" in message, "the error should suggest valid names"


def test_is_strain_field():
    assert is_strain_field("strain_exx")
    assert not is_strain_field("disp_u")


def test_field_unit_distinguishes_length_from_dimensionless():
    # Displacements carry a length unit and must follow the physical-unit
    # setting; strains are dimensionless and must not be scaled by it.
    assert field_unit("disp_u", "px") == "px"
    assert field_unit("disp_u", "mm") == "mm"
    assert field_unit("strain_exx", "mm") == ""
    assert field_unit("strain_von_mises", "px") == ""
    # compute_strain stores rotation via np.degrees. This line used to assert
    # "rad", pinning the mislabel the chart and CSV then showed.
    assert field_unit("strain_rotation", "mm") == "°"
