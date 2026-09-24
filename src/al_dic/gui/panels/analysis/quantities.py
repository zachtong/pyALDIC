"""What the Analysis tab can plot, and the rules for showing it.

No widgets here: the quantity a chart plots, the fields and gauge readings on
offer, how a value is scaled for display, and which frames a profile draws
behind the current one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from al_dic.core.fields import is_strain_field
from al_dic.utils.locale_format import format_number

# Fields a probe can read, in the Strain Field tab's order.
FIELDS = (
    "disp_u", "disp_v", "disp_magnitude",
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
)

# Gauge quantities that are strains (and so take the strain display unit).
STRAIN_GAUGES = frozenset({"strain", "true_strain"})

# Display scale and unit for dimensionless strain.
STRAIN_UNITS = {"ratio": (1.0, ""), "percent": (100.0, "%"),
                "microstrain": (1e6, "µε")}

# What a gauge tool plots once it has placed its line, and the readings that
# already count as its own -- placing a second extensometer keeps a chart of
# elongation rather than resetting it to strain.
GAUGE_TOOLS = {
    "extensometer": ("strain", frozenset({"strain", "true_strain", "elongation"})),
    "crack_gauge": ("cod", frozenset({"cod", "cod_sliding", "cod_magnitude"})),
}

# The chart's views, in tab order.
VIEWS = ("time", "profile", "kymograph", "stress_strain")

# A profile draws at most this many other frames: past a dozen grey lines
# the family stops being readable.
MAX_OTHER_FRAMES = 12


@dataclass(frozen=True)
class Quantity:
    """What the chart is plotting: a field, or a gauge reading."""

    kind: str            # "field" | "gauge"
    name: str            # field name, or gauge quantity

    @property
    def is_gauge(self) -> bool:
        return self.kind == "gauge"

    @property
    def key(self) -> str:
        """Combo item data. A string, because QComboBox.findData compares
        Python objects by identity, so an equal dataclass is never found."""
        return f"{self.kind}:{self.name}"

    @staticmethod
    def from_key(key) -> "Quantity | None":
        if not isinstance(key, str) or ":" not in key:
            return None
        kind, name = key.split(":", 1)
        return Quantity(kind, name)


def is_strainlike(quantity: Quantity | None, statistic: str | None) -> bool:
    """True when the values are a dimensionless strain (so %, µε apply)."""
    if quantity is None:
        return False
    if quantity.is_gauge:
        return quantity.name in STRAIN_GAUGES
    return (is_strain_field(quantity.name)
            and quantity.name != "strain_rotation"
            and statistic != "valid_fraction")


def display_scale(quantity: Quantity, statistic: str | None, *,
                  strain_unit: str, length_unit: str) -> tuple[float, str]:
    """Factor and unit a value is shown in. Lengths arrive already scaled."""
    if is_strainlike(quantity, statistic):
        return STRAIN_UNITS[strain_unit or "ratio"]
    if quantity.is_gauge:
        return 1.0, length_unit
    if statistic == "valid_fraction":
        return 1.0, ""
    if quantity.name == "strain_rotation":
        return 1.0, "°"
    if is_strain_field(quantity.name):
        return 1.0, ""
    return 1.0, length_unit


def other_frames(n: int, current: int) -> list[int]:
    """The frames a profile draws behind *current*, evenly thinned."""
    others = [i for i in range(n) if i != current]
    if len(others) > MAX_OTHER_FRAMES:
        picks = np.linspace(0, len(others) - 1, MAX_OTHER_FRAMES)
        others = [others[i] for i in sorted(set(np.round(picks).astype(int)))]
    return others


def format_length(px: float, pixel_size: float, unit: str) -> str:
    """A gauge length for its label: three significant figures, at least one
    decimal, in the display unit."""
    value = px * pixel_size
    decimals = 1
    if value != 0.0 and math.isfinite(value):
        decimals = max(1, 2 - int(math.floor(math.log10(abs(value)))))
    return f"{format_number(value, decimals)} {unit}"


__all__ = [
    "FIELDS", "GAUGE_TOOLS", "MAX_OTHER_FRAMES", "Quantity", "STRAIN_GAUGES",
    "STRAIN_UNITS", "VIEWS", "display_scale", "format_length", "is_strainlike",
    "other_frames",
]
