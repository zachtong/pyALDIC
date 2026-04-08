"""Export DIC computation parameters as a structured JSON file.

The parameters file is always written regardless of which data formats are
selected.  It records every DICPara field plus run-level metadata so that
results can be reproduced and interpreted without the original GUI session.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from staq_dic.core.data_structures import PipelineResult
from staq_dic.export.export_utils import ensure_dir


def _to_json_value(v: Any) -> Any:
    """Convert a DICPara field value to a JSON-serializable type."""
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_to_json_value(x) for x in v]
    if isinstance(v, np.ndarray):
        return None  # skip large arrays; summarised in n_nodes etc.
    # dataclass (e.g. GridxyROIRange, FrameSchedule) -> dict
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {
            f.name: _to_json_value(getattr(v, f.name))
            for f in dataclasses.fields(v)
        }
    return str(v)


def export_params(
    dest_dir: Path,
    prefix: str,
    timestamp: str,
    results: PipelineResult,
) -> Path:
    """Write a JSON file containing all DIC parameters and run metadata.

    Args:
        dest_dir: Directory to write into (created if absent).
        prefix: Filename prefix (e.g. derived from image folder name).
        timestamp: 14-digit timestamp string (YYYYMMDDHHMMSS).
        results: Full pipeline results object.

    Returns:
        Path to the written JSON file.
    """
    ensure_dir(dest_dir)
    p = results.dic_para
    n_nodes = results.dic_mesh.coordinates_fem.shape[0]

    # Serialize every DICPara field
    dic_para_dict: dict[str, Any] = {}
    for f in dataclasses.fields(p):
        dic_para_dict[f.name] = _to_json_value(getattr(p, f.name))

    data: dict[str, Any] = {
        "export_timestamp": timestamp,
        "n_frames": len(results.result_disp),
        "n_nodes": n_nodes,
        "has_strain": len(results.result_strain) > 0,
        **dic_para_dict,
    }

    out = dest_dir / f"{prefix}_parameters_{timestamp}.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out
