"""Utilities for reproducible residual metrics and experiment result exports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .propagator import R_EARTH


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, numpy values, and paths to JSON-compatible objects."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def residual_series(
    reference_states: np.ndarray,
    model_states: np.ndarray,
    *,
    earth_radius_m: float = R_EARTH,
) -> dict[str, np.ndarray]:
    """Compute aligned residual time series.

    Parameters
    ----------
    reference_states:
        Array of reference states `[x, y, z, vx, vy, vz]` in `[m, m/s]`.
    model_states:
        Array of model states with the same shape and units as `reference_states`.
    earth_radius_m:
        Spherical Earth radius used only for altitude residuals [m].
    """
    reference = np.asarray(reference_states, dtype=float)
    model = np.asarray(model_states, dtype=float)
    if reference.shape != model.shape:
        raise ValueError("reference_states and model_states must have the same shape.")
    if reference.ndim != 2 or reference.shape[1] != 6:
        raise ValueError("state arrays must have shape (N, 6).")

    dr = model[:, :3] - reference[:, :3]
    dv = model[:, 3:] - reference[:, 3:]
    model_altitude = np.linalg.norm(model[:, :3], axis=1) - earth_radius_m
    reference_altitude = np.linalg.norm(reference[:, :3], axis=1) - earth_radius_m
    return {
        "delta_r_km": np.linalg.norm(dr, axis=1) / 1000.0,
        "delta_h_km": np.abs(model_altitude - reference_altitude) / 1000.0,
        "delta_v_km_s": np.linalg.norm(dv, axis=1) / 1000.0,
    }


def summarize_residuals(
    reference_states: np.ndarray,
    model_states: np.ndarray,
    *,
    earth_radius_m: float = R_EARTH,
) -> dict[str, float]:
    """Return median/max residual metrics for aligned state arrays."""
    series = residual_series(reference_states, model_states, earth_radius_m=earth_radius_m)
    return {
        "median_altitude_km": float(
            np.median(np.linalg.norm(reference_states[:, :3], axis=1) - earth_radius_m)
            / 1000.0
        ),
        "median_delta_r_km": float(np.median(series["delta_r_km"])),
        "max_delta_r_km": float(np.max(series["delta_r_km"])),
        "median_delta_h_km": float(np.median(series["delta_h_km"])),
        "max_delta_h_km": float(np.max(series["delta_h_km"])),
        "median_delta_v_km_s": float(np.median(series["delta_v_km_s"])),
        "max_delta_v_km_s": float(np.max(series["delta_v_km_s"])),
    }


def make_experiment_record(
    *,
    name: str,
    source: dict[str, Any],
    propagation_config: Any,
    reference_states: np.ndarray,
    model_states: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one serializable experiment-summary record."""
    record = {
        "name": name,
        "source": _jsonable(source),
        "propagation": _jsonable(propagation_config),
        "metrics": summarize_residuals(reference_states, model_states),
    }
    if extra:
        record["extra"] = _jsonable(extra)
    return record


def write_experiment_summary(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    stem: str = "orbit_prediction_summary",
) -> tuple[Path, Path]:
    """Write experiment records to JSON and compact CSV files.

    Returns
    -------
    tuple[Path, Path]
        Paths to the JSON and CSV files.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / f"{stem}.json"
    csv_path = path / f"{stem}.csv"

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records": _jsonable(records),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "name",
        "median_altitude_km",
        "median_delta_r_km",
        "max_delta_r_km",
        "median_delta_h_km",
        "max_delta_h_km",
        "median_delta_v_km_s",
        "max_delta_v_km_s",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics", {})
            writer.writerow(
                {
                    "name": record.get("name", ""),
                    **{key: metrics.get(key, "") for key in fieldnames[1:]},
                }
            )

    return json_path, csv_path
