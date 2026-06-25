from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.time import Time

from .sp3 import itrs_states_to_gcrs


@dataclass(slots=True)
class EofOrbit:
    """Разобранные векторы состояния орбиты Sentinel EOF.

    Файлы Sentinel POD EOF хранят Orbit State Vectors (OSV) в земной системе
    координат. Положения уже заданы в метрах, скорости - в метрах в секунду,
    а эпохи подписаны в UTC.
    """

    epochs: Time
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    qualities: list[str]


def read_sentinel_eof(path: str | Path) -> EofOrbit:
    """Прочитать номинальные OSV положения и скорости из файла Sentinel EOF."""
    root = ET.parse(Path(path)).getroot()

    epochs_raw: list[datetime] = []
    positions_m: list[list[float]] = []
    velocities_m_s: list[list[float]] = []
    qualities: list[str] = []

    for osv in root.findall(".//OSV"):
        quality = _node_text(osv, "Quality")
        if quality != "NOMINAL":
            continue

        utc_text = _node_text(osv, "UTC")
        epochs_raw.append(_parse_prefixed_datetime(utc_text, "UTC="))
        positions_m.append(
            [
                float(_node_text(osv, "X")),
                float(_node_text(osv, "Y")),
                float(_node_text(osv, "Z")),
            ]
        )
        velocities_m_s.append(
            [
                float(_node_text(osv, "VX")),
                float(_node_text(osv, "VY")),
                float(_node_text(osv, "VZ")),
            ]
        )
        qualities.append(quality)

    if len(epochs_raw) < 2:
        raise ValueError(f"At least two NOMINAL Sentinel EOF OSVs are required: {path}")

    return EofOrbit(
        epochs=Time(epochs_raw, scale="utc"),
        positions_m=np.asarray(positions_m, dtype=float),
        velocities_m_s=np.asarray(velocities_m_s, dtype=float),
        qualities=qualities,
    )


def eof_state_samples(
    path: str | Path,
    duration_hours: float = 12.0,
    step_seconds: float = 900.0,
) -> tuple[Time, np.ndarray]:
    """Вернуть равномерно выбранные состояния GCRS `[r, v]` из OSV Sentinel EOF.

    Выходные состояния используют единицы SI: положения в метрах и скорости
    в метрах в секунду. Входные положения и скорости Sentinel EOF уже заданы в SI.
    """
    orbit = read_sentinel_eof(path)
    source_seconds = (orbit.epochs - orbit.epochs[0]).sec
    max_duration = min(duration_hours * 3600.0, float(source_seconds[-1]))
    target_seconds = np.arange(0.0, max_duration + 0.5 * step_seconds, step_seconds)
    target_epochs = orbit.epochs[0] + target_seconds * u.s

    target_itrs_m = np.column_stack(
        [np.interp(target_seconds, source_seconds, orbit.positions_m[:, axis]) for axis in range(3)]
    )
    target_itrs_velocity_m_s = np.column_stack(
        [np.interp(target_seconds, source_seconds, orbit.velocities_m_s[:, axis]) for axis in range(3)]
    )
    target_gcrs_m, velocities_gcrs_m_s = itrs_states_to_gcrs(
        target_epochs,
        target_itrs_m,
        target_itrs_velocity_m_s,
    )
    return target_epochs, np.column_stack((target_gcrs_m, velocities_gcrs_m_s))


def _node_text(parent: ET.Element, tag: str) -> str:
    node = parent.find(tag)
    if node is None or node.text is None:
        raise ValueError(f"Missing Sentinel EOF OSV field: {tag}")
    return node.text.strip()


def _parse_prefixed_datetime(value: str, prefix: str) -> datetime:
    if not value.startswith(prefix):
        raise ValueError(f"Expected {prefix!r} datetime field, got: {value!r}")
    return datetime.fromisoformat(value[len(prefix) :])
