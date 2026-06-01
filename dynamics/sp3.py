from __future__ import annotations

import gzip
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from io import StringIO

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers

iers.conf.auto_download = False
iers.conf.auto_max_age = None
iers.conf.iers_degraded_accuracy = "warn"


@dataclass(slots=True)
class Sp3Orbit:
    """Parsed SP3 satellite positions and optional velocities.

    SP3 precise orbit files usually store GNSS satellite positions in an
    Earth-fixed terrestrial frame. Position records are converted from
    kilometres to metres while reading. Velocity records, when present, are
    converted from decimetres per second to metres per second.
    """

    epochs: Time
    positions_m: dict[str, np.ndarray]
    velocities_m_s: dict[str, np.ndarray]
    time_system: str = "GPS"


def download_sp3(url: str, output_path: str | Path) -> Path:
    """Download an SP3 file if it is not already present locally."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    return path


def read_sp3(path: str | Path) -> Sp3Orbit:
    """Read positions and optional velocities from an SP3 precise orbit file.

    Supported inputs are plain text `.sp3` files, gzip-compressed `.sp3.gz`
    files, and `.zip` archives that contain one `.sp3` member. Position and
    velocity records with invalid SP3 sentinel values are skipped.
    """
    path = Path(path)

    epochs_raw: list[datetime] = []
    positions_by_sat: dict[str, list[np.ndarray]] = {}
    velocities_by_sat: dict[str, list[np.ndarray]] = {}
    current_epoch_index: int | None = None
    time_system = "GPS"

    with _open_sp3_text(path) as file:
        for line in file:
            if line.startswith("%c"):
                parts = line.split()
                for part in parts:
                    if part.upper() in {"GPS", "UTC", "TAI", "GAL", "GLO", "QZS"}:
                        time_system = part.upper()
                        break
                continue

            if line.startswith("*"):
                parts = line.split()
                year, month, day = map(int, parts[1:4])
                hour, minute = map(int, parts[4:6])
                second_float = float(parts[6])
                second = int(second_float)
                microsecond = int(round((second_float - second) * 1_000_000))
                epochs_raw.append(datetime(year, month, day, hour, minute, second, microsecond))
                current_epoch_index = len(epochs_raw) - 1
                continue

            if line.startswith("P") and current_epoch_index is not None:
                sat_id, position_km = _parse_sp3_vector_record(line)
                if sat_id is None or position_km is None:
                    continue

                if np.any(np.abs(position_km) >= 999_999.0):
                    continue

                sat_positions = positions_by_sat.setdefault(sat_id, [])
                while len(sat_positions) < current_epoch_index:
                    sat_positions.append(np.full(3, np.nan))
                sat_positions.append(position_km * 1000.0)
                continue

            if line.startswith("V") and current_epoch_index is not None:
                sat_id, velocity_dm_s = _parse_sp3_vector_record(line)
                if sat_id is None or velocity_dm_s is None:
                    continue

                if np.any(np.abs(velocity_dm_s) >= 999_999.0):
                    continue

                sat_velocities = velocities_by_sat.setdefault(sat_id, [])
                while len(sat_velocities) < current_epoch_index:
                    sat_velocities.append(np.full(3, np.nan))
                sat_velocities.append(velocity_dm_s * 0.1)

    epochs = _sp3_epochs_to_time(epochs_raw, time_system)
    positions_m = {
        sat_id: np.asarray(values, dtype=float)
        for sat_id, values in positions_by_sat.items()
        if len(values) == len(epochs_raw)
    }
    velocities_m_s = {
        sat_id: np.asarray(values, dtype=float)
        for sat_id, values in velocities_by_sat.items()
        if len(values) == len(epochs_raw)
    }
    return Sp3Orbit(
        epochs=epochs,
        positions_m=positions_m,
        velocities_m_s=velocities_m_s,
        time_system=time_system,
    )


def sp3_state_samples(
    path: str | Path,
    satellite_id: str,
    duration_hours: float = 12.0,
    step_seconds: float = 900.0,
) -> tuple[Time, np.ndarray]:
    """Return uniformly sampled GCRS states `[r, v]` from SP3 positions.

    SP3 velocity records are used when they are available for the requested
    satellite. Otherwise, the velocity is estimated by finite differences after
    converting positions to GCRS.
    """
    orbit = read_sp3(path)
    if satellite_id not in orbit.positions_m:
        available = ", ".join(sorted(orbit.positions_m)[:12])
        raise KeyError(f"Satellite {satellite_id!r} not found in SP3 file. Available examples: {available}")

    source_seconds = (orbit.epochs - orbit.epochs[0]).sec
    max_duration = min(duration_hours * 3600.0, float(source_seconds[-1]))
    target_seconds = np.arange(0.0, max_duration + 0.5 * step_seconds, step_seconds)
    target_epochs = orbit.epochs[0] + target_seconds * u.s

    positions_itrs_m = orbit.positions_m[satellite_id]
    target_itrs_m = np.column_stack(
        [np.interp(target_seconds, source_seconds, positions_itrs_m[:, axis]) for axis in range(3)]
    )

    velocity_records = sp3_velocity_records(orbit)
    if satellite_id in velocity_records:
        velocities_itrs_m_s = velocity_records[satellite_id]
        target_itrs_velocity_m_s = np.column_stack(
            [np.interp(target_seconds, source_seconds, velocities_itrs_m_s[:, axis]) for axis in range(3)]
        )
        target_gcrs_m, velocities_gcrs_m_s = itrs_states_to_gcrs(
            target_epochs,
            target_itrs_m,
            target_itrs_velocity_m_s,
        )
    else:
        target_gcrs_m = itrs_positions_to_gcrs(target_epochs, target_itrs_m)
        velocities_gcrs_m_s = finite_difference_velocity(target_seconds, target_gcrs_m)
    return target_epochs, np.column_stack((target_gcrs_m, velocities_gcrs_m_s))


def sp3_velocity_records(orbit: Sp3Orbit) -> dict[str, np.ndarray]:
    """Return SP3 velocity records [m/s], or an empty mapping for old objects.

    This keeps notebooks robust after editing `dynamics.sp3` in a live Jupyter
    kernel where a previously imported `Sp3Orbit` class may still be present.
    """
    return getattr(orbit, "velocities_m_s", {})


def itrs_positions_to_gcrs(epochs: Time, positions_itrs_m: np.ndarray) -> np.ndarray:
    """Convert Earth-fixed SP3 positions [m] to GCRS positions [m]."""
    representation = CartesianRepresentation(
        x=positions_itrs_m[:, 0] * u.m,
        y=positions_itrs_m[:, 1] * u.m,
        z=positions_itrs_m[:, 2] * u.m,
    )
    itrs = ITRS(representation, obstime=epochs)
    gcrs = itrs.transform_to(GCRS(obstime=epochs))
    return gcrs.cartesian.xyz.to_value(u.m).T


def itrs_states_to_gcrs(
    epochs: Time,
    positions_itrs_m: np.ndarray,
    velocities_itrs_m_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Earth-fixed SP3 position/velocity states to GCRS states."""
    representation = CartesianRepresentation(
        x=positions_itrs_m[:, 0] * u.m,
        y=positions_itrs_m[:, 1] * u.m,
        z=positions_itrs_m[:, 2] * u.m,
        differentials=CartesianDifferential(
            d_x=velocities_itrs_m_s[:, 0] * u.m / u.s,
            d_y=velocities_itrs_m_s[:, 1] * u.m / u.s,
            d_z=velocities_itrs_m_s[:, 2] * u.m / u.s,
        ),
    )
    itrs = ITRS(representation, obstime=epochs)
    gcrs = itrs.transform_to(GCRS(obstime=epochs))
    return (
        gcrs.cartesian.xyz.to_value(u.m).T,
        gcrs.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s).T,
    )


def finite_difference_velocity(elapsed_seconds: np.ndarray, positions_m: np.ndarray) -> np.ndarray:
    """Estimate velocity [m/s] from sampled positions with finite differences."""
    if len(elapsed_seconds) < 2:
        raise ValueError("At least two SP3 samples are required to estimate velocity.")
    edge_order = 2 if len(elapsed_seconds) >= 3 else 1
    return np.gradient(positions_m, elapsed_seconds, axis=0, edge_order=edge_order)


def _sp3_epochs_to_time(epochs_raw: list[datetime], time_system: str) -> Time:
    """Convert SP3 calendar labels to Astropy Time.

    IGS SP3 orbit products are commonly tagged in GPS time. GPS is offset from
    TAI by a constant 19 seconds, so the conversion below avoids treating GPS
    labels as UTC labels.
    """
    if time_system.upper() == "GPS":
        return (Time(epochs_raw, scale="tai") + 19.0 * u.s).utc
    return Time(epochs_raw, scale="utc")


@contextmanager
def _open_sp3_text(path: Path):
    """Open a plain, gzip-compressed, or zip-contained SP3 text stream."""
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="ascii", errors="replace") as file:
            yield file
        return

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            sp3_members = [name for name in archive.namelist() if name.lower().endswith(".sp3")]
            if not sp3_members:
                raise ValueError(f"No .sp3 member found in ZIP archive: {path}")
            text = archive.read(sp3_members[0]).decode("ascii", errors="replace")
            yield StringIO(text)
        return

    with open(path, "rt", encoding="ascii", errors="replace") as file:
        yield file


def _parse_sp3_vector_record(line: str) -> tuple[str | None, np.ndarray | None]:
    """Parse the first three numeric fields from a P or V SP3 record."""
    sat_id = line[1:4].strip()
    if not sat_id:
        return None, None

    try:
        values = np.array(
            [
                float(line[4:18]),
                float(line[18:32]),
                float(line[32:46]),
            ],
            dtype=float,
        )
    except ValueError:
        parts = line.split()
        if len(parts) < 4:
            return None, None
        sat_id = parts[0][1:]
        values = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
    return sat_id, values
