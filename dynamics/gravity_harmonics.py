"""Статические сферические гармоники гравитации Земли в связанной с телом системе."""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from scipy.special import lpmv


DEFAULT_EGM2008_GFC_URL = (
    "https://icgem.gfz-potsdam.de/getmodel/gfc/"
    "c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340/EGM2008.gfc"
)


@dataclass(frozen=True, slots=True)
class GravityHarmonicCoefficients:
    """Полностью нормированные статические коэффициенты гравитационного поля.

    `c[n, m]` и `s[n, m]` полностью нормированы и совместимы с соглашением
    ICGEM `.gfc`. Для `mu_m3_s2` и `radius_m` используются единицы SI.
    """

    mu_m3_s2: float
    radius_m: float
    c: np.ndarray
    s: np.ndarray

    @property
    def max_degree(self) -> int:
        return int(self.c.shape[0] - 1)


def _parse_icgem_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def download_icgem_gfc(
    url: str,
    cache_path: str | Path,
) -> Path:
    """Скачать файл ICGEM `.gfc` или архив `.zip` в локальный кеш."""
    target = Path(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    download_path = target.with_name(f"{target.name}.download")
    urlretrieve(url, download_path)

    try:
        if zipfile.is_zipfile(download_path):
            with zipfile.ZipFile(download_path) as archive:
                gfc_names = [name for name in archive.namelist() if name.lower().endswith(".gfc")]
                if not gfc_names:
                    raise ValueError(f"{url} zip archive does not contain a .gfc file.")
                with archive.open(gfc_names[0]) as source, target.open("wb") as output:
                    output.write(source.read())
        else:
            download_path.replace(target)
    finally:
        if download_path.exists():
            download_path.unlink()

    return target


def download_egm2008_gfc(
    cache_path: str | Path = Path("data") / "cache" / "egm2008.gfc",
    *,
    url: str = DEFAULT_EGM2008_GFC_URL,
) -> Path:
    """Скачать коэффициенты гравитационного поля EGM2008 в локальный кеш."""
    return download_icgem_gfc(url, cache_path)


@lru_cache(maxsize=4)
def read_icgem_gfc(
    path: str | Path,
    *,
    max_degree: int | None = None,
    max_order: int | None = None,
) -> GravityHarmonicCoefficients:
    """Прочитать полностью нормированные коэффициенты ICGEM `.gfc` из локального файла."""
    source = Path(path)
    mu = math.nan
    radius = math.nan
    norm = ""
    rows: list[tuple[int, int, float, float]] = []

    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0].lower()
        if key == "earth_gravity_constant" and len(parts) >= 2:
            mu = _parse_icgem_float(parts[1])
        elif key == "radius" and len(parts) >= 2:
            radius = _parse_icgem_float(parts[1])
        elif key == "norm" and len(parts) >= 2:
            norm = parts[1].lower()
        elif key == "gfc" and len(parts) >= 5:
            n = int(parts[1])
            m = int(parts[2])
            if max_degree is not None and n > max_degree:
                continue
            if max_order is not None and m > max_order:
                continue
            rows.append((n, m, _parse_icgem_float(parts[3]), _parse_icgem_float(parts[4])))

    if not math.isfinite(mu) or not math.isfinite(radius):
        raise ValueError(f"{source} does not contain ICGEM gravity constant/radius metadata.")
    if norm and norm not in {"fully_normalized", "fully-normalized"}:
        raise ValueError(f"{source} uses unsupported normalization {norm!r}.")

    degree = max((n for n, _, _, _ in rows), default=0)
    c = np.zeros((degree + 1, degree + 1), dtype=float)
    s = np.zeros_like(c)
    for n, m, c_nm, s_nm in rows:
        c[n, m] = c_nm
        s[n, m] = s_nm
    return GravityHarmonicCoefficients(mu_m3_s2=mu, radius_m=radius, c=c, s=s)


def _fully_normalized_legendre(max_degree: int, sin_latitude: float) -> np.ndarray:
    """Вычислить полностью нормированные присоединённые функции Лежандра."""
    p = np.zeros((max_degree + 1, max_degree + 1), dtype=float)
    x = float(np.clip(sin_latitude, -1.0, 1.0))
    for n in range(max_degree + 1):
        for m in range(n + 1):
            delta = 1.0 if m == 0 else 0.0
            norm = math.sqrt((2.0 - delta) * (2 * n + 1) * math.factorial(n - m))
            norm /= math.sqrt(math.factorial(n + m))
            p[n, m] = norm * lpmv(m, n, x)
    return p


def harmonic_perturbing_potential(
    r_itrs: np.ndarray,
    coefficients: GravityHarmonicCoefficients,
    *,
    max_degree: int | None = None,
    max_order: int | None = None,
) -> float:
    """Возмущение статического нецентрального гармонического потенциала [м^2/с^2]."""
    r = np.asarray(r_itrs, dtype=float)
    radius = np.linalg.norm(r)
    if radius <= 0.0:
        raise ValueError("r_itrs norm must be positive.")

    nmax = coefficients.max_degree if max_degree is None else max_degree
    nmax = min(nmax, coefficients.max_degree)
    mmax = nmax if max_order is None else min(max_order, nmax)

    sin_lat = r[2] / radius
    lon = math.atan2(r[1], r[0])
    p = _fully_normalized_legendre(nmax, sin_lat)

    total = 0.0
    for n in range(2, nmax + 1):
        degree_scale = (coefficients.radius_m / radius) ** n
        for m in range(0, min(mmax, n) + 1):
            total += (
                degree_scale
                * p[n, m]
                * (
                    coefficients.c[n, m] * math.cos(m * lon)
                    + coefficients.s[n, m] * math.sin(m * lon)
                )
            )
    return coefficients.mu_m3_s2 / radius * total


def harmonic_perturbing_acceleration(
    r_itrs: np.ndarray,
    coefficients: GravityHarmonicCoefficients,
    *,
    max_degree: int | None = None,
    max_order: int | None = None,
    finite_difference_step_m: float = 10.0,
) -> np.ndarray:
    """Численный градиент статического гармонического возмущающего потенциала [м/с^2]."""
    r = np.asarray(r_itrs, dtype=float)
    step = float(finite_difference_step_m)
    if step <= 0.0:
        raise ValueError("finite_difference_step_m must be positive.")

    acceleration = np.zeros(3, dtype=float)
    for axis in range(3):
        delta = np.zeros(3, dtype=float)
        delta[axis] = step
        plus = harmonic_perturbing_potential(
            r + delta,
            coefficients,
            max_degree=max_degree,
            max_order=max_order,
        )
        minus = harmonic_perturbing_potential(
            r - delta,
            coefficients,
            max_degree=max_degree,
            max_order=max_order,
        )
        acceleration[axis] = (plus - minus) / (2.0 * step)
    return acceleration
