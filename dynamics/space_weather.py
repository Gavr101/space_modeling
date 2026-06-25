"""Разбор CSV с космической погодой CelesTrak и вспомогательные функции кеша."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve


CELESTRAK_SPACE_WEATHER_ALL_CSV_URL = "https://celestrak.org/SpaceData/SW-All.csv"
DEFAULT_QUIET_F107A = 150.0
DEFAULT_QUIET_F107 = 150.0
DEFAULT_QUIET_AP = 4.0


@dataclass(frozen=True, slots=True)
class SpaceWeatherRecord:
    """Одна суточная строка CSV CelesTrak с параметрами космической погоды."""

    date: date
    ap_3h: tuple[float, float, float, float, float, float, float, float]
    ap_avg: float
    f107_obs: float
    f107_adj: float
    f107_obs_center81: float
    f107_obs_last81: float
    f107_adj_center81: float
    f107_adj_last81: float


@dataclass(frozen=True, slots=True)
class SpaceWeatherSample:
    """Скалярные входные параметры NRLMSISE-00, выбранные для UTC-эпохи."""

    f107a: float
    f107: float
    ap: float
    ap_daily: float
    source: str


def _parse_float(value: str) -> float:
    text = value.strip()
    if text == "":
        return math.nan
    return float(text)


def parse_celestrak_space_weather_csv(text: str) -> dict[date, SpaceWeatherRecord]:
    """Разобрать содержимое CelesTrak `SW-All.csv`/`SW-Last5Years.csv`."""
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    reader = csv.DictReader(lines)
    records: dict[date, SpaceWeatherRecord] = {}
    for row in reader:
        if not row.get("DATE"):
            continue
        record_date = date.fromisoformat(row["DATE"].strip())
        ap_3h = tuple(_parse_float(row[f"AP{idx}"]) for idx in range(1, 9))
        records[record_date] = SpaceWeatherRecord(
            date=record_date,
            ap_3h=ap_3h,  # type: ignore[arg-type]
            ap_avg=_parse_float(row.get("AP_AVG", "")),
            f107_obs=_parse_float(row.get("F10.7_OBS", "")),
            f107_adj=_parse_float(row.get("F10.7_ADJ", "")),
            f107_obs_center81=_parse_float(row.get("F10.7_OBS_CENTER81", "")),
            f107_obs_last81=_parse_float(row.get("F10.7_OBS_LAST81", "")),
            f107_adj_center81=_parse_float(row.get("F10.7_ADJ_CENTER81", "")),
            f107_adj_last81=_parse_float(row.get("F10.7_ADJ_LAST81", "")),
        )
    return records


@lru_cache(maxsize=8)
def load_celestrak_space_weather_csv(path: str | Path) -> dict[date, SpaceWeatherRecord]:
    """Прочитать и закешировать локальный CSV CelesTrak с космической погодой."""
    csv_path = Path(path)
    return parse_celestrak_space_weather_csv(csv_path.read_text(encoding="utf-8"))


def download_celestrak_space_weather_csv(
    cache_path: str | Path = Path("data") / "cache" / "SW-All.csv",
    *,
    url: str = CELESTRAK_SPACE_WEATHER_ALL_CSV_URL,
) -> Path:
    """Скачать CSV CelesTrak с космической погодой в воспроизводимый локальный кеш."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)
    return path


def quiet_space_weather_sample() -> SpaceWeatherSample:
    """Вернуть явные константы спокойной атмосферы при отсутствии данных индексов."""
    return SpaceWeatherSample(
        f107a=DEFAULT_QUIET_F107A,
        f107=DEFAULT_QUIET_F107,
        ap=DEFAULT_QUIET_AP,
        ap_daily=DEFAULT_QUIET_AP,
        source="quiet_constants",
    )


def sample_space_weather(
    epoch: datetime,
    records: dict[date, SpaceWeatherRecord],
) -> SpaceWeatherSample:
    """Выбрать входные F10.7/F10.7a/Ap для UTC-эпохи."""
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    utc = epoch.astimezone(timezone.utc)
    record = records.get(utc.date())
    if record is None:
        return quiet_space_weather_sample()

    ap_index = min(7, max(0, utc.hour // 3))
    previous_record = records.get(utc.date() - timedelta(days=1))
    f107_record = previous_record if previous_record is not None else record
    f107 = f107_record.f107_adj if math.isfinite(f107_record.f107_adj) else f107_record.f107_obs
    f107a = (
        record.f107_adj_center81
        if math.isfinite(record.f107_adj_center81)
        else record.f107_obs_center81
    )
    ap = record.ap_3h[ap_index]
    if not math.isfinite(f107) or not math.isfinite(f107a) or not math.isfinite(ap):
        return quiet_space_weather_sample()
    ap_daily = record.ap_avg if math.isfinite(record.ap_avg) else ap
    return SpaceWeatherSample(
        f107a=f107a,
        f107=f107,
        ap=ap,
        ap_daily=ap_daily,
        source=f"celestrak:{record.date.isoformat()}:ap{ap_index + 1}",
    )
