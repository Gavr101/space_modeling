from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import lru_cache
from typing import Literal
from tqdm import tqdm

import numpy as np
from astropy import units as u # orekit, sp3 библиотека с экспериментальными данными
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, get_body_barycentric
from astropy.coordinates.builtin_frames.intermediate_rotation_transforms import (
    cirs_to_itrs_mat,
    gcrs_to_cirs_mat,
)
from astropy.utils import iers
from astropy.time import Time
from scipy.integrate import solve_ivp

from .environment import EnvironmentConfig
from .gravity_harmonics import harmonic_perturbing_acceleration, read_icgem_gfc
from .space_weather import (
    load_celestrak_space_weather_csv,
    quiet_space_weather_sample,
    sample_space_weather,
)


class PropagatorBackend(str, Enum):
    TUDATPY = "tudatpy"
    OREKIT = "orekit"


@dataclass(slots=True)
class SpacecraftProperties:
    """Параметры КА, влияющие на негравитационные силы.

    mass: масса аппарата [кг]
    cd: коэффициент аэродинамического сопротивления (безразмерный)
    cr: коэффициент отражения для давления света (безразмерный)
    reference_area: эффективная площадь, на которую действуют drag/SRP [м^2]
    """

    mass: float
    cd: float
    cr: float
    reference_area: float


@dataclass(slots=True)
class PropagationConfig:
    initial_state: np.ndarray
    epoch_seconds: float
    duration_seconds: float
    step_seconds: float = 10.0
    integrator: Literal["rk4_fixed", "dop853"] | str = "dop853"
    rtol: float = 1e-10
    atol: np.ndarray = field(
        default_factory=lambda: np.array(
            [1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6],
            dtype=float,
        )
    )
    spacecraft: SpacecraftProperties = field(
        default_factory=lambda: SpacecraftProperties(5.0, 2.2, 1.3, 0.05)
    )
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    output_times_seconds: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.initial_state = np.asarray(self.initial_state, dtype=float)
        if self.initial_state.shape != (6,):
            raise ValueError(
                "initial_state must have shape (6,) with [m, m, m, m/s, m/s, m/s]."
            )
        if self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be non-negative.")
        if self.step_seconds <= 0.0:
            raise ValueError("step_seconds must be positive.")
        if self.rtol <= 0.0:
            raise ValueError("rtol must be positive.")

        self.integrator = _normalize_integrator_name(self.integrator)
        self.atol = _normalize_atol(self.atol)
        if self.output_times_seconds is not None:
            self.output_times_seconds = _normalize_output_times(
                self.output_times_seconds,
                self.duration_seconds,
            )


# --- Физические константы (SI) ---
MU_EARTH = 3.986004418e14
R_EARTH = 6378136.3
J2_EARTH = 1.08262668e-3
OMEGA_EARTH = np.array([0.0, 0.0, 7.2921150e-5])
MU_SUN = 1.32712440018e20
MU_MOON = 4.9048695e12
R_SUN = 695700000.0
AU_METERS = 149597870700.0
SOLAR_PRESSURE_AT_1AU = 4.56e-6
SPEED_OF_LIGHT = 299792458.0
EARTH_IR_FLUX_W_M2 = 239.0
SOLID_EARTH_LOVE_K2 = 0.3


def _normalize_integrator_name(integrator: str) -> Literal["rk4_fixed", "dop853"]:
    """Вернуть внутреннее имя интегратора с поддержкой исторических псевдонимов."""
    name = str(integrator).strip().lower()
    aliases = {
        "rk4": "rk4_fixed",
        "rk4_fixed": "rk4_fixed",
        "fixed_rk4": "rk4_fixed",
        "dop853": "dop853",
    }
    try:
        return aliases[name]  # type: ignore[return-value]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unsupported integrator {integrator!r}. Supported values: {allowed}."
        ) from exc


def _normalize_atol(atol: float | np.ndarray) -> np.ndarray:
    """Проверить абсолютные допуски для состояния [x, y, z, vx, vy, vz]."""
    arr = np.asarray(atol, dtype=float)
    if arr.shape == ():
        if float(arr) <= 0.0:
            raise ValueError("atol must be positive.")
        return np.full(6, float(arr), dtype=float)
    if arr.shape != (6,):
        raise ValueError("atol must be a positive scalar or a 6-vector for [m, m/s] state.")
    if not np.all(arr > 0.0):
        raise ValueError("all atol components must be positive.")
    return arr


def _normalize_j2_frame(j2_frame: str) -> Literal["gcrs_fixed_axis", "itrs_body_fixed"]:
    """Проверить выбор системы координат для J2."""
    name = str(j2_frame).strip().lower()
    if name in {"gcrs", "gcrs_fixed_axis"}:
        return "gcrs_fixed_axis"
    if name in {"itrs", "itrf", "itrs_body_fixed", "body_fixed"}:
        return "itrs_body_fixed"
    raise ValueError(
        "Unsupported j2_frame "
        f"{j2_frame!r}. Supported values: 'gcrs_fixed_axis', 'itrs_body_fixed'."
    )


def _normalize_earth_gravity_model(earth_gravity_model: str) -> Literal["j2", "egm2008"]:
    """Проверить выбор модели гравитационного поля Земли."""
    name = str(earth_gravity_model).strip().lower()
    if name in {"j2", "central_j2"}:
        return "j2"
    if name in {"egm2008", "spherical_harmonics"}:
        return "egm2008"
    raise ValueError(
        "Unsupported earth_gravity_model "
        f"{earth_gravity_model!r}. Supported values: 'j2', 'egm2008'."
    )


def _normalize_srp_shadow_model(srp_shadow_model: str) -> Literal["none", "cylindrical", "conical"]:
    """Проверить выбор модели тени для SRP."""
    name = str(srp_shadow_model).strip().lower()
    if name in {"none", "off", "false"}:
        return "none"
    if name in {"cylindrical", "cylinder"}:
        return "cylindrical"
    if name in {"conical", "cone"}:
        return "conical"
    raise ValueError(
        "Unsupported srp_shadow_model "
        f"{srp_shadow_model!r}. Supported values: 'none', 'cylindrical', 'conical'."
    )


def _normalize_earth_radiation_model(earth_radiation_model: str) -> Literal["none", "isotropic_ir"]:
    """Проверить выбор модели давления излучения Земли."""
    name = str(earth_radiation_model).strip().lower()
    if name in {"none", "off", "false"}:
        return "none"
    if name in {"isotropic_ir", "ir", "earth_ir"}:
        return "isotropic_ir"
    raise ValueError(
        "Unsupported earth_radiation_model "
        f"{earth_radiation_model!r}. Supported values: 'none', 'isotropic_ir'."
    )


def _normalize_relativity_model(relativity_model: str) -> Literal["none", "schwarzschild"]:
    """Проверить выбор модели релятивистской поправки."""
    name = str(relativity_model).strip().lower()
    if name in {"none", "off", "false"}:
        return "none"
    if name in {"schwarzschild", "post_newtonian", "1pn"}:
        return "schwarzschild"
    raise ValueError(
        "Unsupported relativity_model "
        f"{relativity_model!r}. Supported values: 'none', 'schwarzschild'."
    )


def _normalize_tide_model(tide_model: str) -> Literal["none", "solid_earth_degree2"]:
    """Проверить выбор модели твёрдотельных/океанических приливов."""
    name = str(tide_model).strip().lower()
    if name in {"none", "off", "false"}:
        return "none"
    if name in {"solid_earth_degree2", "solid_earth", "degree2_solid"}:
        return "solid_earth_degree2"
    raise ValueError(
        "Unsupported tide_model "
        f"{tide_model!r}. Supported values: 'none', 'solid_earth_degree2'."
    )


def _select_density_model(force_models: object) -> Literal["exponential", "nrlmsise00"]:
    """Вернуть активную модель плотности с учётом исторического булевого флага."""
    if not getattr(force_models, "nrlmsise00_atmosphere", True):
        return "exponential"

    name = str(getattr(force_models, "density_model", "nrlmsise00")).strip().lower()
    if name in {"exponential", "exp"}:
        return "exponential"
    if name in {"nrlmsise00", "msise", "nrlmsise-00"}:
        return "nrlmsise00"
    raise ValueError(
        "Unsupported density_model "
        f"{name!r}. Supported values: 'exponential', 'nrlmsise00'."
    )


def _output_times(duration_seconds: float, step_seconds: float) -> np.ndarray:
    """Построить сетку сохраняемых состояний; DOP853 может использовать меньшие внутренние шаги."""
    if duration_seconds == 0.0:
        return np.array([0.0], dtype=float)
    times = np.arange(0.0, duration_seconds + 0.5 * step_seconds, step_seconds, dtype=float)
    times = times[times <= duration_seconds]
    if times.size == 0 or times[0] != 0.0:
        times = np.insert(times, 0, 0.0)
    if not np.isclose(times[-1], duration_seconds):
        times = np.append(times, duration_seconds)
    else:
        times[-1] = duration_seconds
    return times


def _normalize_output_times(output_times_seconds: np.ndarray, duration_seconds: float) -> np.ndarray:
    """Проверить и отсортировать пользовательскую сетку сохранения в секундах от эпохи."""
    times = np.asarray(output_times_seconds, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("output_times_seconds must contain at least one time.")
    if np.any(~np.isfinite(times)):
        raise ValueError("output_times_seconds must be finite.")
    if np.any(times < -1e-9) or np.any(times > duration_seconds + 1e-9):
        raise ValueError("output_times_seconds must lie within [0, duration_seconds].")
    times = np.clip(times, 0.0, duration_seconds)
    times = np.unique(np.round(times, decimals=9))
    if times[0] != 0.0:
        times = np.insert(times, 0, 0.0)
    if not np.isclose(times[-1], duration_seconds):
        times = np.append(times, duration_seconds)
    else:
        times[-1] = duration_seconds
    return times


@lru_cache(maxsize=8192)
def _gcrs_to_itrs_rotation(epoch_seconds: float, t_seconds: float) -> np.ndarray:
    """Матрица поворота из координат GCRS в ITRS для заданной UTC-эпохи."""
    obstime = Time(epoch_seconds + t_seconds, format="unix", scale="utc")
    return np.asarray(cirs_to_itrs_mat(obstime) @ gcrs_to_cirs_mat(obstime), dtype=float)


try:
    # Опциональная зависимость: при наличии используем физичную модель атмосферы NRLMSISE-00.
    from nrlmsise00 import msise_model as _nrlmsise_model
except Exception:  # pragma: no cover - optional dependency
    _nrlmsise_model = None

# Для ноутбуков (в т.ч. Colab) часто встречаются эпохи вне текущего IERS диапазона
# или ограниченный интернет-доступ. Разрешаем использовать встроенные таблицы без
# фатальных исключений и отключаем ограничение "свежести" авто-таблиц.
iers.conf.auto_max_age = None
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "warn"


def _datetime_from_epoch_seconds(epoch_seconds: float, t_seconds: float) -> datetime:
    """Переход от UNIX-эпохи к абсолютному UTC времени текущего шага."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=epoch_seconds)
    return epoch + timedelta(seconds=t_seconds)


def _central_earth_gravity_acceleration(r: np.ndarray) -> np.ndarray:
    """Ускорение центрального поля сферической Земли в единицах SI."""
    rn = np.linalg.norm(r)
    return -MU_EARTH * r / (rn**3)


def _j2_acceleration_fixed_axis(r: np.ndarray) -> np.ndarray:
    """Возмущающее ускорение J2 для системы, где ось z совпадает с осью симметрии Земли."""
    rn = np.linalg.norm(r)
    x, y, z = r
    r2 = rn * rn
    z2 = z * z
    factor = 1.5 * J2_EARTH * MU_EARTH * (R_EARTH**2) / (rn**5)
    common = 5.0 * z2 / r2
    ax = factor * x * (common - 1.0)
    ay = factor * y * (common - 1.0)
    az = factor * z * (common - 3.0)
    return np.array([ax, ay, az])


def _earth_gravity_acceleration(
    r: np.ndarray,
    with_j2: bool,
    j2_frame: str = "gcrs_fixed_axis",
    epoch_seconds: float | None = None,
    t_seconds: float = 0.0,
    j2_scale: float = 1.0,
) -> np.ndarray:
    """Ускорение от поля Земли.

    Базовая часть: центральное поле -μ r / |r|^3.
    Поправка J2: учитывает сплюснутость Земли и даёт прецессию орбитальных элементов.

    `gcrs_fixed_axis` воспроизводит старый режим: J2 считается прямо в
    координатах состояния. `itrs_body_fixed` поворачивает положение в ITRS,
    считает J2 в земной системе и поворачивает ускорение обратно в GCRS.
    """
    a = _central_earth_gravity_acceleration(r)
    if not with_j2:
        return a

    frame = _normalize_j2_frame(j2_frame)
    if frame == "gcrs_fixed_axis":
        return a + j2_scale * _j2_acceleration_fixed_axis(r)

    if epoch_seconds is None:
        raise ValueError("epoch_seconds is required for j2_frame='itrs_body_fixed'.")
    rotation = _gcrs_to_itrs_rotation(float(epoch_seconds), float(t_seconds))
    r_itrs = rotation @ r
    a_j2_itrs = _j2_acceleration_fixed_axis(r_itrs)
    return a + j2_scale * (rotation.T @ a_j2_itrs)


def _egm2008_acceleration(
    r_gcrs: np.ndarray,
    config: PropagationConfig,
    t_seconds: float,
    harmonic_scale: float = 1.0,
) -> np.ndarray:
    """Центральная гравитация плюс статические гармоники типа EGM2008 из локального файла ICGEM."""
    if config.environment.gravity_coefficients_file is None:
        raise ValueError(
            "earth_gravity_model='egm2008' requires "
            "EnvironmentConfig.gravity_coefficients_file."
        )

    fm = config.environment.force_models
    coefficients = read_icgem_gfc(
        str(config.environment.gravity_coefficients_file),
        max_degree=fm.gravity_max_degree,
        max_order=fm.gravity_max_order,
    )
    rotation = _gcrs_to_itrs_rotation(float(config.epoch_seconds), float(t_seconds))
    r_itrs = rotation @ r_gcrs
    a_harmonic_itrs = harmonic_perturbing_acceleration(
        r_itrs,
        coefficients,
        max_degree=fm.gravity_max_degree,
        max_order=fm.gravity_max_order,
    )
    return _central_earth_gravity_acceleration(r_gcrs) + harmonic_scale * (
        rotation.T @ a_harmonic_itrs
    )


def _body_position_gcrs(body: str, epoch_seconds: float, t_seconds: float) -> np.ndarray:
    """Положение Солнца/Луны относительно центра Земли в GCRS [м]."""
    dt = _datetime_from_epoch_seconds(epoch_seconds, t_seconds)
    obs_time = Time(dt)
    body_bary = get_body_barycentric(body, obs_time)
    earth_bary = get_body_barycentric("earth", obs_time)
    pos = (body_bary.xyz - earth_bary.xyz).to(u.m).value
    return pos


def _third_body_acceleration(r_sc: np.ndarray, r_body: np.ndarray, mu_body: float) -> np.ndarray:
    """Дифференциальное ускорение от третьего тела.

    Берём разность: притяжение тела на КА минус притяжение тела на центр Земли.
    Так мы корректно описываем возмущение в геоцентрической системе координат.
    """
    d_sc_body = r_body - r_sc
    return mu_body * (
        d_sc_body / np.linalg.norm(d_sc_body) ** 3
        - r_body / np.linalg.norm(r_body) ** 3
    )


def _degree2_solid_tide_body_acceleration(
    r_sc: np.ndarray,
    r_body: np.ndarray,
    mu_body: float,
) -> np.ndarray:
    """Ускорение прилива твёрдой Земли степени 2 от одного внешнего тела [м/с^2].

    Выражение является градиентом
    k2 * mu_body * R_E^5 / |r_body|^3 / |r_sc|^3 * P2(cos(psi)).
    Координаты должны быть заданы в одной геоцентрической системе; propagator
    вызывает функцию в ITRS и поворачивает результат обратно в GCRS.
    """
    r_norm = np.linalg.norm(r_sc)
    body_norm = np.linalg.norm(r_body)
    if r_norm <= R_EARTH or body_norm < 1.0:
        return np.zeros(3)

    body_hat = r_body / body_norm
    projection = float(np.dot(r_sc, body_hat))
    coefficient = SOLID_EARTH_LOVE_K2 * mu_body * R_EARTH**5 / body_norm**3
    return 0.5 * coefficient * (
        6.0 * projection * body_hat / r_norm**5
        - 15.0 * projection**2 * r_sc / r_norm**7
        + 3.0 * r_sc / r_norm**5
    )


def _circle_overlap_area(radius_a: float, radius_b: float, distance: float) -> float:
    """Площадь пересечения двух окружностей в локальной угловой плоскости."""
    if distance >= radius_a + radius_b:
        return 0.0
    if distance <= abs(radius_a - radius_b):
        return np.pi * min(radius_a, radius_b) ** 2

    arg_a = (distance**2 + radius_a**2 - radius_b**2) / (2.0 * distance * radius_a)
    arg_b = (distance**2 + radius_b**2 - radius_a**2) / (2.0 * distance * radius_b)
    arg_a = np.clip(arg_a, -1.0, 1.0)
    arg_b = np.clip(arg_b, -1.0, 1.0)
    area_a = radius_a**2 * np.arccos(arg_a)
    area_b = radius_b**2 * np.arccos(arg_b)
    area_c = 0.5 * np.sqrt(
        max(
            0.0,
            (-distance + radius_a + radius_b)
            * (distance + radius_a - radius_b)
            * (distance - radius_a + radius_b)
            * (distance + radius_a + radius_b),
        )
    )
    return float(area_a + area_b - area_c)


def _srp_illumination_factor(r_sc: np.ndarray, r_sun: np.ndarray, shadow_model: str) -> float:
    """Доля солнечного диска, видимая с космического аппарата.

    `none` возвращает 1.0. `cylindrical` использует бесконечный цилиндр за Землёй.
    `conical` моделирует Солнце и Землю как угловые диски и возвращает гладкий
    коэффициент полутени в диапазоне [0, 1].
    """
    model = _normalize_srp_shadow_model(shadow_model)
    if model == "none":
        return 1.0

    sun_distance = np.linalg.norm(r_sun)
    sc_distance = np.linalg.norm(r_sc)
    if sun_distance < 1.0 or sc_distance <= R_EARTH:
        return 0.0

    sun_hat = r_sun / sun_distance
    projection = np.dot(r_sc, sun_hat)
    behind_earth = projection < 0.0
    perpendicular_distance = np.linalg.norm(r_sc - projection * sun_hat)
    if model == "cylindrical":
        return 0.0 if behind_earth and perpendicular_distance < R_EARTH else 1.0

    if not behind_earth:
        return 1.0

    sun_from_sc = r_sun - r_sc
    sun_from_sc_distance = np.linalg.norm(sun_from_sc)
    if sun_from_sc_distance < 1.0:
        return 1.0

    earth_from_sc = -r_sc
    cos_separation = np.dot(sun_from_sc, earth_from_sc) / (
        sun_from_sc_distance * sc_distance
    )
    separation = float(np.arccos(np.clip(cos_separation, -1.0, 1.0)))
    earth_angular_radius = float(np.arcsin(np.clip(R_EARTH / sc_distance, 0.0, 1.0)))
    sun_angular_radius = float(
        np.arcsin(np.clip(R_SUN / sun_from_sc_distance, 0.0, 1.0))
    )

    overlap = _circle_overlap_area(earth_angular_radius, sun_angular_radius, separation)
    sun_disk_area = np.pi * sun_angular_radius**2
    if sun_disk_area <= 0.0:
        return 1.0
    return float(np.clip(1.0 - overlap / sun_disk_area, 0.0, 1.0))


def _space_weather_sample_for_config(config: PropagationConfig, dt: datetime):
    """Вернуть настроенные индексы космической погоды или явные спокойные константы."""
    if config.environment.space_weather_file is None:
        return quiet_space_weather_sample()
    try:
        records = load_celestrak_space_weather_csv(str(config.environment.space_weather_file))
        return sample_space_weather(dt, records)
    except Exception:
        return quiet_space_weather_sample()


def _nrlmsise_density_kg_m3(
    config: PropagationConfig,
    t_seconds: float,
    r_gcrs: np.ndarray,
) -> float:
    """Плотность атмосферы из NRLMSISE-00 в кг/м^3.

    Модель принимает геодезические широту/долготу/высоту, поэтому выполняем
    преобразование координат из инерциальной GCRS в земную ITRS.
    """
    if _nrlmsise_model is None:
        return np.nan

    dt = _datetime_from_epoch_seconds(config.epoch_seconds, t_seconds)
    obs_time = Time(dt)

    gcrs = GCRS(
        obstime=obs_time,
        representation_type=CartesianRepresentation,
        x=r_gcrs[0] * u.m,
        y=r_gcrs[1] * u.m,
        z=r_gcrs[2] * u.m,
    )
    itrs = gcrs.transform_to(ITRS(obstime=obs_time))
    loc = itrs.earth_location
    lat = loc.lat.to_value(u.deg)
    lon = loc.lon.to_value(u.deg)
    alt_km = max(0.0, loc.height.to_value(u.km))

    doy = int(obs_time.yday.split(":")[1])
    sec = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond * 1e-6

    space_weather = _space_weather_sample_for_config(config, dt)
    try:
        # Совместимо с python-пакетом nrlmsise00:
        # msise_model(time, alt, lat, lon, f107a, f107, ap, ...)
        out = _nrlmsise_model(
            dt,
            alt_km,
            lat,
            lon,
            space_weather.f107a,
            space_weather.f107,
            space_weather.ap,
            method="gtd7d",
        )
        densities = out[0] if isinstance(out, tuple) else out
        rho_g_cm3 = densities[5]
        return rho_g_cm3 * 1000.0
    except Exception:
        # Если конкретная сборка/версия nrlmsise00 ожидает иной формат аргументов
        # или не смогла вычислить плотность, переключаемся на резервную модель.
        return np.nan


def _exponential_density_kg_m3(r: np.ndarray) -> float:
    """Простая резервная модель плотности (экспонента по высоте)."""
    alt = max(0.0, np.linalg.norm(r) - R_EARTH)
    rho0 = 1.225
    h0 = 0.0
    scale_height = 8500.0
    return rho0 * np.exp(-(alt - h0) / scale_height)


def _drag_acceleration(
    state: np.ndarray,
    config: PropagationConfig,
    t_seconds: float,
) -> np.ndarray:
    """Аэродинамическое сопротивление.

    Формула: a_drag = -0.5 * Cd * A/m * rho * |v_rel| * v_rel.
    v_rel считается относительно вращающейся атмосферы Земли.
    """
    r = state[:3]
    v = state[3:]
    rho = np.nan
    if _select_density_model(config.environment.force_models) == "nrlmsise00":
        rho = _nrlmsise_density_kg_m3(config, t_seconds, r)
    if not np.isfinite(rho):
        rho = _exponential_density_kg_m3(r)

    v_rel = v - np.cross(OMEGA_EARTH, r)
    speed = np.linalg.norm(v_rel)
    if speed < 1e-9:
        return np.zeros(3)

    bc = 0.5 * config.spacecraft.cd * config.spacecraft.reference_area / config.spacecraft.mass
    return -bc * rho * speed * v_rel


def _solar_radiation_pressure_acceleration(
    state: np.ndarray,
    config: PropagationConfig,
    t_seconds: float,
) -> np.ndarray:
    """Ускорение от давления солнечного излучения (SRP) с опциональной тенью Земли."""
    r = state[:3]
    r_sun = _body_position_gcrs("sun", config.epoch_seconds, t_seconds)
    illumination = _srp_illumination_factor(
        r,
        r_sun,
        config.environment.force_models.srp_shadow_model,
    )
    if illumination <= 0.0:
        return np.zeros(3)

    d_sun_sc = r - r_sun
    dist = np.linalg.norm(d_sun_sc)
    if dist < 1.0:
        return np.zeros(3)
    p = SOLAR_PRESSURE_AT_1AU * (AU_METERS / dist) ** 2
    coeff = p * config.spacecraft.cr * config.spacecraft.reference_area / config.spacecraft.mass
    return illumination * coeff * (d_sun_sc / dist)


def _earth_ir_pressure_acceleration(state: np.ndarray, config: PropagationConfig) -> np.ndarray:
    """Ускорение от изотропного теплового ИК-излучения Земли [м/с^2].

    Модель рассматривает Землю как сферический изотропный излучатель со средним
    исходящим длинноволновым потоком `EARTH_IR_FLUX_W_M2` на радиусе `R_EARTH`.
    """
    model = _normalize_earth_radiation_model(
        config.environment.force_models.earth_radiation_model
    )
    if model == "none":
        return np.zeros(3)

    r = state[:3]
    rn = np.linalg.norm(r)
    if rn <= R_EARTH:
        return np.zeros(3)

    flux = EARTH_IR_FLUX_W_M2 * (R_EARTH / rn) ** 2
    pressure = flux / SPEED_OF_LIGHT
    coeff = (
        pressure
        * config.spacecraft.cr
        * config.spacecraft.reference_area
        / config.spacecraft.mass
    )
    return coeff * (r / rn)


def _relativistic_acceleration(state: np.ndarray, config: PropagationConfig) -> np.ndarray:
    """Постньютоновская поправка Шварцшильда для центральной гравитации Земли [м/с^2]."""
    model = _normalize_relativity_model(config.environment.force_models.relativity_model)
    if model == "none":
        return np.zeros(3)

    r = state[:3]
    v = state[3:]
    rn = np.linalg.norm(r)
    if rn < 1.0:
        return np.zeros(3)

    v2 = float(np.dot(v, v))
    rv = float(np.dot(r, v))
    factor = MU_EARTH / (SPEED_OF_LIGHT**2 * rn**3)
    return factor * (
        (4.0 * MU_EARTH / rn - v2) * r
        + 4.0 * rv * v
    )


def _solid_earth_tide_acceleration(
    state: np.ndarray,
    config: PropagationConfig,
    t_seconds: float,
) -> np.ndarray:
    """Поправка прилива твёрдой Земли степени 2 от Луны и Солнца [м/с^2]."""
    model = _normalize_tide_model(config.environment.force_models.tide_model)
    if model == "none":
        return np.zeros(3)

    rotation = _gcrs_to_itrs_rotation(float(config.epoch_seconds), float(t_seconds))
    r_itrs = rotation @ state[:3]
    r_moon_itrs = rotation @ _body_position_gcrs("moon", config.epoch_seconds, t_seconds)
    r_sun_itrs = rotation @ _body_position_gcrs("sun", config.epoch_seconds, t_seconds)

    a_itrs = (
        _degree2_solid_tide_body_acceleration(r_itrs, r_moon_itrs, MU_MOON)
        + _degree2_solid_tide_body_acceleration(r_itrs, r_sun_itrs, MU_SUN)
    )
    return rotation.T @ a_itrs


def _total_acceleration(
    state: np.ndarray,
    config: PropagationConfig,
    t_seconds: float,
) -> np.ndarray:
    """Сумма всех включённых в конфигурации ускорений-возмущений."""
    fm = config.environment.force_models
    scales = config.environment.force_scale_factors
    r = state[:3]

    a = np.zeros(3)
    if fm.spherical_earth_gravity:
        gravity_model = _normalize_earth_gravity_model(fm.earth_gravity_model)
        if gravity_model == "j2":
            a += _earth_gravity_acceleration(
                r,
                with_j2=fm.earth_j2,
                j2_frame=fm.j2_frame,
                epoch_seconds=config.epoch_seconds,
                t_seconds=t_seconds,
                j2_scale=float(scales.get("j2", 1.0)),
            )
        elif gravity_model == "egm2008":
            a += _egm2008_acceleration(
                r,
                config,
                t_seconds,
                harmonic_scale=float(scales.get("gravity_harmonics", 1.0)),
            )

    if fm.third_body_sun:
        r_sun = _body_position_gcrs("sun", config.epoch_seconds, t_seconds)
        a += float(scales.get("third_body_sun", 1.0)) * _third_body_acceleration(
            r,
            r_sun,
            MU_SUN,
        )

    if fm.third_body_moon:
        r_moon = _body_position_gcrs("moon", config.epoch_seconds, t_seconds)
        a += float(scales.get("third_body_moon", 1.0)) * _third_body_acceleration(
            r,
            r_moon,
            MU_MOON,
        )

    if fm.atmospheric_drag:
        a += _drag_acceleration(state, config, t_seconds)

    if fm.solar_radiation_pressure:
        a += _solar_radiation_pressure_acceleration(state, config, t_seconds)

    if fm.earth_radiation_model != "none":
        a += _earth_ir_pressure_acceleration(state, config)

    if fm.relativity_model != "none":
        a += float(scales.get("relativity", 1.0)) * _relativistic_acceleration(
            state,
            config,
        )

    if fm.tide_model != "none":
        a += float(scales.get("solid_earth_tide", 1.0)) * _solid_earth_tide_acceleration(
            state,
            config,
            t_seconds,
        )

    return a


def _state_derivative(state: np.ndarray, config: PropagationConfig, t_seconds: float) -> np.ndarray:
    """Правая часть ОДУ: [r_dot, v_dot] = [v, a_total]."""
    v = state[3:]
    a = _total_acceleration(state, config, t_seconds)
    return np.hstack((v, a))


def _fallback_two_body_propagation(config: PropagationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Численное интегрирование RK4 с постоянным шагом.

    Историческое имя функции сохранено для обратной совместимости,
    но фактически здесь уже не "two-body", а многосиловая модель.
    """
    state = config.initial_state.astype(float).copy()
    times = (
        config.output_times_seconds
        if config.output_times_seconds is not None
        else _output_times(config.duration_seconds, config.step_seconds)
    )
    states = np.zeros((len(times), 6), dtype=float)

    for idx, t in tqdm(enumerate(times), total=len(times)):
        states[idx] = state
        if idx == len(times) - 1:
            break
        h = times[idx + 1] - t
        k1 = _state_derivative(state, config, t)
        k2 = _state_derivative(state + 0.5 * h * k1, config, t + 0.5 * h)
        k3 = _state_derivative(state + 0.5 * h * k2, config, t + 0.5 * h)
        k4 = _state_derivative(state + h * k3, config, t + h)
        state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return times + config.epoch_seconds, states


def _dop853_propagation(config: PropagationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Адаптивное интегрирование DOP853 с сохранением состояний на заданной выходной сетке.

    `step_seconds` управляет только `t_eval`; SciPy выбирает внутренние шаги
    по `rtol` и 6-компонентному вектору `atol`.
    """
    times = (
        config.output_times_seconds
        if config.output_times_seconds is not None
        else _output_times(config.duration_seconds, config.step_seconds)
    )
    if config.duration_seconds == 0.0:
        return times + config.epoch_seconds, config.initial_state.reshape(1, 6).copy()

    def rhs(t_seconds: float, state: np.ndarray) -> np.ndarray:
        return _state_derivative(state, config, t_seconds)

    solution = solve_ivp(
        rhs,
        (0.0, config.duration_seconds),
        config.initial_state,
        method="DOP853",
        t_eval=times,
        rtol=config.rtol,
        atol=config.atol,
    )
    if not solution.success:
        raise RuntimeError(f"DOP853 propagation failed: {solution.message}")

    return solution.t + config.epoch_seconds, solution.y.T


def propagate_orbit(
    config: PropagationConfig,
    backend: PropagatorBackend = PropagatorBackend.TUDATPY,
) -> tuple[np.ndarray, np.ndarray]:
    """Пропагация орбиты.

    Сейчас backend-переключатель оставлен как интерфейсная заготовка:
    расчёт выполняется резервным интегратором из этого модуля.
    """

    _ = backend
    if config.integrator == "rk4_fixed":
        return _fallback_two_body_propagation(config)
    if config.integrator == "dop853":
        return _dop853_propagation(config)
    raise ValueError(f"Unsupported integrator {config.integrator!r}.")
