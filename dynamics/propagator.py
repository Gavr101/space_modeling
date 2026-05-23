from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, get_body_barycentric
from astropy.time import Time

from .environment import EnvironmentConfig


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
    integrator: str = "DOP853"
    spacecraft: SpacecraftProperties = field(default_factory=lambda: SpacecraftProperties(5.0, 2.2, 1.3, 0.05))
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


# --- Физические константы (SI) ---
MU_EARTH = 3.986004418e14
R_EARTH = 6378136.3
J2_EARTH = 1.08262668e-3
OMEGA_EARTH = np.array([0.0, 0.0, 7.2921150e-5])
MU_SUN = 1.32712440018e20
MU_MOON = 4.9048695e12
AU_METERS = 149597870700.0
SOLAR_PRESSURE_AT_1AU = 4.56e-6


try:
    # Опциональная зависимость: при наличии используем физичную модель атмосферы NRLMSISE-00.
    from nrlmsise00 import msise_model as _nrlmsise_model
except Exception:  # pragma: no cover - optional dependency
    _nrlmsise_model = None


def _datetime_from_epoch_seconds(epoch_seconds: float, t_seconds: float) -> datetime:
    """Переход от UNIX-эпохи к абсолютному UTC времени текущего шага."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=epoch_seconds)
    return epoch + timedelta(seconds=t_seconds)


def _earth_gravity_acceleration(r: np.ndarray, with_j2: bool) -> np.ndarray:
    """Ускорение от поля Земли.

    Базовая часть: центральное поле -μ r / |r|^3.
    Поправка J2: учитывает сплюснутость Земли и даёт прецессию орбитальных элементов.
    """
    rn = np.linalg.norm(r)
    a = -MU_EARTH * r / (rn**3)
    if not with_j2:
        return a

    x, y, z = r
    r2 = rn * rn
    z2 = z * z
    factor = 1.5 * J2_EARTH * MU_EARTH * (R_EARTH**2) / (rn**5)
    common = 5.0 * z2 / r2
    ax = factor * x * (common - 1.0)
    ay = factor * y * (common - 1.0)
    az = factor * z * (common - 3.0)
    return a + np.array([ax, ay, az])


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
    return mu_body * (d_sc_body / np.linalg.norm(d_sc_body) ** 3 - r_body / np.linalg.norm(r_body) ** 3)


def _nrlmsise_density_kg_m3(epoch_seconds: float, t_seconds: float, r_gcrs: np.ndarray) -> float:
    """Плотность атмосферы из NRLMSISE-00 в кг/м^3.

    Модель принимает геодезические широту/долготу/высоту, поэтому выполняем
    преобразование координат из инерциальной GCRS в земную ITRS.
    """
    if _nrlmsise_model is None:
        return np.nan

    dt = _datetime_from_epoch_seconds(epoch_seconds, t_seconds)
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

    # При отсутствии внешних индексов солнечной/геомагнитной активности
    # используем типичные "спокойные" значения.
    f107a = 150.0
    f107 = 150.0
    ap = 4.0
    out = _nrlmsise_model(dt.year, doy, sec, alt_km, lat, lon, 16.0, f107a, f107, ap)
    rho_g_cm3 = out[5]
    return rho_g_cm3 * 1000.0


def _exponential_density_kg_m3(r: np.ndarray) -> float:
    """Простая резервная модель плотности (экспонента по высоте)."""
    alt = max(0.0, np.linalg.norm(r) - R_EARTH)
    rho0 = 1.225
    h0 = 0.0
    scale_height = 8500.0
    return rho0 * np.exp(-(alt - h0) / scale_height)


def _drag_acceleration(state: np.ndarray, config: PropagationConfig, t_seconds: float) -> np.ndarray:
    """Аэродинамическое сопротивление.

    Формула: a_drag = -0.5 * Cd * A/m * rho * |v_rel| * v_rel.
    v_rel считается относительно вращающейся атмосферы Земли.
    """
    r = state[:3]
    v = state[3:]
    rho = np.nan
    if config.environment.force_models.nrlmsise00_atmosphere:
        rho = _nrlmsise_density_kg_m3(config.epoch_seconds, t_seconds, r)
    if not np.isfinite(rho):
        rho = _exponential_density_kg_m3(r)

    v_rel = v - np.cross(OMEGA_EARTH, r)
    speed = np.linalg.norm(v_rel)
    if speed < 1e-9:
        return np.zeros(3)

    bc = 0.5 * config.spacecraft.cd * config.spacecraft.reference_area / config.spacecraft.mass
    return -bc * rho * speed * v_rel


def _solar_radiation_pressure_acceleration(state: np.ndarray, config: PropagationConfig, t_seconds: float) -> np.ndarray:
    """Ускорение от давления солнечного излучения (SRP), без модели затенения Землёй."""
    r = state[:3]
    r_sun = _body_position_gcrs("sun", config.epoch_seconds, t_seconds)
    d_sun_sc = r - r_sun
    dist = np.linalg.norm(d_sun_sc)
    if dist < 1.0:
        return np.zeros(3)
    p = SOLAR_PRESSURE_AT_1AU * (AU_METERS / dist) ** 2
    coeff = p * config.spacecraft.cr * config.spacecraft.reference_area / config.spacecraft.mass
    return coeff * (d_sun_sc / dist)


def _total_acceleration(state: np.ndarray, config: PropagationConfig, t_seconds: float) -> np.ndarray:
    """Сумма всех включённых в конфигурации ускорений-возмущений."""
    fm = config.environment.force_models
    r = state[:3]

    a = np.zeros(3)
    if fm.spherical_earth_gravity:
        a += _earth_gravity_acceleration(r, with_j2=fm.earth_j2)

    if fm.third_body_sun:
        r_sun = _body_position_gcrs("sun", config.epoch_seconds, t_seconds)
        a += _third_body_acceleration(r, r_sun, MU_SUN)

    if fm.third_body_moon:
        r_moon = _body_position_gcrs("moon", config.epoch_seconds, t_seconds)
        a += _third_body_acceleration(r, r_moon, MU_MOON)

    if fm.atmospheric_drag:
        a += _drag_acceleration(state, config, t_seconds)

    if fm.solar_radiation_pressure:
        a += _solar_radiation_pressure_acceleration(state, config, t_seconds)

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
    n_steps = int(config.duration_seconds // config.step_seconds) + 1
    times = np.linspace(0.0, config.duration_seconds, n_steps)
    states = np.zeros((n_steps, 6), dtype=float)

    h = config.step_seconds
    for idx, t in enumerate(times):
        states[idx] = state
        k1 = _state_derivative(state, config, t)
        k2 = _state_derivative(state + 0.5 * h * k1, config, t + 0.5 * h)
        k3 = _state_derivative(state + 0.5 * h * k2, config, t + 0.5 * h)
        k4 = _state_derivative(state + h * k3, config, t + h)
        state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return times + config.epoch_seconds, states


def propagate_orbit(
    config: PropagationConfig,
    backend: PropagatorBackend = PropagatorBackend.TUDATPY,
) -> tuple[np.ndarray, np.ndarray]:
    """Пропагация орбиты.

    Сейчас backend-переключатель оставлен как интерфейсная заготовка:
    расчёт выполняется fallback-интегратором из этого модуля.
    """

    _ = backend
    return _fallback_two_body_propagation(config)
