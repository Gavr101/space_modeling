from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .environment import EnvironmentConfig


class PropagatorBackend(str, Enum):
    TUDATPY = "tudatpy"
    OREKIT = "orekit"


@dataclass(slots=True)
class SpacecraftProperties:
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


def _two_body_derivative(state: np.ndarray, mu_earth: float) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    rn = np.linalg.norm(r)
    a = -mu_earth * r / (rn**3)
    return np.hstack((v, a))


def _fallback_two_body_propagation(config: PropagationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Numerical fallback propagation using fixed-step RK4.

    NOTE: this is a placeholder for architecture bootstrap only.
    Production propagation is expected from TudatPy integrators (RKF78 / DOP853).
    """

    mu_earth = 3.986004418e14
    state = config.initial_state.astype(float).copy()
    n_steps = int(config.duration_seconds // config.step_seconds) + 1
    times = np.linspace(0.0, config.duration_seconds, n_steps)
    states = np.zeros((n_steps, 6), dtype=float)

    h = config.step_seconds
    for idx, _ in enumerate(times):
        states[idx] = state
        k1 = _two_body_derivative(state, mu_earth)
        k2 = _two_body_derivative(state + 0.5 * h * k1, mu_earth)
        k3 = _two_body_derivative(state + 0.5 * h * k2, mu_earth)
        k4 = _two_body_derivative(state + h * k3, mu_earth)
        state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return times + config.epoch_seconds, states


def propagate_orbit(
    config: PropagationConfig,
    backend: PropagatorBackend = PropagatorBackend.TUDATPY,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate orbit with configured backend.

    Current implementation provides architecture-compatible fallback mode while
    TudatPy/Orekit integration is being incrementally added.
    """

    _ = backend
    return _fallback_two_body_propagation(config)
