from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

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


def _fallback_two_body_propagation(config: PropagationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Numerical fallback propagation using SciPy-like fixed-step Euler.

    NOTE: this is a placeholder for architecture bootstrap only.
    Production propagation is expected from TudatPy integrators (RKF78 / DOP853).
    """

    mu_earth = 3.986004418e14
    state = config.initial_state.astype(float).copy()
    n_steps = int(config.duration_seconds // config.step_seconds) + 1
    times = np.linspace(0.0, config.duration_seconds, n_steps)
    states = np.zeros((n_steps, 6), dtype=float)

    for idx, _ in enumerate(times):
        states[idx] = state
        r = state[:3]
        v = state[3:]
        rn = np.linalg.norm(r)
        a = -mu_earth * r / (rn**3)
        state[:3] = r + v * config.step_seconds
        state[3:] = v + a * config.step_seconds

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
