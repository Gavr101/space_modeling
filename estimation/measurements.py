from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MeasurementNoiseConfig:
    sigma_position_m: float = 30.0
    seed: int | None = 42


def generate_cartesian_measurements(
    truth_states: np.ndarray,
    noise: MeasurementNoiseConfig | None = None,
) -> np.ndarray:
    noise = noise or MeasurementNoiseConfig()
    rng = np.random.default_rng(noise.seed)
    measured = truth_states.copy()
    measured[:, :3] += rng.normal(0.0, noise.sigma_position_m, size=truth_states[:, :3].shape)
    return measured
