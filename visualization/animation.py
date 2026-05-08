import numpy as np


def downsample_for_animation(states: np.ndarray, step: int = 10) -> np.ndarray:
    return states[::step]
