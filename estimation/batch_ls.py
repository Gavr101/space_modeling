import numpy as np


def run_batch_least_squares(initial_guess: np.ndarray, measurements: np.ndarray) -> np.ndarray:
    """Placeholder batch least squares estimator.

    Uses simple averaging-based correction for bootstrap stage.
    """

    correction = np.mean(measurements[:, :6] - initial_guess, axis=0)
    return initial_guess + correction
