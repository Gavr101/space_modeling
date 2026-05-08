import numpy as np


def propagate_covariance(stm: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Covariance propagation: P_k = Phi * P_0 * Phi^T."""

    return stm @ covariance @ stm.T
