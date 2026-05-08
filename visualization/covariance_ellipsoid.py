import numpy as np


def covariance_axes_lengths(covariance_xyz: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(covariance_xyz)
    return sigma * np.sqrt(np.clip(eigvals, 0.0, None))
