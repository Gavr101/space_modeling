from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter


def build_ekf(dim_x: int = 6, dim_z: int = 3) -> ExtendedKalmanFilter:
    ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
    return ekf


def build_ukf(dim_x: int = 6, dim_z: int = 3) -> UnscentedKalmanFilter:
    raise NotImplementedError("UKF setup will be added in stage 5.")
