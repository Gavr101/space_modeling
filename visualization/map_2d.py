from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

EARTH_ROTATION_RATE_RAD_S = 7.2921159e-5


def ecef_to_latlon_deg(positions_ecef_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = positions_ecef_m[:, 0], positions_ecef_m[:, 1], positions_ecef_m[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    hyp = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, hyp))
    return lat, lon


def eci_to_ecef(positions_eci_m: np.ndarray, elapsed_seconds: np.ndarray, theta0_rad: float = 0.0) -> np.ndarray:
    thetas = theta0_rad + EARTH_ROTATION_RATE_RAD_S * elapsed_seconds
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    x = cos_t * positions_eci_m[:, 0] + sin_t * positions_eci_m[:, 1]
    y = -sin_t * positions_eci_m[:, 0] + cos_t * positions_eci_m[:, 1]
    z = positions_eci_m[:, 2]
    return np.column_stack((x, y, z))


def build_groundtrack_figure(
    trajectories_eci: list[np.ndarray] | np.ndarray,
    names: list[str] | None = None,
    elapsed_seconds: list[np.ndarray] | np.ndarray | None = None,
    theta0_rad: float = 0.0,
) -> go.Figure:
    if isinstance(trajectories_eci, np.ndarray):
        trajectories_eci = [trajectories_eci]

    names = names or [f"Track {idx+1}" for idx in range(len(trajectories_eci))]
    fig = go.Figure()

    if elapsed_seconds is None:
        elapsed_seconds = [np.arange(track.shape[0], dtype=float) for track in trajectories_eci]
    elif isinstance(elapsed_seconds, np.ndarray):
        elapsed_seconds = [elapsed_seconds]

    for states, t_seconds, name in zip(trajectories_eci, elapsed_seconds, names, strict=False):
        positions_ecef = eci_to_ecef(states[:, :3], t_seconds, theta0_rad=theta0_rad)
        lat, lon = ecef_to_latlon_deg(positions_ecef)
        fig.add_trace(go.Scattergeo(lon=lon, lat=lat, mode="lines", name=name))

    fig.update_layout(
        title="Satellite ground tracks",
        geo={
            "projection_type": "equirectangular",
            "showland": True,
            "landcolor": "rgb(220, 220, 220)",
            "showocean": True,
            "oceancolor": "rgb(190, 220, 255)",
            "showlakes": True,
            "showcoastlines": True,
        },
    )
    return fig
