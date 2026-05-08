from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def ecef_to_latlon_deg(positions_ecef_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = positions_ecef_m[:, 0], positions_ecef_m[:, 1], positions_ecef_m[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    hyp = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, hyp))
    return lat, lon


def build_groundtrack_figure(
    trajectories_ecef: list[np.ndarray] | np.ndarray,
    names: list[str] | None = None,
) -> go.Figure:
    if isinstance(trajectories_ecef, np.ndarray):
        trajectories_ecef = [trajectories_ecef]

    names = names or [f"Track {idx+1}" for idx in range(len(trajectories_ecef))]
    fig = go.Figure()

    for states, name in zip(trajectories_ecef, names, strict=False):
        lat, lon = ecef_to_latlon_deg(states[:, :3])
        fig.add_trace(
            go.Scattergeo(
                lon=lon,
                lat=lat,
                mode="lines",
                name=name,
            )
        )

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
