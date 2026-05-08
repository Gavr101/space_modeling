from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

EARTH_RADIUS_M = 6_378_137.0
SCALE_M_TO_THOUSAND_KM = 1.0e-6


def _as_trajectories(states: list[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    if isinstance(states, np.ndarray):
        return [states]
    return states


def _axis_limits_from_trajectories(trajectories: list[np.ndarray]) -> tuple[float, float]:
    xyz = np.vstack([track[:, :3] for track in trajectories]) * SCALE_M_TO_THOUSAND_KM
    max_abs = float(np.max(np.abs(xyz)))
    max_abs = max(max_abs, EARTH_RADIUS_M * 1.05 * SCALE_M_TO_THOUSAND_KM)
    padding = 0.08 * max_abs
    extent = max_abs + padding
    return -extent, extent


def _earth_surface_sphere(radius_m: float = EARTH_RADIUS_M, opacity: float = 0.35) -> go.Surface:
    radius = radius_m * SCALE_M_TO_THOUSAND_KM
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale=[[0.0, "rgb(68,120,180)"], [1.0, "rgb(60,145,75)"]],
        showscale=False,
        name="Earth",
        opacity=opacity,
        hoverinfo="skip",
    )


def _north_pole_arrow(radius_m: float = EARTH_RADIUS_M) -> go.Cone:
    radius = radius_m * SCALE_M_TO_THOUSAND_KM
    return go.Cone(
        x=[0.0],
        y=[0.0],
        z=[radius * 1.02],
        u=[0.0],
        v=[0.0],
        w=[radius * 0.35],
        sizemode="absolute",
        sizeref=radius * 0.08,
        showscale=False,
        name="North pole",
        colorscale=[[0.0, "crimson"], [1.0, "crimson"]],
    )


def build_orbit_figure(states: list[np.ndarray] | np.ndarray, names: list[str] | None = None, show_earth: bool = True) -> go.Figure:
    trajectories = _as_trajectories(states)
    names = names or [f"Orbit {idx + 1}" for idx in range(len(trajectories))]

    fig = go.Figure()
    if show_earth:
        fig.add_trace(_earth_surface_sphere())
        fig.add_trace(_north_pole_arrow())

    for track, name in zip(trajectories, names, strict=False):
        scaled = track[:, :3] * SCALE_M_TO_THOUSAND_KM
        fig.add_trace(go.Scatter3d(x=scaled[:, 0], y=scaled[:, 1], z=scaled[:, 2], mode="lines", name=name, line={"width": 4}))
        fig.add_trace(
            go.Scatter3d(
                x=[scaled[-1, 0]],
                y=[scaled[-1, 1]],
                z=[scaled[-1, 2]],
                mode="markers",
                name=f"{name} (current)",
                marker={"size": 4},
                showlegend=False,
            )
        )

    min_axis, max_axis = _axis_limits_from_trajectories(trajectories)
    fig.update_layout(
        title="Orbit visualization",
        width=1100,
        height=950,
        scene={
            "xaxis": {"range": [min_axis, max_axis], "title": "X, thousand km"},
            "yaxis": {"range": [min_axis, max_axis], "title": "Y, thousand km"},
            "zaxis": {"range": [min_axis, max_axis], "title": "Z, thousand km"},
            "aspectmode": "cube",
        },
    )
    return fig
