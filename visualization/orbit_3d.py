from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

EARTH_RADIUS_M = 6_378_137.0


def _as_trajectories(states: list[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    if isinstance(states, np.ndarray):
        return [states]
    return states


def _axis_limits_from_trajectories(trajectories: list[np.ndarray]) -> tuple[float, float]:
    xyz = np.vstack([track[:, :3] for track in trajectories])
    max_abs = float(np.max(np.abs(xyz)))
    max_abs = max(max_abs, EARTH_RADIUS_M * 1.05)
    padding = 0.08 * max_abs
    extent = max_abs + padding
    return -extent, extent


def _earth_surface_sphere(radius_m: float = EARTH_RADIUS_M, opacity: float = 0.35) -> go.Surface:
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x = radius_m * np.outer(np.cos(u), np.sin(v))
    y = radius_m * np.outer(np.sin(u), np.sin(v))
    z = radius_m * np.outer(np.ones_like(u), np.cos(v))
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


def build_orbit_figure(
    states: list[np.ndarray] | np.ndarray,
    names: list[str] | None = None,
    show_earth: bool = True,
) -> go.Figure:
    trajectories = _as_trajectories(states)
    names = names or [f"Orbit {idx + 1}" for idx in range(len(trajectories))]

    fig = go.Figure()
    if show_earth:
        fig.add_trace(_earth_surface_sphere())

    for track, name in zip(trajectories, names, strict=False):
        fig.add_trace(
            go.Scatter3d(
                x=track[:, 0],
                y=track[:, 1],
                z=track[:, 2],
                mode="lines",
                name=name,
                line={"width": 4},
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[track[-1, 0]],
                y=[track[-1, 1]],
                z=[track[-1, 2]],
                mode="markers",
                name=f"{name} (current)",
                marker={"size": 4},
                showlegend=False,
            )
        )

    min_axis, max_axis = _axis_limits_from_trajectories(trajectories)
    fig.update_layout(
        title="Orbit visualization",
        scene={
            "xaxis": {"range": [min_axis, max_axis], "title": "X, m"},
            "yaxis": {"range": [min_axis, max_axis], "title": "Y, m"},
            "zaxis": {"range": [min_axis, max_axis], "title": "Z, m"},
            "aspectmode": "cube",
        },
    )
    return fig
