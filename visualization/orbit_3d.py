from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

EARTH_RADIUS_M = 6_371_000.0


def _axis_bounds(points_xyz: np.ndarray, padding_ratio: float = 0.1) -> tuple[float, float]:
    max_abs = float(np.max(np.abs(points_xyz)))
    max_abs = max(max_abs, EARTH_RADIUS_M)
    padded = max_abs * (1.0 + padding_ratio)
    return -padded, padded


def _earth_surface(resolution: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = EARTH_RADIUS_M * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_M * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_M * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def build_orbit_figure(
    trajectories: list[np.ndarray] | np.ndarray,
    names: list[str] | None = None,
    show_earth: bool = True,
    earth_opacity: float = 0.35,
) -> go.Figure:
    """Render one or multiple trajectories in 3D with Earth sphere."""

    if isinstance(trajectories, np.ndarray):
        trajectories = [trajectories]

    names = names or [f"Trajectory {idx+1}" for idx in range(len(trajectories))]
    fig = go.Figure()

    if show_earth:
        ex, ey, ez = _earth_surface()
        fig.add_trace(
            go.Surface(
                x=ex,
                y=ey,
                z=ez,
                colorscale=[[0, "royalblue"], [1, "royalblue"]],
                opacity=earth_opacity,
                showscale=False,
                name="Earth",
            )
        )

    for states, name in zip(trajectories, names, strict=False):
        fig.add_trace(
            go.Scatter3d(
                x=states[:, 0],
                y=states[:, 1],
                z=states[:, 2],
                mode="lines",
                name=name,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[states[-1, 0]],
                y=[states[-1, 1]],
                z=[states[-1, 2]],
                mode="markers",
                marker={"size": 4},
                name=f"{name} current",
                showlegend=False,
            )
        )

    all_points = np.vstack([traj[:, :3] for traj in trajectories])
    axis_min, axis_max = _axis_bounds(all_points)

    fig.update_layout(
        title="Orbit visualization",
        scene={
            "xaxis": {"title": "X [m]", "range": [axis_min, axis_max]},
            "yaxis": {"title": "Y [m]", "range": [axis_min, axis_max]},
            "zaxis": {"title": "Z [m]", "range": [axis_min, axis_max]},
            "aspectmode": "cube",
        },
    )
    return fig
