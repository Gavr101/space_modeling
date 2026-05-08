import numpy as np
import plotly.graph_objects as go


def build_orbit_figure(states: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            mode="lines",
            name="Orbit",
        )
    )
    fig.update_layout(scene_aspectmode="data", title="Orbit visualization")
    return fig
