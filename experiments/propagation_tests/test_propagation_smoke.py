import numpy as np

from dynamics.propagator import PropagationConfig, propagate_orbit


def test_propagation_smoke_runs() -> None:
    initial_state = np.array([6778e3, 0.0, 0.0, 0.0, 7.67e3, 0.0])
    cfg = PropagationConfig(
        initial_state=initial_state,
        epoch_seconds=0.0,
        duration_seconds=60.0,
        step_seconds=10.0,
    )
    times, states = propagate_orbit(cfg)
    assert len(times) == states.shape[0]
    assert states.shape[1] == 6
