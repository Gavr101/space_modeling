import json
from pathlib import Path

import numpy as np

from dynamics.experiment_results import (
    make_experiment_record,
    residual_series,
    summarize_residuals,
    write_experiment_summary,
)
from dynamics.propagator import PropagationConfig


def test_residual_series_and_summary_use_si_state_contract() -> None:
    reference = np.array(
        [
            [7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0],
            [0.0, 7_000_000.0, 0.0, -7_500.0, 0.0, 0.0],
        ]
    )
    model = reference.copy()
    model[:, 0] += 1000.0
    model[:, 4] += 1.0

    series = residual_series(reference, model)
    summary = summarize_residuals(reference, model)

    np.testing.assert_allclose(series["delta_r_km"], [1.0, 1.0])
    np.testing.assert_allclose(series["delta_v_km_s"], [0.001, 0.001])
    assert summary["median_delta_r_km"] == 1.0
    assert summary["max_delta_v_km_s"] == 0.001


def test_write_experiment_summary_creates_json_and_csv() -> None:
    reference = np.array([[7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0]])
    model = reference.copy()
    config = PropagationConfig(
        initial_state=reference[0],
        epoch_seconds=0.0,
        duration_seconds=0.0,
    )
    record = make_experiment_record(
        name="demo",
        source={"format": "synthetic"},
        propagation_config=config,
        reference_states=reference,
        model_states=model,
        extra={"note": "unit-test"},
    )

    scratch = Path("tests") / "_experiment_results_scratch"
    try:
        json_path, csv_path = write_experiment_summary([record], scratch, stem="summary")
        assert json_path.exists()
        assert csv_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["records"][0]["name"] == "demo"
        assert payload["records"][0]["metrics"]["max_delta_r_km"] == 0.0
        assert "max_delta_r_km" in csv_path.read_text(encoding="utf-8")
    finally:
        for file_name in ("summary.json", "summary.csv"):
            path = scratch / file_name
            if path.exists():
                path.unlink()
        if scratch.exists():
            scratch.rmdir()
