import numpy as np
import pytest
from pathlib import Path

from dynamics.environment import EnvironmentConfig
from dynamics.force_models import ForceModelConfig
from dynamics import make_addition_force_configs, make_recommended_force_config
from dynamics.propagator import (
    AU_METERS,
    EARTH_IR_FLUX_W_M2,
    MU_EARTH,
    R_EARTH,
    SPEED_OF_LIGHT,
    PropagationConfig,
    SOLID_EARTH_LOVE_K2,
    SpacecraftProperties,
    _degree2_solid_tide_body_acceleration,
    _select_density_model,
    _earth_gravity_acceleration,
    _earth_ir_pressure_acceleration,
    _nrlmsise_density_kg_m3,
    _relativistic_acceleration,
    _srp_illumination_factor,
    _total_acceleration,
    propagate_orbit,
)


def _central_gravity_config(integrator: str, duration_seconds: float = 600.0) -> PropagationConfig:
    r0 = np.array([7_000_000.0, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU_EARTH / np.linalg.norm(r0)), 0.0])
    force_models = ForceModelConfig(
        earth_j2=False,
        atmospheric_drag=False,
        nrlmsise00_atmosphere=False,
        third_body_sun=False,
        third_body_moon=False,
        solar_radiation_pressure=False,
    )
    return PropagationConfig(
        initial_state=np.hstack([r0, v0]),
        epoch_seconds=0.0,
        duration_seconds=duration_seconds,
        step_seconds=60.0,
        integrator=integrator,
        spacecraft=SpacecraftProperties(mass=1.0, cd=2.2, cr=1.3, reference_area=1.0),
        environment=EnvironmentConfig(force_models=force_models),
    )


def test_dop853_alias_and_vector_atol_are_normalized() -> None:
    config = _central_gravity_config("DOP853")

    assert config.integrator == "dop853"
    assert config.atol.shape == (6,)
    np.testing.assert_allclose(config.atol[:3], [1e-3, 1e-3, 1e-3])
    np.testing.assert_allclose(config.atol[3:], [1e-6, 1e-6, 1e-6])


def test_propagators_return_saved_state_grid_with_final_epoch() -> None:
    for integrator in ("rk4_fixed", "dop853"):
        config = _central_gravity_config(integrator, duration_seconds=125.0)
        times, states = propagate_orbit(config)

        np.testing.assert_allclose(times, [0.0, 60.0, 120.0, 125.0])
        assert states.shape == (4, 6)


def test_custom_output_times_are_supported_and_sorted() -> None:
    config = _central_gravity_config("dop853", duration_seconds=125.0)
    config.output_times_seconds = np.array([125.0, 0.0, 17.5, 60.0])
    config.__post_init__()

    times, states = propagate_orbit(config)

    np.testing.assert_allclose(times, [0.0, 17.5, 60.0, 125.0])
    assert states.shape == (4, 6)


def test_dop853_central_gravity_keeps_circular_orbit_radius() -> None:
    config = _central_gravity_config("dop853")
    _, states = propagate_orbit(config)

    final_radius_error_m = abs(
        np.linalg.norm(states[-1, :3]) - np.linalg.norm(config.initial_state[:3])
    )
    assert final_radius_error_m < 0.01


def test_j2_body_fixed_frame_returns_finite_si_acceleration() -> None:
    r_gcrs = np.array([6_900_000.0, 1_200_000.0, 1_700_000.0])

    fixed_axis = _earth_gravity_acceleration(r_gcrs, with_j2=True, j2_frame="gcrs_fixed_axis")
    body_fixed = _earth_gravity_acceleration(
        r_gcrs,
        with_j2=True,
        j2_frame="itrs_body_fixed",
        epoch_seconds=1_704_067_200.0,
        t_seconds=123.0,
    )

    assert fixed_axis.shape == (3,)
    assert body_fixed.shape == (3,)
    assert np.all(np.isfinite(body_fixed))
    assert 7.0 < np.linalg.norm(body_fixed) < 9.0


def test_force_scale_factors_scale_j2_perturbation_only() -> None:
    state = np.array([6_900_000.0, 1_200_000.0, 1_700_000.0, 0.0, 7_500.0, 0.0])
    force_models = ForceModelConfig(
        earth_gravity_model="j2",
        earth_j2=True,
        atmospheric_drag=False,
        third_body_sun=False,
        third_body_moon=False,
        solar_radiation_pressure=False,
    )
    config = PropagationConfig(
        initial_state=state,
        epoch_seconds=1_704_067_200.0,
        duration_seconds=0.0,
        integrator="dop853",
        environment=EnvironmentConfig(force_models=force_models),
    )

    default_acceleration = _total_acceleration(state, config, 123.0)
    central_acceleration = _earth_gravity_acceleration(state[:3], with_j2=False)

    config.environment.force_scale_factors = {"j2": 0.0}
    scaled_acceleration = _total_acceleration(state, config, 123.0)

    np.testing.assert_allclose(scaled_acceleration, central_acceleration)
    assert np.linalg.norm(default_acceleration - central_acceleration) > 0.0


def test_srp_shadow_models_cover_light_and_umbra_cases() -> None:
    r_sun = np.array([AU_METERS, 0.0, 0.0])
    sunward_spacecraft = np.array([7_000_000.0, 0.0, 0.0])
    eclipsed_spacecraft = np.array([-7_000_000.0, 0.0, 0.0])

    assert _srp_illumination_factor(sunward_spacecraft, r_sun, "cylindrical") == 1.0
    assert _srp_illumination_factor(eclipsed_spacecraft, r_sun, "cylindrical") == 0.0
    assert _srp_illumination_factor(eclipsed_spacecraft, r_sun, "conical") == 0.0


def test_conical_srp_shadow_has_penumbra_between_umbra_and_light() -> None:
    r_sun = np.array([AU_METERS, 0.0, 0.0])
    penumbra_spacecraft = np.array([-7_000_000.0, 6_400_000.0, 0.0])

    illumination = _srp_illumination_factor(penumbra_spacecraft, r_sun, "conical")

    assert 0.0 < illumination < 1.0


def test_earth_ir_pressure_is_switchable_and_points_away_from_earth() -> None:
    state = np.array([7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0])
    force_models = ForceModelConfig(earth_radiation_model="none")
    config = PropagationConfig(
        initial_state=state,
        epoch_seconds=0.0,
        duration_seconds=0.0,
        integrator="dop853",
        spacecraft=SpacecraftProperties(mass=10.0, cd=2.2, cr=1.3, reference_area=2.0),
        environment=EnvironmentConfig(force_models=force_models),
    )

    np.testing.assert_allclose(_earth_ir_pressure_acceleration(state, config), np.zeros(3))

    config.environment.force_models.earth_radiation_model = "isotropic_ir"
    acceleration = _earth_ir_pressure_acceleration(state, config)
    expected_magnitude = (
        EARTH_IR_FLUX_W_M2
        * (R_EARTH / np.linalg.norm(state[:3])) ** 2
        / SPEED_OF_LIGHT
        * config.spacecraft.cr
        * config.spacecraft.reference_area
        / config.spacecraft.mass
    )

    assert acceleration[0] > 0.0
    np.testing.assert_allclose(acceleration[1:], [0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(acceleration), expected_magnitude)


def test_schwarzschild_relativity_is_switchable_and_small() -> None:
    radius = 7_000_000.0
    speed = np.sqrt(MU_EARTH / radius)
    state = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])
    force_models = ForceModelConfig(relativity_model="none")
    config = PropagationConfig(
        initial_state=state,
        epoch_seconds=0.0,
        duration_seconds=0.0,
        integrator="dop853",
        spacecraft=SpacecraftProperties(mass=10.0, cd=2.2, cr=1.3, reference_area=2.0),
        environment=EnvironmentConfig(force_models=force_models),
    )

    np.testing.assert_allclose(_relativistic_acceleration(state, config), np.zeros(3))

    config.environment.force_models.relativity_model = "schwarzschild"
    acceleration = _relativistic_acceleration(state, config)
    central_acceleration = MU_EARTH / radius**2

    assert acceleration[0] > 0.0
    np.testing.assert_allclose(acceleration[1:], [0.0, 0.0], atol=1e-30)
    assert 0.0 < np.linalg.norm(acceleration) / central_acceleration < 1e-8


def test_degree2_solid_tide_body_acceleration_has_expected_scale() -> None:
    r_sc = np.array([7_000_000.0, 0.0, 0.0])
    r_moon = np.array([384_400_000.0, 0.0, 0.0])

    acceleration = _degree2_solid_tide_body_acceleration(r_sc, r_moon, 4.9048695e12)

    assert SOLID_EARTH_LOVE_K2 > 0.0
    assert acceleration[0] < 0.0
    np.testing.assert_allclose(acceleration[1:], [0.0, 0.0], atol=1e-30)
    assert 1e-9 < np.linalg.norm(acceleration) < 1e-6


def test_density_model_selection_keeps_legacy_bool_compatibility() -> None:
    assert _select_density_model(ForceModelConfig(density_model="exponential")) == "exponential"
    assert (
        _select_density_model(
            ForceModelConfig(density_model="nrlmsise00", nrlmsise00_atmosphere=False)
        )
        == "exponential"
    )


def test_nrlmsise00_density_uses_space_weather_file() -> None:
    csv_path = Path("tests") / "_tmp_SW-All.csv"
    try:
        csv_path.write_text(
            "DATE,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,"
            "F10.7_OBS,F10.7_ADJ,F10.7_OBS_CENTER81,F10.7_ADJ_CENTER81\n"
            "2023-07-31,4,4,4,4,4,4,4,4,4,180.0,182.4,166.0,167.0\n"
            "2023-08-01,7,6,5,4,4,5,6,7,8,177.0,179.9,166.8,167.1\n",
            encoding="utf-8",
        )
        config = PropagationConfig(
            initial_state=np.array([7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0]),
            epoch_seconds=1_690_848_000.0,
            duration_seconds=0.0,
            spacecraft=SpacecraftProperties(mass=500.0, cd=2.2, cr=1.3, reference_area=2.2),
        )
        config.environment.force_models = ForceModelConfig(
            density_model="nrlmsise00",
            nrlmsise00_atmosphere=True,
        )
        config.environment.space_weather_file = csv_path

        rho = _nrlmsise_density_kg_m3(config, 0.0, config.initial_state[:3])

        assert np.isfinite(rho)
        assert rho > 0.0
    finally:
        csv_path.unlink(missing_ok=True)


def test_recommended_force_config_keeps_addition_forces_explicit() -> None:
    config = make_recommended_force_config()

    assert config.j2_frame == "itrs_body_fixed"
    assert config.srp_shadow_model == "conical"
    assert config.earth_radiation_model == "none"
    assert config.relativity_model == "none"
    assert config.tide_model == "none"


def test_addition_force_configs_enable_one_new_force_at_a_time() -> None:
    configs = make_addition_force_configs()

    assert configs["recommended_baseline"].earth_radiation_model == "none"
    assert configs["recommended_baseline"].relativity_model == "none"
    assert configs["recommended_baseline"].tide_model == "none"
    assert configs["plus_earth_ir"].earth_radiation_model == "isotropic_ir"
    assert configs["plus_earth_ir"].relativity_model == "none"
    assert configs["plus_relativity"].relativity_model == "schwarzschild"
    assert configs["plus_relativity"].earth_radiation_model == "none"
    assert configs["plus_solid_earth_tides"].tide_model == "solid_earth_degree2"
    assert configs["plus_solid_earth_tides"].earth_radiation_model == "none"


def test_egm2008_gravity_requires_local_coefficients_file() -> None:
    config = _central_gravity_config("rk4_fixed", duration_seconds=60.0)
    config.environment.force_models.earth_gravity_model = "egm2008"

    with pytest.raises(ValueError, match="gravity_coefficients_file"):
        propagate_orbit(config)
