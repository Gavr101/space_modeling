from dataclasses import dataclass


@dataclass(slots=True)
class ForceModelConfig:
    """Supported force model toggles for stage-1 propagation."""

    spherical_earth_gravity: bool = True
    earth_j2: bool = True
    atmospheric_drag: bool = True
    nrlmsise00_atmosphere: bool = True
    third_body_sun: bool = True
    third_body_moon: bool = True
    solar_radiation_pressure: bool = True
