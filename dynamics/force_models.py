from dataclasses import dataclass, replace
from typing import Literal


@dataclass(slots=True)
class ForceModelConfig:
    """Поддерживаемые переключатели силовых моделей для первого этапа распространения."""

    spherical_earth_gravity: bool = True
    earth_gravity_model: Literal["j2", "egm2008"] | str = "j2"
    earth_j2: bool = True
    gravity_max_degree: int = 8
    gravity_max_order: int = 8
    j2_frame: Literal["gcrs_fixed_axis", "itrs_body_fixed"] | str = "gcrs_fixed_axis"
    atmospheric_drag: bool = True
    density_model: Literal["exponential", "nrlmsise00"] | str = "nrlmsise00"
    nrlmsise00_atmosphere: bool = True
    third_body_sun: bool = True
    third_body_moon: bool = True
    solar_radiation_pressure: bool = True
    srp_shadow_model: Literal["none", "cylindrical", "conical"] | str = "none"
    earth_radiation_model: Literal["none", "isotropic_ir"] | str = "none"
    relativity_model: Literal["none", "schwarzschild"] | str = "none"
    tide_model: Literal["none", "solid_earth_degree2"] | str = "none"


def make_recommended_force_config() -> ForceModelConfig:
    """Вернуть воспроизводимую высокоточную стартовую конфигурацию для экспериментов добавления.

    Рекомендованный вариант оставляет новые малые силы выключенными, пока
    проверка в ноутбуке не покажет устойчивое улучшение. Их можно включать
    по одной в экспериментах добавления.
    """
    return ForceModelConfig(
        spherical_earth_gravity=True,
        earth_gravity_model="j2",
        earth_j2=True,
        gravity_max_degree=8,
        gravity_max_order=8,
        j2_frame="itrs_body_fixed",
        atmospheric_drag=True,
        density_model="nrlmsise00",
        nrlmsise00_atmosphere=True,
        third_body_sun=True,
        third_body_moon=True,
        solar_radiation_pressure=True,
        srp_shadow_model="conical",
        earth_radiation_model="none",
        relativity_model="none",
        tide_model="none",
    )


def make_addition_force_configs() -> dict[str, ForceModelConfig]:
    """Вернуть именованные конфигурации сил для поочерёдных экспериментов добавления."""
    baseline = make_recommended_force_config()
    return {
        "recommended_baseline": baseline,
        "plus_earth_ir": replace(baseline, earth_radiation_model="isotropic_ir"),
        "plus_relativity": replace(baseline, relativity_model="schwarzschild"),
        "plus_solid_earth_tides": replace(baseline, tide_model="solid_earth_degree2"),
    }
