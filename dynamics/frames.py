from dataclasses import dataclass


@dataclass(slots=True)
class FrameConfig:
    """Конфигурация систем отсчёта.

    Для преобразований систем координат намеренно используем внешние
    астродинамические библиотеки (TudatPy/Orekit), а не собственные реализации.
    """

    inertial_frame: str = "J2000"
    body_fixed_frame: str = "ITRF"
