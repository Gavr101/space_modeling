from dataclasses import dataclass


@dataclass(slots=True)
class FrameConfig:
    """Configuration of reference frames.

    We intentionally rely on external astrodynamics libraries (TudatPy/Orekit)
    for frame transformations instead of implementing custom transforms.
    """

    inertial_frame: str = "J2000"
    body_fixed_frame: str = "ITRF"
