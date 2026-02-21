"""Deterministic physics calculations for the F1 2026 simulation engine."""

from f1_engine.core.car import Car
from f1_engine.core.track import Track


def lap_time(
    track: Track,
    car: Car,
    tyre_age: float,
    deploy_level: float,
) -> float:
    """Calculate a deterministic lap time for a given car on a given track.

    The formula combines baseline car speed with aero, tyre degradation,
    and ERS deployment effects:

        lap_time = base_component + aero_component + tyre_component - ers_component

    Where:
        base_component  = car.base_speed
        aero_component  = track.downforce_sensitivity * (1 - car.aero_efficiency)
        tyre_component  = tyre_age * track.tyre_degradation_factor * car.tyre_wear_rate
        ers_component   = deploy_level * car.ers_efficiency

    Args:
        track: The circuit being raced on.
        car: The car being driven.
        tyre_age: Number of laps since last tyre change (>= 0).
        deploy_level: ERS deployment level for this lap (0.0 - 1.0).

    Returns:
        Calculated lap time in seconds.

    Raises:
        ValueError: If tyre_age < 0 or deploy_level is out of [0.0, 1.0].
    """
    if tyre_age < 0.0:
        raise ValueError("tyre_age must be >= 0.")
    if not 0.0 <= deploy_level <= 1.0:
        raise ValueError("deploy_level must be between 0.0 and 1.0.")

    base_component: float = car.base_speed
    aero_component: float = track.downforce_sensitivity * (1.0 - car.aero_efficiency)
    tyre_component: float = (
        tyre_age * track.tyre_degradation_factor * car.tyre_wear_rate
    )
    ers_component: float = deploy_level * car.ers_efficiency

    return base_component + aero_component + tyre_component - ers_component
