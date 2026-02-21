"""Car model for the F1 2026 simulation engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Car:
    """Deterministic representation of a Formula 1 car.

    Attributes:
        team_name: Constructor team name.
        base_speed: Baseline lap time in seconds (lower is faster).
        ers_efficiency: Effectiveness of energy recovery and deployment (0.0-1.0).
        aero_efficiency: Aerodynamic efficiency coefficient (0.0-1.0).
        tyre_wear_rate: Car-specific tyre degradation multiplier (>= 0.0).
        reliability: Mechanical reliability factor (0.0-1.0).
    """

    team_name: str
    base_speed: float
    ers_efficiency: float
    aero_efficiency: float
    tyre_wear_rate: float
    reliability: float

    def __post_init__(self) -> None:
        """Validate car parameters."""
        if not self.team_name:
            raise ValueError("team_name must not be empty.")
        if self.base_speed <= 0.0:
            raise ValueError("base_speed must be > 0.0.")
        if not 0.0 <= self.ers_efficiency <= 1.0:
            raise ValueError("ers_efficiency must be between 0.0 and 1.0.")
        if not 0.0 <= self.aero_efficiency <= 1.0:
            raise ValueError("aero_efficiency must be between 0.0 and 1.0.")
        if self.tyre_wear_rate < 0.0:
            raise ValueError("tyre_wear_rate must be >= 0.0.")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be between 0.0 and 1.0.")
