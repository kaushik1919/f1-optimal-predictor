"""Track model for the F1 2026 simulation engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    """Deterministic representation of a Formula 1 circuit.

    Attributes:
        name: Official circuit name.
        straight_ratio: Proportion of the lap that is straight sections (0.0-1.0).
        overtake_coefficient: Relative ease of overtaking at this circuit (0.0-1.0).
        energy_harvest_factor: ERS energy recovery potential of the circuit (0.0-1.0).
        tyre_degradation_factor: Circuit-specific tyre wear multiplier (>= 0.0).
        downforce_sensitivity: How much downforce affects lap time
            at this circuit (>= 0.0).
    """

    name: str
    straight_ratio: float
    overtake_coefficient: float
    energy_harvest_factor: float
    tyre_degradation_factor: float
    downforce_sensitivity: float

    def __post_init__(self) -> None:
        """Validate track parameters."""
        if not self.name:
            raise ValueError("Track name must not be empty.")
        if not 0.0 <= self.straight_ratio <= 1.0:
            raise ValueError("straight_ratio must be between 0.0 and 1.0.")
        if not 0.0 <= self.overtake_coefficient <= 1.0:
            raise ValueError("overtake_coefficient must be between 0.0 and 1.0.")
        if not 0.0 <= self.energy_harvest_factor <= 1.0:
            raise ValueError("energy_harvest_factor must be between 0.0 and 1.0.")
        if self.tyre_degradation_factor < 0.0:
            raise ValueError("tyre_degradation_factor must be >= 0.0.")
        if self.downforce_sensitivity < 0.0:
            raise ValueError("downforce_sensitivity must be >= 0.0.")
