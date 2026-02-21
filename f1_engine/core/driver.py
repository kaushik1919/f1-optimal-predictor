"""Driver model for the F1 2026 simulation engine.

Each driver belongs to a team and modifies the shared car performance
through an individual skill offset and consistency multiplier.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Driver:
    """Immutable representation of a Formula 1 driver.

    Attributes:
        name: Unique driver name.
        team_name: Constructor team the driver belongs to.
        skill_offset: Additive offset applied to the deterministic lap
            time.  Negative values make the driver faster than the car
            baseline; positive values make the driver slower.
        consistency: Multiplier applied to Gaussian noise standard
            deviation.  A value of 1.0 represents baseline consistency;
            values below 1.0 reduce variance (more consistent) and
            values above 1.0 increase variance (less consistent).
    """

    name: str
    team_name: str
    skill_offset: float
    consistency: float

    def __post_init__(self) -> None:
        """Validate driver parameters."""
        if not self.name:
            raise ValueError("name must not be empty.")
        if not self.team_name:
            raise ValueError("team_name must not be empty.")
        if self.consistency <= 0.0:
            raise ValueError("consistency must be > 0.0.")
