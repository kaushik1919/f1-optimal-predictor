"""Deterministic strategy model for the F1 2026 simulation engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    """Constant ERS deployment and harvest strategy for a stint.

    Both levels are scalar multipliers in the range [0.0, 1.0].

    Attributes:
        deploy_level: Fraction of available ERS energy to deploy per lap.
        harvest_level: Fraction of maximum harvestable energy to recover per lap.
    """

    deploy_level: float
    harvest_level: float

    def __post_init__(self) -> None:
        """Validate strategy parameters."""
        if not 0.0 <= self.deploy_level <= 1.0:
            raise ValueError("deploy_level must be between 0.0 and 1.0.")
        if not 0.0 <= self.harvest_level <= 1.0:
            raise ValueError("harvest_level must be between 0.0 and 1.0.")
