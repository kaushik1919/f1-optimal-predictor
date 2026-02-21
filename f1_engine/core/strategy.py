"""Deterministic strategy model for the F1 2026 simulation engine.

Phase 11B extends the strategy model with tyre compound sequencing and
pit-stop scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass

from f1_engine.core.tyre import MEDIUM, TyreCompound


@dataclass(frozen=True)
class Strategy:
    """Constant ERS deployment and harvest strategy for a stint.

    Both levels are scalar multipliers in the range [0.0, 1.0].

    Phase 11B adds:
        compound_sequence: Ordered list of tyre compounds to use across
            stints.  The first entry is the starting compound, and each
            subsequent entry is fitted after the corresponding pit stop.
            Length must be ``len(pit_laps) + 1``.
        pit_laps: Sorted list of lap numbers on which a pit stop occurs.
            Lap numbering is 1-based (lap 1 is the first racing lap).

    Attributes:
        deploy_level: Fraction of available ERS energy to deploy per lap.
        harvest_level: Fraction of maximum harvestable energy to recover per lap.
        compound_sequence: Tyre compounds per stint (default: single medium stint).
        pit_laps: Laps on which to pit (default: no stops).
    """

    deploy_level: float
    harvest_level: float
    compound_sequence: tuple[TyreCompound, ...] = (MEDIUM,)
    pit_laps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate strategy parameters."""
        if not 0.0 <= self.deploy_level <= 1.0:
            raise ValueError("deploy_level must be between 0.0 and 1.0.")
        if not 0.0 <= self.harvest_level <= 1.0:
            raise ValueError("harvest_level must be between 0.0 and 1.0.")
        if len(self.compound_sequence) != len(self.pit_laps) + 1:
            raise ValueError("compound_sequence length must be len(pit_laps) + 1.")
        if list(self.pit_laps) != sorted(self.pit_laps):
            raise ValueError("pit_laps must be sorted in ascending order.")
        if len(set(self.pit_laps)) != len(self.pit_laps):
            raise ValueError("pit_laps must not contain duplicates.")
