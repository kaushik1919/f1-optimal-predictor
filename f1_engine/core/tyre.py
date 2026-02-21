"""Deterministic tyre state model for the F1 2026 simulation engine.

Phase 11B adds tyre compound modelling.  Each compound defines a pace
delta (negative = faster) and a degradation rate multiplier relative to
the medium baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tyre compound model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TyreCompound:
    """Immutable description of a tyre compound.

    Attributes:
        name: Human-readable compound label (e.g. "SOFT").
        base_pace_delta: Additive lap-time offset in seconds.
            Negative values make the compound faster.
        degradation_rate: Multiplier applied to the track/car
            degradation term.  1.0 = medium baseline.
    """

    name: str
    base_pace_delta: float
    degradation_rate: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Compound name must be non-empty.")
        if self.degradation_rate < 0.0:
            raise ValueError("degradation_rate must be >= 0.")


# Pre-defined compounds -------------------------------------------------------

SOFT = TyreCompound(name="SOFT", base_pace_delta=-0.6, degradation_rate=1.5)
MEDIUM = TyreCompound(name="MEDIUM", base_pace_delta=-0.3, degradation_rate=1.0)
HARD = TyreCompound(name="HARD", base_pace_delta=0.0, degradation_rate=0.7)


# ---------------------------------------------------------------------------
# Tyre state tracker
# ---------------------------------------------------------------------------


class TyreState:
    """Tracks tyre wear over a stint.

    Age increments are strictly integer-based and deterministic.

    Attributes:
        age: Number of laps completed on the current set of tyres.
        wear_rate_multiplier: Car-specific tyre wear scalar (>= 0).
        compound: The tyre compound currently fitted.
    """

    __slots__ = ("age", "wear_rate_multiplier", "compound")

    def __init__(
        self,
        age: int = 0,
        wear_rate_multiplier: float = 1.0,
        compound: TyreCompound | None = None,
    ):
        """Initialise tyre state.

        Args:
            age: Starting tyre age in laps. Must be >= 0.
            wear_rate_multiplier: Multiplier for tyre degradation. Must be >= 0.
            compound: Tyre compound fitted.  Defaults to ``MEDIUM`` if not
                provided.

        Raises:
            ValueError: If constraints are violated.
        """
        if age < 0:
            raise ValueError("age must be >= 0.")
        if wear_rate_multiplier < 0.0:
            raise ValueError("wear_rate_multiplier must be >= 0.")
        self.age: int = age
        self.wear_rate_multiplier: float = wear_rate_multiplier
        self.compound: TyreCompound = compound if compound is not None else MEDIUM

    def increment_age(self) -> None:
        """Advance tyre age by one lap."""
        self.age += 1

    def reset(self, compound: TyreCompound | None = None) -> None:
        """Reset tyre age to zero after a pit stop.

        Args:
            compound: New compound to fit.  If ``None``, the current compound
                is retained.
        """
        self.age = 0
        if compound is not None:
            self.compound = compound
