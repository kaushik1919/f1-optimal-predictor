"""Deterministic tyre state model for the F1 2026 simulation engine."""


class TyreState:
    """Tracks tyre wear over a stint.

    Age increments are strictly integer-based and deterministic.

    Attributes:
        age: Number of laps completed on the current set of tyres.
        wear_rate_multiplier: Car-specific tyre wear scalar (>= 0).
    """

    __slots__ = ("age", "wear_rate_multiplier")

    def __init__(self, age: int = 0, wear_rate_multiplier: float = 1.0):
        """Initialise tyre state.

        Args:
            age: Starting tyre age in laps. Must be >= 0.
            wear_rate_multiplier: Multiplier for tyre degradation. Must be >= 0.

        Raises:
            ValueError: If constraints are violated.
        """
        if age < 0:
            raise ValueError("age must be >= 0.")
        if wear_rate_multiplier < 0.0:
            raise ValueError("wear_rate_multiplier must be >= 0.")
        self.age: int = age
        self.wear_rate_multiplier: float = wear_rate_multiplier

    def increment_age(self) -> None:
        """Advance tyre age by one lap."""
        self.age += 1
