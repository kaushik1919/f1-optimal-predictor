"""Deterministic energy state model for the F1 2026 simulation engine."""


class EnergyState:
    """Tracks ERS battery charge over a stint.

    The energy state is fully deterministic: harvest and deploy operations
    are bounded by physical limits (0 to max_charge) with no stochastic
    component.

    Attributes:
        current_charge: Current battery energy level in MJ.
        max_charge: Maximum battery capacity in MJ.
    """

    __slots__ = ("current_charge", "max_charge")

    def __init__(self, max_charge: float = 4.0, current_charge: float | None = None):
        """Initialise battery state.

        Args:
            max_charge: Maximum battery capacity in MJ. Must be > 0.
            current_charge: Initial charge level. Defaults to max_charge.

        Raises:
            ValueError: If constraints are violated.
        """
        if max_charge <= 0.0:
            raise ValueError("max_charge must be > 0.")
        resolved_charge = max_charge if current_charge is None else current_charge
        if resolved_charge < 0.0:
            raise ValueError("current_charge must be >= 0.")
        if resolved_charge > max_charge:
            raise ValueError("current_charge must be <= max_charge.")
        self.max_charge: float = max_charge
        self.current_charge: float = resolved_charge

    def deploy(self, amount: float) -> float:
        """Deploy energy from the battery.

        Deploys the minimum of the requested amount and the available
        charge.  The battery level is reduced accordingly.

        Args:
            amount: Requested deployment in MJ (>= 0).

        Returns:
            Actual energy deployed in MJ.

        Raises:
            ValueError: If amount is negative.
        """
        if amount < 0.0:
            raise ValueError("deploy amount must be >= 0.")
        actual: float = min(amount, self.current_charge)
        self.current_charge -= actual
        return actual

    def harvest(self, amount: float) -> float:
        """Harvest energy into the battery.

        Increases the charge by the requested amount, capped at
        max_charge.

        Args:
            amount: Requested harvest in MJ (>= 0).

        Returns:
            Actual energy harvested in MJ.

        Raises:
            ValueError: If amount is negative.
        """
        if amount < 0.0:
            raise ValueError("harvest amount must be >= 0.")
        headroom: float = self.max_charge - self.current_charge
        actual: float = min(amount, headroom)
        self.current_charge += actual
        return actual
