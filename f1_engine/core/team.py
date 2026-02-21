"""Team model for the F1 2026 simulation engine.

A team pairs a single Car with exactly two Drivers, enforcing
the constraint that both drivers belong to the same constructor.
"""

from __future__ import annotations

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver


class Team:
    """A constructor team comprising a car and two drivers.

    Attributes:
        name: Constructor team name.
        car: Shared car for both drivers.
        drivers: Exactly two :class:`Driver` instances.
    """

    __slots__ = ("name", "car", "drivers")

    def __init__(self, name: str, car: Car, drivers: list[Driver]) -> None:
        if not name:
            raise ValueError("Team name must not be empty.")
        if len(drivers) != 2:
            raise ValueError(
                f"Team '{name}' must have exactly 2 drivers, got {len(drivers)}."
            )
        for drv in drivers:
            if drv.team_name != name:
                raise ValueError(
                    f"Driver '{drv.name}' team_name '{drv.team_name}' "
                    f"does not match Team name '{name}'."
                )
        self.name: str = name
        self.car: Car = car
        self.drivers: list[Driver] = list(drivers)

    def __repr__(self) -> str:
        driver_names = ", ".join(d.name for d in self.drivers)
        return f"Team(name={self.name!r}, drivers=[{driver_names}])"
