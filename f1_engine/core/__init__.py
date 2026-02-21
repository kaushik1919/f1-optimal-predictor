"""Core simulation modules for the F1 2026 engine."""

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver
from f1_engine.core.energy import EnergyState
from f1_engine.core.monte_carlo import simulate_race_monte_carlo
from f1_engine.core.physics import lap_time
from f1_engine.core.race import RaceResult, simulate_race
from f1_engine.core.season import simulate_season_monte_carlo
from f1_engine.core.sensitivity import (
    compute_championship_entropy,
    compute_ers_sensitivity,
    compute_reliability_sensitivity,
)
from f1_engine.core.stint import find_best_constant_deploy, simulate_stint
from f1_engine.core.strategy import Strategy
from f1_engine.core.team import Team
from f1_engine.core.track import Track
from f1_engine.core.tyre import TyreState
from f1_engine.core.updating import (
    PerformanceState,
    apply_updated_state,
    update_performance_state,
)

__all__ = [
    "Car",
    "Driver",
    "EnergyState",
    "PerformanceState",
    "RaceResult",
    "Strategy",
    "Team",
    "Track",
    "TyreState",
    "apply_updated_state",
    "compute_championship_entropy",
    "compute_ers_sensitivity",
    "compute_reliability_sensitivity",
    "find_best_constant_deploy",
    "lap_time",
    "simulate_race",
    "simulate_race_monte_carlo",
    "simulate_season_monte_carlo",
    "simulate_stint",
    "update_performance_state",
]
