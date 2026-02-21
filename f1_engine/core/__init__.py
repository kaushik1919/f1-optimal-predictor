"""Core simulation modules for the F1 2026 engine."""

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver
from f1_engine.core.energy import EnergyState
from f1_engine.core.kalman_update import (
    KalmanPerformanceState,
    apply_kalman_state_to_team,
    compute_measurement_gradient,
    initialize_kalman_state,
    kalman_update,
)
from f1_engine.core.monte_carlo import simulate_race_monte_carlo
from f1_engine.core.physics import lap_time
from f1_engine.core.pit_dp import compute_optimal_strategy_dp
from f1_engine.core.race import (
    PIT_LOSS,
    SC_GAP_INTERVAL,
    SC_LAP_TIME_FACTOR,
    SC_PIT_MULTIPLIER,
    RaceResult,
    simulate_race,
)
from f1_engine.core.season import simulate_season_monte_carlo
from f1_engine.core.sensitivity import (
    compute_championship_entropy,
    compute_ers_sensitivity,
    compute_reliability_sensitivity,
)
from f1_engine.core.stint import (
    find_best_constant_deploy,
    find_best_pit_strategy,
    simulate_stint,
)
from f1_engine.core.strategy import Strategy
from f1_engine.core.team import Team
from f1_engine.core.track import Track
from f1_engine.core.tyre import HARD, MEDIUM, SOFT, TyreCompound, TyreState
from f1_engine.core.updating import (
    PerformanceState,
    apply_updated_state,
    update_performance_state,
)

__all__ = [
    "Car",
    "Driver",
    "EnergyState",
    "HARD",
    "KalmanPerformanceState",
    "MEDIUM",
    "PIT_LOSS",
    "PerformanceState",
    "RaceResult",
    "SC_GAP_INTERVAL",
    "SC_LAP_TIME_FACTOR",
    "SC_PIT_MULTIPLIER",
    "SOFT",
    "Strategy",
    "Team",
    "Track",
    "TyreCompound",
    "TyreState",
    "apply_kalman_state_to_team",
    "apply_updated_state",
    "compute_championship_entropy",
    "compute_optimal_strategy_dp",
    "compute_ers_sensitivity",
    "compute_measurement_gradient",
    "compute_reliability_sensitivity",
    "find_best_constant_deploy",
    "find_best_pit_strategy",
    "initialize_kalman_state",
    "kalman_update",
    "lap_time",
    "simulate_race",
    "simulate_race_monte_carlo",
    "simulate_season_monte_carlo",
    "simulate_stint",
    "update_performance_state",
]
