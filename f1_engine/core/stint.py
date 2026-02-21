"""Deterministic stint simulation for the F1 2026 simulation engine."""

from typing import Any

from f1_engine.core.car import Car
from f1_engine.core.energy import EnergyState
from f1_engine.core.physics import lap_time
from f1_engine.core.strategy import Strategy
from f1_engine.core.track import Track
from f1_engine.core.tyre import TyreState


def simulate_stint(
    track: Track,
    car: Car,
    strategy: Strategy,
    laps: int,
    initial_charge: float = 4.0,
    max_charge: float = 4.0,
) -> dict[str, Any]:
    """Simulate a deterministic stint and return telemetry traces.

    For each lap the sequence is:
        1. Harvest energy (scaled by track and strategy).
        2. Deploy energy (bounded by battery state).
        3. Compute lap time via the physics model.
        4. Increment tyre age.

    Args:
        track: Circuit to simulate.
        car: Car to simulate.
        strategy: Constant deploy/harvest strategy for the stint.
        laps: Number of laps to simulate (>= 1).
        initial_charge: Starting battery charge in MJ.
        max_charge: Battery capacity in MJ.

    Returns:
        Dictionary containing:
            total_time   -- Sum of all lap times (float).
            lap_times    -- Per-lap times in seconds (list[float]).
            energy_trace -- Battery level after each lap (list[float]).
            tyre_trace   -- Tyre age after each lap (list[int]).

    Raises:
        ValueError: If laps < 1.
    """
    if laps < 1:
        raise ValueError("laps must be >= 1.")

    energy = EnergyState(max_charge=max_charge, current_charge=initial_charge)
    tyre = TyreState(age=0, wear_rate_multiplier=car.tyre_wear_rate)

    lap_times: list[float] = []
    energy_trace: list[float] = []
    tyre_trace: list[int] = []

    for _ in range(laps):
        # 1. Harvest
        harvest_amount: float = track.energy_harvest_factor * strategy.harvest_level
        energy.harvest(harvest_amount)

        # 2. Deploy
        actual_deploy: float = energy.deploy(strategy.deploy_level)

        # 3. Compute lap time
        t: float = lap_time(track, car, float(tyre.age), actual_deploy)
        lap_times.append(t)

        # 4. Advance tyre
        tyre.increment_age()

        # Record traces
        energy_trace.append(energy.current_charge)
        tyre_trace.append(tyre.age)

    return {
        "total_time": sum(lap_times),
        "lap_times": lap_times,
        "energy_trace": energy_trace,
        "tyre_trace": tyre_trace,
    }


def find_best_constant_deploy(
    track: Track,
    car: Car,
    laps: int,
) -> dict[str, Any]:
    """Search over fixed deploy levels and return the fastest stint.

    Deploy levels tested: [0.0, 0.2, 0.4, 0.6, 0.8].
    Harvest level is fixed at 1.0 for all candidates.

    Args:
        track: Circuit to evaluate.
        car: Car to evaluate.
        laps: Stint length in laps.

    Returns:
        Dictionary containing:
            best_strategy -- The Strategy yielding the lowest total time.
            best_time     -- The total stint time for that strategy (float).
    """
    deploy_levels: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8]
    best_strategy: Strategy | None = None
    best_time: float = float("inf")

    for dl in deploy_levels:
        strat = Strategy(deploy_level=dl, harvest_level=1.0)
        result = simulate_stint(track, car, strat, laps)
        if result["total_time"] < best_time:
            best_time = result["total_time"]
            best_strategy = strat

    assert best_strategy is not None
    return {
        "best_strategy": best_strategy,
        "best_time": best_time,
    }
