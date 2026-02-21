"""Deterministic stint simulation for the F1 2026 simulation engine.

Phase 11B adds ``find_best_pit_strategy`` which searches a limited set of
1-stop and 2-stop strategies across tyre compound combinations.
"""

from __future__ import annotations

from typing import Any

from f1_engine.core.car import Car
from f1_engine.core.energy import EnergyState
from f1_engine.core.physics import lap_time
from f1_engine.core.strategy import Strategy
from f1_engine.core.track import Track
from f1_engine.core.tyre import HARD, MEDIUM, SOFT, TyreCompound, TyreState


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


# ---------------------------------------------------------------------------
# Phase 11B: pit-stop strategy search
# ---------------------------------------------------------------------------


def _simulate_compound_stint(
    track: Track,
    car: Car,
    laps: int,
    compound: TyreCompound,
    deploy_level: float,
    harvest_level: float,
) -> float:
    """Return total time for a single-compound stint of *laps* laps.

    Compound pace delta and degradation rate are applied on top of the
    standard physics model.
    """
    energy = EnergyState(max_charge=4.0, current_charge=4.0)
    tyre = TyreState(age=0, wear_rate_multiplier=car.tyre_wear_rate, compound=compound)
    total: float = 0.0
    for _ in range(laps):
        harvest_amount: float = track.energy_harvest_factor * harvest_level
        energy.harvest(harvest_amount)
        actual_deploy: float = energy.deploy(deploy_level)
        # Base lap time with zero tyre_age (we add compound-scaled deg ourselves)
        t: float = lap_time(track, car, 0.0, actual_deploy)
        base_deg: float = (
            float(tyre.age) * track.tyre_degradation_factor * car.tyre_wear_rate
        )
        t += base_deg * compound.degradation_rate
        t += compound.base_pace_delta
        total += t
        tyre.increment_age()
    return total


def find_best_pit_strategy(
    track: Track,
    car: Car,
    total_laps: int,
    pit_loss: float = 20.0,
) -> dict[str, Any]:
    """Search a limited grid of 1-stop and 2-stop strategies.

    Searches the following candidates:

    * **1-stop**: pit on laps 20-30 (or ``total_laps // 2`` if race is
      short), all permutations of SOFT/MEDIUM/HARD for each stint.
    * **2-stop**: pit on laps 15 and 35 (or ``total_laps // 3`` and
      ``2 * total_laps // 3``), all permutations of compounds for three
      stints.

    A best deploy level from ``find_best_constant_deploy`` is used for
    all evaluations.

    Args:
        track: Circuit to evaluate.
        car: Car to evaluate.
        total_laps: Total race distance in laps.
        pit_loss: Time penalty per pit stop (seconds).

    Returns:
        Dictionary containing:
            best_strategy -- ``Strategy`` with compound_sequence and pit_laps.
            best_time     -- Estimated total race time (float).
    """
    # Get best deploy/harvest from existing search
    best_deploy = find_best_constant_deploy(track, car, total_laps)
    deploy: float = best_deploy["best_strategy"].deploy_level
    harvest: float = best_deploy["best_strategy"].harvest_level

    compounds: list[TyreCompound] = [SOFT, MEDIUM, HARD]

    best_time: float = float("inf")
    best_strategy: Strategy | None = None

    # -- 1-stop candidates ---------------------------------------------------
    pit1_lap: int = max(2, min(total_laps - 1, total_laps // 2))
    for pit_offset in range(-5, 6):
        plap = pit1_lap + pit_offset
        if plap < 2 or plap >= total_laps:
            continue
        stint1_laps = plap  # laps before the pit (pit happens after this lap)
        stint2_laps = total_laps - plap
        for c1 in compounds:
            for c2 in compounds:
                t1 = _simulate_compound_stint(
                    track, car, stint1_laps, c1, deploy, harvest
                )
                t2 = _simulate_compound_stint(
                    track, car, stint2_laps, c2, deploy, harvest
                )
                t = t1 + pit_loss + t2
                if t < best_time:
                    best_time = t
                    best_strategy = Strategy(
                        deploy_level=deploy,
                        harvest_level=harvest,
                        compound_sequence=(c1, c2),
                        pit_laps=(plap,),
                    )

    # -- 2-stop candidates ---------------------------------------------------
    p1_base: int = max(2, total_laps // 3)
    p2_base: int = max(p1_base + 1, 2 * total_laps // 3)
    for p1_off in range(-5, 6):
        for p2_off in range(-5, 6):
            p1 = p1_base + p1_off
            p2 = p2_base + p2_off
            if p1 < 2 or p2 <= p1 or p2 >= total_laps:
                continue
            s1_laps = p1
            s2_laps = p2 - p1
            s3_laps = total_laps - p2
            for c1 in compounds:
                for c2 in compounds:
                    for c3 in compounds:
                        t1 = _simulate_compound_stint(
                            track, car, s1_laps, c1, deploy, harvest
                        )
                        t2 = _simulate_compound_stint(
                            track, car, s2_laps, c2, deploy, harvest
                        )
                        t3 = _simulate_compound_stint(
                            track, car, s3_laps, c3, deploy, harvest
                        )
                        t = t1 + pit_loss + t2 + pit_loss + t3
                        if t < best_time:
                            best_time = t
                            best_strategy = Strategy(
                                deploy_level=deploy,
                                harvest_level=harvest,
                                compound_sequence=(c1, c2, c3),
                                pit_laps=(p1, p2),
                            )

    assert best_strategy is not None
    return {
        "best_strategy": best_strategy,
        "best_time": best_time,
    }
