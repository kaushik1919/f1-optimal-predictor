"""Multi-car stochastic race simulator for the F1 2026 simulation engine.

This module introduces Gaussian lap-time noise, a reliability hazard model,
and a logistic overtake probability model.  All randomness is seeded via
a per-call ``numpy.random.Generator`` so that results are fully reproducible
when a seed is supplied and noise_std is held constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator

from f1_engine.core.car import Car
from f1_engine.core.energy import EnergyState
from f1_engine.core.physics import lap_time as compute_lap_time
from f1_engine.core.stint import find_best_constant_deploy
from f1_engine.core.track import Track
from f1_engine.core.tyre import TyreState

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RaceResult:
    """Outcome of a multi-car race simulation.

    Attributes:
        final_classification: Ordered list of team names.  Finishers are
            sorted by cumulative time; DNFs are appended at the end.
        dnf_list: Team names of cars that did not finish.
        lap_times: Mapping from team name to the list of per-lap times.
    """

    final_classification: list[str] = field(default_factory=list)
    dnf_list: list[str] = field(default_factory=list)
    lap_times: dict[str, list[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal per-car state
# ---------------------------------------------------------------------------


class _CarState:
    """Mutable per-car bookkeeping during a race simulation."""

    __slots__ = (
        "car",
        "energy",
        "tyre",
        "deploy_level",
        "harvest_level",
        "cumulative_time",
        "lap_times",
        "active",
        "last_lap_time",
    )

    def __init__(self, car: Car, deploy_level: float, harvest_level: float) -> None:
        self.car: Car = car
        self.energy: EnergyState = EnergyState(max_charge=4.0)
        self.tyre: TyreState = TyreState(age=0, wear_rate_multiplier=car.tyre_wear_rate)
        self.deploy_level: float = deploy_level
        self.harvest_level: float = harvest_level
        self.cumulative_time: float = 0.0
        self.lap_times: list[float] = []
        self.active: bool = True
        self.last_lap_time: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_race(
    track: Track,
    cars: list[Car],
    laps: int,
    noise_std: float = 0.05,
    seed: int | None = None,
) -> RaceResult:
    """Simulate a multi-car race with stochastic elements.

    Each car is initialised with its own energy state, tyre state, and a
    constant deploy strategy selected via ``find_best_constant_deploy``.

    Per lap, for every active car:
        1. Energy is harvested (track factor * strategy harvest level).
        2. Energy is deployed (bounded by battery).
        3. A deterministic lap time is computed.
        4. Gaussian noise (``N(0, noise_std)``) is added.
        5. A reliability hazard check may trigger a DNF.
        6. Tyre age is advanced.

    After all cars have been updated, active cars are sorted by cumulative
    time.  Adjacent pairs are then evaluated for a logistic overtake swap.

    Args:
        track: Circuit to race on.
        cars: List of participating cars.
        laps: Number of race laps (>= 1).
        noise_std: Standard deviation of Gaussian lap-time noise.
            Set to 0.0 for fully deterministic behaviour.
        seed: Random seed for reproducibility.  ``None`` uses
            entropy from the OS.

    Returns:
        A ``RaceResult`` containing classification, DNF list, and lap times.

    Raises:
        ValueError: If laps < 1 or cars is empty.
    """
    if laps < 1:
        raise ValueError("laps must be >= 1.")
    if not cars:
        raise ValueError("cars list must not be empty.")

    rng: Generator = np.random.default_rng(seed)

    # -- Initialise per-car state using Phase 2 strategy search ---------------
    states: list[_CarState] = []
    for car in cars:
        best = find_best_constant_deploy(track, car, laps)
        strat = best["best_strategy"]
        states.append(
            _CarState(
                car=car,
                deploy_level=strat.deploy_level,
                harvest_level=strat.harvest_level,
            )
        )

    # -- Lap loop -------------------------------------------------------------
    for _ in range(laps):
        for cs in states:
            if not cs.active:
                continue

            # 1. Harvest
            harvest_amount: float = track.energy_harvest_factor * cs.harvest_level
            cs.energy.harvest(harvest_amount)

            # 2. Deploy
            actual_deploy: float = cs.energy.deploy(cs.deploy_level)

            # 3. Deterministic lap time
            t: float = compute_lap_time(
                track, cs.car, float(cs.tyre.age), actual_deploy
            )

            # 4. Gaussian noise
            if noise_std > 0.0:
                t += float(rng.normal(0.0, noise_std))

            cs.last_lap_time = t
            cs.cumulative_time += t
            cs.lap_times.append(t)

            # 5. Reliability hazard
            hazard: float = 1.0 - math.exp(-(1.0 - cs.car.reliability))
            if rng.random() < hazard:
                cs.active = False

            # 6. Tyre age
            cs.tyre.increment_age()

        # -- Sort active cars by cumulative time ------------------------------
        active_states = [s for s in states if s.active]
        active_states.sort(key=lambda s: s.cumulative_time)

        # -- Overtake model (adjacent pairs) ----------------------------------
        _apply_overtakes(active_states, track, rng)

    # -- Build result ---------------------------------------------------------
    active_sorted = sorted(
        [s for s in states if s.active], key=lambda s: s.cumulative_time
    )
    dnf_sorted = [s for s in states if not s.active]

    classification: list[str] = [s.car.team_name for s in active_sorted] + [
        s.car.team_name for s in dnf_sorted
    ]
    dnf_names: list[str] = [s.car.team_name for s in dnf_sorted]
    lap_time_map: dict[str, list[float]] = {
        s.car.team_name: s.lap_times for s in states
    }

    return RaceResult(
        final_classification=classification,
        dnf_list=dnf_names,
        lap_times=lap_time_map,
    )


# ---------------------------------------------------------------------------
# Overtake helper
# ---------------------------------------------------------------------------


def _apply_overtakes(
    ranked: list[_CarState],
    track: Track,
    rng: Generator,
) -> None:
    """Evaluate logistic overtake probability for adjacent car pairs.

    For each consecutive pair (leading, trailing) in *ranked*:
        - If the cumulative time gap is < 1.0 s, compute a logistic pass
          probability based on the lap-time delta and the track's overtake
          coefficient.
        - If the random draw succeeds, swap positions in *ranked*.

    The logistic function used is::

        pass_prob = 1 / (1 + exp(-3.0 * delta * track.overtake_coefficient))

    where ``delta = trailing_last_lap - leading_last_lap``.
    """
    i = 0
    while i < len(ranked) - 1:
        leader = ranked[i]
        trailer = ranked[i + 1]

        gap: float = abs(trailer.cumulative_time - leader.cumulative_time)
        if gap < 1.0:
            delta: float = trailer.last_lap_time - leader.last_lap_time
            exponent: float = -3.0 * delta * track.overtake_coefficient
            pass_prob: float = 1.0 / (1.0 + math.exp(exponent))

            if rng.random() < pass_prob:
                ranked[i], ranked[i + 1] = ranked[i + 1], ranked[i]
                # Skip next pair to avoid double-swapping the same car.
                i += 2
                continue
        i += 1
