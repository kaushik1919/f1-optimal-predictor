"""Multi-car stochastic race simulator for the F1 2026 simulation engine.

This module introduces Gaussian lap-time noise, a reliability hazard model,
and a logistic overtake probability model.  All randomness is seeded via
a per-call ``numpy.random.Generator`` so that results are fully reproducible
when a seed is supplied and noise_std is held constant.

Phase 10 extends the simulator to operate at the *driver* level.  Each
team fields two drivers who share the same car but have individual skill
offsets and consistency multipliers.

Phase 11A makes overtakes persistent by applying a cumulative-time
adjustment (pass_time_delta = 0.2 s) on each successful pass, preventing
position oscillation and ensuring time-consistent race order.

Phase 11B adds tyre compound modelling and pit-stop strategy.  Each
driver follows a pit schedule that dictates when to stop and which
compound to fit.  A ``PIT_LOSS`` of 20.0 seconds is added to cumulative
time on every pit lap.  Lap times incorporate the compound's pace delta
and degradation rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator

from f1_engine.core.car import Car
from f1_engine.core.driver import Driver
from f1_engine.core.energy import EnergyState
from f1_engine.core.physics import lap_time as compute_lap_time
from f1_engine.core.stint import find_best_constant_deploy
from f1_engine.core.strategy import Strategy
from f1_engine.core.team import Team
from f1_engine.core.track import Track
from f1_engine.core.tyre import MEDIUM, TyreCompound, TyreState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIT_LOSS: float = 20.0  # seconds added to cumulative time on every pit stop

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RaceResult:
    """Outcome of a multi-car race simulation.

    Attributes:
        final_classification: Ordered list of driver names.  Finishers are
            sorted by cumulative time; DNFs are appended at the end.
        dnf_list: Driver names of entries that did not finish.
        lap_times: Mapping from driver name to the list of per-lap times.
    """

    final_classification: list[str] = field(default_factory=list)
    dnf_list: list[str] = field(default_factory=list)
    lap_times: dict[str, list[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal per-driver state
# ---------------------------------------------------------------------------


class _DriverState:
    """Mutable per-driver bookkeeping during a race simulation."""

    __slots__ = (
        "driver",
        "car",
        "energy",
        "tyre",
        "deploy_level",
        "harvest_level",
        "cumulative_time",
        "lap_times",
        "active",
        "last_lap_time",
        "pit_laps",
        "compound_sequence",
        "stint_index",
    )

    def __init__(
        self,
        driver: Driver,
        car: Car,
        deploy_level: float,
        harvest_level: float,
        compound_sequence: tuple[TyreCompound, ...] = (MEDIUM,),
        pit_laps: tuple[int, ...] = (),
    ) -> None:
        self.driver: Driver = driver
        self.car: Car = car
        self.energy: EnergyState = EnergyState(max_charge=4.0)
        starting_compound: TyreCompound = compound_sequence[0]
        self.tyre: TyreState = TyreState(
            age=0, wear_rate_multiplier=car.tyre_wear_rate, compound=starting_compound
        )
        self.deploy_level: float = deploy_level
        self.harvest_level: float = harvest_level
        self.cumulative_time: float = 0.0
        self.lap_times: list[float] = []
        self.active: bool = True
        self.last_lap_time: float = 0.0
        self.pit_laps: tuple[int, ...] = pit_laps
        self.compound_sequence: tuple[TyreCompound, ...] = compound_sequence
        self.stint_index: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_race(
    track: Track,
    teams: list[Team],
    laps: int,
    noise_std: float = 0.05,
    seed: int | None = None,
    strategies: dict[str, Strategy] | None = None,
) -> RaceResult:
    """Simulate a multi-driver race with stochastic elements.

    Each driver is initialised with its own energy state, tyre state, and
    a constant deploy strategy selected via ``find_best_constant_deploy``
    using the shared team car.

    Per lap, for every active driver:
        1. Energy is harvested (track factor * strategy harvest level).
        2. Energy is deployed (bounded by battery).
        3. A deterministic lap time is computed, then the driver's
           ``skill_offset`` and the compound's ``base_pace_delta`` are
           added.  Tyre degradation is scaled by the compound's
           ``degradation_rate``.
        4. Gaussian noise ``N(0, noise_std * driver.consistency)`` is added.
        5. A reliability hazard check (car-based) may trigger a DNF.
        6. Tyre age is advanced.
        7. If the current lap is in the driver's pit schedule, a pit stop
           is performed: ``PIT_LOSS`` is added to cumulative time, tyres
           are reset, and the next compound in the sequence is fitted.

    After all drivers have been updated, active entries are sorted by
    cumulative time.  Adjacent pairs are then evaluated for a logistic
    overtake swap.

    Args:
        track: Circuit to race on.
        teams: List of participating teams (each with 2 drivers).
        laps: Number of race laps (>= 1).
        noise_std: Baseline standard deviation of Gaussian lap-time noise.
            Each driver's effective noise_std is ``noise_std * driver.consistency``.
            Set to 0.0 for fully deterministic behaviour.
        seed: Random seed for reproducibility.  ``None`` uses
            entropy from the OS.
        strategies: Optional mapping from driver *name* to a ``Strategy``
            instance that includes ``compound_sequence`` and ``pit_laps``.
            If a driver is not present in this mapping (or if ``strategies``
            is ``None``), the driver uses the default strategy returned by
            ``find_best_constant_deploy`` with a single medium-compound
            stint and no pit stops.

    Returns:
        A ``RaceResult`` containing classification, DNF list, and lap times.

    Raises:
        ValueError: If laps < 1 or teams is empty.
    """
    if laps < 1:
        raise ValueError("laps must be >= 1.")
    if not teams:
        raise ValueError("teams list must not be empty.")

    rng: Generator = np.random.default_rng(seed)
    strat_map: dict[str, Strategy] = strategies if strategies is not None else {}

    # -- Initialise per-driver state using Phase 2 strategy search ------------
    states: list[_DriverState] = []
    for team in teams:
        best = find_best_constant_deploy(track, team.car, laps)
        default_strat: Strategy = best["best_strategy"]
        for driver in team.drivers:
            strat = strat_map.get(driver.name, default_strat)
            states.append(
                _DriverState(
                    driver=driver,
                    car=team.car,
                    deploy_level=strat.deploy_level,
                    harvest_level=strat.harvest_level,
                    compound_sequence=strat.compound_sequence,
                    pit_laps=strat.pit_laps,
                )
            )

    # -- Lap loop -------------------------------------------------------------
    for lap_number in range(1, laps + 1):
        for ds in states:
            if not ds.active:
                continue

            # 1. Harvest
            harvest_amount: float = track.energy_harvest_factor * ds.harvest_level
            ds.energy.harvest(harvest_amount)

            # 2. Deploy
            actual_deploy: float = ds.energy.deploy(ds.deploy_level)

            # 3. Deterministic lap time + driver skill offset + compound delta
            #    The physics model already applies base tyre degradation;
            #    we additionally scale by the compound degradation rate.
            compound: TyreCompound = ds.tyre.compound
            base_tyre_component: float = (
                float(ds.tyre.age)
                * track.tyre_degradation_factor
                * ds.car.tyre_wear_rate
            )
            # Compute lap time with tyre_age = 0 so we can add the
            # compound-scaled degradation ourselves.
            t: float = compute_lap_time(track, ds.car, 0.0, actual_deploy)
            t += base_tyre_component * compound.degradation_rate
            t += compound.base_pace_delta
            t += ds.driver.skill_offset

            # 4. Gaussian noise scaled by driver consistency
            if noise_std > 0.0:
                effective_std: float = noise_std * ds.driver.consistency
                t += float(rng.normal(0.0, effective_std))

            ds.last_lap_time = t
            ds.cumulative_time += t
            ds.lap_times.append(t)

            # 5. Reliability hazard (car-based)
            hazard: float = 1.0 - math.exp(-(1.0 - ds.car.reliability))
            if rng.random() < hazard:
                ds.active = False

            # 6. Tyre age
            ds.tyre.increment_age()

            # 7. Pit stop check (Phase 11B)
            if lap_number in ds.pit_laps:
                ds.cumulative_time += PIT_LOSS
                ds.stint_index += 1
                next_compound: TyreCompound | None = None
                if ds.stint_index < len(ds.compound_sequence):
                    next_compound = ds.compound_sequence[ds.stint_index]
                ds.tyre.reset(next_compound)

        # -- Sort active drivers by cumulative time ---------------------------
        active_states = [s for s in states if s.active]
        active_states.sort(key=lambda s: s.cumulative_time)

        # -- Overtake model (adjacent pairs) ----------------------------------
        _apply_overtakes(active_states, track, rng)

    # -- Build result ---------------------------------------------------------
    active_sorted = sorted(
        [s for s in states if s.active], key=lambda s: s.cumulative_time
    )
    dnf_sorted = [s for s in states if not s.active]

    classification: list[str] = [s.driver.name for s in active_sorted] + [
        s.driver.name for s in dnf_sorted
    ]
    dnf_names: list[str] = [s.driver.name for s in dnf_sorted]
    lap_time_map: dict[str, list[float]] = {s.driver.name: s.lap_times for s in states}

    return RaceResult(
        final_classification=classification,
        dnf_list=dnf_names,
        lap_times=lap_time_map,
    )


# ---------------------------------------------------------------------------
# Overtake helper
# ---------------------------------------------------------------------------


_PASS_TIME_DELTA: float = 0.2  # seconds transferred on a successful overtake


def _apply_overtakes(
    ranked: list[_DriverState],
    track: Track,
    rng: Generator,
) -> None:
    """Evaluate logistic overtake probability for adjacent car pairs.

    For each consecutive pair (leading, trailing) in *ranked*:
        - If the cumulative time gap is < 1.0 s, compute a logistic pass
          probability based on the lap-time delta and the track's overtake
          coefficient.
        - If the random draw succeeds, the overtake is made *persistent* by
          adjusting cumulative times:

            trailer.cumulative_time = leader.cumulative_time - pass_time_delta
            leader.cumulative_time += pass_time_delta

          Cumulative times are clamped to a minimum of 0.0.
        - The two entries are swapped in *ranked* and the next comparison is
          skipped to prevent immediate re-swap oscillation.

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
                # Persistent time adjustment: place the overtaker ahead.
                trailer.cumulative_time = max(
                    0.0, leader.cumulative_time - _PASS_TIME_DELTA
                )
                leader.cumulative_time += _PASS_TIME_DELTA

                ranked[i], ranked[i + 1] = ranked[i + 1], ranked[i]
                # Skip next pair to avoid immediate re-swap oscillation.
                i += 2
                continue
        i += 1
