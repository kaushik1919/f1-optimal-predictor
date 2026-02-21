"""Finite-horizon dynamic programming pit-stop optimiser.

Replaces the grid-search approach in ``stint.py`` with backward induction
over a compact state space to find the globally optimal pit-stop strategy.

State tuple::

    (lap, tyre_age, compound_name)

* ``lap`` -- current lap number (0-based; lap 0 is the start, the race
  covers laps 0 .. total_laps - 1).
* ``tyre_age`` -- laps completed on the current set of tyres (0 after
  a fresh fit, increments by 1 each lap).
* ``compound_name`` -- string name of the currently fitted compound
  (``"SOFT"``, ``"MEDIUM"``, or ``"HARD"``).

Energy state and Safety Car effects are deliberately excluded to keep
the state space tractable.  The optimiser assumes ``deploy_level = 0``
(no ERS deployment) for cost evaluation, matching the conservative
baseline used by the grid-search optimiser.

The solver uses dictionary-based memoisation and backward induction.
State space size is bounded by
``total_laps * total_laps * 3`` entries (lap x tyre_age x compound),
which is comfortably small for any realistic race length (< 100 laps).
"""

from __future__ import annotations

from f1_engine.core.car import Car
from f1_engine.core.physics import lap_time as compute_lap_time
from f1_engine.core.race import PIT_LOSS
from f1_engine.core.strategy import Strategy
from f1_engine.core.track import Track
from f1_engine.core.tyre import HARD, MEDIUM, SOFT, TyreCompound

# ---------------------------------------------------------------------------
# Compound registry (name -> TyreCompound)
# ---------------------------------------------------------------------------

_COMPOUNDS: dict[str, TyreCompound] = {c.name: c for c in (SOFT, MEDIUM, HARD)}

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

# State: (lap, tyre_age, compound_name)
_State = tuple[int, int, str]

# Memo value: (cost-to-go, action)
#   action = None  means "continue on current tyres"
#   action = str   means "pit and switch to compound with this name"
_Action = str | None


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------


def _lap_cost(
    track: Track,
    car: Car,
    tyre_age: int,
    compound: TyreCompound,
) -> float:
    """Deterministic cost of completing one lap on the given tyres.

    Mirrors the race-engine calculation:
        base lap time (with deploy=0)
      + compound-scaled tyre degradation
      + compound pace delta
    """
    # Base lap time with zero tyre age (we add degradation ourselves)
    base: float = compute_lap_time(track, car, 0.0, 0.0)

    # Tyre degradation component, scaled by compound
    deg: float = (
        float(tyre_age)
        * track.tyre_degradation_factor
        * car.tyre_wear_rate
        * compound.degradation_rate
    )

    return base + deg + compound.base_pace_delta


# ---------------------------------------------------------------------------
# DP solver
# ---------------------------------------------------------------------------


def compute_optimal_strategy_dp(
    track: Track,
    car: Car,
    total_laps: int,
    starting_compound: TyreCompound = MEDIUM,
) -> Strategy:
    """Find the optimal pit-stop strategy via backward induction.

    Uses finite-horizon dynamic programming over the state space
    ``(lap, tyre_age, compound_name)``.  At each state the solver
    chooses the action that minimises total remaining race time:

    * **Continue** -- drive the current lap on existing tyres, then
      transition to ``(lap+1, tyre_age+1, compound)``.
    * **Pit to compound X** -- incur ``PIT_LOSS`` plus a fresh-tyre
      lap, then transition to ``(lap+1, 1, X)``.

    Pitting is not allowed on the first lap (lap 0) or the last lap
    (lap ``total_laps - 1``) to avoid degenerate solutions.

    Args:
        track: Circuit to optimise for.
        car: Car whose physics parameters drive the cost model.
        total_laps: Number of race laps (>= 1).
        starting_compound: Compound fitted at the start of the race.

    Returns:
        A :class:`Strategy` with ``deploy_level=0.0``,
        ``harvest_level=0.0``, the optimal ``compound_sequence``,
        and the optimal ``pit_laps`` (1-based lap numbers).

    Raises:
        ValueError: If ``total_laps < 1``.
    """
    if total_laps < 1:
        raise ValueError("total_laps must be >= 1.")

    compound_names: list[str] = list(_COMPOUNDS.keys())

    # -- Memoisation table ----------------------------------------------------
    # V[state] = (cost_to_go, action)
    #   action = None  -> continue
    #   action = "SOFT" / "MEDIUM" / "HARD" -> pit to that compound
    memo: dict[_State, tuple[float, _Action]] = {}

    # -- Base case: after the last lap, cost-to-go is zero -------------------
    for age in range(total_laps + 1):
        for cname in compound_names:
            memo[(total_laps, age, cname)] = (0.0, None)

    # -- Backward induction ---------------------------------------------------
    for lap in range(total_laps - 1, -1, -1):
        for tyre_age in range(total_laps + 1):
            for cname in compound_names:
                compound = _COMPOUNDS[cname]

                # Option 1: Continue on current tyres
                continue_cost: float = _lap_cost(track, car, tyre_age, compound)
                future_age = min(tyre_age + 1, total_laps)
                continue_cost += memo[(lap + 1, future_age, cname)][0]

                best_cost: float = continue_cost
                best_action: _Action = None

                # Option 2: Pit to each available compound
                # Disallow pitting on the first or last lap.
                if 0 < lap < total_laps - 1:
                    for new_cname in compound_names:
                        new_compound = _COMPOUNDS[new_cname]
                        pit_cost: float = PIT_LOSS
                        # Lap driven on fresh tyres (age 0)
                        pit_cost += _lap_cost(track, car, 0, new_compound)
                        pit_cost += memo[(lap + 1, 1, new_cname)][0]

                        if pit_cost < best_cost:
                            best_cost = pit_cost
                            best_action = new_cname

                memo[(lap, tyre_age, cname)] = (best_cost, best_action)

    # -- Policy extraction (forward pass) -------------------------------------
    pit_laps: list[int] = []
    compounds: list[TyreCompound] = [starting_compound]

    current_cname: str = starting_compound.name
    current_age: int = 0

    for lap in range(total_laps):
        _, action = memo[(lap, current_age, current_cname)]

        if action is not None:
            # Pit on this lap: record 1-based lap number
            pit_laps.append(lap + 1)  # convert 0-based to 1-based
            current_cname = action
            compounds.append(_COMPOUNDS[current_cname])
            current_age = 1  # just drove one lap on fresh tyres
        else:
            current_age += 1

    return Strategy(
        deploy_level=0.0,
        harvest_level=0.0,
        compound_sequence=tuple(compounds),
        pit_laps=tuple(pit_laps),
    )
