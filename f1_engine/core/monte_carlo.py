"""Monte Carlo race analytics engine for the F1 2026 simulation engine.

Runs many seeded replications of ``simulate_race`` and aggregates the
outcomes into probability distributions over winner, podium, finishing
position, and championship points.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from f1_engine.core.car import Car
from f1_engine.core.race import simulate_race
from f1_engine.core.track import Track

# Standard F1 points for positions 1-10.
_POINTS_TABLE: list[int] = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def simulate_race_monte_carlo(
    track: Track,
    cars: list[Car],
    laps: int,
    simulations: int,
    base_seed: int = 42,
) -> dict[str, Any]:
    """Run a Monte Carlo ensemble of race simulations.

    Each replication uses ``seed = base_seed + i`` so that:
      - Results are fully reproducible given the same ``base_seed``.
      - No global random state is modified.

    Collected statistics per team:
      - **Winner probability** -- fraction of simulations won.
      - **Podium probability** -- fraction of simulations finishing in the
        top 3.
      - **Expected finishing position** -- mean position across all runs.
      - **Expected championship points** -- mean points per the standard
        F1 scoring table ``[25, 18, 15, 12, 10, 8, 6, 4, 2, 1]``.
      - **Finish distribution** -- mapping from each finishing position to
        the probability of finishing there.

    Args:
        track: Circuit to simulate.
        cars: List of participating cars.
        laps: Race length in laps (>= 1).
        simulations: Number of Monte Carlo replications (>= 1).
        base_seed: Starting seed value.  Replication *i* uses
            ``base_seed + i``.

    Returns:
        Dictionary with keys:
            winner_probabilities  -- ``{team_name: float}``
            podium_probabilities  -- ``{team_name: float}``
            expected_position     -- ``{team_name: float}``
            expected_points       -- ``{team_name: float}``
            finish_distribution   -- ``{team_name: {position: float}}``

    Raises:
        ValueError: If simulations < 1.
    """
    if simulations < 1:
        raise ValueError("simulations must be >= 1.")

    team_names: list[str] = [c.team_name for c in cars]

    # Accumulators
    win_counts: dict[str, int] = defaultdict(int)
    podium_counts: dict[str, int] = defaultdict(int)
    position_sums: dict[str, int] = defaultdict(int)
    points_sums: dict[str, float] = defaultdict(float)
    position_counts: dict[str, dict[int, int]] = {
        name: defaultdict(int) for name in team_names
    }

    for i in range(simulations):
        seed: int = base_seed + i
        result = simulate_race(track, cars, laps, seed=seed)

        classification: list[str] = result.final_classification
        for pos_idx, name in enumerate(classification):
            position: int = pos_idx + 1  # 1-based

            # Winner
            if position == 1:
                win_counts[name] += 1

            # Podium
            if position <= 3:
                podium_counts[name] += 1

            # Position accumulator
            position_sums[name] += position

            # Points
            if pos_idx < len(_POINTS_TABLE):
                points_sums[name] += _POINTS_TABLE[pos_idx]

            # Position histogram
            position_counts[name][position] += 1

    # -- Normalise to probabilities -------------------------------------------
    inv: float = 1.0 / simulations

    winner_probabilities: dict[str, float] = {
        name: win_counts[name] * inv for name in team_names
    }
    podium_probabilities: dict[str, float] = {
        name: podium_counts[name] * inv for name in team_names
    }
    expected_position: dict[str, float] = {
        name: position_sums[name] * inv for name in team_names
    }
    expected_points: dict[str, float] = {
        name: points_sums[name] * inv for name in team_names
    }
    finish_distribution: dict[str, dict[int, float]] = {}
    for name in team_names:
        finish_distribution[name] = {
            pos: count * inv for pos, count in sorted(position_counts[name].items())
        }

    return {
        "winner_probabilities": winner_probabilities,
        "podium_probabilities": podium_probabilities,
        "expected_position": expected_position,
        "expected_points": expected_points,
        "finish_distribution": finish_distribution,
    }
